"""
train_checkworthy_model.py
----------------------------
Trains a BINARY check-worthiness classifier on the real, human-annotated
CLEF CheckThat! 2019 dataset (~17.6k claim-like sentences from US political
debates/speeches, 2016-2019, professionally fact-checked; see
data/prepare_checkthat_data.py for source, license, and label-meaning notes).

Why a separate model from the 3-way classifier (train_models.py):
the two datasets label different things. CheckThat labels "would a
professional fact-checker select this for verification" (binary,
real-world-grounded). Our synthetic seed set labels a finer-grained
"context_sufficient / needs_verification / high_priority" scheme that no
public dataset provides. Conflating them would mean pretending the CheckThat
label IS our 3-way scheme, which it isn't (see prepare_checkthat_data.py
docstring). Instead we train this model honestly on its own real label,
then feed its probability output into the 3-way model as one additional
engineered feature (see train_models.py / claim_analyzer.py) -- a
legitimate way to blend a small real signal into the larger synthetic task
without misrepresenting either dataset.

Evaluation notes (imbalanced data -- ~3% positive class):
  - We use the dataset's OWN train/test split, which is split by debate
    FILE (not by random sentence), so no vocabulary/topic leaks between
    train and test -- the correct way to evaluate on this kind of data.
  - Accuracy is not reported as the headline metric (a model that always
    predicts "not check-worthy" would score ~97% and be useless). We
    report precision/recall/F1 on the check-worthy (positive) class, plus
    ROC-AUC and average precision (PR-AUC), which are the right measures
    for a rare positive class.
  - class_weight='balanced' is used to counter the ~32:1 imbalance instead
    of naively undersampling, which would throw away real, hard-won label
    information.

Run:
    python3 train_checkworthy_model.py
Produces (models/):
    checkworthy_model.pkl
    checkworthy_metrics.json
"""

import csv
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                              average_precision_score, confusion_matrix, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from features import engineer_features

HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "checkthat_sentences.csv"
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_KEYS = [
    "n_words", "future_marker_count", "present_temporal_count", "conditional_marker_count",
    "hedge_marker_count", "superlative_count", "implicit_causal_count", "stable_fact_hint_count",
    "numeric_count", "year_count", "future_year_flag", "money_hits", "has_and",
    "has_conjunction_split", "is_health", "is_financial", "is_political",
]


def load_split():
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts = train_texts if row["split"] == "train" else test_texts
            labels = train_labels if row["split"] == "train" else test_labels
            texts.append(row["text"])
            labels.append(int(row["checkworthy"]))
    return train_texts, train_labels, test_texts, test_labels


def build_matrix(texts, vectorizer, scaler, fit=False):
    tfidf = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
    eng_rows = []
    for t in texts:
        feats, _, _ = engineer_features(t)
        eng_rows.append([feats[k] for k in FEATURE_KEYS])
    eng_arr = np.array(eng_rows, dtype=float)
    eng_arr = scaler.fit_transform(eng_arr) if fit else scaler.transform(eng_arr)
    return hstack([tfidf, csr_matrix(eng_arr)])


def main():
    train_texts, y_train, test_texts, y_test = load_split()
    print(f"Train: {len(train_texts)} sentences ({sum(y_train)} check-worthy)")
    print(f"Test:  {len(test_texts)} sentences ({sum(y_test)} check-worthy)")

    # Carve a validation split OUT OF TRAINING ONLY, used solely to pick a
    # decision threshold. The test set stays untouched until final scoring --
    # tuning the threshold on the test set would be a subtle form of leakage.
    tr_texts, val_texts, y_tr, y_val = train_test_split(
        train_texts, y_train, test_size=0.15, random_state=42, stratify=train_labels_safe(y_train))

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                                  stop_words="english", min_df=2)
    scaler = MinMaxScaler()

    X_tr = build_matrix(tr_texts, vectorizer, scaler, fit=True)
    X_val = build_matrix(val_texts, vectorizer, scaler, fit=False)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(X_tr, y_tr)

    val_probs = clf.predict_proba(X_val)[:, 1]
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_val, val_probs)
    f1_curve = np.where((prec_curve + rec_curve) > 0,
                         2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-12), 0)
    best_idx = int(np.argmax(f1_curve[:-1])) if len(thresholds) else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
    print(f"Threshold tuned on validation split: {best_threshold:.3f} "
          f"(val F1={f1_curve[best_idx]:.3f})")

    # Refit on ALL training data (tr+val) now that the threshold is fixed,
    # then evaluate once, cleanly, on the untouched test set.
    X_train_full = build_matrix(train_texts, vectorizer, scaler, fit=True)
    X_test = build_matrix(test_texts, vectorizer, scaler, fit=False)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(X_train_full, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    preds_default = (probs >= 0.5).astype(int)
    preds_tuned = (probs >= best_threshold).astype(int)

    def score(preds):
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        return round(p, 4), round(r, 4), round(f1, 4)

    p05, r05, f105 = score(preds_default)
    pT, rT, f1T = score(preds_tuned)
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    cm_tuned = confusion_matrix(y_test, preds_tuned).tolist()

    metrics = {
        "n_train": len(train_texts), "n_train_positive": int(sum(y_train)),
        "n_test": len(test_texts), "n_test_positive": int(sum(y_test)),
        "roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4),
        "threshold_0.5": {"precision": p05, "recall": r05, "f1": f105},
        "threshold_tuned": {"value": round(best_threshold, 3),
                             "precision": pT, "recall": rT, "f1": f1T},
        "confusion_matrix_tuned": cm_tuned,
        "confusion_matrix_labels": ["not_checkworthy", "checkworthy"],
        "source": "CLEF CheckThat! 2019 Task 1 (Elsayed et al., CLEF 2019) -- real human annotations",
    }

    print(f"\n@0.5 threshold:   P={p05:.3f} R={r05:.3f} F1={f105:.3f}")
    print(f"@tuned threshold ({best_threshold:.3f}): P={pT:.3f} R={rT:.3f} F1={f1T:.3f}")
    print(f"ROC-AUC={roc_auc:.3f}  PR-AUC={pr_auc:.3f} (baseline/random ~= {sum(y_test)/len(y_test):.3f})")
    print("Confusion matrix (tuned) [[TN,FP],[FN,TP]]:", cm_tuned)

    with open(MODELS_DIR / "checkworthy_model.pkl", "wb") as f:
        pickle.dump({"model": clf, "vectorizer": vectorizer, "scaler": scaler,
                     "feature_keys": FEATURE_KEYS, "threshold": best_threshold}, f)
    with open(MODELS_DIR / "checkworthy_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved checkworthy_model.pkl and checkworthy_metrics.json")


def train_labels_safe(labels):
    """Guard stratify= against the (unlikely) case of too few positives."""
    from collections import Counter
    c = Counter(labels)
    if min(c.values()) < 2:
        return None
    return labels


if __name__ == "__main__":
    main()
