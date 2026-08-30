"""
train_models.py  (v2)
-----------------------
Research deliverables implemented here:
  - Baseline implementation
  - Controlled experiments across FIVE models (Logistic Regression, Random
    Forest, Linear SVM, Multinomial Naive Bayes, and a dense-embedding
    baseline: TF-IDF -> Truncated SVD/LSA -> Logistic Regression)
  - Proper evaluation via stratified 5-fold cross-validation (not a single
    lucky/unlucky train-test split) given the still-modest dataset size
  - Blending in a REAL signal: each claim's probability under the
    check-worthiness model trained on real CLEF CheckThat! 2019 data
    (train_checkworthy_model.py) is added as an extra engineered feature,
    so the fine-grained classifier is no longer trained on synthetic
    labels in total isolation from real-world annotated data.
  - Evaluation & error analysis (per-fold metrics, confusion matrix on a
    held-out split, and a misclassification log with a guessed error
    category)

On an "advanced experiment" transformer/embedding comparison: real
pretrained transformer embeddings (e.g. sentence-transformers) require
downloading model weights from huggingface.co, which this development
sandbox cannot reach (see README). The self-contained LSA/SVD dense
embedding baseline below IS run and compared here. A guarded, OPTIONAL
code path for real sentence-transformer embeddings is provided in
train_models_transformer_experiment.py -- it will run correctly on a
machine with normal internet access (i.e. the student's own laptop) and
is skipped gracefully otherwise.

Run:
    python3 train_checkworthy_model.py   # must run first
    python3 train_models.py
"""

import csv
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from features import engineer_features

HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "seed_claims_combined.csv"
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)
CHECKWORTHY_MODEL_PATH = MODELS_DIR / "checkworthy_model.pkl"

LABELS = ["context_sufficient", "needs_verification", "high_priority"]
BASE_FEATURE_KEYS = [
    "n_words", "future_marker_count", "present_temporal_count", "conditional_marker_count",
    "hedge_marker_count", "superlative_count", "implicit_causal_count", "stable_fact_hint_count",
    "numeric_count", "year_count", "future_year_flag", "money_hits", "has_and",
    "has_conjunction_split", "is_health", "is_financial", "is_political",
]
# The extra feature blended in from the real check-worthiness model.
FEATURE_KEYS = BASE_FEATURE_KEYS + ["checkworthy_prob"]


def load_dataset():
    claims, categories, labels = [], [], []
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            claims.append(row["claim"])
            categories.append(row["category"])
            labels.append(row["label"])
    return claims, categories, labels


def load_checkworthy_scorer():
    if not CHECKWORTHY_MODEL_PATH.exists():
        print("WARNING: checkworthy_model.pkl not found -- run train_checkworthy_model.py first. "
              "Falling back to 0.0 for the checkworthy_prob feature.")
        return None
    with open(CHECKWORTHY_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def score_checkworthy(texts, bundle):
    if bundle is None:
        return np.zeros(len(texts))
    tfidf = bundle["vectorizer"].transform(texts)
    eng_rows = []
    for t in texts:
        feats, _, _ = engineer_features(t)
        eng_rows.append([feats[k] for k in bundle["feature_keys"]])
    eng_arr = bundle["scaler"].transform(np.array(eng_rows, dtype=float))
    X = hstack([tfidf, csr_matrix(eng_arr)])
    return bundle["model"].predict_proba(X)[:, 1]


def build_feature_matrix(claims, vectorizer, scaler, cw_bundle, fit=False):
    tfidf = vectorizer.fit_transform(claims) if fit else vectorizer.transform(claims)
    cw_probs = score_checkworthy(claims, cw_bundle)
    eng_rows = []
    for c, cw in zip(claims, cw_probs):
        feats, _, _ = engineer_features(c)
        row = [feats[k] for k in BASE_FEATURE_KEYS] + [cw]
        eng_rows.append(row)
    eng_arr = np.array(eng_rows, dtype=float)
    eng_arr = scaler.fit_transform(eng_arr) if fit else scaler.transform(eng_arr)
    return hstack([tfidf, csr_matrix(eng_arr)])


def guess_error_category(feats, pred_label, true_label):
    if feats["future_marker_count"] or feats["future_year_flag"]:
        if pred_label == "context_sufficient":
            return "Temporal misunderstanding (missed future/time-dependent marker)"
    if feats["numeric_count"] >= 1 and pred_label != true_label:
        return "Numerical claim error (statistical stakes misjudged)"
    if feats["has_conjunction_split"]:
        return "Compound claim error (sub-claims likely conflated)"
    if feats["hedge_marker_count"] or feats["conditional_marker_count"]:
        return "Conditional/hedged language misread"
    return "Ambiguous language / insufficient lexical signal"


def make_lsa_pipeline():
    """The self-contained 'dense embedding' baseline: TF-IDF -> Truncated
    SVD (Latent Semantic Analysis) collapses sparse TF-IDF vectors into a
    dense low-dimensional embedding space -- the classical pre-transformer
    way of getting dense text embeddings without downloading pretrained
    weights. Documented explicitly as NOT a transformer -- see module
    docstring for why real transformer embeddings aren't run in this
    sandboxed environment."""
    return Pipeline([
        ("svd", TruncatedSVD(n_components=40, random_state=42)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def extract_feature_importance(model, vectorizer, top_k=12):
    """Best-effort feature importance for the Research Analytics Dashboard.
    Only meaningful for linear models (coef_) or tree ensembles
    (feature_importances_); returns None otherwise rather than fabricating
    numbers for a model type that doesn't support it."""
    if model is None:
        return None
    tfidf_terms = vectorizer.get_feature_names_out().tolist()
    all_terms = tfidf_terms + FEATURE_KEYS

    if hasattr(model, "coef_"):
        coefs = model.coef_
        classes = model.classes_
        result = {}
        for i, cls in enumerate(classes):
            row = coefs[i] if coefs.ndim > 1 else coefs
            if len(row) != len(all_terms):
                continue
            top_idx = np.argsort(-np.abs(row))[:top_k]
            result[str(cls)] = [{"term": all_terms[j], "weight": round(float(row[j]), 4)}
                                 for j in top_idx]
        return result if result else None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if len(importances) != len(all_terms):
            return None
        top_idx = np.argsort(-importances)[:top_k]
        return {"overall": [{"term": all_terms[j], "weight": round(float(importances[j]), 4)}
                             for j in top_idx]}
    return None


def main():
    claims, categories, labels = load_dataset()
    cw_bundle = load_checkworthy_scorer()
    print(f"Loaded {len(claims)} claims. Checkworthy scorer: "
          f"{'loaded' if cw_bundle else 'MISSING (feature will be 0)'}")

    idx = list(range(len(claims)))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
    train_claims = [claims[i] for i in train_idx]
    test_claims = [claims[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
    scaler = MinMaxScaler()

    X_train = build_feature_matrix(train_claims, vectorizer, scaler, cw_bundle, fit=True)
    X_test = build_feature_matrix(test_claims, vectorizer, scaler, cw_bundle, fit=False)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced", max_iter=5000),
        "MultinomialNB": MultinomialNB(),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, clf in candidates.items():
        f1_scores, acc_scores = [], []
        for tr_i, va_i in skf.split(X_train, y_train):
            clf_fold = type(clf)(**clf.get_params())
            Xtr, Xva = X_train[tr_i], X_train[va_i]
            ytr = [y_train[i] for i in tr_i]
            yva = [y_train[i] for i in va_i]
            clf_fold.fit(Xtr, ytr)
            preds = clf_fold.predict(Xva)
            acc_scores.append(accuracy_score(yva, preds))
            _, _, f1, _ = precision_recall_fscore_support(
                yva, preds, average="macro", zero_division=0, labels=LABELS)
            f1_scores.append(f1)
        cv_results[name] = {"cv_f1_mean": round(float(np.mean(f1_scores)), 4),
                             "cv_f1_std": round(float(np.std(f1_scores)), 4),
                             "cv_acc_mean": round(float(np.mean(acc_scores)), 4),
                             "cv_acc_std": round(float(np.std(acc_scores)), 4)}
        print(f"{name:20s} 5-fold CV: acc={np.mean(acc_scores):.3f}+/-{np.std(acc_scores):.3f}  "
              f"f1_macro={np.mean(f1_scores):.3f}+/-{np.std(f1_scores):.3f}")

    # LSA/embedding baseline needs its own dense pipeline (SVD doesn't
    # combine cleanly with the sparse hstack matrix above), so it is
    # evaluated the same way but on a TF-IDF-only representation.
    tfidf_only_vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = tfidf_only_vec.fit_transform(train_claims)
    f1_scores, acc_scores = [], []
    for tr_i, va_i in skf.split(X_train_tfidf, y_train):
        pipe = make_lsa_pipeline()
        Xtr, Xva = X_train_tfidf[tr_i], X_train_tfidf[va_i]
        ytr = [y_train[i] for i in tr_i]
        yva = [y_train[i] for i in va_i]
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xva)
        acc_scores.append(accuracy_score(yva, preds))
        _, _, f1, _ = precision_recall_fscore_support(yva, preds, average="macro", zero_division=0, labels=LABELS)
        f1_scores.append(f1)
    cv_results["TFIDF_LSA_Embedding"] = {"cv_f1_mean": round(float(np.mean(f1_scores)), 4),
                                          "cv_f1_std": round(float(np.std(f1_scores)), 4),
                                          "cv_acc_mean": round(float(np.mean(acc_scores)), 4),
                                          "cv_acc_std": round(float(np.std(acc_scores)), 4)}
    print(f"{'TFIDF_LSA_Embedding':20s} 5-fold CV: acc={np.mean(acc_scores):.3f}+/-{np.std(acc_scores):.3f}  "
          f"f1_macro={np.mean(f1_scores):.3f}+/-{np.std(f1_scores):.3f}")

    best_name = max(cv_results, key=lambda n: cv_results[n]["cv_f1_mean"])
    print(f"\nBest by CV F1: {best_name}")

    results = {}
    fitted_models = {}
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0, labels=LABELS)
        cm = confusion_matrix(y_test, preds, labels=LABELS).tolist()
        results[name] = {**cv_results[name], "test_accuracy": round(acc, 4),
                          "test_precision_macro": round(precision, 4),
                          "test_recall_macro": round(recall, 4),
                          "test_f1_macro": round(f1, 4),
                          "confusion_matrix": cm, "labels_order": LABELS}
        fitted_models[name] = clf

    lsa_pipe = make_lsa_pipeline()
    lsa_pipe.fit(X_train_tfidf, y_train)
    X_test_tfidf = tfidf_only_vec.transform(test_claims)
    lsa_preds = lsa_pipe.predict(X_test_tfidf)
    acc = accuracy_score(y_test, lsa_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, lsa_preds, average="macro", zero_division=0, labels=LABELS)
    results["TFIDF_LSA_Embedding"] = {**cv_results["TFIDF_LSA_Embedding"],
                                       "test_accuracy": round(acc, 4),
                                       "test_precision_macro": round(precision, 4),
                                       "test_recall_macro": round(recall, 4),
                                       "test_f1_macro": round(f1, 4),
                                       "confusion_matrix": confusion_matrix(y_test, lsa_preds, labels=LABELS).tolist(),
                                       "labels_order": LABELS}

    best_model = fitted_models.get(best_name)
    print(classification_report(y_test, (best_model.predict(X_test) if best_model else lsa_preds),
                                 labels=LABELS, zero_division=0))

    error_log = []
    if best_model is not None:
        preds_best = best_model.predict(X_test)
        for claim, true_label, pred_label in zip(test_claims, y_test, preds_best):
            if true_label != pred_label:
                feats, _, _ = engineer_features(claim)
                error_log.append({"claim": claim, "true_label": true_label,
                                   "predicted_label": pred_label,
                                   "error_category": guess_error_category(feats, pred_label, true_label)})

    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump({"results": results, "best_model": best_name, "n_train": len(train_claims),
                    "n_test": len(test_claims),
                    "evaluation": "5-fold stratified CV on the train split (mean+/-std reported), "
                                   "plus a final held-out 20% test split",
                    "dataset": str(DATA_PATH.name),
                    "checkworthy_feature_blended": cw_bundle is not None}, f, indent=2)

    with open(MODELS_DIR / "error_analysis.json", "w") as f:
        json.dump({"n_errors": len(error_log), "errors": error_log}, f, indent=2)

    # ---- feature importance (best-effort; only meaningful for linear models) ----
    feature_importance = extract_feature_importance(best_model, vectorizer)
    if feature_importance:
        with open(MODELS_DIR / "feature_importance.json", "w") as f:
            json.dump(feature_importance, f, indent=2)
        print(f"Saved feature_importance.json ({len(feature_importance)} labels)")

    if best_model is not None:
        with open(MODELS_DIR / "best_model.pkl", "wb") as f:
            pickle.dump({"model": best_model, "model_name": best_name, "vectorizer": vectorizer,
                         "scaler": scaler, "feature_keys": FEATURE_KEYS, "labels": LABELS,
                         "uses_checkworthy_feature": True}, f)
        print(f"\nSaved best_model.pkl ({best_name})")

        # Save ALL fitted classical models (not just the best) for the
        # Model Decision Comparison / consensus feature -- lets the UI show
        # what each model individually predicted and flag disagreement.
        with open(MODELS_DIR / "all_models.pkl", "wb") as f:
            pickle.dump({"models": fitted_models, "vectorizer": vectorizer, "scaler": scaler,
                         "feature_keys": FEATURE_KEYS, "labels": LABELS}, f)
        print(f"Saved all_models.pkl ({len(fitted_models)} models: {', '.join(fitted_models.keys())})")
    else:
        print("\nBest model was the LSA pipeline -- best_model.pkl not overwritten "
              "(claim_analyzer.py expects the hstack-feature format used by the classical models).")

    print(f"Saved model_comparison.json, error_analysis.json ({len(error_log)} errors)")


if __name__ == "__main__":
    main()
