"""
train_models_transformer_experiment.py
-----------------------------------------
OPTIONAL advanced experiment: compares the classical models against a
REAL pretrained transformer embedding (sentence-transformers,
'all-MiniLM-L6-v2'). Needs internet access to huggingface.co to download
weights -- degrades gracefully if unavailable, never required for the
rest of the app to run.

Usage:
    pip install sentence-transformers --break-system-packages
    python3 train_models_transformer_experiment.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "seed_claims_combined.csv"
LABELS = ["context_sufficient", "needs_verification", "high_priority"]
MODEL_NAME = "all-MiniLM-L6-v2"


def load_dataset():
    claims, labels = [], []
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            claims.append(row["claim"])
            labels.append(row["label"])
    return claims, labels


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers is not installed.")
        print("Install: pip install sentence-transformers --break-system-packages")
        sys.exit(0)

    print(f"Loading pretrained model '{MODEL_NAME}'...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Could not load/download the pretrained model: {e}")
        sys.exit(0)

    claims, labels = load_dataset()
    print(f"Encoding {len(claims)} claims with {MODEL_NAME}...")
    embeddings = model.encode(claims, show_progress_bar=False)

    idx = list(range(len(claims)))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores, acc_scores = [], []
    for tr_i, va_i in skf.split(X_train_s, y_train):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_train_s[tr_i], [y_train[i] for i in tr_i])
        preds = clf.predict(X_train_s[va_i])
        yva = [y_train[i] for i in va_i]
        acc_scores.append(accuracy_score(yva, preds))
        _, _, f1, _ = precision_recall_fscore_support(yva, preds, average="macro", zero_division=0, labels=LABELS)
        f1_scores.append(f1)

    print(f"\nSentence-Transformer ({MODEL_NAME}) + Logistic Regression")
    print(f"5-fold CV: acc={np.mean(acc_scores):.3f}+/-{np.std(acc_scores):.3f}  f1_macro={np.mean(f1_scores):.3f}+/-{np.std(f1_scores):.3f}")

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    _, _, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0, labels=LABELS)
    print(f"Held-out test: acc={acc:.3f}  f1_macro={f1:.3f}")
    print("\nCompare against models/model_comparison.json -- with n~175, transformer "
          "embeddings often do NOT outperform TF-IDF + engineered features.")


if __name__ == "__main__":
    main()
