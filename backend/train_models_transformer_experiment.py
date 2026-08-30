"""
train_models_transformer_experiment.py
-----------------------------------------
OPTIONAL "advanced experiment": compares the classical models against a
REAL pretrained transformer embedding (sentence-transformers,
'all-MiniLM-L6-v2') on the same 3-way classification task.

Why this is a separate, optional script rather than folded into
train_models.py: sentence-transformers needs to download ~90MB of
pretrained weights from huggingface.co the first time it runs. The
development sandbox this project was built in only allow-lists a short list
of package registries (pypi, npm, GitHub) and does NOT allow-list
huggingface.co, so this script could not be run or verified there. It WILL
run correctly on a normal machine with unrestricted internet access (e.g.
your own laptop). This is disclosed here rather than silently assumed --
see README "Dataset & Evaluation" for the full explanation.

Usage:
    pip install sentence-transformers --break-system-packages
    python3 train_models_transformer_experiment.py

If sentence-transformers isn't installed, or the model can't be downloaded
(no internet / blocked host), this script prints a clear message and exits
without crashing the rest of the project -- it is never required for
app.py to run.
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
        print("Install it with: pip install sentence-transformers --break-system-packages")
        print("Skipping the transformer-embedding experiment (this is optional; the rest "
              "of ClaimGuard does not depend on it).")
        sys.exit(0)

    print(f"Loading pretrained model '{MODEL_NAME}' (downloads from huggingface.co on first run)...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Could not load/download the pretrained model: {e}")
        print("This usually means no internet access to huggingface.co from this machine. "
              "Skipping the transformer-embedding experiment.")
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
        _, _, f1, _ = precision_recall_fscore_support(yva, preds, average="macro",
                                                        zero_division=0, labels=LABELS)
        f1_scores.append(f1)

    print(f"\nSentence-Transformer ({MODEL_NAME}) + Logistic Regression")
    print(f"5-fold CV: acc={np.mean(acc_scores):.3f}+/-{np.std(acc_scores):.3f}  "
          f"f1_macro={np.mean(f1_scores):.3f}+/-{np.std(f1_scores):.3f}")

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    _, _, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro",
                                                    zero_division=0, labels=LABELS)
    print(f"Held-out test: acc={acc:.3f}  f1_macro={f1:.3f}")
    print("\nCompare these numbers against models/model_comparison.json (classical models) "
          "to see whether real transformer embeddings actually help on this dataset size -- "
          "with n~175, they often do NOT outperform TF-IDF + engineered features, which is "
          "itself a useful, reportable finding for the write-up.")


if __name__ == "__main__":
    main()
