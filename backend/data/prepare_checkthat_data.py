"""
prepare_checkthat_data.py
--------------------------
Processes the real, human-annotated CLEF CheckThat! 2019 Task 1 dataset
into a clean CSV. Source: Elsayed, Nakov, Barron-Cedeno, Hasanain,
Suwaileh, Da San Martino, Atanasova, CLEF 2019.
Repo: https://github.com/apepa/clef2019-factchecking-task1
License: "free for general research use" (per repo README).

check_worthy=1 means fact-checkers selected this sentence for
verification. check_worthy=0 does NOT mean "verified true" -- it just
means fact-checkers didn't select it. We use this to train a dedicated
binary check-worthiness model, not to relabel our 3-way scheme.

Run: python3 prepare_checkthat_data.py
Produces: checkthat_sentences.csv
"""

import csv
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "checkthat_raw")
OUT_PATH = os.path.join(HERE, "checkthat_sentences.csv")


def clean_text(t):
    t = t.replace("\r", "").strip()
    return re.sub(r"\s+", " ", t)


def looks_claimlike(t):
    words = t.split()
    if len(words) < 4:
        return False
    if t.strip().endswith("?"):
        return False
    low = t.lower()
    if re.match(r"^(thank you|thanks|good evening|good morning|welcome|please|"
                r"let's|let us|next question|mr\.|mrs\.|ms\.)\b", low):
        return False
    return True


def main():
    rows = []
    for split_name, subdir in [("train", "training"), ("test", "test_annotated")]:
        pattern = os.path.join(RAW_DIR, subdir, "*.tsv")
        for path in sorted(glob.glob(pattern)):
            fname = os.path.basename(path)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 4:
                        continue
                    _, speaker, text, label = parts[0], parts[1], parts[2], parts[3]
                    text = clean_text(text)
                    label = label.strip()
                    if label not in ("0", "1") or not looks_claimlike(text):
                        continue
                    rows.append((text, fname, split_name, int(label)))

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "source_file", "split", "checkworthy"])
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r[3] == 1)
    print(f"Wrote {len(rows)} claim-like sentences to {OUT_PATH}")
    print(f"  check-worthy (1): {n_pos}  ({n_pos/len(rows)*100:.1f}%)")
    print(f"  not check-worthy (0): {len(rows)-n_pos}  ({(len(rows)-n_pos)/len(rows)*100:.1f}%)")
    n_train = sum(1 for r in rows if r[2] == "train")
    n_test = sum(1 for r in rows if r[2] == "test")
    print(f"  train split: {n_train}   test split: {n_test}")


if __name__ == "__main__":
    main()
