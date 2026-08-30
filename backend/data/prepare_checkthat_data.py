"""
prepare_checkthat_data.py
--------------------------
Processes the real, human-annotated CLEF CheckThat! 2019 Task 1 dataset
(political debate/speech transcripts, sentence-level check-worthiness labels
assigned by professional fact-checkers referencing FactCheck.org) into a
clean CSV usable for training.

SOURCE & LICENSE:
  Elsayed, Nakov, Barron-Cedeno, Hasanain, Suwaileh, Da San Martino, Atanasova.
  "Overview of the CLEF-2019 CheckThat!: Automatic Identification and
  Verification of Claims." CLEF 2019.
  Repo: https://github.com/apepa/clef2019-factchecking-task1
  License (per repo README): "These datasets are free for general research use."
  Raw files are bundled under backend/data/checkthat_raw/ for offline reproducibility.

WHAT THE LABEL MEANS (and does NOT mean):
  check_worthy=1 means professional fact-checkers selected this sentence for
  verification against FactCheck.org. This is real-world evidence for our
  "needs external verification" concept. It is NOT the same as our full
  3-way scheme (context_sufficient / needs_verification / high_priority) --
  see README for the documented mapping methodology. check_worthy=0 does
  NOT mean "verified true" or "stable fact" -- it just means fact-checkers
  didn't select it, which includes filler, opinion, and mundane claims.
  We therefore use this dataset primarily to train a dedicated, honestly-
  scoped BINARY check-worthiness model (checkworthy_model.pkl), and only
  secondarily blend it into the 3-way scheme as an additional engineered
  feature -- we do not silently relabel check_worthy=0 as "sufficient."

Run:
    python3 prepare_checkthat_data.py
Produces:
    checkthat_sentences.csv  (columns: text, source_file, split, checkworthy)
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
    t = re.sub(r"\s+", " ", t)
    return t


def looks_claimlike(t):
    """Minimal quality filter -- drop fragments too short to be a claim,
    pure greetings/procedural debate chatter, and questions."""
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
    # training/ files -> split="train", test_annotated/ files -> split="test"
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
                    if label not in ("0", "1"):
                        continue
                    if not looks_claimlike(text):
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
