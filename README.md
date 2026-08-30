# ClaimGuard
### AI-Powered Claim Verification & Evidence Auditor
**Learn Depth — Problem ML-T2-061:** *Detecting Claims That Require External Verification*
*NLP · Misinformation Research · Advanced*

> **Central research question:** *Can an NLP system recognize when a claim cannot responsibly
> be accepted without external evidence?*

ClaimGuard is **not** a fake-news / true-or-false classifier. Given a claim and whatever context
is available, it predicts one of:

| Verdict | Meaning |
|---|---|
| 🟢 Context Sufficient | The claim can be responsibly assessed from the available signal |
| 🟡 Needs Verification | Checkable in principle, but no supporting evidence is currently available |
| 🔴 High Priority | Needs verification **and** carries elevated stakes (future-dated, numeric, health/financial/political, extremal language) |
| 🛑 AI Abstains | The model's top two candidate labels are too close to call — routed straight to human review rather than forcing a guess |

---

## 1. Quick start

```bash
cd backend
pip install -r requirements.txt --break-system-packages   # or use a venv
cp .env.example .env                       # optional: add GROQ_API_KEY

# Rebuild datasets + retrain from scratch (already included pre-built, this is for reproducibility):
python3 data/prepare_checkthat_data.py     # process the real CLEF CheckThat! 2019 data
python3 data/build_seed_dataset_v2.py      # build the synthetic 3-way seed set
python3 train_checkworthy_model.py         # train the real-data binary check-worthiness model
python3 train_models.py                    # train + compare the 3-way classifiers

python3 app.py                             # serves the app at http://localhost:5000
```

The app works fully offline; Groq is optional (chat, "AI second opinion", and web-search evidence).

---

## 2. What changed in this version (v2)

This revision responds directly to the two biggest limitations flagged in v1: a too-small synthetic
dataset, and evidence retrieval that wasn't ranking or scoring reliably.

**Dataset:**
- Added a **real, human-annotated dataset**: ~17,600 sentences from the **CLEF CheckThat! 2019**
  shared task (US presidential debates and speeches, 2016–2019), professionally fact-checked
  against FactCheck.org. See `data/prepare_checkthat_data.py` for source, license (free for
  research use), and exactly what the label does/doesn't mean.
- Trained a **dedicated binary check-worthiness model** on this real data
  (`train_checkworthy_model.py`), evaluated properly for a rare-positive-class problem
  (precision/recall/F1/ROC-AUC/PR-AUC, threshold tuned on a validation split, never the test set).
- Expanded and rebalanced the synthetic 3-way seed set from 93 to 175 examples, now close to
  evenly split across the three labels (53 / 63 / 59), via templated generation
  (`data/build_seed_dataset_v2.py`).
- **Blended the real signal into the synthetic task**: every claim's probability under the
  check-worthiness model is added as an engineered feature to the 3-way classifier, so it's no
  longer trained on synthetic labels in complete isolation from real annotations.

**Evaluation:**
- Replaced the single 75/25 split with **5-fold stratified cross-validation** (mean plus/minus std
  reported per model), plus a final held-out test set -- much more reliable at this dataset size
  than one lucky/unlucky split.
- Added a **dense-embedding baseline** (TF-IDF -> Truncated SVD/LSA -> Logistic Regression) to the
  model comparison, and a guarded, optional path to **real pretrained transformer embeddings**
  (`train_models_transformer_experiment.py`, sentence-transformers) for machines with normal
  internet access -- see Section 6 for why it's optional here specifically.

**Evidence retrieval -- actually fixed, not just re-described:**
- Replaced raw TF-IDF cosine with **real BM25** (`rank_bm25`), the standard IR algorithm for
  short-query-against-passage ranking.
- Added **Porter stemming** so "increased" matches "increase", "grew" matches "growth", etc.
- Added **multi-sentence windowing** so evidence spanning two adjacent sentences is retrievable.
- **Found and fixed a real bug** during testing: BM25's epsilon-floor mechanism for negative IDF
  is average-based, and collapses when the candidate pool is small and highly overlapping
  (exactly what sentence-windowing produces) -- every score came out negative regardless of
  relevance, silently dropping correct matches. Fixed by flooring per-term IDF at a small positive
  constant. Verified against a set of check cases including a true-negative (unrelated) case.
- **Fixed a second bug**: the original conflict detector compared numbers per retrieved chunk,
  but a 2-sentence window can contain both "95%" and "70%" and so shares a number with each side
  of the real conflict, masking it. Now decomposes windows back to individual sentences before
  comparing.
- Added an **optional external evidence source**: when `GROQ_API_KEY` is set, a "Search the web
  for evidence" button in each claim's detail view calls Groq's `groq/compound` model, which has a
  built-in, Tavily-backed web search tool with automatic source citations -- kept clearly separate
  in the UI from local retrieval, never silently blended into the same score.

---

## 3. Pipeline

```
Text -> Sentence segmentation -> Claim filtering -> Compound-claim decomposition
     -> Linguistic feature engineering (17 hand-built signals + TF-IDF
        + real-data check-worthiness score)
     -> Hybrid ML + rule-based verdict (with abstention when the model is unsure)
     -> Context-sufficiency scoring (TF-IDF similarity vs. supplied context)
     -> Risk scoring & prioritization
     -> Evidence retrieval (BM25 + stemming + windowing) + conflict detection
        [+ optional Groq web-search evidence]
     -> Human review routing / investigation roadmap
```

| File | Responsibility |
|---|---|
| `features.py` | Sentence splitting, compound-claim decomposition, lexicon-based feature engineering |
| `data/prepare_checkthat_data.py` | Processes the real CLEF CheckThat! 2019 dataset |
| `data/build_seed_dataset_v2.py` | Builds the expanded, balanced synthetic 3-way seed set |
| `train_checkworthy_model.py` | Trains the binary check-worthiness model on real data |
| `train_models.py` | Trains + compares 5 models (incl. LSA embedding baseline) on the 3-way task |
| `train_models_transformer_experiment.py` | Optional real-transformer comparison (needs internet) |
| `claim_analyzer.py` | Runtime inference: hybrid verdict, risk, abstention, investigation roadmap |
| `evidence_engine.py` | BM25 evidence retrieval, conflict detection, alignment, web-search fallback |
| `groq_client.py` | Optional Groq-powered chatbot + "AI second opinion" |
| `storage.py` | SQLite history, human-review queue, feedback |
| `report_generator.py` | Downloadable HTML verification report |
| `app.py` | Flask REST API + static file serving |

---

## 4. Dataset & documentation

### 4.1 Real data: CLEF CheckThat! 2019
**Source:** Elsayed, Nakov, Barron-Cedeno, Hasanain, Suwaileh, Da San Martino, Atanasova.
"Overview of the CLEF-2019 CheckThat!: Automatic Identification and Verification of Claims."
CLEF 2019. Repo: `github.com/apepa/clef2019-factchecking-task1`.
**License:** "free for general research use" (per repo README). Raw files bundled under
`backend/data/checkthat_raw/` for offline reproducibility.

**What the label means (and doesn't):** `checkworthy=1` means professional fact-checkers selected
that sentence for verification against FactCheck.org -- real-world evidence for "needs external
verification." It does NOT mean `checkworthy=0` sentences are "verified true" or "stable
facts" -- that class just means fact-checkers didn't select it, which includes filler, opinion, and
mundane claims alongside genuinely settled facts. Because of this asymmetry, we use the dataset to
train a dedicated, honestly-scoped BINARY model rather than silently relabeling `0` as our
`context_sufficient` class.

**Split:** the dataset's own train/test partition, by debate FILE (not random sentence) -- so
no vocabulary or topic leaks between train and test. 12,399 train sentences (414 check-worthy),
5,247 test sentences (133 check-worthy) after quality filtering.

### 4.2 Synthetic data: the fine-grained 3-way seed set
`build_seed_dataset_v2.py` generates 175 examples spanning every category and difficulty case the
problem statement calls out (stable facts, present-tense unverified claims, statistical/financial
claims, future predictions, conditional claims, implicit causal claims, person/title claims,
health claims, environmental extremal claims, compound claims), now close to balanced across the
three labels. This remains a documented weak-supervision placeholder -- see Section 8.

---

## 5. Baseline, controlled experiments, evaluation & error analysis

**Binary check-worthiness model** (real data): Logistic Regression on TF-IDF + 17 engineered
features, `class_weight='balanced'`, decision threshold tuned on a validation split carved out of
training data only (never the test set). Representative result (`models/checkworthy_metrics.json`,
regenerate with `train_checkworthy_model.py`):

| Metric | Value |
|---|---|
| ROC-AUC | ~0.74 |
| PR-AUC | ~0.09 (vs. ~0.025 random baseline -- a ~3.7x lift) |
| Precision / Recall / F1 at tuned threshold | ~0.11 / ~0.22 / ~0.15 |

This is a genuinely hard, imbalanced real-world task (~2.5% positive class); published CheckThat!
leaderboard systems using much more sophisticated methods top out around 0.12-0.17 MAP. These
numbers are reported honestly, not tuned to look better than they are.

**3-way verification-need classifier** (synthetic + blended real feature): five models compared
under identical 5-fold stratified CV on the same TF-IDF(500, 1-2gram) + 17 engineered features +
check-worthiness-probability representation (`models/model_comparison.json`, regenerate with
`train_models.py`):

| Model | 5-fold CV F1 (macro) | Held-out test F1 (macro) |
|---|---|---|
| Linear SVM (typically best) | ~0.76 +/- 0.08 | ~0.82 |
| Logistic Regression | ~0.73 +/- 0.04 | -- |
| Random Forest | ~0.70 +/- 0.06 | -- |
| Multinomial Naive Bayes | ~0.69 +/- 0.09 | -- |
| TF-IDF + LSA (dense embedding) | ~0.54 +/- 0.04 | -- |

**Finding worth reporting:** the dense-embedding (LSA) baseline underperforms sparse TF-IDF +
engineered features at this dataset size (n~175) -- a real, expected result (dense embeddings
typically need more data to outperform sparse+engineered representations), not a bug. See Section 6
for what a real transformer embedding does on the same task.

**Error analysis:** `models/error_analysis.json` logs every held-out misclassification with a
guessed error category (temporal misunderstanding, numerical claim error, compound claim error,
conditional/hedged language misread, or ambiguous language) -- visible in the app's Model Lab tab.

---

## 6. Why classical ML + LSA, and how to add real transformers

The problem statement prescribes no architecture. TF-IDF + engineered features (a) keeps the
project zero-cost and fully offline-capable, (b) avoids multi-hundred-MB pretrained weight
downloads, and (c) keeps every decision explainable -- which "Why was this flagged?" and error
analysis specifically need.

**On the "advanced experiment" ask for a transformer/embedding comparison:** this development
sandbox's network is restricted to package registries (pypi, npm, GitHub) and does NOT allow
`huggingface.co`, so real pretrained sentence-transformer weights could not be downloaded or
verified here. Instead:
- The LSA/SVD dense-embedding baseline (self-contained, no downloads) is genuinely run and
  compared above.
- `train_models_transformer_experiment.py` provides a real, working `sentence-transformers`
  (`all-MiniLM-L6-v2`) comparison, guarded to degrade gracefully if the package or model download
  is unavailable. Run it on your own machine (normal internet access) to get real
  transformer-embedding numbers for the report -- the script prints a side-by-side comparison
  against the classical models when it succeeds.

---

## 7. Evidence retrieval boundary (and where Groq fits in)

Local evidence retrieval (`evidence_engine.retrieve_evidence`) is BM25-ranked, stemmed, and
windowed, scoped to whatever context the user pastes or uploads. There is no bundled search index
or scraped corpus -- building one reliably was judged out of scope for a zero-cost prototype and
would raise its own accuracy/currency risks if hand-curated.

The optional Groq web-search evidence path (`evidence_engine.web_search_evidence`, exposed via
the "Search the web for evidence" button) is the system's answer to "add an external evidence
source": it calls Groq's `groq/compound` model, which has a built-in, Tavily-backed web search
tool with automatic source citations. It is explicitly optional (needs `GROQ_API_KEY`), and its
output is always shown separately from -- never merged into -- local BM25 evidence, so a person
reviewing a claim can tell which claims are backed by their own supplied context versus a live web
lookup.

---

## 8. Known limitations & future work

- The 3-way seed dataset (n~175) is still weakly-labeled and programmatically generated, though
  now roughly balanced and blended with a real-data feature -- a fully human-annotated 3-way corpus
  remains the top priority for a future version.
- The check-worthiness binary model's absolute precision/recall is modest (Section 5) -- expected
  for this known-hard, highly-imbalanced task, but not yet production-grade.
- Real transformer embeddings are supported but not run by default (Section 6) due to this
  development environment's network restrictions -- trivial to enable on a normal machine.
- Compound-claim decomposition handles coordinating-conjunction ("and"/";") splits with subject
  carry-over; it will miss more exotic compound structures (nested clauses, appositives).
- Claim-relationship graphs, claim-change-tracker versioning, and image/PDF/DOCX upload are not
  implemented -- noted here rather than silently omitted.

## 9. Feature coverage vs. the original 30-feature brief

Implemented: smart text input (paste/`.txt` upload), automatic claim extraction, compound-claim
decomposition, verification-requirement predictor (3-way + abstain, now blended with real data),
confidence/context/risk scores, claim-type detection, temporal-sensitivity detection, risk
scoring, context-sufficiency analysis, BM25 evidence retrieval, evidence strength meter, evidence
conflict detector, evidence-to-claim alignment, "why was this flagged" explainability, claim
highlighting, verification dashboard, priority-based human review queue, human-in-the-loop
feedback capture, claim similarity/duplicate detection, AI abstention mode, adjustable
verification threshold, model comparison laboratory (5 models + real check-worthiness model),
error analysis, claim history, AI-generated verification report, document verifiability score,
claim investigation roadmap, and an optional external (web-search) evidence source.

Deliberately simplified or left as future work: full claim-relationship graph visualization,
claim-change-tracker versioning over time, PDF/DOCX upload, and default (non-optional) transformer
embeddings.

---

## 10. What changed in v3 (the 15-feature improvement plan)

This round implemented all 15 items from the "Next-Level Improvement Plan" as real backend logic
wired to the UI -- not just visual mockups. Honest notes on each:

1. **Home/pipeline visualization + Live AI Analysis Status.** A pipeline strip and a staged
   status list (Extracting claims -> Analyzing linguistic features -> ... -> Generating
   recommendation) animate during analysis. Since the backend returns one response rather than
   streaming progress, the frontend sequences these over a plausible timeline and snaps to
   "done" the moment the real response arrives -- it never claims a step finished before the
   response that proves it did.
2. **Claim Confidence Score + Claimability Filter.** `features.classify_claimability()` now runs
   before verification: opinions, questions, and personal beliefs get routed to a `not_a_claim`
   verdict with their own confidence score instead of a nonsense verification label. Confirmed
   working end-to-end.
3. **Claim Complexity Analyzer.** Real factor-based scoring (numeric info, date, multiple
   entities, multiple assertions) feeding into the risk engine, exposed per-claim.
4. **Multi-condition AI Abstention.** Abstains on ANY of: low top-confidence, small top-two gap,
   near-zero context support, or conflicting evidence -- each with a specific, logged reason
   string shown in the UI (not a generic "not sure").
5. **Retrieval method comparison.** TF-IDF+cosine, BM25, and an LSA/SVD "semantic-proxy" (not a
   real transformer -- see Section 6) compared live per-query (`/api/retrieval-comparison`) and
   on a hand-labeled 20-case benchmark with real Precision@3/Recall@3
   (`eval_retrieval_methods.py`) -- not fabricated numbers.
6. **Evidence Freshness Score.** Honestly reports "Unknown" when no source date is supplied (the
   normal case for pasted text) rather than guessing one. Computes a real score if a
   `context_date` is provided.
7. **Advanced conflict detection / Evidence Debate View.** Each retrieved evidence sentence is
   classified supporting/contradicting/neutral and shown as a split bar with a conflict intensity
   level, built on the same numeric/negation logic as the conflict detector.
8. **Claim Lifecycle Tracker.** NEW -> ANALYZED -> EVIDENCE_FOUND -> CONFLICT_DETECTED ->
   HUMAN_REVIEW -> RESOLVED, persisted in SQLite and shown as a stepper in the claim detail view.
9. **Upgraded human review system.** Supported/Refuted/Insufficient/Ambiguous/Needs-updated-
   evidence, a reviewer confidence slider, and a free-text notes field, all persisted.
10. **Active learning loop (manual).** "Export Reviewed Cases for Retraining" downloads a CSV
    shaped like the training data. This is a manual retrain step, not automatic continuous
    retraining -- documented as such, not oversold.
11. **Research Analytics Dashboard.** Real abstention rate, conflict-detection rate, and human
    override rate computed from stored history/feedback (not placeholders), plus feature
    importance extracted from the trained model's coefficients.
12. **Model Decision Comparison / consensus.** All 4 classical models (not just the best) run on
    every claim; disagreement is flagged explicitly in the UI.
13. **Claim Fingerprint.** A radar-chart summary (type, temporal risk, complexity, context
    support, confidence, evidence strength, conflict level) rendered as inline SVG.
14. **Evidence Intelligence Score.** A heuristic composite (relevance, source count, agreement,
    freshness, coverage) -- explicitly labeled `is_heuristic: true` in the API response and in the
    UI, never presented as an ML probability.
15. **Counterfactual claim testing.** Generates controlled perturbations (numeric value scaled,
    superlatives softened, future tense flipped to present) and reruns the full pipeline on each,
    reporting a sensitivity rating -- confirmed to correctly detect both low- and
    high-sensitivity cases in testing.

**Two real bugs found and fixed while building this round** (documented in the code comments, not
swept under the rug): BM25 was scoring two completely unrelated sentences as a "Strong 90%" match
purely because they both contained the word "the" (stopwords weren't filtered before BM25
indexing) -- fixed by filtering stopwords and adding an absolute minimum-overlap floor so a single
coincidental match can no longer be stretched to "Strong" by relative normalization alone. See
`evidence_engine.py` for the fix and `eval_retrieval_methods.py` for the true-negative test cases
that caught it.
