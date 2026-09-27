# VeriScope AI
### AI-Powered Claim Verification & Evidence Auditor
**Learn Depth — Problem ML-T2-061:** *Detecting Claims That Require External Verification*

> **Central research question:** *Can an NLP system recognize when a claim cannot responsibly
> be accepted without external evidence?*

VeriScope AI is **not** a fake-news / true-or-false classifier. Given a claim and whatever context
is available, it predicts one of:

| Verdict | Meaning |
|---|---|
| 🟢 Context Sufficient | The claim can be responsibly assessed from the available signal |
| 🟡 Needs Verification | Checkable in principle, but no supporting evidence is currently available |
| 🔴 High Priority | Needs verification **and** carries elevated stakes |
| 🛑 AI Abstains | Multi-condition check found the model's confidence too low to call — routed to human review |
| ⏭️ Not a Claim | Filtered out as opinion, question, or personal belief before verification even runs |

---

## 1. Quick start (local)

```bash
cd backend
pip install -r requirements.txt --break-system-packages   # or use a venv
cp .env.example .env                       # optional: add GROQ_API_KEY
python3 app.py                             # serves at http://localhost:5000
```

Models are pre-trained and included (`backend/models/*.pkl`). To regenerate from scratch:
```bash
python3 data/prepare_checkthat_data.py
python3 data/build_seed_dataset_v2.py
# then merge seed_claims.csv + seed_claims_v2.csv into seed_claims_combined.csv
python3 train_checkworthy_model.py
python3 train_models.py
python3 eval_retrieval_methods.py
```

## 2. Deployment (Render + Netlify)

This project is set up for a split deployment: **Render** hosts the Flask backend,
**Netlify** hosts the static frontend.

**Render:** Root Directory = `backend`, Build Command = `pip install -r requirements.txt`,
Start Command = `gunicorn app:app`. Add `GROQ_API_KEY` (optional) and `PYTHON_VERSION=3.12.3`
as environment variables.

**Netlify:** Base/Publish Directory = `frontend`, no build command (plain HTML/CSS/JS).

**`frontend/app.js` auto-detects environment** (see the top of the file): on `localhost` or on
the Render domain itself, it uses relative API calls; on a `*.netlify.app` domain, it calls the
Render backend directly via `RENDER_API_URL`. Update that constant if your Render URL changes.

---

## 3. Pipeline

```
Text -> Sentence segmentation -> Claimability filter (opinion/question/belief vs. verifiable claim)
     -> Compound-claim decomposition -> Complexity analysis
     -> Linguistic feature engineering (17 signals + TF-IDF + real check-worthiness score)
     -> Hybrid ML + rule-based verdict (multi-condition abstention)
     -> Context-sufficiency scoring -> Risk scoring & claim fingerprint
     -> Multi-model consensus (4 classifiers voted, disagreement flagged)
     -> Evidence retrieval (BM25 + stemming + windowing) + conflict detection + debate view
        [+ optional Groq web-search evidence, with automatic fallback if the live tool call fails]
     -> Human review routing / investigation roadmap / counterfactual testing
```

| File | Responsibility |
|---|---|
| `features.py` | Sentence splitting, compound decomposition, claimability filter, complexity analyzer |
| `data/prepare_checkthat_data.py` | Processes the real CLEF CheckThat! 2019 dataset |
| `data/build_seed_dataset_v2.py` | Builds the balanced synthetic 3-way seed set (175 examples) |
| `train_checkworthy_model.py` | Trains the binary check-worthiness model on real data |
| `train_models.py` | Trains + compares 5 models (incl. LSA baseline), saves all 4 for consensus |
| `train_models_transformer_experiment.py` | Optional real-transformer comparison (needs internet) |
| `eval_retrieval_methods.py` | Hand-labeled Precision@K/Recall@K benchmark for retrieval methods |
| `claim_analyzer.py` | Runtime inference: verdict, abstention, fingerprint, consensus, complexity |
| `evidence_engine.py` | BM25 retrieval, conflict detection, debate view, freshness, web-search fallback |
| `counterfactual.py` | Controlled claim perturbations for robustness testing |
| `groq_client.py` | Chatbot + "AI second opinion" (optional, degrades gracefully) |
| `storage.py` | SQLite history, review queue, feedback, lifecycle tracking, research stats |
| `report_generator.py` | Downloadable HTML verification report |
| `app.py` | Flask REST API + static file serving + CORS |

---

## 4. Dataset & documentation

**Real data:** ~17,600 sentences from CLEF CheckThat! 2019 (US presidential debates/speeches,
2016–2019), professionally fact-checked. License: free for research use. Trains a dedicated
binary check-worthiness model whose output is blended into the 3-way classifier as a feature.

**Synthetic data:** 175 balanced examples (53/63/59 across the three labels), programmatically
authored to cover every difficulty case the problem statement names (compound, temporal,
conditional, implicit). Documented weak-supervision placeholder — see Section 8.

## 5. Evaluation

Binary check-worthiness model (real data): ROC-AUC ~0.74, PR-AUC ~0.09 (vs ~0.025 random
baseline — a ~3.7x lift). Modest in absolute terms, consistent with published CheckThat!
leaderboard results for this genuinely hard, ~2.5%-positive-class task.

3-way classifier (5-fold CV): Linear SVM best at ~0.76±0.05 CV F1 macro, ~0.74 held-out test
accuracy. TF-IDF+LSA dense embedding baseline underperforms (~0.56) — expected at this dataset
size; sparse+engineered features win over dense embeddings with n~175.

Retrieval benchmark (hand-labeled, n=20): TF-IDF, BM25, and LSA-semantic-proxy all score
Precision@3/Recall@3 = 0.95 after fixing two real bugs found during development (see Section 6).

## 6. Real bugs found and fixed during development

1. **BM25 negative-IDF collapse on small candidate pools:** every score came out negative
   regardless of relevance, silently dropping correct matches. Fixed by flooring per-term IDF.
2. **Stopword false-positive matches:** two unrelated sentences scored as a "Strong 90%" match
   purely because they shared the word "the." Fixed by filtering stopwords before BM25 indexing
   and requiring genuine minimum term overlap.
3. **Windowed-evidence conflict masking:** a 2-sentence window containing both "95%" and "70%"
   shared a number with each side of a real conflict, hiding it. Fixed by decomposing windows
   back to atomic sentences before comparing.
4. **`groq/compound` decommissioned (2026-09-21):** the web-search feature's underlying model was
   retired by Groq. Migrated to the `browser_search` tool on `openai/gpt-oss-120b`, with an
   automatic fallback to a plain (non-tool) chat completion if the tool call fails for any
   reason — so the feature degrades gracefully instead of surfacing a raw API error.

## 7. Why classical ML, and the transformer/embedding path

TF-IDF + engineered features keeps the project zero-cost, offline-capable, and every decision
explainable. A self-contained LSA/SVD dense-embedding baseline is run and compared (Section 5).
Real pretrained transformer embeddings need `huggingface.co` access this dev sandbox couldn't
reach — `train_models_transformer_experiment.py` provides a real, working, guarded path for a
machine with normal internet access.

## 8. Known limitations & future work

- The 3-way seed dataset (n~175) is weakly-labeled/programmatically generated, though balanced
  and blended with a real-data feature. A human-annotated 3-way corpus is the top future priority.
- Check-worthiness model precision/recall is modest — expected for this hard, imbalanced task.
- Local evidence retrieval is scoped to user-supplied context; the optional Groq web-search path
  is the route to broader knowledge, and now degrades gracefully rather than erroring out.
- Claim-relationship graphs, claim-change-tracker versioning, and PDF/DOCX upload are not
  implemented.
- SQLite resets on every Render redeploy (ephemeral filesystem on the free tier) — fine for a
  demo, not for long-term persistence without adding a real hosted database.

## 9. Feature coverage

Implemented and tested: claim extraction, compound decomposition, claimability filter, complexity
analyzer, verification-requirement predictor (3-way + abstain, blended with real data),
confidence/context/risk scores, claim-type/temporal detection, BM25 evidence retrieval with
strength meter, conflict detector, evidence debate view, evidence freshness (honest "unknown"
without a date), evidence intelligence score (explicitly heuristic), claim highlighting,
verification + research analytics dashboards, priority-based human review queue with
confidence/notes, claim lifecycle tracker, AI abstention mode, adjustable verification threshold,
model comparison lab (5 models + real check-worthiness model + feature importance), model
consensus/disagreement flagging, claim fingerprint (radar chart), counterfactual testing, claim
history, AI-generated verification report, document verifiability score, investigation roadmap,
CSV export for retraining, optional Groq chat/AI-review/web-search-with-fallback.

Deliberately simplified or left as future work: full claim-relationship graph visualization,
claim-change-tracker versioning over time, PDF/DOCX upload.
