"""
evidence_engine.py  (v2)
--------------------------
Evidence retrieval, rewritten to actually rank well instead of relying on
raw TF-IDF cosine similarity alone (which was the reported problem: short
claims vs. short context sentences give noisy, unbounded, poorly-calibrated
cosine scores). Improvements:

  1. Real BM25 ranking (rank_bm25's BM25Okapi) as the primary retrieval
     algorithm -- the standard IR algorithm for exactly this "short query
     against a small sentence collection" setting, better calibrated than
     raw cosine similarity for short texts.
  2. Stemming normalization (Porter stemmer, no corpus download needed) so
     "grew" matches "growth", "increased" matches "increase", etc. -- this
     was silently killing recall before.
  3. Multi-granularity indexing: both single sentences AND overlapping
     2-sentence windows are indexed, so evidence spanning two sentences is
     still retrievable.
  4. Proper score normalization: BM25's raw scores are unbounded, so we
     min-max normalize within the retrieved candidate set before turning
     them into a 0-100% relevance number and a Strong/Moderate/Weak band.
  5. An OPTIONAL, explicitly-labeled web-search evidence path via Groq's
     groq/compound model (built-in web search, Tavily-backed, automatic
     citations) for when local context doesn't contain an answer. This is
     the "external evidence source" requested -- gated behind GROQ_API_KEY,
     clearly distinguished in the UI from local retrieval, and never
     silently blended with local evidence scores.

Also covers: Evidence Strength Meter, Evidence Conflict Detector,
Evidence-to-Claim Alignment, Claim Similarity / Duplicate Detection, and
document-level verifiability scoring.
"""

import os
import re
import requests
import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

from features import split_sentences

NEGATION_WORDS = {"not", "no", "never", "n't", "without", "fails", "failed", "denies", "denied"}
_stemmer = PorterStemmer()
_token_re = re.compile(r"[a-zA-Z]+")
# Negation words carry real meaning for conflict detection, so keep them
# out of the generic stopword-drop list even though sklearn's list includes some.
_STOPWORDS = ENGLISH_STOP_WORDS - NEGATION_WORDS

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_WEB_SEARCH_MODEL = "groq/compound"


def _tokenize_stem(text):
    """Lowercase, alpha-only tokenize, drop stopwords, then Porter-stem.
    Pure algorithm, no corpus/model download required.

    Stopword removal matters here specifically: without it, two completely
    unrelated sentences that only share "the"/"was"/"will" were scoring as
    BM25 matches (a real bug found via eval_retrieval_methods.py -- see
    README), because the small-corpus IDF floor (below) treats any matched
    term as a positive contribution."""
    tokens = _token_re.findall(text.lower())
    return [_stemmer.stem(t) for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def _build_windows(sentences, window=2):
    """Overlapping N-sentence windows, so two-sentence evidence spans are
    retrievable even though each index entry still maps back to readable text."""
    windows = []
    for i in range(len(sentences)):
        windows.append(sentences[i])
        if i + 1 < len(sentences) and window >= 2:
            windows.append(sentences[i] + " " + sentences[i + 1])
    # de-duplicate while preserving order
    seen = set()
    out = []
    for w in windows:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def retrieve_evidence(claim_text, context_text, top_k=5):
    """BM25-ranked evidence retrieval over the user-supplied context, with
    stemming normalization and sentence+window granularity. Returns a list
    of dicts with a 0-100 normalized relevance score and a strength band."""
    sentences = split_sentences(context_text) if context_text else []
    if not sentences:
        return []

    candidates = _build_windows(sentences, window=2)
    tokenized_corpus = [_tokenize_stem(c) for c in candidates]
    # BM25Okapi errors on an all-empty corpus (e.g. context of only numbers/symbols)
    if not any(tokenized_corpus):
        return []

    bm25 = BM25Okapi(tokenized_corpus)
    # Known BM25 quirk: with a tiny, highly-overlapping candidate pool (our
    # windowed chunks deliberately repeat content from their constituent
    # sentences), most terms appear in >half the "documents", so raw IDF
    # goes negative for nearly every term -- and rank_bm25's own epsilon
    # floor (0.25 * average_idf) is *also* negative when the average itself
    # is negative, so it fails to rescue anything. Every score ends up
    # negative regardless of relevance, and real matches get silently
    # dropped. Floor every term's IDF at a small positive constant so a
    # genuine term match always contributes positively to the score.
    bm25.idf = {term: max(val, 0.15) for term, val in bm25.idf.items()}

    query_tokens = _tokenize_stem(claim_text)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    order = np.argsort(-scores)

    # Only keep candidates with a nonzero score (BM25 gives 0 for no term overlap at all)
    ranked = [(i, scores[i]) for i in order if scores[i] > 0][:top_k]
    if not ranked:
        return []

    # Require a MEANINGFUL minimum overlap, not just "the best of a bad
    # bunch": count actual stemmed term overlap between query and each
    # candidate. Without this, min-max normalizing purely relative scores
    # can stretch a single coincidental match up to "Strong" even when
    # nothing in the context is actually relevant (the true bug behind the
    # stopword issue above -- this is a second, independent safeguard).
    filtered = []
    for i, raw in ranked:
        overlap = len(set(query_tokens) & set(_tokenize_stem(candidates[i])))
        if overlap >= 1:
            filtered.append((i, raw, overlap))
    if not filtered:
        return []

    raw_scores = np.array([s for _, s, _ in filtered], dtype=float)
    lo, hi = raw_scores.min(), raw_scores.max()
    if hi - lo < 1e-9:
        normalized = np.full_like(raw_scores, 0.6)
    else:
        normalized = (raw_scores - lo) / (hi - lo) * 0.55 + 0.35

    results = []
    for (idx, raw, overlap), norm in zip(filtered, normalized):
        # a single-term overlap is capped below "Strong" regardless of how
        # it ranks relative to its (possibly all-weak) neighbors
        capped = min(norm, 0.6) if overlap == 1 else norm
        results.append({
            "sentence": candidates[idx],
            "relevance_pct": round(float(capped) * 100, 1),
            "strength": _strength_band(float(capped)),
            "raw_bm25_score": round(float(raw), 3),
        })
    return results


def _strength_band(norm_score):
    if norm_score >= 0.75:
        return "Strong"
    if norm_score >= 0.55:
        return "Moderate"
    if norm_score >= 0.35:
        return "Weak"
    return "Insufficient"


def detect_conflict(evidence_list):
    """Compare numeric values and negation polarity across retrieved evidence
    to flag likely contradictions. Each evidence "sentence" may actually be a
    2-sentence window (see _build_windows), so we first decompose every
    evidence item back into its finest-grain sentences before comparing --
    otherwise a window containing BOTH "95%" and "70%" would share a number
    with each single-sentence source and mask a real conflict between them."""
    atoms = []  # list of (text, numbers, has_negation), deduplicated
    seen_text = set()
    for ev in evidence_list:
        for sub in split_sentences(ev["sentence"]) or [ev["sentence"]]:
            key = sub.strip().lower()
            if key in seen_text:
                continue
            seen_text.add(key)
            nums = set(re.findall(r"\d+(?:\.\d+)?", sub))
            low = sub.lower()
            has_neg = any(w in low.split() or w in low for w in NEGATION_WORDS)
            atoms.append((sub, nums, has_neg))

    conflicts = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            text_a, nums_a, neg_a = atoms[i]
            text_b, nums_b, neg_b = atoms[j]
            if nums_a and nums_b and nums_a.isdisjoint(nums_b):
                conflicts.append({
                    "a": text_a, "b": text_b,
                    "reason": "Differing numeric values reported for what appears to be the same claim.",
                })
            elif neg_a != neg_b and (nums_a or nums_b) == set():
                # Only flag negation mismatches when there's no numeric
                # signal already driving the comparison, to avoid double-
                # flagging the same pair for two different reasons.
                conflicts.append({
                    "a": text_a, "b": text_b,
                    "reason": "One source appears to negate what the other asserts.",
                })
    return conflicts


def align_evidence_to_claim(claim_text, evidence_sentence):
    """Highlight overlapping salient (stemmed) tokens between claim and
    evidence for explainability -- stemmed so 'grew'/'growth' still align."""
    claim_tokens = set(_tokenize_stem(claim_text))
    ev_raw_tokens = re.findall(r"[a-zA-Z]{3,}", evidence_sentence)
    overlap = [t for t in ev_raw_tokens if _stemmer.stem(t.lower()) in claim_tokens]
    return sorted(set(overlap))


def claim_similarity(claim_a, claim_b):
    try:
        vec = TfidfVectorizer(stop_words="english").fit([claim_a, claim_b])
        mat = vec.transform([claim_a, claim_b])
        sim = float((mat[0] @ mat[1].T).toarray()[0][0])
    except ValueError:
        sim = 0.0
    return round(sim * 100, 1)


def find_duplicates(new_claim, history_claims, threshold=70.0):
    matches = []
    for h in history_claims:
        sim = claim_similarity(new_claim, h.get("claim", ""))
        if sim >= threshold:
            matches.append({"claim": h.get("claim"), "similarity_pct": sim,
                             "previous_verdict": h.get("verdict")})
    return sorted(matches, key=lambda m: -m["similarity_pct"])


# ---------------------------------------------------------------------------
# Retrieval method comparison: TF-IDF+cosine (Model A) vs BM25 (Model B,
# the production default above) vs a self-contained "semantic-ish" proxy
# (Model C -- TF-IDF -> LSA/SVD cosine, i.e. dense embeddings without a
# downloaded transformer; see README for why a real transformer isn't run
# in this sandboxed dev environment). All three share the same candidate
# pool (sentences + 2-sentence windows) so results are directly comparable.
# ---------------------------------------------------------------------------

def _retrieve_tfidf_cosine(claim_text, candidates, top_k=5):
    if not candidates:
        return []
    corpus = [claim_text] + candidates
    try:
        vec = TfidfVectorizer(stop_words="english").fit(corpus)
        mat = vec.transform(corpus)
        sims = (mat[1:] @ mat[0].T).toarray().flatten()
    except ValueError:
        return []
    order = np.argsort(-sims)[:top_k]
    return [{"sentence": candidates[i], "relevance_pct": round(float(sims[i]) * 100, 1),
              "strength": _strength_band(float(sims[i]))} for i in order if sims[i] > 0.02]


def _retrieve_lsa_semantic(claim_text, candidates, top_k=5):
    """Dense-embedding proxy: TF-IDF -> Truncated SVD -> cosine similarity.
    NOT a real transformer/sentence-embedding model -- documented explicitly
    (see train_models.py / README) as the offline-friendly stand-in."""
    if len(candidates) < 2:
        return _retrieve_tfidf_cosine(claim_text, candidates, top_k)  # too few docs for SVD
    from sklearn.decomposition import TruncatedSVD
    corpus = [claim_text] + candidates
    try:
        vec = TfidfVectorizer(stop_words="english").fit(corpus)
        mat = vec.transform(corpus)
        n_comp = max(1, min(20, mat.shape[1] - 1, mat.shape[0] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        dense = svd.fit_transform(mat)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        dense = dense / norms
        sims = dense[1:] @ dense[0]
    except (ValueError, Exception):
        return []
    order = np.argsort(-sims)[:top_k]
    return [{"sentence": candidates[i], "relevance_pct": round(float(max(sims[i], 0)) * 100, 1),
              "strength": _strength_band(float(max(sims[i], 0)))} for i in order if sims[i] > 0.05]


def compare_retrieval_methods(claim_text, context_text, top_k=3):
    """Runs all three retrieval methods on the same claim+context and
    returns each one's top results side by side, for the Model
    Comparison / research write-up. Precision@K/Recall@K against a labeled
    benchmark is computed separately -- see eval_retrieval_methods.py,
    since that needs ground-truth relevance judgments, which a single live
    query doesn't have."""
    sentences = split_sentences(context_text) if context_text else []
    candidates = _build_windows(sentences, window=2)
    return {
        "tfidf_cosine": _retrieve_tfidf_cosine(claim_text, candidates, top_k),
        "bm25": retrieve_evidence(claim_text, context_text, top_k),
        "lsa_semantic": _retrieve_lsa_semantic(claim_text, candidates, top_k),
    }


# ---------------------------------------------------------------------------
# Evidence Debate View + Conflict Intensity: classify each retrieved
# evidence sentence as supporting, contradicting, or neutral toward the
# claim, then summarize as percentages the UI can render as a split view.
# ---------------------------------------------------------------------------

def evidence_debate_view(claim_text, evidence_list):
    if not evidence_list:
        return {"supporting_pct": 0, "contradicting_pct": 0, "neutral_pct": 0,
                "conflict_intensity": "None", "items": []}

    # Decompose windowed evidence back to atomic sentences and de-duplicate
    # (same fix as detect_conflict) so a 2-sentence window doesn't get
    # counted as a second, redundant "vote" alongside its own constituent
    # sentences -- otherwise support/contradict percentages get inflated.
    atoms = []
    seen = set()
    for ev in evidence_list:
        for sub in split_sentences(ev["sentence"]) or [ev["sentence"]]:
            key = sub.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            atoms.append({"sentence": sub, "relevance_pct": ev.get("relevance_pct", 50)})

    claim_nums = set(re.findall(r"\d+(?:\.\d+)?", claim_text))
    claim_low = claim_text.lower()
    claim_negated = any(w in claim_low.split() for w in NEGATION_WORDS)

    items = []
    support_score = 0.0
    contradict_score = 0.0
    for ev in atoms:
        sent = ev["sentence"]
        sent_nums = set(re.findall(r"\d+(?:\.\d+)?", sent))
        sent_low = sent.lower()
        sent_negated = any(w in sent_low.split() for w in NEGATION_WORDS)
        weight = ev.get("relevance_pct", 50) / 100.0

        stance = "neutral"
        if claim_nums and sent_nums:
            if claim_nums & sent_nums:
                stance = "supporting"
            elif claim_nums.isdisjoint(sent_nums):
                stance = "contradicting"
        elif sent_negated != claim_negated and claim_nums == set() and sent_nums == set():
            stance = "contradicting"
        else:
            overlap = align_evidence_to_claim(claim_text, sent)
            stance = "supporting" if len(overlap) >= 2 else "neutral"

        if stance == "supporting":
            support_score += weight
        elif stance == "contradicting":
            contradict_score += weight
        items.append({"sentence": sent, "stance": stance, "relevance_pct": ev.get("relevance_pct", 0)})

    total = support_score + contradict_score
    if total == 0:
        supporting_pct, contradicting_pct = 0, 0
    else:
        supporting_pct = round(support_score / total * 100, 1)
        contradicting_pct = round(contradict_score / total * 100, 1)
    neutral_pct = round(max(0, 100 - supporting_pct - contradicting_pct), 1) if total else 100.0

    if contradicting_pct >= 40:
        intensity = "HIGH"
    elif contradicting_pct >= 15:
        intensity = "MEDIUM"
    elif contradicting_pct > 0:
        intensity = "LOW"
    else:
        intensity = "None"

    return {"supporting_pct": supporting_pct, "contradicting_pct": contradicting_pct,
            "neutral_pct": neutral_pct, "conflict_intensity": intensity, "items": items}


# ---------------------------------------------------------------------------
# Evidence Freshness Score: honest version -- we have no publication dates
# for pasted context by default, so this reports "Unknown" rather than
# fabricating a number. If the caller supplies a context_date, a real
# freshness score is computed against it.
# ---------------------------------------------------------------------------

def evidence_freshness(temporal_sensitive, context_date=None):
    from datetime import datetime
    if not context_date:
        return {"status": "unknown", "label": "Unknown \u2014 no source date supplied",
                "freshness_pct": None,
                "note": "Provide a context/source date to compute real freshness; "
                        "without one, this is honestly reported as unknown rather than guessed."}
    try:
        src_date = datetime.fromisoformat(context_date)
    except ValueError:
        return {"status": "unknown", "label": "Unknown \u2014 unparseable date", "freshness_pct": None}

    days_old = (datetime.now() - src_date).days
    if not temporal_sensitive:
        # stable facts age slowly
        pct = max(40, 100 - days_old / 30)
    else:
        # time-sensitive claims age fast
        pct = max(0, 100 - days_old / 3)
    pct = round(min(pct, 100), 1)
    if pct >= 70:
        label = f"Fresh ({days_old} days old)"
    elif pct >= 35:
        label = f"Aging ({days_old} days old) \u2014 consider reverification"
    else:
        label = f"Stale ({days_old} days old) \u2014 reverification recommended"
    return {"status": "computed", "label": label, "freshness_pct": pct, "days_old": days_old}


# ---------------------------------------------------------------------------
# Evidence Intelligence Score: a heuristic composite, explicitly documented
# as NOT an ML probability -- combines relevance, source count, agreement,
# freshness (if known), and context coverage into one number for a quick read.
# ---------------------------------------------------------------------------

def evidence_intelligence_score(evidence_list, debate, freshness, context_sufficiency_pct):
    if not evidence_list:
        return {"score": 0, "label": "No evidence retrieved", "is_heuristic": True}

    avg_relevance = sum(e.get("relevance_pct", 0) for e in evidence_list) / len(evidence_list)
    source_count_score = min(len(evidence_list) / 3, 1.0) * 100  # saturates at 3+ sources
    agreement_score = max(0, 100 - debate.get("contradicting_pct", 0) * 1.5)
    freshness_score = freshness.get("freshness_pct") if freshness.get("freshness_pct") is not None else 60  # neutral default when unknown
    coverage_score = context_sufficiency_pct

    weights = {"relevance": 0.30, "sources": 0.15, "agreement": 0.25, "freshness": 0.10, "coverage": 0.20}
    score = (avg_relevance * weights["relevance"] + source_count_score * weights["sources"] +
             agreement_score * weights["agreement"] + freshness_score * weights["freshness"] +
             coverage_score * weights["coverage"])
    score = round(min(score, 100), 1)

    if score >= 75:
        label = "Strong evidence base"
    elif score >= 50:
        label = "Moderate evidence base"
    elif score >= 25:
        label = "Weak evidence base"
    else:
        label = "Insufficient evidence base"

    return {"score": score, "label": label, "is_heuristic": True,
            "components": {"avg_relevance": round(avg_relevance, 1),
                            "source_count_score": round(source_count_score, 1),
                            "agreement_score": round(agreement_score, 1),
                            "freshness_score": round(freshness_score, 1),
                            "coverage_score": round(coverage_score, 1)}}


def document_verifiability_score(analyses):
    """0-100 score: higher = more of the document is responsibly assessable
    from available context, i.e. fewer/less-severe outstanding verification needs."""
    if not analyses:
        return 100, "No claims detected."
    penalty = 0.0
    for a in analyses:
        if a["verdict"] == "high_priority":
            penalty += 12
        elif a["verdict"] == "needs_verification":
            penalty += 6
        elif a["verdict"] == "abstain":
            penalty += 9
    score = max(0, 100 - penalty)
    score = round(score, 1)
    if score >= 90:
        band = "Highly supported"
    elif score >= 70:
        band = "Mostly supported"
    elif score >= 40:
        band = "Significant verification required"
    else:
        band = "High verification risk"
    return score, band


# ---------------------------------------------------------------------------
# Optional external evidence: Groq's web-search-enabled compound model.
# ---------------------------------------------------------------------------

def web_search_evidence(claim_text):
    """Ask Groq's groq/compound model (built-in, Tavily-backed web search
    with automatic citations) to find real-world evidence for a claim.
    Explicitly optional and clearly distinct from local retrieval -- callers
    must label results as web-sourced in the UI, never merge silently with
    local BM25 evidence. Returns (result_dict_or_None, error_or_None)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None, "GROQ_API_KEY is not set -- web-search evidence is unavailable offline."

    system_prompt = (
        "You help audit factual claims. Given a claim, use web search to find what "
        "reliable, current sources say. Report specifically: (1) what sources say "
        "about this claim, in 2-4 sentences, (2) whether sources agree or conflict, "
        "(3) how recent/authoritative the sources are. Do not declare the claim "
        "definitively true or false -- describe what the evidence shows and let the "
        "reader judge. Keep the whole answer under 150 words."
    )
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_WEB_SEARCH_MODEL,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": f'Claim to check: "{claim_text}"'}],
                "temperature": 0.2,
                "max_completion_tokens": 400,
            },
            timeout=45,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        content = message.get("content", "")
        executed_tools = message.get("executed_tools", []) or data.get("executed_tools", [])
        sources = []
        for tool in executed_tools or []:
            results = tool.get("search_results") or tool.get("results") or []
            for r in results:
                url = r.get("url") if isinstance(r, dict) else None
                title = r.get("title") if isinstance(r, dict) else None
                if url:
                    sources.append({"title": title or url, "url": url})
        return {"summary": content, "sources": sources[:5]}, None
    except requests.exceptions.RequestException as e:
        return None, f"Groq web-search request failed: {e}"
    except (KeyError, IndexError, ValueError) as e:
        return None, f"Unexpected Groq web-search response: {e}"
