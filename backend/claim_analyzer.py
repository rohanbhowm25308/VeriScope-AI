"""
claim_analyzer.py
------------------
Runtime inference pipeline used by the Flask API. Combines:
  - the trained baseline classifier (models/best_model.pkl) for a
    verification-requirement PRIOR, with
  - transparent rule-based adjustments for context sufficiency, risk,
    temporal sensitivity and abstention, with
  - a REAL-data signal: the probability that this claim would be selected
    as "check-worthy" by the binary classifier trained on the human-
    annotated CLEF CheckThat! 2019 dataset (models/checkworthy_model.pkl),
    blended in as one of the engineered features the 3-way model sees.

This hybrid (ML prior + real check-worthiness signal + explainable rule
layer) is a deliberate design choice: the synthetic seed dataset is modest
in size, so grounding part of its feature space in real, professionally
fact-checked annotations reduces (without eliminating) the risk of the
model learning purely synthetic artifacts.
"""

import pickle
import re
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from features import (engineer_features, split_sentences, split_compound_claim,
                       classify_claim_type, CLAIM_TYPE_LABELS,
                       classify_claimability, CLAIMABILITY_LABELS, analyze_complexity)

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "best_model.pkl"
CHECKWORTHY_MODEL_PATH = HERE / "models" / "checkworthy_model.pkl"
ALL_MODELS_PATH = HERE / "models" / "all_models.pkl"

_bundle = None
_cw_bundle = None
_cw_missing_warned = False
_all_models_bundle = None


def _load_bundle():
    global _bundle
    if _bundle is None:
        with open(MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


def _load_checkworthy_bundle():
    global _cw_bundle, _cw_missing_warned
    if _cw_bundle is None and CHECKWORTHY_MODEL_PATH.exists():
        with open(CHECKWORTHY_MODEL_PATH, "rb") as f:
            _cw_bundle = pickle.load(f)
    elif not CHECKWORTHY_MODEL_PATH.exists() and not _cw_missing_warned:
        print("NOTE: checkworthy_model.pkl not found -- run train_checkworthy_model.py. "
              "The real-data checkworthiness signal will read as 0 until then.")
        _cw_missing_warned = True
    return _cw_bundle


def checkworthy_score(claim_text):
    """Probability under the CLEF CheckThat!-trained binary model that a
    professional fact-checker would flag this sentence as check-worthy.
    Returns 0.0 if the model hasn't been trained yet."""
    bundle = _load_checkworthy_bundle()
    if bundle is None:
        return 0.0
    tfidf = bundle["vectorizer"].transform([claim_text])
    feats, _, _ = engineer_features(claim_text)
    eng_arr = np.array([[feats[k] for k in bundle["feature_keys"]]], dtype=float)
    eng_arr = bundle["scaler"].transform(eng_arr)
    X = hstack([tfidf, csr_matrix(eng_arr)])
    return float(bundle["model"].predict_proba(X)[0, 1])


def _load_all_models_bundle():
    global _all_models_bundle
    if _all_models_bundle is None and ALL_MODELS_PATH.exists():
        with open(ALL_MODELS_PATH, "rb") as f:
            _all_models_bundle = pickle.load(f)
    return _all_models_bundle


def model_consensus(claim_text):
    """Runs every candidate model (not just the best one) on this claim and
    reports each prediction, so the person can see whether the models agree
    or disagree -- disagreement is itself a signal worth surfacing, not
    just an internal training-time detail."""
    bundle = _load_all_models_bundle()
    if bundle is None:
        return None
    vectorizer = bundle["vectorizer"]
    scaler = bundle["scaler"]
    feature_keys = bundle["feature_keys"]

    tfidf = vectorizer.transform([claim_text])
    feats, _, _ = engineer_features(claim_text)
    base_keys = [k for k in feature_keys if k != "checkworthy_prob"]
    row = [feats[k] for k in base_keys] + [checkworthy_score(claim_text)]
    eng_arr = scaler.transform(np.array([row], dtype=float))
    X = hstack([tfidf, csr_matrix(eng_arr)])

    predictions = {}
    for name, model in bundle["models"].items():
        pred = model.predict(X)[0]
        predictions[name] = pred

    votes = {}
    for pred in predictions.values():
        votes[pred] = votes.get(pred, 0) + 1
    consensus_label = max(votes, key=votes.get)
    consensus_count = votes[consensus_label]
    total = len(predictions)
    agreement_pct = round(consensus_count / total * 100, 1) if total else 0.0

    return {
        "predictions": predictions,
        "consensus_label": consensus_label,
        "agreement": f"{consensus_count}/{total}",
        "agreement_pct": agreement_pct,
        "disagreement": consensus_count < total,
    }


# Sentences that look like claims vs. filler / questions / commands
NON_CLAIM_PATTERNS = [
    r"^\s*$",
    r"^(what|who|why|how|when|where|is|are|do|does|did|can|could|should)\b.*\?\s*$",
    r"^(please|thank you|thanks|hello|hi|hey)\b",
]


def _looks_like_claim(sentence):
    s = sentence.strip()
    if len(s.split()) < 3:
        return False
    for pat in NON_CLAIM_PATTERNS:
        if re.match(pat, s, re.IGNORECASE):
            return False
    return True


def _model_predict(claim_text):
    bundle = _load_bundle()
    vectorizer = bundle["vectorizer"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    feature_keys = bundle["feature_keys"]
    uses_cw = bundle.get("uses_checkworthy_feature", False)

    tfidf = vectorizer.transform([claim_text])
    feats, _, _ = engineer_features(claim_text)
    if uses_cw:
        # feature_keys ends with "checkworthy_prob" -- compute it live.
        base_keys = [k for k in feature_keys if k != "checkworthy_prob"]
        row = [feats[k] for k in base_keys] + [checkworthy_score(claim_text)]
    else:
        row = [feats[k] for k in feature_keys]
    eng_arr = np.array([row], dtype=float)
    eng_arr = scaler.transform(eng_arr)
    X = hstack([tfidf, csr_matrix(eng_arr)])

    labels = bundle["labels"]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        prob_map = dict(zip(model.classes_, proba))
    else:
        # LinearSVC has no predict_proba; use decision_function as a pseudo-confidence
        decision = model.decision_function(X)[0]
        decision = np.atleast_1d(decision)
        exp = np.exp(decision - np.max(decision))
        soft = exp / exp.sum()
        classes = model.classes_ if len(model.classes_) > 2 else labels
        prob_map = dict(zip(classes, soft))
    for lab in labels:
        prob_map.setdefault(lab, 0.0)
    return prob_map


def context_sufficiency(claim_text, context_text):
    """Estimate whether supplied context contains support for the claim.
    Returns (sufficiency_pct, best_matching_sentence_or_None).
    Uses TF-IDF cosine similarity between the claim and each context sentence
    -- a transparent, dependency-light proxy for entailment/support.
    """
    if not context_text or not context_text.strip():
        return 0.0, None

    ctx_sentences = split_sentences(context_text)
    if not ctx_sentences:
        return 0.0, None

    corpus = [claim_text] + ctx_sentences
    try:
        vec = TfidfVectorizer(stop_words="english").fit(corpus)
        mat = vec.transform(corpus)
        claim_vec = mat[0]
        ctx_mat = mat[1:]
        sims = (ctx_mat @ claim_vec.T).toarray().flatten()
    except ValueError:
        return 0.0, None

    if sims.size == 0:
        return 0.0, None
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    return round(float(best_score * 100), 1), (ctx_sentences[best_idx] if best_score > 0.05 else None)


def compute_risk_score(features, claim_type, sufficiency_pct, cw_score=0.0):
    """0-100 risk score combining stakes, complexity, missing context, uncertainty."""
    score = 0.0
    score += 18 if features["future_marker_count"] or features["future_year_flag"] else 0
    score += 15 if features["numeric_count"] or features["money_hits"] else 0
    score += 12 if features["superlative_count"] else 0
    score += 10 if features["implicit_causal_count"] else 0
    score += 8 if features["conditional_marker_count"] else 0
    score += 8 if features["has_conjunction_split"] else 0
    score += 12 if features["is_health"] else 0
    score += 10 if features["is_financial"] else 0
    score += 8 if features["is_political"] else 0
    score += max(0, (60 - sufficiency_pct) / 60 * 15)  # missing-context contribution
    score += cw_score * 10  # real-data check-worthiness nudges risk up
    return round(float(min(score, 100)), 1)


def risk_band(score):
    if score >= 70:
        return "Critical", "\U0001F534"
    if score >= 50:
        return "High Risk", "\U0001F7E0"
    if score >= 25:
        return "Medium Risk", "\U0001F7E1"
    return "Low Risk", "\U0001F7E2"


def decide_verdict(prob_map, sufficiency_pct, threshold=0.5, abstain_margin=0.12,
                    evidence_conflict=False):
    """Turn model probabilities + context sufficiency into a final verdict.

    Multi-condition abstention: the AI abstains if ANY of the following hold,
    not just the single "top-two probabilities are close" rule from v1:
      1. Top prediction confidence is below the safety threshold.
      2. The gap between the top two candidate labels is too small to call.
      3. Context similarity/support is very low (nothing to anchor on).
      4. Retrieved evidence is internally conflicting (when known).
    Each triggered condition is recorded so the UI can show a specific,
    defensible reason rather than a generic "not sure" message.
    """
    ordered = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
    top_label, top_prob = ordered[0]
    second_label, second_prob = ordered[1] if len(ordered) > 1 else (None, 0.0)
    gap = top_prob - second_prob

    # Strong contextual support can downgrade an otherwise-flagged claim
    if sufficiency_pct >= 70 and top_label != "context_sufficient":
        if prob_map.get("context_sufficient", 0) >= threshold - 0.25:
            top_label = "context_sufficient"
            top_prob = max(top_prob, prob_map.get("context_sufficient", 0))

    triggers = []
    if top_prob < threshold:
        triggers.append(f"top prediction confidence ({round(top_prob*100)}%) is below the "
                         f"{round(threshold*100)}% safety threshold")
    if gap < abstain_margin:
        triggers.append(f"the top two candidate labels are too close to call "
                         f"(gap of {round(gap*100,1)} points)")
    if 0 < sufficiency_pct < 15:
        triggers.append("the supplied context gives almost no support either way")
    if evidence_conflict:
        triggers.append("retrieved evidence directly conflicts with itself")

    abstain = bool(triggers)
    reason = ("The AI abstained because " + "; and ".join(triggers) + "."
               if triggers else None)
    return top_label, round(float(top_prob) * 100, 1), abstain, reason


VERDICT_META = {
    "context_sufficient": {"emoji": "\U0001F7E2", "label": "Context Sufficient",
                            "action": ["Continue analysis"]},
    "needs_verification": {"emoji": "\U0001F7E1", "label": "External Verification Recommended",
                            "action": ["Retrieve Evidence", "Send for Human Review"]},
    "high_priority": {"emoji": "\U0001F534", "label": "High-Priority Verification",
                       "action": ["Retrieve Evidence", "Send for Human Review"]},
}


def build_fingerprint(claim_type_label, temporal_sensitive, complexity, sufficiency_pct,
                       confidence_pct, evidence_strength_avg, conflict_level):
    """Feature 13: Claim Fingerprint -- a compact, radar-chart-friendly
    summary of everything the pipeline learned about this claim."""
    return {
        "type": claim_type_label,
        "temporal_risk": "High" if temporal_sensitive else "Low",
        "complexity": complexity["level"],
        "context_support": ("High" if sufficiency_pct >= 60 else
                             "Medium" if sufficiency_pct >= 25 else "Low"),
        "model_confidence_pct": confidence_pct,
        "evidence_strength": evidence_strength_avg,
        "conflict_level": conflict_level,
        # 0-100 axes for a radar chart: complexity/temporal/conflict are
        # inverted (higher = more caution needed) to keep "bigger = riskier"
        # consistent across all axes.
        "radar": {
            "Temporal Risk": 100 if temporal_sensitive else 15,
            "Complexity": {"LOW": 20, "MEDIUM": 55, "HIGH": 90}[complexity["level"]],
            "Context Support": sufficiency_pct,
            "Model Confidence": confidence_pct,
            "Evidence Strength": evidence_strength_avg or 0,
            "Conflict Level": conflict_level,
        },
    }


def analyze_claim(claim_text, context_text=None, threshold=0.5, evidence_conflict=None,
                   evidence_strength_avg=None):
    claim_text = claim_text.strip()
    features, reason_tags, category = engineer_features(claim_text)

    # If context was supplied, actually check it for evidence/conflicts rather
    # than leaving these as inert placeholders -- this is what makes the
    # "evidence conflicting" abstention trigger and the fingerprint's
    # evidence/conflict axes meaningful instead of always reading zero.
    if evidence_conflict is None or evidence_strength_avg is None:
        evidence_conflict, evidence_strength_avg = False, 0.0
        if context_text:
            try:
                import evidence_engine
                ev = evidence_engine.retrieve_evidence(claim_text, context_text, top_k=5)
                conflicts = evidence_engine.detect_conflict(ev)
                evidence_conflict = bool(conflicts)
                evidence_strength_avg = round(sum(e["relevance_pct"] for e in ev) / len(ev), 1) if ev else 0.0
            except Exception:
                pass

    claimability_label, claimability_conf = classify_claimability(claim_text)
    claimability_display, is_verifiable = CLAIMABILITY_LABELS[claimability_label]
    complexity = analyze_complexity(claim_text, features)

    claim_type_key = classify_claim_type(features, category)
    claim_type_label = CLAIM_TYPE_LABELS.get(claim_type_key, CLAIM_TYPE_LABELS["general"])

    sufficiency_pct, best_ctx_sentence = context_sufficiency(claim_text, context_text)

    if not is_verifiable:
        # Claimability filter: don't run verification machinery on opinions,
        # questions, or personal beliefs -- report what it IS instead.
        return {
            "claim": claim_text, "category": category,
            "claim_type_key": claim_type_key, "claim_type_label": claim_type_label,
            "temporal_sensitive": False,
            "claimability": claimability_label, "claimability_label": claimability_display,
            "claimability_confidence_pct": claimability_conf,
            "verdict": "not_a_claim", "verdict_label": f"Not Verifiable \u2014 {claimability_display}",
            "verdict_emoji": "\u23ED\uFE0F", "actions": [],
            "confidence_pct": claimability_conf, "context_sufficiency_pct": sufficiency_pct,
            "risk_score": 0.0, "risk_band": "N/A", "risk_band_emoji": "\u26AA",
            "abstain": False, "abstain_reason": None,
            "reasons": [f"Classified as {claimability_display.split(' ',1)[-1].lower()}, "
                        f"not a verifiable factual claim, so no verification verdict is produced."],
            "probabilities": {}, "best_context_sentence": best_ctx_sentence,
            "is_compound_source": False, "checkworthy_score_pct": 0.0,
            "complexity": complexity, "lifecycle_status": "NOT_A_CLAIM",
            "fingerprint": None, "model_consensus": None,
        }

    prob_map = _model_predict(claim_text)
    verdict, confidence_pct, abstain, abstain_reason = decide_verdict(
        prob_map, sufficiency_pct, threshold=threshold, evidence_conflict=evidence_conflict)

    cw_score = checkworthy_score(claim_text)
    risk = compute_risk_score(features, claim_type_key, sufficiency_pct, cw_score=cw_score)
    if complexity["level"] == "HIGH":
        risk = min(risk + 6, 100)
    elif complexity["level"] == "MEDIUM":
        risk = min(risk + 2, 100)
    risk = round(risk, 1)
    band, band_emoji = risk_band(risk)

    temporal_sensitive = bool(features["future_marker_count"] or features["present_temporal_count"]
                               or features["future_year_flag"])

    if not reason_tags:
        reason_tags = ["No strong linguistic risk signals detected; treated as a plain factual statement"]
    if best_ctx_sentence:
        reason_tags.append("Partial support found in supplied context (see evidence)")
    elif context_text:
        reason_tags.append("Supplied context does not clearly support this claim")
    if cw_score >= 0.5:
        reason_tags.append(f"Resembles sentences professional fact-checkers flagged as check-worthy "
                            f"in real debate transcripts ({round(cw_score*100)}% match)")
    if complexity["level"] == "HIGH":
        reason_tags.append("High claim complexity (multiple entities/assertions/numbers) increases risk")

    conflict_level = 80 if evidence_conflict else (10 if evidence_strength_avg else 0)

    meta = VERDICT_META[verdict]
    lifecycle_status = "ANALYZED"
    if best_ctx_sentence:
        lifecycle_status = "EVIDENCE_FOUND"
    if evidence_conflict:
        lifecycle_status = "CONFLICT_DETECTED"
    if verdict in ("needs_verification", "high_priority") or abstain:
        lifecycle_status = "HUMAN_REVIEW" if lifecycle_status in ("ANALYZED", "EVIDENCE_FOUND") else lifecycle_status

    result = {
        "claim": claim_text,
        "category": category,
        "claim_type_key": claim_type_key,
        "claim_type_label": claim_type_label,
        "temporal_sensitive": temporal_sensitive,
        "claimability": claimability_label,
        "claimability_label": claimability_display,
        "claimability_confidence_pct": claimability_conf,
        "verdict": verdict,
        "verdict_label": meta["label"],
        "verdict_emoji": meta["emoji"],
        "actions": meta["action"],
        "confidence_pct": confidence_pct,
        "context_sufficiency_pct": sufficiency_pct,
        "risk_score": risk,
        "risk_band": band,
        "risk_band_emoji": band_emoji,
        "abstain": abstain,
        "abstain_reason": abstain_reason,
        "reasons": reason_tags,
        "probabilities": {k: round(float(v) * 100, 1) for k, v in prob_map.items()},
        "best_context_sentence": best_ctx_sentence,
        "is_compound_source": False,
        "checkworthy_score_pct": round(cw_score * 100, 1),
        "complexity": complexity,
        "lifecycle_status": lifecycle_status,
        "model_consensus": model_consensus(claim_text),
    }
    result["fingerprint"] = build_fingerprint(
        claim_type_label, temporal_sensitive, complexity, sufficiency_pct,
        confidence_pct, evidence_strength_avg, conflict_level)

    if abstain:
        result["verdict"] = "abstain"
        result["verdict_label"] = "AI Abstains \u2014 Confidence Below Safety Threshold"
        result["verdict_emoji"] = "\U0001F6D1"
        result["actions"] = ["Retrieve Evidence", "Send for Human Review"]
        result["lifecycle_status"] = "HUMAN_REVIEW"
    return result


def extract_and_analyze(document_text, context_text=None, threshold=0.5, decompose=True):
    """Full pipeline: sentence split -> claim filter -> compound decomposition ->
    per-claim analysis. Returns a list of analysis dicts plus offsets for
    highlighting the original text.
    """
    sentences = split_sentences(document_text)
    results = []
    for sent in sentences:
        if not _looks_like_claim(sent):
            continue
        sub_claims = split_compound_claim(sent) if decompose else [sent]
        is_compound = len(sub_claims) > 1
        for sub in sub_claims:
            analysis = analyze_claim(sub, context_text=context_text, threshold=threshold)
            analysis["source_sentence"] = sent
            analysis["is_compound_source"] = is_compound
            results.append(analysis)
    return results


def investigation_roadmap(analysis):
    """Feature 30: Claim Investigation Roadmap -- concrete next checks."""
    steps = []
    if analysis["claim_type_key"] == "financial":
        steps += ["Find official financial reports or regulatory filings.",
                  "Check the exact reporting period referenced.",
                  "Compare against the prior period's figures."]
    if analysis["claim_type_key"] == "health":
        steps += ["Check for peer-reviewed clinical trial data.",
                  "Verify regulatory approval status (e.g. FDA/EMA equivalent).",
                  "Look for independent replication of the result."]
    if analysis["claim_type_key"] in ("political", "person"):
        steps += ["Check official government or organizational records.",
                  "Verify current officeholder/status via a primary source."]
    if analysis["temporal_sensitive"]:
        steps.append("Confirm the claim's timeframe and check for more recent updates.")
    if analysis["risk_score"] >= 50:
        steps.append("Prioritize cross-referencing with at least two independent sources.")
    if not steps:
        steps.append("Search for a primary or authoritative source that directly states this fact.")
    steps.append("Resolve any conflicting figures before accepting the claim.")
    # de-duplicate while preserving order
    seen = set()
    ordered = []
    for s in steps:
        if s not in seen:
            ordered.append(s)
            seen.add(s)
    return ordered
