"""
counterfactual.py
-------------------
Feature 15: Counterfactual Claim Testing. Given a claim, generates a small
set of controlled perturbations (numeric value changed, temporal marker
flipped, superlative removed/added) and reruns the full pipeline on each,
so a reviewer can see how sensitive the model's verdict is to specific
surface changes. This is genuinely useful for the error-analysis /
robustness-testing section of a report -- it's not just a UI gimmick, the
perturbations and resulting verdict changes are real model outputs.
"""

import re


def _perturb_numeric(text):
    """Replace the first number found with a few different values."""
    m = re.search(r"\b\d+(?:\.\d+)?\b", text)
    if not m:
        return []
    original = m.group(0)
    try:
        val = float(original)
    except ValueError:
        return []
    variants = []
    for factor, label in [(0.5, "halved"), (2.0, "doubled"), (0.1, "far smaller")]:
        new_val = val * factor
        new_val_str = str(int(new_val)) if new_val == int(new_val) else str(round(new_val, 1))
        variant_text = text[:m.start()] + new_val_str + text[m.end():]
        variants.append({"variant": variant_text, "change": f"numeric value {label} ({original} -> {new_val_str})"})
    return variants


def _perturb_superlative(text):
    """Remove/soften a superlative or extremal word if present."""
    pairs = [
        (r"\bmost\b", "a", "removed superlative 'most'"),
        (r"\bbest\b", "a good", "softened 'best' -> 'a good'"),
        (r"\bworst\b", "a poor", "softened 'worst' -> 'a poor'"),
        (r"\bever recorded\b", "recently recorded", "softened 'ever recorded' -> 'recently recorded'"),
        (r"\brecord\b", "notable", "softened 'record' -> 'notable'"),
    ]
    variants = []
    low = text.lower()
    for pattern, replacement, change_desc in pairs:
        if re.search(pattern, low):
            new_text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
            variants.append({"variant": new_text, "change": change_desc})
    return variants


def _perturb_temporal(text):
    """Flip future <-> past/present tense markers to test temporal sensitivity."""
    variants = []
    low = text.lower()
    if re.search(r"\bwill\b", low):
        new_text = re.sub(r"\bwill\b", "has", text, count=1, flags=re.IGNORECASE)
        variants.append({"variant": new_text, "change": "future tense 'will' -> present-perfect 'has' (removes future-dating)"})
    m = re.search(r"\bby (\d{4})\b", text)
    if m:
        year = int(m.group(1))
        new_text = text[:m.start()] + f"in {year - 10}" + text[m.end():]
        variants.append({"variant": new_text, "change": f"target year moved from {year} to {year-10} (past instead of future)"})
    return variants


def generate_counterfactuals(claim_text, max_variants=5):
    variants = []
    variants += _perturb_numeric(claim_text)
    variants += _perturb_superlative(claim_text)
    variants += _perturb_temporal(claim_text)
    # de-duplicate identical variant text
    seen = set()
    unique = []
    for v in variants:
        if v["variant"] not in seen and v["variant"] != claim_text:
            seen.add(v["variant"])
            unique.append(v)
    return unique[:max_variants]


def run_counterfactual_analysis(claim_text, analyze_fn):
    """analyze_fn: claim_analyzer.analyze_claim (injected to avoid a circular
    import, since counterfactual.py is a small, independent utility module)."""
    baseline = analyze_fn(claim_text)
    variants = generate_counterfactuals(claim_text)
    results = []
    for v in variants:
        result = analyze_fn(v["variant"])
        results.append({
            "variant_claim": v["variant"],
            "change": v["change"],
            "verdict": result["verdict"],
            "verdict_changed": result["verdict"] != baseline["verdict"],
            "risk_score": result["risk_score"],
            "risk_delta": round(result["risk_score"] - baseline["risk_score"], 1),
            "confidence_pct": result["confidence_pct"],
        })
    n_changed = sum(1 for r in results if r["verdict_changed"])
    sensitivity = "HIGH" if len(results) and n_changed / len(results) >= 0.5 else \
                  "MEDIUM" if n_changed > 0 else "LOW"
    return {
        "baseline_claim": claim_text, "baseline_verdict": baseline["verdict"],
        "baseline_risk_score": baseline["risk_score"],
        "variants": results, "n_verdict_changes": n_changed, "n_variants": len(results),
        "sensitivity": sensitivity,
    }
