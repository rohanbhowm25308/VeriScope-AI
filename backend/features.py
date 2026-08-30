"""
features.py
------------
Rule-based / lexical feature engineering shared between:
  - training the baseline scikit-learn classifiers (train_models.py)
  - live inference in the Flask API (claim_analyzer.py)

Design choice (documented): we deliberately avoid heavy pretrained
transformer downloads (no BERT/sentence-transformers) because (a) the
problem statement prescribes no architecture, (b) it keeps the project
zero-cost / fully offline-capable per the resource boundary, and (c) it
keeps every feature explainable, which the "Why was this flagged?" and
error-analysis deliverables need. TF-IDF + engineered linguistic features
is the justified baseline; the README documents heavier alternatives
(sentence embeddings, BERT fine-tuning) as future work.
"""

import re
from datetime import datetime

CURRENT_YEAR = datetime.now().year

# --- lexicons -----------------------------------------------------------

FUTURE_MARKERS = [
    r"\bwill\b", r"\bshall\b", r"\bgoing to\b", r"\bexpected to\b",
    r"\bby \d{4}\b", r"\bnext (year|month|quarter|week|decade|season)\b",
    r"\bupcoming\b", r"\bplans to\b", r"\bforecast\b", r"\bprojected\b",
    r"\bsoon\b", r"\bwithin \w+ (year|month|week)s?\b",
]

PRESENT_TEMPORAL_MARKERS = [
    r"\bcurrently\b", r"\bis the\b", r"\bare the\b", r"\bnow\b",
    r"\bat present\b", r"\bthis (year|season|quarter)\b", r"\btoday\b",
]

CONDITIONAL_MARKERS = [
    r"\bif\b", r"\bcould\b", r"\bmight\b", r"\bmay\b", r"\bwould\b",
    r"\bunless\b", r"\bprovided that\b", r"\bin case\b",
]

HEDGE_MARKERS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bpossibly\b", r"\blikely\b",
    r"\ballegedly\b", r"\breportedly\b", r"\bsuggests\b", r"\bappears to\b",
]

SUPERLATIVE_MARKERS = [
    r"\bmost\b", r"\bbest\b", r"\bworst\b", r"\blargest\b", r"\bsmallest\b",
    r"\bfirst ever\b", r"\bnever before\b", r"\bhighest\b", r"\blowest\b",
    r"\brecord\b", r"\bunprecedented\b", r"\ball[- ]time\b", r"\bever recorded\b",
]

IMPLICIT_CAUSAL_MARKERS = [
    r"\bclearly\b", r"\bdestroyed\b", r"\bruined\b", r"\bdamaged\b",
    r"\bended\b", r"\bcost .*(million|billion)\b", r"\bsplit\b", r"\bcaused\b",
]

NUMERIC_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s?(%|percent|million|billion|thousand|kg|kilograms|km)?\b")
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
MONEY_PATTERN = re.compile(r"[$₹€£]\s?\d|\bdollars?\b|\brupees?\b|\beuros?\b")

CATEGORY_KEYWORDS = {
    "financial": [r"\brevenue\b", r"\bstock\b", r"\bprofit\b", r"\bgdp\b", r"\bmarket\b",
                  r"\bprice\b", r"\bshares?\b", r"\binterest rate", r"\beconomy\b", r"\bdollars?\b"],
    "health": [r"\bvaccine\b", r"\bcancer\b", r"\bhospital\b", r"\bdisease\b", r"\btreatment\b",
               r"\bdrug\b", r"\bmortality\b", r"\bdiabetes\b", r"\btherapy\b", r"\bmedicine\b"],
    "political": [r"\bgovernment\b", r"\bpolicy\b", r"\bminister\b", r"\belection\b",
                  r"\bsenator\b", r"\bparliament\b", r"\bmayor\b", r"\btreaty\b", r"\bparty\b"],
    "scientific": [r"\bstudy\b", r"\bresearch\b", r"\bsatellite\b", r"\bexperiment\b",
                   r"\btemperature\b", r"\bspecies\b", r"\bboils?\b", r"\borbit\b"],
    "environmental": [r"\bclimate\b", r"\bcarbon\b", r"\brainforest\b", r"\bemissions?\b",
                       r"\bsea levels?\b", r"\bdrought\b", r"\brecycling\b", r"\bpollution\b"],
    "corporate": [r"\bcompany\b", r"\bstartup\b", r"\bceo\b", r"\bproduct\b", r"\bfactory\b",
                  r"\bemployees?\b", r"\bworkforce\b", r"\bmerger\b", r"\boffice\b"],
    "person": [r"\bhe \b", r"\bshe \b", r"\bcaptain\b", r"\bactor\b", r"\bplayer\b"],
    "statistical": [r"\bpopulation\b", r"\brate\b", r"\baverage\b", r"\bpercent\b", r"\bpassengers\b"],
    "geography": [r"\bmountain\b", r"\bocean\b", r"\briver\b", r"\bcapital\b", r"\bcontinent\b",
                  r"\bstates?\b", r"\bunion territor"],
}

STABLE_FACT_HINTS = [
    r"\bhas \d+ (states|sides|chambers|days)\b", r"\bboils? at\b", r"\bfreezing point\b",
    r"\borbits\b", r"\bwas founded in\b", r"\bwas adopted in\b", r"\bended in\b",
    r"\bbegan in\b", r"\bis (a|an) programming language\b", r"\bis located in\b",
    r"\breduces the risk\b", r"\bincreases the risk\b",
]


def _count_matches(patterns, text_lower):
    return sum(1 for p in patterns if re.search(p, text_lower))


def split_sentences(text):
    """Lightweight sentence splitter (no external tokenizer download needed).
    Handles standard terminal punctuation while being conservative about
    abbreviations (Mr., Dr., e.g., U.S., etc.) so claims aren't chopped mid-way.
    """
    text = text.strip()
    if not text:
        return []
    abbrev_guard = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|U\.S|U\.K)\.",
                           lambda m: m.group(0).replace(".", "\u2024"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\u201c])", abbrev_guard)
    sentences = [p.replace("\u2024", ".").strip() for p in parts if p.strip()]
    return sentences


def split_compound_claim(sentence):
    """Attempt to decompose a compound sentence into independent sub-claims.
    Heuristic: split on coordinating conjunctions ('and', ';') only when both
    sides look like independent clauses (each has a verb-ish token and is not
    trivially short) -- avoids destroying noun-phrase lists like
    "revenue and profit both increased".
    """
    sentence = sentence.strip().rstrip(".")
    candidates = re.split(r"\s*;\s*|\s+\band\b\s+(?=[a-z])", sentence, flags=re.IGNORECASE)
    if len(candidates) <= 1:
        return [sentence + "."]
    verb_ish = re.compile(
        r"\b(is|are|was|were|will|has|have|had|launched|increased|hired|weighs|grew|"
        r"rose|fell|reported|announced|approved|signed|expects?|becomes?|reduced|"
        r"caused|cost|ended|damaged|destroyed|passed|built|added|opened)\b", re.IGNORECASE)
    parts = []
    buffer = ""
    for i, cand in enumerate(candidates):
        cand = cand.strip()
        if not buffer:
            buffer = cand
        else:
            buffer = buffer + " and " + cand
        if verb_ish.search(cand) and len(cand.split()) >= 3:
            parts.append(buffer.strip())
            buffer = ""
    if buffer:
        if parts:
            parts[-1] = parts[-1] + " and " + buffer
        else:
            parts.append(buffer.strip())
    if len(parts) <= 1:
        return [sentence + "."]

    # Carry the subject forward when a later part starts directly with a verb/modal
    # (e.g. "India has 28 states and will become ..." -> second part has no subject).
    subject_verb_split = re.compile(
        r"^(.*?)\s+\b(is|are|was|were|will|shall|has|have|had|increased|reported|"
        r"launched|weighs|grew|rose|fell|announced|approved|signed|expects?|"
        r"becomes?|reduced|caused|cost|ended|damaged|destroyed|passed|built|"
        r"added|opened|hired)\b(.*)$", re.IGNORECASE)

    starts_with_verb = re.compile(
        r"^(is|are|was|were|will|shall|has|have|had|increased|reported|"
        r"launched|weighs|grew|rose|fell|announced|approved|signed|expects?|"
        r"becomes?|reduced|caused|cost|ended|damaged|destroyed|passed|built|"
        r"added|opened|hired)\b", re.IGNORECASE)

    fixed = []
    current_subject = None
    for p in parts:
        if starts_with_verb.match(p):
            has_own_subject = False
            m = None
        else:
            m = subject_verb_split.match(p)
            has_own_subject = bool(m and m.group(1).strip())
        if has_own_subject:
            current_subject = m.group(1).strip()
            fixed.append(p)
        elif current_subject:
            fixed.append(f"{current_subject} {p[0].lower()}{p[1:]}")
        else:
            fixed.append(p)

    return [p.strip().rstrip(".") + "." for p in fixed]


def detect_category(text_lower):
    scores = {cat: _count_matches(pats, text_lower) for cat, pats in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general"
    return best


def engineer_features(claim_text):
    """Return (feature_dict, reason_tags) for a single claim string."""
    text_lower = claim_text.lower()
    words = text_lower.split()
    n_words = max(len(words), 1)

    future_hits = _count_matches(FUTURE_MARKERS, text_lower)
    present_hits = _count_matches(PRESENT_TEMPORAL_MARKERS, text_lower)
    conditional_hits = _count_matches(CONDITIONAL_MARKERS, text_lower)
    hedge_hits = _count_matches(HEDGE_MARKERS, text_lower)
    superlative_hits = _count_matches(SUPERLATIVE_MARKERS, text_lower)
    implicit_hits = _count_matches(IMPLICIT_CAUSAL_MARKERS, text_lower)
    stable_hits = _count_matches(STABLE_FACT_HINTS, text_lower)

    numeric_hits = len(NUMERIC_PATTERN.findall(text_lower))
    years_found = YEAR_PATTERN.findall(claim_text)
    money_hits = len(MONEY_PATTERN.findall(text_lower))
    future_year_flag = 0
    for y in YEAR_PATTERN.finditer(claim_text):
        year_val = int(y.group(0))
        if year_val > CURRENT_YEAR:
            future_year_flag = 1

    has_and = 1 if re.search(r"\band\b", text_lower) else 0
    has_conjunction_split = 1 if len(split_compound_claim(claim_text)) > 1 else 0

    category = detect_category(text_lower)

    features = {
        "n_words": n_words,
        "future_marker_count": future_hits,
        "present_temporal_count": present_hits,
        "conditional_marker_count": conditional_hits,
        "hedge_marker_count": hedge_hits,
        "superlative_count": superlative_hits,
        "implicit_causal_count": implicit_hits,
        "stable_fact_hint_count": stable_hits,
        "numeric_count": numeric_hits,
        "year_count": len(years_found),
        "future_year_flag": future_year_flag,
        "money_hits": money_hits,
        "has_and": has_and,
        "has_conjunction_split": has_conjunction_split,
        "is_health": 1 if category == "health" else 0,
        "is_financial": 1 if category == "financial" else 0,
        "is_political": 1 if category == "political" else 0,
    }

    reason_tags = []
    if future_hits or future_year_flag:
        reason_tags.append("Refers to a future or time-dependent event")
    if conditional_hits:
        reason_tags.append("Conditional / predictive phrasing detected")
    if numeric_hits or money_hits:
        reason_tags.append("Contains a numerical or statistical assertion")
    if superlative_hits:
        reason_tags.append("Uses superlative or extremal language (e.g. 'most', 'first ever')")
    if implicit_hits:
        reason_tags.append("Implicit causal claim embedded in the sentence")
    if hedge_hits:
        reason_tags.append("Contains hedging language signaling uncertainty")
    if stable_hits and not (future_hits or numeric_hits > 2):
        reason_tags.append("Matches patterns typical of stable/definitional facts")
    if present_hits:
        reason_tags.append("States a current status that may change over time")

    return features, reason_tags, category


CLAIM_TYPE_LABELS = {
    "future": "\U0001F52E Future prediction",
    "temporal": "\U0001F4C5 Time-sensitive",
    "statistical": "\U0001F4CA Statistical claim",
    "financial": "\U0001F4B0 Financial claim",
    "health": "\U0001F3E5 Health-related claim",
    "political": "\U0001F3DB\uFE0F Political / public-policy claim",
    "scientific": "\U0001F52C Scientific claim",
    "environmental": "\U0001F30D Environmental claim",
    "corporate": "\U0001F3E2 Corporate claim",
    "person": "\U0001F464 Person-related claim",
    "general": "\U0001F4CC General factual claim",
}


def classify_claim_type(features, category):
    if features["future_marker_count"] or features["future_year_flag"]:
        return "future"
    if features["present_temporal_count"]:
        return "temporal"
    if category in ("financial", "health", "political", "scientific", "environmental", "corporate", "person"):
        return category
    if features["numeric_count"] >= 2:
        return "statistical"
    return "general"


# ---------------------------------------------------------------------------
# Claimability filter: is this even a verifiable factual claim, or opinion /
# question / personal belief / too ambiguous to check? Running verification
# machinery on non-claims wastes review effort and produces nonsense verdicts,
# so this runs BEFORE the verification-need model.
# ---------------------------------------------------------------------------

OPINION_MARKERS = [
    r"\bi (think|believe|feel|suspect|guess|reckon)\b", r"\bin my (opinion|view)\b",
    r"\bpersonally\b", r"\bi'd say\b", r"\bmy take\b", r"\bshould be\b", r"\bought to\b",
    r"\bbetter than\b", r"\bworse than\b", r"\bbeautiful\b", r"\bugly\b", r"\bamazing\b",
    r"\bterrible\b", r"\bthe best\b", r"\bthe worst\b(?!.*\d)", r"\bfavorite\b",
    r"\blove\b", r"\bhate\b", r"\bdisgusting\b", r"\bwonderful\b",
]
BELIEF_MARKERS = [
    r"\bi (hope|wish|assume|expect|doubt)\b", r"\bwe believe\b", r"\bit seems to me\b",
    r"\bi'm not sure but\b", r"\bmaybe\b.{0,3}$",
]


def classify_claimability(text):
    """Returns (label, confidence_pct) where label is one of:
    'verifiable_claim', 'opinion', 'question', 'personal_belief', 'ambiguous'."""
    t = text.strip()
    low = t.lower()

    if t.endswith("?") or re.match(r"^(what|who|why|how|when|where|is|are|do|does|did|can|could|should|would)\b", low):
        return "question", 92.0

    if _count_matches(BELIEF_MARKERS, low) and not re.search(r"\d", t):
        return "personal_belief", 80.0

    opinion_hits = _count_matches(OPINION_MARKERS, low)
    if opinion_hits and not re.search(r"\d", t):
        # subjective language with no factual anchor (no number/date/named stat)
        return "opinion", min(60 + opinion_hits * 10, 95.0)

    words = t.split()
    if len(words) < 4:
        return "ambiguous", 50.0

    # A real POS tagger would identify the main verb reliably; without one,
    # use a broad common-verb list as a CONFIDENCE booster, not a hard gate
    # -- gating on it caused false "ambiguous" verdicts for legitimate claims
    # using verbs outside the list (e.g. "boils", "contains", "shows").
    has_verb = bool(re.search(
        r"\b(is|are|was|were|will|has|have|had|launched|increased|hired|weighs|grew|"
        r"rose|fell|reported|announced|approved|signed|expects?|becomes?|reduced|"
        r"caused|cost|ended|damaged|destroyed|passed|built|added|opened|reached|"
        r"stated|said|claims?|shows?|found|boils?|contains?|orbits?|consists?|"
        r"occurs?|requires?|produces?|releases?|forms?|remains?)\b", low))
    has_entity = bool(re.search(r"\b[A-Z][a-z]+\b", t))
    has_number_or_date = bool(re.search(r"\d", t))

    anchor_count = sum([has_entity, has_number_or_date])
    if opinion_hits == 0 and (has_verb or anchor_count >= 1):
        conf = 72 + anchor_count * 8 + (10 if has_verb else 0)
        return "verifiable_claim", max(min(conf, 98.0), 55.0)
    if has_verb and anchor_count >= 1:
        conf = 80 + anchor_count * 8 - (15 if opinion_hits else 0)
        return "verifiable_claim", max(min(conf, 98.0), 55.0)
    if has_verb:
        return "verifiable_claim", 62.0
    return "ambiguous", 50.0


CLAIMABILITY_LABELS = {
    "verifiable_claim": ("\u2705 Verifiable Claim", True),
    "opinion": ("\U0001F4AC Opinion / Subjective", False),
    "question": ("\u2753 Question", False),
    "personal_belief": ("\U0001F52E Personal Belief", False),
    "ambiguous": ("\U0001F937 Ambiguous Statement", False),
}


# ---------------------------------------------------------------------------
# Claim complexity analyzer -- feeds into the risk engine and is reported
# separately so "does complexity affect verification accuracy" can be
# investigated in the error-analysis writeup.
# ---------------------------------------------------------------------------

def analyze_complexity(text, features):
    factors = {
        "has_numeric": bool(features["numeric_count"] or features["money_hits"]),
        "has_date": bool(features["year_count"]),
        "has_multiple_entities": len(re.findall(r"\b[A-Z][a-z]{2,}\b", text)) >= 2,
        "has_multiple_assertions": bool(features["has_conjunction_split"]) or
            len(re.findall(r"[,;]", text)) >= 2,
    }
    score = sum(factors.values())
    if score >= 3:
        level = "HIGH"
    elif score >= 1:
        level = "MEDIUM"
    else:
        level = "LOW"
    return {"level": level, "score": score, "factors": factors}
