"""
build_seed_dataset.py
----------------------
Generates the ClaimGuard seed dataset: a weakly-labeled collection of claims
used to train and evaluate the baseline "verification requirement" classifier.

WHY A GENERATED DATASET (documented limitation):
ML-T2-061 explicitly prescribes no dataset. There is no off-the-shelf, freely
licensed corpus labeled specifically for "does this claim require EXTERNAL
verification" (as opposed to true/false fact-checking corpora like FEVER or
CheckThat!, which label veracity, not verification-need). Building one by
hand for every category/type combination lets us control class balance and
guarantee coverage of the difficult cases the problem statement calls out
(compound, temporal, conditional, implicit). This is a weak-supervision seed
set: labels are assigned by a human-written rule (see `label_rule` below),
not by an external ground truth. This is a known, documented limitation of
the baseline -- see README.md "Dataset & Limitations".

Run:
    python3 build_seed_dataset.py
Produces:
    seed_claims.csv  (columns: claim, category, label)
labels are one of: context_sufficient, needs_verification, high_priority
"""

import csv
import itertools
import random

random.seed(7)

# ---------------------------------------------------------------------------
# Building blocks per category. Each entry is (claim_text, label)
# label_rule used for consistency (documented, not hidden):
#   context_sufficient -> claim is a stable, definitional, or already-settled
#                          fact that does not depend on retrieval
#   needs_verification -> claim is checkable in principle but this system has
#                          no supplied context/evidence for it right now
#   high_priority       -> needs_verification AND (future-dated OR carries
#                          numeric/financial/health/political stakes OR is
#                          extremal/superlative language)
# ---------------------------------------------------------------------------

rows = []

def add(claim, category, label):
    rows.append((claim.strip(), category, label))

# ---- Stable / definitional facts (context_sufficient) ----
stable_facts = [
    ("India has 28 states and 8 union territories.", "geography"),
    ("Water boils at 100 degrees Celsius at sea level.", "scientific"),
    ("The Earth orbits the Sun once every 365.25 days.", "scientific"),
    ("The human heart has four chambers.", "scientific"),
    ("Mount Everest is the tallest mountain above sea level.", "geography"),
    ("The Pacific Ocean is the largest ocean on Earth.", "geography"),
    ("A triangle has three sides.", "scientific"),
    ("The United Nations was founded in 1945.", "political"),
    ("Python is a programming language.", "corporate"),
    ("The French Revolution began in 1789.", "political"),
    ("Oxygen is required for human respiration.", "scientific"),
    ("The capital of Japan is Tokyo.", "geography"),
    ("A leap year has 366 days.", "scientific"),
    ("The Great Wall of China is located in China.", "geography"),
    ("Light travels faster than sound.", "scientific"),
    ("The Constitution of India was adopted in 1949.", "political"),
    ("A hexagon has six sides.", "scientific"),
    ("The Amazon River flows through South America.", "geography"),
    ("World War II ended in 1945.", "political"),
    ("The freezing point of water is 0 degrees Celsius.", "scientific"),
]
for c, cat in stable_facts:
    add(c, cat, "context_sufficient")

# ---- Present-tense claims about current, checkable-but-unverified state
#      (needs_verification, not future, not extreme) ----
present_claims = [
    ("The company reported a profit last quarter.", "corporate"),
    ("The new smartphone model has a larger battery than its predecessor.", "corporate"),
    ("The local hospital has added twenty new beds.", "health"),
    ("The city's population grew slightly this year.", "statistical"),
    ("The bridge renovation is currently underway.", "corporate"),
    ("The museum's new exhibit features contemporary art.", "corporate"),
    ("The regional airline added a new domestic route.", "corporate"),
    ("The university published a new research paper on climate patterns.", "scientific"),
    ("The startup recently moved to a larger office.", "corporate"),
    ("The library extended its weekend hours.", "corporate"),
    ("The team signed a new player this season.", "person"),
    ("The restaurant chain opened three new locations.", "corporate"),
    ("The city council approved a new park design.", "political"),
    ("The software update fixed several bugs.", "corporate"),
    ("The farmers reported a better harvest this year.", "statistical"),
]
for c, cat in present_claims:
    add(c, cat, "needs_verification")

# ---- Numeric / statistical claims without evidence (needs_verification / high_priority) ----
stat_claims = [
    ("The company increased its revenue by 40 percent in 2025.", "financial", "high_priority"),
    ("Unemployment fell by 2 percent last month.", "statistical", "high_priority"),
    ("The city's crime rate dropped by 15 percent this year.", "statistical", "high_priority"),
    ("The satellite weighs 5,000 kilograms.", "scientific", "needs_verification"),
    ("The company's stock price rose by 8 percent yesterday.", "financial", "high_priority"),
    ("The vaccine is 95 percent effective against the virus.", "health", "high_priority"),
    ("The factory produces 10,000 units per day.", "corporate", "needs_verification"),
    ("Average rainfall increased by 20 percent this decade.", "environmental", "needs_verification"),
    ("The country's GDP grew by 6 percent last year.", "financial", "high_priority"),
    ("The airline carried 2 million passengers last month.", "statistical", "needs_verification"),
]
for c, cat, lab in stat_claims:
    add(c, cat, lab)

# ---- Future / prediction claims (high_priority: time-dependent) ----
future_claims = [
    ("India's GDP will surpass the United States by 2030.", "financial"),
    ("The company will launch its new product next month.", "corporate"),
    ("The government will introduce a new AI policy next year.", "political"),
    ("This treatment will reduce cancer mortality by 30 percent by 2030.", "health"),
    ("The stock market will rise sharply next quarter.", "financial"),
    ("The city will complete the new metro line by 2027.", "political"),
    ("The team will win the championship next season.", "person"),
    ("The country will become carbon neutral by 2050.", "environmental"),
    ("The startup will go public within two years.", "corporate"),
    ("Interest rates will increase next quarter.", "financial"),
    ("The population will double by 2060.", "statistical"),
    ("A new species of frog will be discovered in the region soon.", "environmental"),
    ("The company expects to double its workforce next year.", "corporate"),
    ("The new drug will be approved by regulators next year.", "health"),
]
for c, cat in future_claims:
    add(c, cat, "high_priority")

# ---- Conditional / predictive claims (needs_verification, sometimes high_priority) ----
conditional_claims = [
    ("If interest rates increase, housing prices could fall.", "financial", "needs_verification"),
    ("If the treaty is signed, tensions may ease in the region.", "political", "needs_verification"),
    ("If demand continues to rise, prices could increase further.", "financial", "needs_verification"),
    ("If the drought continues, crop yields may decline sharply.", "environmental", "high_priority"),
    ("If the merger goes through, thousands of jobs could be affected.", "corporate", "high_priority"),
    ("If approved, the vaccine could reach markets within a year.", "health", "high_priority"),
]
for c, cat, lab in conditional_claims:
    add(c, cat, lab)

# ---- Implicit claims (needs_verification: factual assertion embedded in opinion-like text) ----
implicit_claims = [
    ("The company's decision clearly destroyed investor confidence.", "corporate", "high_priority"),
    ("The new policy has already damaged small businesses.", "political", "high_priority"),
    ("The scandal ended the senator's political career.", "political", "high_priority"),
    ("The product recall ruined the brand's reputation overnight.", "corporate", "needs_verification"),
    ("The outage cost the company millions in lost sales.", "financial", "high_priority"),
    ("The controversy split the party into two factions.", "political", "needs_verification"),
]
for c, cat, lab in implicit_claims:
    add(c, cat, lab)

# ---- Person / title claims (temporally sensitive) ----
person_claims = [
    ("Tesla is the world's most valuable automobile company.", "corporate", "high_priority"),
    ("She is the CEO of the company.", "person", "needs_verification"),
    ("He currently serves as the finance minister.", "political", "needs_verification"),
    ("The team's captain is the league's top scorer this season.", "person", "needs_verification"),
    ("The city's mayor was re-elected for a third term.", "political", "needs_verification"),
    ("The actor is the highest-paid performer in the industry this year.", "person", "high_priority"),
]
for c, cat, lab in person_claims:
    add(c, cat, lab)

# ---- Health-related claims (usually high stakes) ----
health_claims = [
    ("The new diet cures diabetes within weeks.", "health", "high_priority"),
    ("Regular exercise reduces the risk of heart disease.", "health", "context_sufficient"),
    ("The supplement eliminates the need for prescribed medication.", "health", "high_priority"),
    ("The hospital reported zero infections last month.", "health", "needs_verification"),
    ("Smoking increases the risk of lung cancer.", "health", "context_sufficient"),
    ("The new therapy has no side effects.", "health", "high_priority"),
]
for c, cat, lab in health_claims:
    add(c, cat, lab)

# ---- Environmental / scientific extremal claims ----
env_claims = [
    ("This is the hottest year ever recorded on Earth.", "environmental", "high_priority"),
    ("Global sea levels rose by 3.4 millimeters last year.", "environmental", "high_priority"),
    ("The rainforest lost 10 percent of its cover this decade.", "environmental", "high_priority"),
    ("Recycling rates improved slightly in the region.", "environmental", "needs_verification"),
    ("Carbon emissions in the region fell for the first time in a decade.", "environmental", "high_priority"),
]
for c, cat, lab in env_claims:
    add(c, cat, lab)

# ---- Compound claims (kept whole here; decomposition handled separately in code,
#      but useful for the classifier to see conjunction patterns) ----
compound_claims = [
    ("India launched a new satellite in 2025 and the satellite weighs 5,000 kilograms.", "scientific", "needs_verification"),
    ("The company increased its revenue by 40 percent in 2025 and hired 10,000 employees.", "corporate", "high_priority"),
    ("India has 28 states and will become the world's largest economy by 2030.", "political", "high_priority"),
    ("The city built a new stadium and it will host the finals next year.", "corporate", "needs_verification"),
    ("The vaccine passed clinical trials and will be available by next spring.", "health", "high_priority"),
]
for c, cat, lab in compound_claims:
    add(c, cat, lab)

# Write CSV
out_path = "seed_claims.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["claim", "category", "label"])
    for claim, cat, lab in rows:
        writer.writerow([claim, cat, lab])

# Simple class balance report
from collections import Counter
label_counts = Counter(r[2] for r in rows)
cat_counts = Counter(r[1] for r in rows)
print(f"Wrote {len(rows)} rows to {out_path}")
print("Label distribution:", dict(label_counts))
print("Category distribution:", dict(cat_counts))
