"""
build_seed_dataset_v2.py
--------------------------
V2 of the synthetic seed dataset: same weak-supervision methodology as v1
(build_seed_dataset.py, kept for provenance), but roughly 4x larger and
explicitly balanced across the three labels using templated entity
substitution -- documented, not hidden, per the project's honesty
commitments (see README "Dataset & Limitations").

This dataset trains the fine-grained 3-way (context_sufficient /
needs_verification / high_priority) classifier. The separate, REAL,
human-annotated CLEF CheckThat! 2019 dataset (see prepare_checkthat_data.py)
trains a dedicated binary check-worthiness model, and its probability output
is blended in as an additional engineered feature at inference time
(see claim_analyzer.py) -- so the fine-grained model is no longer trained
in isolation from real-world signal, even though its own labels remain
synthetic.

Run:
    python3 build_seed_dataset_v2.py
Produces:
    seed_claims_v2.csv  (columns: claim, category, label)
"""

import csv
import random
from collections import Counter

random.seed(11)

rows = []


def add(claim, category, label):
    rows.append((claim.strip(), category, label))


# =====================================================================
# 1. STABLE / DEFINITIONAL FACTS -> context_sufficient
# =====================================================================
stable_templates = [
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
    ("The Nile is a river in Africa.", "geography"),
    ("A century has 100 years.", "scientific"),
    ("The moon orbits the Earth.", "scientific"),
    ("Canada is located in North America.", "geography"),
    ("The human body has 206 bones in adulthood.", "scientific"),
    ("Photosynthesis converts sunlight into chemical energy in plants.", "scientific"),
    ("The Sahara is a desert in Africa.", "geography"),
    ("Democracy is a system of government by the people.", "political"),
    ("A square has four equal sides.", "scientific"),
    ("The speed of light is approximately 300,000 kilometers per second.", "scientific"),
    ("Regular exercise reduces the risk of heart disease.", "health"),
    ("Smoking increases the risk of lung cancer.", "health"),
    ("A balanced diet includes proteins, carbohydrates, and fats.", "health"),
    ("Sleep is essential for physical and mental recovery.", "health"),
    ("Vaccines work by training the immune system to recognize pathogens.", "health"),
    ("The stock market consists of exchanges where shares are traded.", "financial"),
    ("Inflation refers to a general rise in prices over time.", "financial"),
    ("A budget deficit occurs when spending exceeds revenue.", "financial"),
    ("Renewable energy comes from naturally replenished sources.", "environmental"),
    ("Deforestation reduces the number of trees in a region.", "environmental"),
    ("Recycling reduces the amount of waste sent to landfills.", "environmental"),
    ("A corporation is a legal entity separate from its owners.", "corporate"),
    ("A merger occurs when two companies combine into one.", "corporate"),
    ("An algorithm is a step-by-step procedure for solving a problem.", "scientific"),
]
for c, cat in stable_templates:
    add(c, cat, "context_sufficient")

definitional_subjects = [
    ("Brazil", "South America"), ("Egypt", "Africa"), ("Germany", "Europe"),
    ("Thailand", "Asia"), ("Australia", "Oceania"), ("Kenya", "Africa"),
    ("Peru", "South America"), ("Norway", "Europe"), ("Vietnam", "Asia"),
]
for country, continent in definitional_subjects:
    add(f"{country} is located in {continent}.", "geography", "context_sufficient")

# =====================================================================
# 2. PRESENT-TENSE UNVERIFIED CLAIMS -> needs_verification
# =====================================================================
present_templates = [
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
    ("The factory hired additional staff for the holiday season.", "corporate"),
    ("The clinic introduced a new appointment booking system.", "health"),
    ("The school district revised its curriculum this year.", "political"),
    ("The nonprofit raised funds for a new community center.", "corporate"),
    ("The airport terminal underwent minor renovations.", "corporate"),
    ("The company's new office reduced its carbon footprint.", "environmental"),
    ("The research team observed unusual migration patterns.", "scientific"),
    ("The city introduced a new recycling program.", "environmental"),
    ("The bank updated its mobile app interface.", "financial"),
    ("The film festival added a new category this year.", "corporate"),
]
for c, cat in present_templates:
    add(c, cat, "needs_verification")

companies = ["Acme Corp", "Nexora Technologies", "Bluepeak Industries", "Verdant Foods",
             "Solstice Media", "Ironclad Logistics", "Fernbrook Health", "Latticework Robotics"]
actions = ["opened a new manufacturing plant", "launched a redesigned website",
           "expanded into a neighboring market", "upgraded its customer support system",
           "relocated its headquarters", "partnered with a regional distributor"]
for comp in companies:
    act = random.choice(actions)
    add(f"{comp} {act} this year.", "corporate", "needs_verification")

# =====================================================================
# 3. NUMERIC / STATISTICAL CLAIMS
# =====================================================================
stat_templates = [
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
    ("The hospital reduced patient wait times by 30 percent.", "health", "high_priority"),
    ("The city's water usage dropped by 12 percent this summer.", "environmental", "needs_verification"),
    ("The startup raised 50 million dollars in its latest funding round.", "financial", "high_priority"),
    ("Voter turnout increased by 9 percent compared to the previous election.", "political", "high_priority"),
    ("The retailer's online sales grew by 25 percent this quarter.", "financial", "needs_verification"),
    ("The school's graduation rate rose by 5 percent this year.", "statistical", "needs_verification"),
    ("The country's exports increased by 18 percent last year.", "financial", "high_priority"),
    ("The clinical trial enrolled 3,000 participants.", "health", "needs_verification"),
]
for c, cat, lab in stat_templates:
    add(c, cat, lab)

# =====================================================================
# 4. FUTURE / PREDICTION CLAIMS -> high_priority
# =====================================================================
future_templates = [
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
    ("The bridge will open to traffic by the end of next year.", "corporate"),
    ("The country will hold elections within six months.", "political"),
    ("The airline will add five new international routes next year.", "corporate"),
    ("The region will experience severe drought conditions next summer.", "environmental"),
    ("The company will report record profits next quarter.", "financial"),
    ("The vaccine will be distributed to the public by next spring.", "health"),
]
for c, cat in future_templates:
    add(c, cat, "high_priority")

# =====================================================================
# 5. CONDITIONAL / PREDICTIVE CLAIMS
# =====================================================================
conditional_templates = [
    ("If interest rates increase, housing prices could fall.", "financial", "needs_verification"),
    ("If the treaty is signed, tensions may ease in the region.", "political", "needs_verification"),
    ("If demand continues to rise, prices could increase further.", "financial", "needs_verification"),
    ("If the drought continues, crop yields may decline sharply.", "environmental", "high_priority"),
    ("If the merger goes through, thousands of jobs could be affected.", "corporate", "high_priority"),
    ("If approved, the vaccine could reach markets within a year.", "health", "high_priority"),
    ("If the bill passes, small businesses could face new tax rules.", "political", "high_priority"),
    ("If funding is secured, the project may begin next year.", "corporate", "needs_verification"),
    ("If the trend continues, the lake could dry up within a decade.", "environmental", "high_priority"),
    ("If elected, the candidate says she would lower taxes.", "political", "high_priority"),
]
for c, cat, lab in conditional_templates:
    add(c, cat, lab)

# =====================================================================
# 6. IMPLICIT CAUSAL CLAIMS -> mostly high_priority
# =====================================================================
implicit_templates = [
    ("The company's decision clearly destroyed investor confidence.", "corporate", "high_priority"),
    ("The new policy has already damaged small businesses.", "political", "high_priority"),
    ("The scandal ended the senator's political career.", "political", "high_priority"),
    ("The product recall ruined the brand's reputation overnight.", "corporate", "needs_verification"),
    ("The outage cost the company millions in lost sales.", "financial", "high_priority"),
    ("The controversy split the party into two factions.", "political", "needs_verification"),
    ("The layoffs devastated the local economy.", "financial", "high_priority"),
    ("The new regulation crushed small farmers' profits.", "financial", "high_priority"),
    ("The data breach eroded customer trust in the platform.", "corporate", "high_priority"),
    ("The strike paralyzed the city's public transport system.", "corporate", "needs_verification"),
]
for c, cat, lab in implicit_templates:
    add(c, cat, lab)

# =====================================================================
# 7. PERSON / TITLE CLAIMS (temporally sensitive)
# =====================================================================
person_templates = [
    ("Tesla is the world's most valuable automobile company.", "corporate", "high_priority"),
    ("She is the CEO of the company.", "person", "needs_verification"),
    ("He currently serves as the finance minister.", "political", "needs_verification"),
    ("The team's captain is the league's top scorer this season.", "person", "needs_verification"),
    ("The city's mayor was re-elected for a third term.", "political", "needs_verification"),
    ("The actor is the highest-paid performer in the industry this year.", "person", "high_priority"),
    ("The professor is currently the department chair.", "person", "needs_verification"),
    ("He is the youngest CEO in the company's history.", "corporate", "needs_verification"),
    ("The senator is the longest-serving member of the committee.", "political", "needs_verification"),
]
for c, cat, lab in person_templates:
    add(c, cat, lab)

# =====================================================================
# 8. HEALTH-RELATED CLAIMS (usually high stakes)
# =====================================================================
health_templates = [
    ("The new diet cures diabetes within weeks.", "health", "high_priority"),
    ("The supplement eliminates the need for prescribed medication.", "health", "high_priority"),
    ("The hospital reported zero infections last month.", "health", "needs_verification"),
    ("The new therapy has no side effects.", "health", "high_priority"),
    ("The clinic's new program improved patient recovery times.", "health", "needs_verification"),
    ("The drug reverses the effects of aging.", "health", "high_priority"),
    ("The treatment is available at no cost to patients.", "health", "needs_verification"),
    ("The surgery has a 99 percent success rate.", "health", "high_priority"),
]
for c, cat, lab in health_templates:
    add(c, cat, lab)

# =====================================================================
# 9. ENVIRONMENTAL / SCIENTIFIC EXTREMAL CLAIMS -> high_priority
# =====================================================================
env_templates = [
    ("This is the hottest year ever recorded on Earth.", "environmental", "high_priority"),
    ("Global sea levels rose by 3.4 millimeters last year.", "environmental", "high_priority"),
    ("The rainforest lost 10 percent of its cover this decade.", "environmental", "high_priority"),
    ("Recycling rates improved slightly in the region.", "environmental", "needs_verification"),
    ("Carbon emissions in the region fell for the first time in a decade.", "environmental", "high_priority"),
    ("The glacier has retreated more than any other on record.", "environmental", "high_priority"),
    ("Air quality in the city improved marginally this year.", "environmental", "needs_verification"),
]
for c, cat, lab in env_templates:
    add(c, cat, lab)

# =====================================================================
# 10. COMPOUND CLAIMS
# =====================================================================
compound_templates = [
    ("India launched a new satellite in 2025 and the satellite weighs 5,000 kilograms.", "scientific", "needs_verification"),
    ("The company increased its revenue by 40 percent in 2025 and hired 10,000 employees.", "corporate", "high_priority"),
    ("India has 28 states and will become the world's largest economy by 2030.", "political", "high_priority"),
    ("The city built a new stadium and it will host the finals next year.", "corporate", "needs_verification"),
    ("The vaccine passed clinical trials and will be available by next spring.", "health", "high_priority"),
    ("The startup raised new funding and plans to double its team next year.", "corporate", "high_priority"),
    ("The country signed the treaty and tensions have already begun to ease.", "political", "needs_verification"),
]
for c, cat, lab in compound_templates:
    add(c, cat, lab)

# =====================================================================
# de-duplicate and write
# =====================================================================
seen = set()
unique_rows = []
for claim, cat, lab in rows:
    key = claim.lower()
    if key not in seen:
        seen.add(key)
        unique_rows.append((claim, cat, lab))

random.shuffle(unique_rows)

with open("seed_claims_v2.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["claim", "category", "label"])
    writer.writerows(unique_rows)

label_counts = Counter(r[2] for r in unique_rows)
cat_counts = Counter(r[1] for r in unique_rows)
print(f"Wrote {len(unique_rows)} rows to seed_claims_v2.csv")
print("Label distribution:", dict(label_counts))
print("Category distribution:", dict(cat_counts))
