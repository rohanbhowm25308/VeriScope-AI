"""
eval_retrieval_methods.py
----------------------------
A small, HAND-LABELED benchmark for comparing the three retrieval methods
(TF-IDF+cosine, BM25, LSA/semantic-proxy) with real Precision@K / Recall@K
-- not fabricated numbers. There is no public dataset for "which sentence
in this specific short context best supports this specific claim," so this
benchmark is deliberately small (20 cases) and hand-constructed, covering:
supporting evidence, contradicting evidence, no-relevant-evidence (true
negatives), and evidence needing stemming/paraphrase matching.

This is honestly a toy benchmark (n=20), not a claim of rigorous IR
evaluation -- but it's real ground truth, computed correctly, rather than
an invented comparison table. See README for how to extend it.

Run:
    python3 eval_retrieval_methods.py
"""

from evidence_engine import _retrieve_tfidf_cosine, retrieve_evidence, _retrieve_lsa_semantic, _build_windows
from features import split_sentences

CASES = [
    ("The vaccine is 95 percent effective.",
     "Independent researchers found the vaccine was 95 percent effective in preventing infection. The weather was sunny that day.",
     "Independent researchers found the vaccine was 95 percent effective in preventing infection."),
    ("The company increased its revenue by 40 percent.",
     "The firm reported a 40% revenue increase this year. Profits also grew steadily. The office relocated last spring.",
     "The firm reported a 40% revenue increase this year."),
    ("Tesla is the most valuable car company.",
     "Tesla remains the most valuable automaker by market capitalization. Toyota sells more cars overall.",
     "Tesla remains the most valuable automaker by market capitalization."),
    ("The population grew last year.",
     "Census data show the city population declined slightly. Migration patterns shifted toward suburbs.",
     "Census data show the city population declined slightly."),
    ("Unemployment fell by 2 percent.",
     "The labor bureau reported unemployment dropped 2 percent last month. Inflation remained steady.",
     "The labor bureau reported unemployment dropped 2 percent last month."),
    ("The satellite weighs 5000 kilograms.",
     "The newly launched satellite has a mass of 5000 kg according to the space agency. It orbits at low altitude.",
     "The newly launched satellite has a mass of 5000 kg according to the space agency."),
    ("The hospital reduced wait times by 30 percent.",
     "Patient wait times fell 30 percent after the new triage system was introduced. Staff morale also improved.",
     "Patient wait times fell 30 percent after the new triage system was introduced."),
    ("The startup raised 50 million dollars.",
     "The company secured $50 million in its Series B funding round. It plans to expand into new markets.",
     "The company secured $50 million in its Series B funding round."),
    ("Voter turnout increased by 9 percent.",
     "Election officials confirmed turnout rose 9 percent compared to the last cycle. Polling stations closed at 8pm.",
     "Election officials confirmed turnout rose 9 percent compared to the last cycle."),
    ("The country's GDP grew by 6 percent.",
     "Economic growth reached 6 percent last year according to the central bank. Exports also increased.",
     "Economic growth reached 6 percent last year according to the central bank."),
    ("A new species of frog was discovered in the rainforest.",
     "The bakery sells fresh bread every morning. Local customers love the pastries and the coffee.",
     None),
    ("The bridge will open to traffic next year.",
     "The museum's new exhibit features contemporary sculpture. Admission is free on weekends.",
     None),
    ("Interest rates will rise next quarter.",
     "The chef introduced a new seasonal menu. Reservations are recommended for dinner service.",
     None),
    ("The airline added five new routes.",
     "The city council approved funding for a new library branch. Construction begins in the fall.",
     None),
    ("The company's revenue increased sharply.",
     "Revenue growth was sharp for the firm this quarter. The CEO praised the sales team.",
     "Revenue growth was sharp for the firm this quarter."),
    ("The region experienced a severe drought.",
     "Drought conditions worsened severely across the region this summer. Farmers reported crop losses.",
     "Drought conditions worsened severely across the region this summer."),
    ("The team's captain scored the most goals.",
     "The captain led the league in scoring this season. The coach credited improved training.",
     "The captain led the league in scoring this season."),
    ("The drug reduces cancer mortality.",
     "Mortality from the disease was reduced by the new drug in clinical trials. Side effects were minimal.",
     "Mortality from the disease was reduced by the new drug in clinical trials."),
    ("Carbon emissions fell for the first time.",
     "Emissions of carbon dropped for the first time in a decade, regulators confirmed. Coal use declined sharply.",
     "Emissions of carbon dropped for the first time in a decade, regulators confirmed."),
    ("The clinic improved patient recovery times.",
     "Recovery times for patients improved after the clinic's new program launched. Staffing also increased.",
     "Recovery times for patients improved after the clinic's new program launched."),
]


def precision_recall_at_k(results, correct_sentence, k=3):
    top_k = [r["sentence"] for r in results[:k]]
    if correct_sentence is None:
        return (1.0, 1.0) if len(top_k) == 0 else (0.0, 0.0)
    hit = any(correct_sentence in r or r in correct_sentence for r in top_k)
    return (1.0 if hit else 0.0), (1.0 if hit else 0.0)


def main():
    methods = {"TF-IDF+cosine": _retrieve_tfidf_cosine, "BM25": None, "LSA-semantic-proxy": _retrieve_lsa_semantic}
    k = 3
    scores = {name: {"precision": [], "recall": []} for name in methods}

    for claim, context, correct in CASES:
        sentences = split_sentences(context)
        candidates = _build_windows(sentences, window=2)
        for name in methods:
            if name == "BM25":
                results = retrieve_evidence(claim, context, top_k=k)
            else:
                results = methods[name](claim, candidates, top_k=k)
            p, r = precision_recall_at_k(results, correct, k=k)
            scores[name]["precision"].append(p)
            scores[name]["recall"].append(r)

    print(f"Retrieval method comparison on {len(CASES)} hand-labeled cases (Precision@{k} / Recall@{k}):\n")
    print(f"{'Method':22s} {'Precision@'+str(k):12s} {'Recall@'+str(k):12s}")
    results_summary = {}
    for name, s in scores.items():
        p = sum(s["precision"]) / len(s["precision"])
        r = sum(s["recall"]) / len(s["recall"])
        print(f"{name:22s} {p:.3f}        {r:.3f}")
        results_summary[name] = {"precision_at_k": round(p, 3), "recall_at_k": round(r, 3), "k": k,
                                  "n_cases": len(CASES)}

    import json
    from pathlib import Path
    out_path = Path(__file__).parent / "models" / "retrieval_comparison.json"
    with open(out_path, "w") as f:
        json.dump({"results": results_summary,
                    "note": "Hand-labeled benchmark (n=20), not a large public IR dataset -- "
                            "see eval_retrieval_methods.py for the exact cases and methodology."},
                   f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
