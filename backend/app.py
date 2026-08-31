"""
app.py
------
VeriScope AI Flask API. Run with:
    python3 app.py
Serves the API on http://localhost:5000 and the frontend (static files) at
http://localhost:5000/.

Environment variables (see .env.example): GROQ_API_KEY, GROQ_MODEL, PORT.
"""

import os
import io
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

import storage
import groq_client
import evidence_engine
from claim_analyzer import extract_and_analyze, analyze_claim, investigation_roadmap
from evidence_engine import (retrieve_evidence, detect_conflict, align_evidence_to_claim,
                              find_duplicates, document_verifiability_score,
                              compare_retrieval_methods, evidence_debate_view,
                              evidence_freshness, evidence_intelligence_score)
from counterfactual import run_counterfactual_analysis
from report_generator import generate_report_html

HERE = Path(__file__).parent
FRONTEND_DIR = HERE.parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

# CORS: needed once the frontend is hosted on a different origin than the
# API (e.g. Netlify frontend + Render backend). ALLOWED_ORIGINS is a comma-
# separated list read from an env var so you don't have to hardcode a
# domain into the source -- set it to your Netlify URL in Render's
# environment settings. Defaults to "*" (open) for local/dev convenience;
# tighten this once you know your real Netlify URL.
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
CORS(app, resources={r"/api/*": {"origins": _allowed_origins.split(",") if _allowed_origins != "*" else "*"}})

storage.init_db()


# ---------------------------------------------------------------- frontend
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ------------------------------------------------------------------ health
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "groq_configured": groq_client.is_configured()})


# ---------------------------------------------------------------- analyze
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    payload = request.get_json(force=True)
    text = (payload.get("text") or "").strip()
    context = (payload.get("context") or "").strip()
    threshold = float(payload.get("threshold", 0.5))
    decompose = bool(payload.get("decompose", True))

    if not text:
        return jsonify({"error": "No text provided."}), 400

    analyses = extract_and_analyze(text, context_text=context, threshold=threshold, decompose=decompose)

    history = storage.get_history(limit=500)
    for a in analyses:
        if a["verdict"] == "not_a_claim":
            storage.save_history(a)
            continue
        dupes = find_duplicates(a["claim"], history, threshold=75.0)
        a["duplicates"] = dupes
        storage.save_history(a)
        if a["verdict"] in ("high_priority",) or a.get("abstain"):
            storage.enqueue_review(a, priority=("Critical" if a["verdict"] == "high_priority" else "High"),
                                    reason="; ".join(a["reasons"][:2]))
        elif a["verdict"] == "needs_verification" and a["risk_score"] >= 45:
            storage.enqueue_review(a, priority="Medium", reason="; ".join(a["reasons"][:2]))

    doc_score, doc_band = document_verifiability_score(analyses)
    return jsonify({
        "claims": analyses,
        "document_score": doc_score,
        "document_band": doc_band,
        "total": len(analyses),
    })


@app.route("/api/analyze-single", methods=["POST"])
def api_analyze_single():
    payload = request.get_json(force=True)
    claim = (payload.get("claim") or "").strip()
    context = (payload.get("context") or "").strip()
    threshold = float(payload.get("threshold", 0.5))
    if not claim:
        return jsonify({"error": "No claim provided."}), 400
    analysis = analyze_claim(claim, context_text=context, threshold=threshold)
    storage.save_history(analysis)
    return jsonify(analysis)


# ---------------------------------------------------------------- evidence
@app.route("/api/evidence", methods=["POST"])
def api_evidence():
    payload = request.get_json(force=True)
    claim = (payload.get("claim") or "").strip()
    context = (payload.get("context") or "").strip()
    use_web = bool(payload.get("use_web", False))
    context_date = (payload.get("context_date") or "").strip() or None
    temporal_sensitive = bool(payload.get("temporal_sensitive", False))
    context_sufficiency_pct = float(payload.get("context_sufficiency_pct", 0))
    if not claim:
        return jsonify({"error": "No claim provided."}), 400

    evidence = retrieve_evidence(claim, context, top_k=5)
    conflicts = detect_conflict(evidence)
    for ev in evidence:
        ev["aligned_terms"] = align_evidence_to_claim(claim, ev["sentence"])

    debate = evidence_debate_view(claim, evidence)
    freshness = evidence_freshness(temporal_sensitive, context_date)
    intelligence = evidence_intelligence_score(evidence, debate, freshness, context_sufficiency_pct)

    web_result, web_error = (None, None)
    if use_web:
        web_result, web_error = evidence_engine.web_search_evidence(claim)

    return jsonify({
        "evidence": evidence,
        "conflicts": conflicts,
        "local_evidence_found": len(evidence) > 0,
        "debate": debate,
        "freshness": freshness,
        "intelligence_score": intelligence,
        "web_search": web_result,
        "web_search_error": web_error,
        "web_search_available": groq_client.is_configured(),
    })


@app.route("/api/retrieval-comparison", methods=["POST"])
def api_retrieval_comparison():
    """Feature 5: live side-by-side of TF-IDF+cosine / BM25 / LSA-semantic-
    proxy on THIS claim+context. For the aggregate Precision@K/Recall@K
    numbers over the hand-labeled benchmark, see /api/model-info's
    retrieval_benchmark field (computed offline by eval_retrieval_methods.py)."""
    payload = request.get_json(force=True)
    claim = (payload.get("claim") or "").strip()
    context = (payload.get("context") or "").strip()
    if not claim or not context:
        return jsonify({"error": "Both claim and context are required to compare retrieval methods."}), 400
    return jsonify(compare_retrieval_methods(claim, context))


@app.route("/api/roadmap", methods=["POST"])
def api_roadmap():
    payload = request.get_json(force=True)
    claim = (payload.get("claim") or "").strip()
    context = (payload.get("context") or "").strip()
    if not claim:
        return jsonify({"error": "No claim provided."}), 400
    analysis = analyze_claim(claim, context_text=context)
    steps = investigation_roadmap(analysis)
    return jsonify({"roadmap": steps, "analysis": analysis})


# ------------------------------------------------------------ review queue
@app.route("/api/review-queue", methods=["GET"])
def api_review_queue():
    status = request.args.get("status", "pending")
    return jsonify({"queue": storage.get_review_queue(status)})


@app.route("/api/review-queue/<int:review_id>", methods=["POST"])
def api_update_review(review_id):
    payload = request.get_json(force=True)
    status = payload.get("status", "resolved")
    storage.update_review_status(review_id, status)
    return jsonify({"ok": True})


# ----------------------------------------------------------------feedback
@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    payload = request.get_json(force=True)
    claim = (payload.get("claim") or "").strip()
    decision = payload.get("decision", "")
    notes = payload.get("notes", "")
    reviewer_confidence_pct = payload.get("reviewer_confidence_pct")
    original_verdict = payload.get("original_verdict")
    if not claim or not decision:
        return jsonify({"error": "claim and decision are required."}), 400
    storage.save_feedback(claim, decision, notes,
                           reviewer_confidence_pct=reviewer_confidence_pct,
                           original_verdict=original_verdict)
    return jsonify({"ok": True})


@app.route("/api/feedback", methods=["GET"])
def api_get_feedback():
    return jsonify({"feedback": storage.get_feedback()})


@app.route("/api/feedback/export", methods=["GET"])
def api_export_feedback():
    """Feature 10: 'Export Reviewed Cases for Retraining' -- downloads
    reviewed cases as a CSV shaped like the training data, ready to append
    to seed_claims_combined.csv by hand. Manual retrain step, not automatic
    continuous retraining -- documented as such, not oversold."""
    import csv as csv_module
    rows = storage.export_feedback_csv()
    buf = io.StringIO()
    writer = csv_module.writer(buf)
    writer.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv",
                     headers={"Content-Disposition": "attachment; filename=claimguard_reviewed_cases.csv"})


# ------------------------------------------------------------------history
@app.route("/api/history", methods=["GET"])
def api_history():
    return jsonify({"history": storage.get_history()})


@app.route("/api/dashboard", methods=["GET"])
def api_dashboard():
    return jsonify(storage.dashboard_stats())


@app.route("/api/research-dashboard", methods=["GET"])
def api_research_dashboard():
    """Feature 11: Research Analytics Dashboard -- dataset stats, model
    performance, feature importance, and AI-behavior metrics (abstention
    rate, override rate, conflict rate) all in one place for presentation."""
    import json
    stats = storage.research_dashboard_stats()
    fi_path = HERE / "models" / "feature_importance.json"
    feature_importance = json.loads(fi_path.read_text()) if fi_path.exists() else None
    return jsonify({"ai_behavior": stats, "feature_importance": feature_importance})


@app.route("/api/counterfactual", methods=["POST"])
def api_counterfactual():
    payload = request.get_json(force=True)
    claim = (payload.get("claim") or "").strip()
    if not claim:
        return jsonify({"error": "No claim provided."}), 400
    return jsonify(run_counterfactual_analysis(claim, analyze_claim))


# -------------------------------------------------------------------report
@app.route("/api/report", methods=["POST"])
def api_report():
    payload = request.get_json(force=True)
    claims = payload.get("claims", [])
    source_label = payload.get("source_label", "Pasted text")
    if not claims:
        return jsonify({"error": "No claims to report on."}), 400
    html = generate_report_html(claims, source_label=source_label)
    return Response(html, mimetype="text/html",
                     headers={"Content-Disposition": "inline; filename=claimguard_report.html"})


# ---------------------------------------------------------- model metadata
@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    import json
    comp_path = HERE / "models" / "model_comparison.json"
    err_path = HERE / "models" / "error_analysis.json"
    cw_path = HERE / "models" / "checkworthy_metrics.json"
    retrieval_path = HERE / "models" / "retrieval_comparison.json"
    comp = json.loads(comp_path.read_text()) if comp_path.exists() else {}
    err = json.loads(err_path.read_text()) if err_path.exists() else {}
    cw = json.loads(cw_path.read_text()) if cw_path.exists() else {}
    retrieval = json.loads(retrieval_path.read_text()) if retrieval_path.exists() else {}
    return jsonify({"comparison": comp, "error_analysis": err, "checkworthy_model": cw,
                     "retrieval_benchmark": retrieval})


# --------------------------------------------------------------------- ai
@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(force=True)
    message = (payload.get("message") or "").strip()
    history = payload.get("history", [])
    if not message:
        return jsonify({"error": "No message provided."}), 400
    reply, error = groq_client.chat(message, history=history)
    if error and not reply:
        return jsonify({"reply": None, "error": error}), 200
    return jsonify({"reply": reply, "error": None})


@app.route("/api/ai-review", methods=["POST"])
def api_ai_review():
    payload = request.get_json(force=True)
    claim = (payload.get("claim") or "").strip()
    if not claim:
        return jsonify({"error": "No claim provided."}), 400
    analysis = analyze_claim(claim)
    summary = (f"{analysis['verdict_label']} (confidence {analysis['confidence_pct']}%), "
               f"risk {analysis['risk_score']}/100, reasons: {'; '.join(analysis['reasons'])}")
    reply, error = groq_client.ai_review(claim, summary)
    return jsonify({"reply": reply, "error": error, "heuristic_summary": summary})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)