"""
storage.py
----------
Lightweight SQLite persistence -- zero external services, satisfies the
zero-cost resource boundary. Stores analyzed claims (history), the
human-review queue, and reviewer feedback (for the human-in-the-loop
feedback -> retraining-dataset loop described in the feature list).
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "data" / "claimguard.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS claim_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim TEXT NOT NULL,
        verdict TEXT NOT NULL,
        risk_score REAL,
        confidence_pct REAL,
        category TEXT,
        claim_type_key TEXT,
        lifecycle_status TEXT DEFAULT 'NEW',
        abstained INTEGER DEFAULT 0,
        evidence_conflict INTEGER DEFAULT 0,
        analysis_json TEXT,
        created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS review_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim TEXT NOT NULL,
        priority TEXT,
        reason TEXT,
        analysis_json TEXT,
        status TEXT DEFAULT 'pending',
        created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim TEXT NOT NULL,
        reviewer_decision TEXT,
        reviewer_confidence_pct REAL,
        notes TEXT,
        original_verdict TEXT,
        is_override INTEGER DEFAULT 0,
        created_at TEXT
    );
    """)
    # Lightweight migration for anyone with a pre-existing v1 database:
    # add columns that may not exist yet rather than requiring a fresh DB.
    existing_history_cols = {r["name"] for r in conn.execute("PRAGMA table_info(claim_history)")}
    for col, decl in [("lifecycle_status", "TEXT DEFAULT 'NEW'"), ("abstained", "INTEGER DEFAULT 0"),
                        ("evidence_conflict", "INTEGER DEFAULT 0")]:
        if col not in existing_history_cols:
            conn.execute(f"ALTER TABLE claim_history ADD COLUMN {col} {decl}")
    existing_feedback_cols = {r["name"] for r in conn.execute("PRAGMA table_info(feedback)")}
    for col, decl in [("reviewer_confidence_pct", "REAL"), ("original_verdict", "TEXT"),
                        ("is_override", "INTEGER DEFAULT 0")]:
        if col not in existing_feedback_cols:
            conn.execute(f"ALTER TABLE feedback ADD COLUMN {col} {decl}")
    conn.commit()
    conn.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def save_history(analysis):
    conn = get_conn()
    conn.execute(
        "INSERT INTO claim_history (claim, verdict, risk_score, confidence_pct, category, "
        "claim_type_key, lifecycle_status, abstained, evidence_conflict, analysis_json, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (analysis["claim"], analysis["verdict"], analysis.get("risk_score", 0),
         analysis.get("confidence_pct", 0), analysis.get("category", ""),
         analysis.get("claim_type_key", ""), analysis.get("lifecycle_status", "NEW"),
         int(bool(analysis.get("abstain"))),
         int(bool(analysis.get("fingerprint", {}) and analysis["fingerprint"].get("conflict_level", 0) and
                   analysis["fingerprint"]["conflict_level"] >= 50)),
         json.dumps(analysis), _now())
    )
    conn.commit()
    conn.close()


def get_history(limit=200):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM claim_history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def enqueue_review(analysis, priority, reason):
    conn = get_conn()
    conn.execute(
        "INSERT INTO review_queue (claim, priority, reason, analysis_json, created_at) VALUES (?,?,?,?,?)",
        (analysis["claim"], priority, reason, json.dumps(analysis), _now())
    )
    conn.commit()
    conn.close()


def get_review_queue(status="pending"):
    conn = get_conn()
    if status == "all":
        rows = conn.execute("SELECT * FROM review_queue ORDER BY id DESC").fetchall()
    else:
        rows = conn.execute("SELECT * FROM review_queue WHERE status=? ORDER BY id DESC", (status,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_review_status(review_id, status):
    conn = get_conn()
    conn.execute("UPDATE review_queue SET status=? WHERE id=?", (status, review_id))
    conn.commit()
    conn.close()


def update_lifecycle_status(claim_text, status):
    """Advances a claim's lifecycle status (Feature 8: Claim Lifecycle
    Tracker). Updates the most recent history row for this exact claim text."""
    conn = get_conn()
    row = conn.execute(
        "SELECT id FROM claim_history WHERE claim=? ORDER BY id DESC LIMIT 1", (claim_text,)
    ).fetchone()
    if row:
        conn.execute("UPDATE claim_history SET lifecycle_status=? WHERE id=?", (status, row["id"]))
        conn.commit()
    conn.close()
    return bool(row)


def save_feedback(claim, decision, notes="", reviewer_confidence_pct=None, original_verdict=None):
    is_override = 0
    if original_verdict and decision:
        # An "override" is when the human's decision doesn't match what the
        # AI verdict implied (e.g. AI said needs_verification, human says Refuted).
        implied_ok = {"context_sufficient": "supported", "needs_verification": None,
                      "high_priority": None, "abstain": None}
        is_override = int(bool(original_verdict in implied_ok and implied_ok[original_verdict]
                                and decision != implied_ok[original_verdict]))
    conn = get_conn()
    conn.execute(
        "INSERT INTO feedback (claim, reviewer_decision, reviewer_confidence_pct, notes, "
        "original_verdict, is_override, created_at) VALUES (?,?,?,?,?,?,?)",
        (claim, decision, reviewer_confidence_pct, notes, original_verdict, is_override, _now())
    )
    conn.commit()
    conn.close()
    update_lifecycle_status(claim, "RESOLVED")


def get_feedback(limit=200):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def dashboard_stats():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) c FROM claim_history").fetchone()["c"]
    by_verdict = conn.execute(
        "SELECT verdict, COUNT(*) c FROM claim_history GROUP BY verdict").fetchall()
    by_category = conn.execute(
        "SELECT category, COUNT(*) c FROM claim_history GROUP BY category").fetchall()
    by_type = conn.execute(
        "SELECT claim_type_key, COUNT(*) c FROM claim_history GROUP BY claim_type_key").fetchall()
    conn.close()
    return {
        "total": total,
        "by_verdict": {r["verdict"]: r["c"] for r in by_verdict},
        "by_category": {r["category"]: r["c"] for r in by_category},
        "by_type": {r["claim_type_key"]: r["c"] for r in by_type},
    }


def research_dashboard_stats():
    """Feature 11: Research Analytics Dashboard -- AI-behavior metrics
    (abstention rate, human override rate, conflict detection rate)
    computed from real stored history/feedback, not placeholders."""
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) c FROM claim_history").fetchone()["c"]
    abstained = conn.execute("SELECT COUNT(*) c FROM claim_history WHERE abstained=1").fetchone()["c"]
    conflicted = conn.execute("SELECT COUNT(*) c FROM claim_history WHERE evidence_conflict=1").fetchone()["c"]
    by_lifecycle = conn.execute(
        "SELECT lifecycle_status, COUNT(*) c FROM claim_history GROUP BY lifecycle_status").fetchall()

    n_feedback = conn.execute("SELECT COUNT(*) c FROM feedback").fetchone()["c"]
    n_overrides = conn.execute("SELECT COUNT(*) c FROM feedback WHERE is_override=1").fetchone()["c"]
    by_decision = conn.execute(
        "SELECT reviewer_decision, COUNT(*) c FROM feedback GROUP BY reviewer_decision").fetchall()
    conn.close()

    return {
        "total_claims": total,
        "abstention_rate_pct": round(abstained / total * 100, 1) if total else 0.0,
        "conflict_detection_rate_pct": round(conflicted / total * 100, 1) if total else 0.0,
        "human_override_rate_pct": round(n_overrides / n_feedback * 100, 1) if n_feedback else 0.0,
        "n_reviewed": n_feedback,
        "n_overrides": n_overrides,
        "by_lifecycle_status": {r["lifecycle_status"]: r["c"] for r in by_lifecycle},
        "by_reviewer_decision": {r["reviewer_decision"]: r["c"] for r in by_decision},
    }


def export_feedback_csv():
    """Feature 10: Active Learning Loop -- 'Export Reviewed Cases for
    Retraining'. Maps stored reviewer feedback into the same claim/category/
    label CSV shape used by build_seed_dataset_v2.py, so a person can append
    it directly to the training data by hand (manual retrain step, not
    automatic continuous retraining -- documented, not oversold)."""
    decision_to_label = {
        "supported": "context_sufficient", "verified": "context_sufficient",
        "refuted": "needs_verification", "insufficient": "needs_verification",
        "insufficient_evidence": "needs_verification", "ambiguous": "needs_verification",
        "needs_updated_evidence": "high_priority", "more_research": "needs_verification",
    }
    conn = get_conn()
    rows = conn.execute("SELECT * FROM feedback ORDER BY id").fetchall()
    conn.close()
    out_rows = [["claim", "category", "label", "reviewer_decision", "reviewer_confidence_pct",
                 "notes", "original_verdict", "is_override", "created_at"]]
    for r in rows:
        mapped_label = decision_to_label.get((r["reviewer_decision"] or "").lower().replace(" ", "_"), "")
        out_rows.append([r["claim"], "", mapped_label, r["reviewer_decision"],
                          r["reviewer_confidence_pct"], r["notes"], r["original_verdict"],
                          r["is_override"], r["created_at"]])
    return out_rows
