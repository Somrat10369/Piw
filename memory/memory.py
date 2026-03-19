# file: memory/memory.py
"""
PIW Memory Layer — SQLite-backed persistence.

Stores:
  - articles seen (dedup by URL hash)
  - topic run history (what was analyzed, when)
  - entity mentions over time
  - full report JSON per run

All queries are synchronous/blocking — suitable for CLI use.
"""

from __future__ import annotations
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    from config import MEMORY_DB_PATH, MEMORY_DEDUP_HOURS, ENTITY_MIN_MENTIONS
except ImportError:
    MEMORY_DB_PATH     = "memory/piw_memory.db"
    MEMORY_DEDUP_HOURS = 48
    ENTITY_MIN_MENTIONS = 2

# ── DB setup ──────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles_seen (
    id          TEXT PRIMARY KEY,      -- MD5 hash of URL
    url         TEXT NOT NULL,
    title       TEXT,
    source      TEXT,
    first_seen  TEXT NOT NULL,         -- ISO datetime
    last_seen   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topic_runs (
    run_id      TEXT PRIMARY KEY,      -- timestamp slug
    topic       TEXT NOT NULL,
    sources     TEXT,                  -- JSON list
    article_count INTEGER,
    run_at      TEXT NOT NULL,
    report_json TEXT                   -- full report serialized
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,         -- COUNTRY | ORG | PERSON | OTHER
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    mention_count INTEGER DEFAULT 1,
    last_context TEXT                  -- short excerpt from last mention
);

CREATE TABLE IF NOT EXISTS entity_topic_links (
    entity_id   INTEGER,
    run_id      TEXT,
    PRIMARY KEY (entity_id, run_id)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type  TEXT NOT NULL,         -- NEW_ENTITY | TOPIC_SPIKE | TREND_CHANGE
    message     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    acknowledged INTEGER DEFAULT 0
);
"""

def _db_path() -> str:
    path = MEMORY_DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path

@contextmanager
def _conn():
    db = sqlite3.connect(_db_path())
    db.row_factory = sqlite3.Row
    try:
        yield db
        db.commit()
    finally:
        db.close()

def init_db():
    """Create tables if they don't exist. Safe to call multiple times."""
    with _conn() as db:
        db.executescript(SCHEMA)

# ── Article dedup ─────────────────────────────────────────────────────────────

def mark_article_seen(article_id: str, url: str, title: str, source: str) -> None:
    now = datetime.now().isoformat()
    with _conn() as db:
        existing = db.execute(
            "SELECT id FROM articles_seen WHERE id=?", (article_id,)
        ).fetchone()
        if existing:
            db.execute(
                "UPDATE articles_seen SET last_seen=? WHERE id=?", (now, article_id)
            )
        else:
            db.execute(
                "INSERT INTO articles_seen VALUES (?,?,?,?,?,?)",
                (article_id, url, title, source, now, now)
            )

def is_article_fresh(article_id: str) -> bool:
    """Return True if this article hasn't been seen within MEMORY_DEDUP_HOURS."""
    with _conn() as db:
        row = db.execute(
            "SELECT last_seen FROM articles_seen WHERE id=?", (article_id,)
        ).fetchone()
        if not row:
            return True
        last = datetime.fromisoformat(row["last_seen"])
        return datetime.now() - last > timedelta(hours=MEMORY_DEDUP_HOURS)

def filter_fresh_articles(articles: list) -> list:
    """Return only articles not recently analyzed. Mark all as seen."""
    fresh = []
    for a in articles:
        if is_article_fresh(a.id):
            fresh.append(a)
        mark_article_seen(a.id, a.url, a.title, a.source)
    return fresh

# ── Topic run history ─────────────────────────────────────────────────────────

def save_report_to_memory(report, run_id: str | None = None) -> str:
    """Persist a full IntelligenceReport to memory. Returns run_id."""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + report.topic[:20].replace(" ","_")

    report_dict = {
        "topic":         report.topic,
        "sources_used":  report.sources_used,
        "article_count": report.article_count,
        "agents": {
            key: {"output": getattr(report, key).output, "error": getattr(report, key).error}
            for key in ["reality","bias","missing","incentives","trends","scenarios","personal"]
            if getattr(report, key) is not None
        },
    }

    with _conn() as db:
        db.execute(
            "INSERT OR REPLACE INTO topic_runs VALUES (?,?,?,?,?,?)",
            (
                run_id,
                report.topic,
                json.dumps(report.sources_used),
                report.article_count,
                datetime.now().isoformat(),
                json.dumps(report_dict),
            )
        )
    return run_id

def get_topic_history(topic: str | None = None, limit: int = 20) -> list[dict]:
    """Fetch past runs, optionally filtered by topic substring."""
    with _conn() as db:
        if topic:
            rows = db.execute(
                "SELECT run_id, topic, sources, article_count, run_at "
                "FROM topic_runs WHERE topic LIKE ? ORDER BY run_at DESC LIMIT ?",
                (f"%{topic}%", limit)
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT run_id, topic, sources, article_count, run_at "
                "FROM topic_runs ORDER BY run_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

def get_report_by_run_id(run_id: str) -> dict | None:
    with _conn() as db:
        row = db.execute(
            "SELECT report_json FROM topic_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return json.loads(row["report_json"]) if row else None

# ── Entity tracking ───────────────────────────────────────────────────────────

def upsert_entity(name: str, etype: str, context: str = "", run_id: str = "") -> int:
    """Insert or update an entity. Returns entity_id."""
    now = datetime.now().isoformat()
    name_lower = name.lower().strip()
    with _conn() as db:
        row = db.execute(
            "SELECT entity_id, mention_count FROM entities WHERE LOWER(name)=?",
            (name_lower,)
        ).fetchone()
        if row:
            eid = row["entity_id"]
            db.execute(
                "UPDATE entities SET last_seen=?, mention_count=mention_count+1, last_context=? WHERE entity_id=?",
                (now, context[:200], eid)
            )
        else:
            db.execute(
                "INSERT INTO entities (name, type, first_seen, last_seen, mention_count, last_context) "
                "VALUES (?,?,?,?,1,?)",
                (name, etype, now, now, context[:200])
            )
            eid = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        if run_id:
            db.execute(
                "INSERT OR IGNORE INTO entity_topic_links VALUES (?,?)", (eid, run_id)
            )
    return eid

def get_top_entities(etype: str | None = None, limit: int = 20) -> list[dict]:
    with _conn() as db:
        if etype:
            rows = db.execute(
                "SELECT name, type, mention_count, first_seen, last_seen, last_context "
                "FROM entities WHERE type=? AND mention_count >= ? "
                "ORDER BY mention_count DESC LIMIT ?",
                (etype, ENTITY_MIN_MENTIONS, limit)
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT name, type, mention_count, first_seen, last_seen, last_context "
                "FROM entities WHERE mention_count >= ? "
                "ORDER BY mention_count DESC LIMIT ?",
                (ENTITY_MIN_MENTIONS, limit)
            ).fetchall()
        return [dict(r) for r in rows]

def get_entity_timeline(entity_name: str) -> list[dict]:
    """Get all topic runs this entity appeared in."""
    with _conn() as db:
        rows = db.execute(
            """SELECT tr.run_id, tr.topic, tr.run_at
               FROM topic_runs tr
               JOIN entity_topic_links etl ON tr.run_id = etl.run_id
               JOIN entities e ON e.entity_id = etl.entity_id
               WHERE LOWER(e.name) = ?
               ORDER BY tr.run_at DESC""",
            (entity_name.lower(),)
        ).fetchall()
        return [dict(r) for r in rows]

# ── Alerts ────────────────────────────────────────────────────────────────────

def create_alert(alert_type: str, message: str) -> None:
    with _conn() as db:
        db.execute(
            "INSERT INTO alerts (alert_type, message, created_at) VALUES (?,?,?)",
            (alert_type, message, datetime.now().isoformat())
        )

def get_pending_alerts() -> list[dict]:
    with _conn() as db:
        rows = db.execute(
            "SELECT alert_id, alert_type, message, created_at "
            "FROM alerts WHERE acknowledged=0 ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

def acknowledge_alert(alert_id: int) -> None:
    with _conn() as db:
        db.execute("UPDATE alerts SET acknowledged=1 WHERE alert_id=?", (alert_id,))

def acknowledge_all_alerts() -> None:
    with _conn() as db:
        db.execute("UPDATE alerts SET acknowledged=1")

# ── Stats ─────────────────────────────────────────────────────────────────────

def get_memory_stats() -> dict:
    with _conn() as db:
        articles_total = db.execute("SELECT COUNT(*) FROM articles_seen").fetchone()[0]
        runs_total     = db.execute("SELECT COUNT(*) FROM topic_runs").fetchone()[0]
        entities_total = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        alerts_pending = db.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged=0").fetchone()[0]
        oldest_run     = db.execute("SELECT MIN(run_at) FROM topic_runs").fetchone()[0]
        return {
            "articles_seen":  articles_total,
            "total_runs":     runs_total,
            "entities_known": entities_total,
            "pending_alerts": alerts_pending,
            "tracking_since": oldest_run or "never",
        }