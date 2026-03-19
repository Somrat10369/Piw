# file: memory/trend_tracker.py
"""
Long-Term Trend Tracker

Tracks entity mention frequency over time in configurable windows,
computes velocity (rate of change) and acceleration (velocity change),
classifies each entity's trend state, and fires alerts on state transitions.

Trend States:
  RISING      — mentions increasing consistently
  SURGING     — sudden large spike (velocity > surge threshold)
  PEAKED      — was rising, now falling
  DECLINING   — mentions decreasing consistently
  DORMANT     — no mentions in recent window
  STABLE      — low variance, no clear direction
  EMERGING    — first seen recently, too early to classify

Schema additions (appended to existing DB via migrate_db()):
  entity_snapshots  — one row per entity per time bucket
  trend_states      — current classified state per entity
"""

from __future__ import annotations
import math
from datetime import datetime, timedelta
from typing import NamedTuple

from memory.memory import _conn, init_db, create_alert

# ── Schema extension ──────────────────────────────────────────────────────────

TREND_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_snapshots (
    snapshot_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id    INTEGER NOT NULL,
    bucket       TEXT NOT NULL,        -- ISO date of window start e.g. "2025-01-06"
    window_days  INTEGER NOT NULL,     -- window size (1=daily, 7=weekly)
    mention_count INTEGER DEFAULT 0,
    UNIQUE(entity_id, bucket, window_days)
);

CREATE TABLE IF NOT EXISTS trend_states (
    entity_id    INTEGER PRIMARY KEY,
    state        TEXT NOT NULL,        -- RISING|SURGING|PEAKED|DECLINING|DORMANT|STABLE|EMERGING
    velocity     REAL DEFAULT 0.0,     -- mentions/window change (positive=rising)
    acceleration REAL DEFAULT 0.0,    -- change in velocity
    prev_state   TEXT,                 -- what state was before this one
    state_since  TEXT NOT NULL,        -- when this state started
    updated_at   TEXT NOT NULL
);
"""

def migrate_db() -> None:
    """Add trend tables to existing DB. Safe to call repeatedly."""
    init_db()
    with _conn() as db:
        db.executescript(TREND_SCHEMA)


# ── Snapshot recording ────────────────────────────────────────────────────────

def _bucket_date(dt: datetime, window_days: int) -> str:
    """Align a datetime to the start of its window bucket."""
    if window_days == 1:
        return dt.date().isoformat()
    # Align to Monday for weekly buckets
    days_since_monday = dt.weekday()
    monday = (dt - timedelta(days=days_since_monday)).date()
    return monday.isoformat()


def record_snapshot(entity_id: int, window_days: int = 7) -> None:
    """
    Increment the mention count for this entity in the current time bucket.
    Called every time an entity is extracted from a new report.
    """
    bucket = _bucket_date(datetime.now(), window_days)
    with _conn() as db:
        db.execute("""
            INSERT INTO entity_snapshots (entity_id, bucket, window_days, mention_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(entity_id, bucket, window_days)
            DO UPDATE SET mention_count = mention_count + 1
        """, (entity_id, bucket, window_days))


def record_snapshot_for_name(entity_name: str, window_days: int = 7) -> None:
    """Convenience: record snapshot by entity name (looks up ID)."""
    with _conn() as db:
        row = db.execute(
            "SELECT entity_id FROM entities WHERE LOWER(name)=?",
            (entity_name.lower().strip(),)
        ).fetchone()
        if row:
            record_snapshot(row["entity_id"], window_days)


# ── Time-series retrieval ─────────────────────────────────────────────────────

def get_entity_series(entity_id: int,
                      window_days: int = 7,
                      lookback_windows: int = 12) -> list[dict]:
    """
    Return mention counts per bucket for the last N windows.
    Fills in 0 for missing buckets so the series is continuous.
    Returns list of {bucket, count} sorted oldest → newest.
    """
    cutoff = (datetime.now() - timedelta(days=window_days * lookback_windows)).date()

    with _conn() as db:
        rows = db.execute("""
            SELECT bucket, mention_count
            FROM entity_snapshots
            WHERE entity_id=? AND window_days=? AND bucket >= ?
            ORDER BY bucket ASC
        """, (entity_id, window_days, cutoff.isoformat())).fetchall()

    # Build complete bucket list (fill gaps with 0)
    existing = {r["bucket"]: r["mention_count"] for r in rows}
    series = []
    current = cutoff
    today = datetime.now().date()
    while current <= today:
        b = _bucket_date(datetime(current.year, current.month, current.day), window_days)
        if b not in [s["bucket"] for s in series]:  # avoid dupes from alignment
            series.append({"bucket": b, "count": existing.get(b, 0)})
        current += timedelta(days=window_days)

    return series


def get_series_by_name(entity_name: str,
                       window_days: int = 7,
                       lookback_windows: int = 12) -> tuple[list[dict], str | None]:
    """
    Returns (series, entity_type) by name.
    series is [] if entity not found.
    """
    with _conn() as db:
        row = db.execute(
            "SELECT entity_id, type FROM entities WHERE LOWER(name)=?",
            (entity_name.lower().strip(),)
        ).fetchone()
        if not row:
            return [], None
        series = get_entity_series(row["entity_id"], window_days, lookback_windows)
        return series, row["type"]


# ── Trend computation ─────────────────────────────────────────────────────────

class TrendMetrics(NamedTuple):
    state:        str
    velocity:     float   # avg mentions/window change over recent half
    acceleration: float   # velocity change: recent half vs older half
    recent_avg:   float   # avg mentions in most recent N/2 windows
    baseline_avg: float   # avg mentions in older N/2 windows
    peak:         int     # highest single-window count
    total:        int     # total mentions across all windows


# Thresholds — tune these in config or leave as defaults
_SURGE_MULTIPLIER   = 3.0   # recent_avg > baseline * this → SURGING
_RISING_MIN_VEL     = 0.3   # velocity above this → RISING
_DECLINING_MAX_VEL  = -0.3  # velocity below this → DECLINING
_DORMANT_WINDOWS    = 3     # no mentions in last N windows → DORMANT
_EMERGING_MAX_AGE   = 2     # seen in ≤ N windows → EMERGING


def compute_trend(series: list[dict]) -> TrendMetrics:
    """
    Given a time-series of {bucket, count}, compute trend state and metrics.
    Requires at least 4 data points for meaningful classification.
    """
    counts = [s["count"] for s in series]
    n = len(counts)

    if n == 0:
        return TrendMetrics("DORMANT", 0.0, 0.0, 0.0, 0.0, 0, 0)

    total     = sum(counts)
    peak      = max(counts)
    nonzero   = sum(1 for c in counts if c > 0)

    # Not enough data
    if n < 4 or nonzero <= 1:
        if nonzero == 0:
            return TrendMetrics("DORMANT", 0.0, 0.0, 0.0, 0.0, 0, total)
        if nonzero <= _EMERGING_MAX_AGE:
            return TrendMetrics("EMERGING", 0.0, 0.0, float(counts[-1]), 0.0, peak, total)
        return TrendMetrics("STABLE", 0.0, 0.0, float(counts[-1]), float(sum(counts[:-1])/(n-1)), peak, total)

    # Split into recent half and baseline half
    mid          = n // 2
    baseline     = counts[:mid]
    recent       = counts[mid:]
    baseline_avg = sum(baseline) / len(baseline) if baseline else 0.0
    recent_avg   = sum(recent)   / len(recent)   if recent   else 0.0

    # Velocity: slope of recent half using simple linear regression
    velocity     = _slope(recent)
    # Acceleration: slope(recent) - slope(baseline)
    accel        = velocity - _slope(baseline)

    # Dormant check: last N windows all zero
    last_n       = counts[-_DORMANT_WINDOWS:]
    if all(c == 0 for c in last_n) and total > 0:
        return TrendMetrics("DORMANT", velocity, accel, recent_avg, baseline_avg, peak, total)
    if total == 0:
        return TrendMetrics("DORMANT", 0.0, 0.0, 0.0, 0.0, 0, 0)

    # Surge: recent avg much higher than baseline
    if baseline_avg > 0 and recent_avg >= baseline_avg * _SURGE_MULTIPLIER and recent_avg > 1:
        return TrendMetrics("SURGING", velocity, accel, recent_avg, baseline_avg, peak, total)

    # Rising / declining by velocity
    if velocity >= _RISING_MIN_VEL:
        return TrendMetrics("RISING", velocity, accel, recent_avg, baseline_avg, peak, total)
    if velocity <= _DECLINING_MAX_VEL:
        # Peaked: was higher in baseline, now falling
        if baseline_avg > recent_avg and peak in baseline:
            return TrendMetrics("PEAKED", velocity, accel, recent_avg, baseline_avg, peak, total)
        return TrendMetrics("DECLINING", velocity, accel, recent_avg, baseline_avg, peak, total)

    return TrendMetrics("STABLE", velocity, accel, recent_avg, baseline_avg, peak, total)


def _slope(values: list[float | int]) -> float:
    """Simple linear regression slope (least squares)."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num    = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    den    = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


# ── State persistence + alerting ──────────────────────────────────────────────

def update_trend_state(entity_id: int, metrics: TrendMetrics) -> bool:
    """
    Persist the computed trend state. Returns True if state changed.
    Fires a TREND_CHANGE alert on transition.
    """
    now = datetime.now().isoformat()
    with _conn() as db:
        existing = db.execute(
            "SELECT state FROM trend_states WHERE entity_id=?", (entity_id,)
        ).fetchone()

        prev_state = existing["state"] if existing else None
        state_changed = prev_state != metrics.state

        db.execute("""
            INSERT INTO trend_states
                (entity_id, state, velocity, acceleration, prev_state, state_since, updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(entity_id) DO UPDATE SET
                state=excluded.state,
                velocity=excluded.velocity,
                acceleration=excluded.acceleration,
                prev_state=excluded.prev_state,
                state_since=CASE WHEN state != excluded.state THEN excluded.state_since ELSE state_since END,
                updated_at=excluded.updated_at
        """, (entity_id, metrics.state, metrics.velocity, metrics.acceleration,
              prev_state, now, now))

    if state_changed and prev_state is not None:
        # Fetch entity name for alert message
        with _conn() as db:
            row = db.execute(
                "SELECT name, type FROM entities WHERE entity_id=?", (entity_id,)
            ).fetchone()
            if row:
                create_alert(
                    "TREND_CHANGE",
                    f"[{row['type']}] {row['name']}: {prev_state} → {metrics.state}  "
                    f"(velocity={metrics.velocity:+.2f}, accel={metrics.acceleration:+.2f})"
                )
    return state_changed


# ── Batch update all entities ─────────────────────────────────────────────────

def refresh_all_trends(window_days: int = 7, lookback_windows: int = 12) -> list[dict]:
    """
    Recompute trend state for every known entity.
    Called after each analysis run or on-demand via CLI.
    Returns list of {name, type, state, velocity, acceleration} for display.
    """
    migrate_db()
    with _conn() as db:
        entities = db.execute(
            "SELECT entity_id, name, type FROM entities ORDER BY mention_count DESC"
        ).fetchall()

    results = []
    for ent in entities:
        series  = get_entity_series(ent["entity_id"], window_days, lookback_windows)
        metrics = compute_trend(series)
        update_trend_state(ent["entity_id"], metrics)
        results.append({
            "name":         ent["name"],
            "type":         ent["type"],
            "state":        metrics.state,
            "velocity":     metrics.velocity,
            "acceleration": metrics.acceleration,
            "recent_avg":   metrics.recent_avg,
            "baseline_avg": metrics.baseline_avg,
            "peak":         metrics.peak,
            "total":        metrics.total,
        })
    return results


# ── Display helpers ───────────────────────────────────────────────────────────

_STATE_ICONS = {
    "RISING":    ("📈", "green"),
    "SURGING":   ("🚀", "bold green"),
    "PEAKED":    ("🏔️ ", "yellow"),
    "DECLINING": ("📉", "red"),
    "DORMANT":   ("💤", "dim"),
    "STABLE":    ("➡️ ", "cyan"),
    "EMERGING":  ("🌱", "bright_green"),
}

def format_trend_table(results: list[dict], min_total: int = 1) -> str:
    """
    Format trend results as a plain-text table for terminal display.
    Rich markup included for colored output.
    """
    if not results:
        return "No trend data. Run an analysis first."

    filtered = [r for r in results if r["total"] >= min_total]
    if not filtered:
        return f"No entities with ≥{min_total} total mentions yet."

    lines = [
        f"{'ENTITY':<28} {'TYPE':<8} {'STATE':<10} {'VEL':>6} {'ACCEL':>6} {'RECENT':>7} {'TOTAL':>6}",
        "─" * 80,
    ]
    for r in filtered:
        icon, _ = _STATE_ICONS.get(r["state"], ("  ", "white"))
        vel_str   = f"{r['velocity']:+.2f}"
        accel_str = f"{r['acceleration']:+.2f}"
        lines.append(
            f"{r['name']:<28} {r['type']:<8} {icon} {r['state']:<8} "
            f"{vel_str:>6} {accel_str:>6} {r['recent_avg']:>7.1f} {r['total']:>6}"
        )
    return "\n".join(lines)


def format_entity_trend_detail(entity_name: str,
                                window_days: int = 7,
                                lookback_windows: int = 12) -> str:
    """
    Full trend breakdown for a single entity including ASCII sparkline.
    """
    series, etype = get_series_by_name(entity_name, window_days, lookback_windows)
    if not series:
        return f"Entity '{entity_name}' not found in memory."

    counts = [s["count"] for s in series]
    metrics = compute_trend(series)
    icon, _ = _STATE_ICONS.get(metrics.state, ("  ", "white"))

    lines = [
        f"Trend Report: {entity_name}  [{etype}]",
        "─" * 50,
        f"  State       : {icon} {metrics.state}",
        f"  Velocity    : {metrics.velocity:+.3f}  (mentions/window change)",
        f"  Acceleration: {metrics.acceleration:+.3f}  (velocity change)",
        f"  Recent avg  : {metrics.recent_avg:.1f} mentions/window",
        f"  Baseline avg: {metrics.baseline_avg:.1f} mentions/window",
        f"  Peak window : {metrics.peak} mentions",
        f"  Total seen  : {metrics.total} mentions",
        "",
        f"  Window size : {window_days} day(s)  |  Lookback: {lookback_windows} windows",
        "",
        "  Sparkline (oldest → newest):",
        "  " + _sparkline(counts),
        "",
        "  Bucket history:",
    ]
    for s in series[-10:]:   # last 10 buckets
        bar = "█" * min(s["count"], 30) + (f" +{s['count']-30}" if s["count"] > 30 else "")
        lines.append(f"    {s['bucket']}  {bar or '·'}  ({s['count']})")

    return "\n".join(lines)


def _sparkline(values: list[int]) -> str:
    """Render a unicode sparkline for a list of counts."""
    if not values or max(values) == 0:
        return "▁" * len(values)
    blocks = " ▁▂▃▄▅▆▇█"
    max_v  = max(values)
    return "".join(blocks[min(int(v / max_v * 8), 8)] for v in values)