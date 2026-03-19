# file: monitor/scheduler.py
"""
PIW Continuous Monitor — scheduled feed polling with change detection.

Runs a background loop that:
  1. Polls RSS feeds every MONITOR_INTERVAL_MIN minutes
  2. Deduplicates against memory (skips already-analyzed articles)
  3. Detects topic spikes (same topic in 3+ sources at once)
  4. Runs pipeline on new clusters
  5. Fires alerts for new entities and topic spikes
  6. Prints a live status line between cycles

Run with:
  python piw.py monitor
  python piw.py monitor --interval 30   # poll every 30 min
  python piw.py monitor --interval 60 --top 2
"""

from __future__ import annotations
import signal
import sys
import time
from datetime import datetime

from rich import print as rprint
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

try:
    from config import (
        MONITOR_INTERVAL_MIN, MONITOR_TOP_N,
        ALERT_NEW_ENTITY, ALERT_TOPIC_SPIKE,
    )
except ImportError:
    MONITOR_INTERVAL_MIN = 60
    MONITOR_TOP_N        = 2
    ALERT_NEW_ENTITY     = True
    ALERT_TOPIC_SPIKE    = True

console = Console()

_running = True

def _handle_sigint(sig, frame):
    global _running
    _running = False
    rprint("\n[yellow]Monitor stopping after current cycle...[/yellow]")

signal.signal(signal.SIGINT, _handle_sigint)


# ── Single monitor cycle ──────────────────────────────────────────────────────

def run_monitor_cycle(top_n: int = MONITOR_TOP_N,
                      verbose: bool = True) -> dict:
    """
    One full collection → dedup → cluster → alert → pipeline cycle.
    Returns summary dict.
    """
    from collector.collector import collect_all
    from clustering.cluster import cluster_articles, pick_top_clusters
    from agents.agents import run_pipeline
    from memory.memory import init_db, filter_fresh_articles, get_top_entities
    from memory.entity_tracker import (
        extract_entities_from_report, check_and_alert_new_entities,
        check_topic_spike,
    )
    from output.renderer import render_report, save_report

    init_db()
    cycle_start = datetime.now()

    if verbose:
        rprint(f"\n[cyan]━━━ Monitor Cycle: {cycle_start.strftime('%Y-%m-%d %H:%M')} ━━━[/cyan]")

    # Collect
    articles = collect_all(scrape=True, verbose=verbose)
    if not articles:
        return {"status": "no_articles", "new": 0, "clusters": 0}

    # Dedup via memory
    fresh = filter_fresh_articles(articles)
    if verbose:
        rprint(f"  Fresh articles (not seen before): {len(fresh)} / {len(articles)}")

    if not fresh:
        return {"status": "all_seen", "new": 0, "clusters": 0}

    # Cluster fresh articles
    from clustering.cluster import cluster_articles, pick_top_clusters
    clusters = cluster_articles(fresh)
    top = pick_top_clusters(clusters, top_n=top_n)

    # Topic spike detection
    if ALERT_TOPIC_SPIKE:
        for cluster in top:
            source_count = len({a.source for a in cluster})
            topic_hint = _topic_from_cluster(cluster)
            check_topic_spike(topic_hint, source_count)

    # Known entities before this run
    known_names = {e["name"].lower() for e in get_top_entities(limit=1000)}

    reports = []
    for i, cluster in enumerate(top, 1):
        topic = _topic_from_cluster(cluster)
        if verbose:
            rprint(f"\n  [bold]Cluster {i}/{len(top)}:[/bold] {topic} ({len(cluster)} articles)")

        report = run_pipeline(cluster, topic=topic, verbose=verbose)
        render_report(report)
        save_report(report, label=topic)
        reports.append(report)

        # Entity extraction + new-entity alerts
        if ALERT_NEW_ENTITY:
            extracted = extract_entities_from_report(report)
            check_and_alert_new_entities(extracted, known_names)
            # Update known for next cluster in same cycle
            known_names.update(e["name"].lower() for e in extracted)

    # Print any pending alerts
    _print_pending_alerts()

    elapsed = (datetime.now() - cycle_start).seconds
    return {
        "status":    "ok",
        "new":       len(fresh),
        "clusters":  len(top),
        "elapsed_s": elapsed,
    }


# ── Monitor loop ──────────────────────────────────────────────────────────────

def run_monitor_loop(interval_min: int = MONITOR_INTERVAL_MIN,
                     top_n: int = MONITOR_TOP_N) -> None:
    """
    Run continuous monitor loop until Ctrl-C.
    Polls every interval_min minutes.
    """
    global _running
    _running = True

    rprint(f"\n[bold green]PIW Monitor started[/bold green]")
    rprint(f"  Poll interval : [cyan]{interval_min} min[/cyan]")
    rprint(f"  Topics/cycle  : [cyan]{top_n}[/cyan]")
    rprint(f"  Press [bold]Ctrl-C[/bold] to stop\n")

    cycle_count = 0
    while _running:
        cycle_count += 1
        rprint(f"[dim]Cycle #{cycle_count}[/dim]")
        result = run_monitor_cycle(top_n=top_n, verbose=True)

        if not _running:
            break

        next_run = datetime.now().replace(second=0, microsecond=0)
        wait_secs = interval_min * 60

        rprint(f"\n[dim]Cycle result: {result}[/dim]")
        rprint(f"[dim]Next cycle in {interval_min} min — press Ctrl-C to stop[/dim]")

        # Sleep in 10s chunks so Ctrl-C is responsive
        for _ in range(wait_secs // 10):
            if not _running:
                break
            time.sleep(10)

    rprint("[yellow]Monitor stopped.[/yellow]")


# ── Alert display ─────────────────────────────────────────────────────────────

def _print_pending_alerts() -> None:
    from memory.memory import get_pending_alerts, acknowledge_all_alerts
    alerts = get_pending_alerts()
    if not alerts:
        return

    rprint(f"\n[bold yellow]⚡ {len(alerts)} ALERT(S)[/bold yellow]")
    for a in alerts:
        icon = "🆕" if a["alert_type"] == "NEW_ENTITY" else "📈"
        rprint(f"  {icon} [{a['alert_type']}] {a['message']}")
    acknowledge_all_alerts()


def _topic_from_cluster(articles) -> str:
    from collections import Counter
    import re
    stopwords = {"the","a","an","in","on","at","to","for","of","and","or",
                 "is","are","was","were","says","said","new","over","after"}
    words = []
    for a in articles[:5]:
        words += [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", a.title)
                  if w.lower() not in stopwords]
    if not words:
        return articles[0].title[:50]
    return " ".join(w for w, _ in Counter(words).most_common(4)).title()