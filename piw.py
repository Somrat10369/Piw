#!/usr/bin/env python3
# file: piw.py
"""
PIW — Personal Intelligence Workstation  (Phase 1 + 2)
Main CLI entrypoint.

USAGE:
  python piw.py                           # Interactive menu
  python piw.py analyze                   # Fetch feeds + analyze top topics
  python piw.py url <URL>                 # Deep-analyze a single article/URL
  python piw.py topic "<query>"           # Filter + analyze a topic
  python piw.py check                     # Check Ollama connection
  python piw.py monitor                   # Continuous feed monitoring loop
  python piw.py monitor --interval 30     # Poll every 30 min
  python piw.py history                   # Show past analysis runs
  python piw.py history --topic "IMF"     # Filter history by topic
  python piw.py entities                  # Show tracked entities
  python piw.py entities --type COUNTRY   # Filter by COUNTRY/ORG/PERSON
  python piw.py entity-timeline "IMF"     # Show all runs mentioning an entity
  python piw.py alerts                    # Show pending alerts
  python piw.py memory-stats              # Memory DB summary

INSTALL DEPS (run once):
  pip install feedparser beautifulsoup4 requests rich typer httpx
  pip install sentence-transformers faiss-cpu   # optional: semantic clustering

OLLAMA SETUP:
  ollama pull qwen2.5:14b             # recommended model (~9GB, fits RTX 4070)
  ollama serve                        # must be running before piw.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich import print as rprint

from config import OLLAMA_MODEL, SAVE_REPORTS, MONITOR_INTERVAL_MIN, MONITOR_TOP_N
from agents.ollama_client import check_ollama_connection
from agents.agents import run_pipeline
from collector.collector import collect_all, collect_from_url
from clustering.cluster import cluster_articles, pick_top_clusters
from output.renderer import render_report, save_report
from memory.memory import (
    init_db, get_topic_history, get_memory_stats,
    get_pending_alerts, acknowledge_all_alerts,
)
from memory.entity_tracker import (
    format_entity_table, format_entity_timeline,
    extract_entities_from_report,
)
from memory.entity_tracker import get_top_entities

console = Console()
cli = typer.Typer(
    help="PIW — Personal Intelligence Workstation",
    no_args_is_help=False,
)


# ── check ─────────────────────────────────────────────────────────────────────

@cli.command()
def check():
    """Check Ollama connection and available models."""
    ok, msg = check_ollama_connection()
    icon = "✅" if ok else "❌"
    rprint(f"{icon}  {msg}")
    if not ok:
        rprint("\n[yellow]Start Ollama:[/yellow]  ollama serve")
        rprint(f"[yellow]Pull model:[/yellow]    ollama pull {OLLAMA_MODEL}")


# ── analyze (full pipeline) ───────────────────────────────────────────────────

@cli.command()
def analyze(
    top_n: int = typer.Option(3, "--top", "-n", help="Number of top topic clusters to analyze"),
    scrape: bool = typer.Option(True, "--scrape/--no-scrape", help="Scrape full article bodies"),
    save: bool = typer.Option(SAVE_REPORTS, "--save/--no-save", help="Save reports to output/"),
):
    """Fetch RSS feeds, cluster by topic, analyze top N clusters."""
    ok, msg = check_ollama_connection()
    if not ok:
        rprint(f"[red]❌ {msg}[/red]")
        raise typer.Exit(1)
    rprint(f"[green]✓[/green] {msg}\n")

    # Collect
    articles = collect_all(scrape=scrape)
    if not articles:
        rprint("[red]No articles collected. Check RSS feed URLs.[/red]")
        raise typer.Exit(1)

    # Cluster
    rprint(f"[cyan]Clustering {len(articles)} articles...[/cyan]")
    clusters = cluster_articles(articles)
    top_clusters = pick_top_clusters(clusters, top_n=top_n)
    rprint(f"[cyan]Found {len(clusters)} clusters — analyzing top {len(top_clusters)}[/cyan]\n")

    # Pipeline per cluster
    for i, cluster in enumerate(top_clusters, 1):
        topic_hint = _guess_topic(cluster)
        rprint(f"\n[bold white]══ Cluster {i}/{len(top_clusters)}: {topic_hint} ({len(cluster)} articles) ══[/bold white]")
        report = run_pipeline(cluster, topic=topic_hint)
        render_report(report)
        if save:
            path = save_report(report, label=topic_hint)
            rprint(f"[dim]💾 Saved → {path}[/dim]")


# ── url (single article deep-dive) ───────────────────────────────────────────

@cli.command()
def url(
    target_url: str = typer.Argument(..., help="URL to scrape and analyze"),
    save: bool = typer.Option(SAVE_REPORTS, "--save/--no-save"),
):
    """Deep-analyze a single URL or article."""
    ok, msg = check_ollama_connection()
    if not ok:
        rprint(f"[red]❌ {msg}[/red]")
        raise typer.Exit(1)

    rprint(f"\n[cyan]Scraping:[/cyan] {target_url}")
    article = collect_from_url(target_url)
    if not article.body:
        rprint("[yellow]Warning: Could not scrape body text. Proceeding with URL only.[/yellow]")

    report = run_pipeline([article], topic=article.title[:60])
    render_report(report)
    if save:
        path = save_report(report, label="url_analysis")
        rprint(f"[dim]💾 Saved → {path}[/dim]")


# ── topic search ──────────────────────────────────────────────────────────────

@cli.command()
def topic(
    query: str = typer.Argument(..., help="Topic keyword to filter articles on"),
    scrape: bool = typer.Option(True, "--scrape/--no-scrape"),
    save: bool = typer.Option(SAVE_REPORTS, "--save/--no-save"),
):
    """Collect articles, filter by topic keyword, then analyze."""
    ok, msg = check_ollama_connection()
    if not ok:
        rprint(f"[red]❌ {msg}[/red]")
        raise typer.Exit(1)

    articles = collect_all(scrape=scrape)
    q = query.lower()
    filtered = [a for a in articles
                if q in a.title.lower() or q in a.summary.lower()]

    if not filtered:
        rprint(f"[yellow]No articles found matching '{query}'. Try a broader term.[/yellow]")
        raise typer.Exit(1)

    rprint(f"\n[cyan]Found {len(filtered)} articles matching '{query}'[/cyan]")
    report = run_pipeline(filtered, topic=query)
    render_report(report)
    if save:
        path = save_report(report, label=query)
        rprint(f"[dim]💾 Saved → {path}[/dim]")


# ── interactive menu ──────────────────────────────────────────────────────────

@cli.command()
def interactive():
    """Interactive menu-driven session."""
    init_db()
    rprint("\n[bold cyan]╔══════════════════════════════════════╗[/bold cyan]")
    rprint("[bold cyan]║   PIW — Intelligence Workstation     ║[/bold cyan]")
    rprint("[bold cyan]╚══════════════════════════════════════╝[/bold cyan]\n")

    ok, msg = check_ollama_connection()
    rprint(f"{'[green]✓[/green]' if ok else '[red]✗[/red]'}  Ollama: {msg}")
    stats = get_memory_stats()
    rprint(f"[dim]  Memory: {stats['total_runs']} runs | {stats['entities_known']} entities | "
           f"{stats['pending_alerts']} alerts pending[/dim]\n")

    while True:
        rprint("[bold]What do you want to do?[/bold]")
        rprint("  [cyan]1[/cyan]  Full feed scan (analyze top topics)")
        rprint("  [cyan]2[/cyan]  Analyze a specific topic/keyword")
        rprint("  [cyan]3[/cyan]  Analyze a specific URL")
        rprint("  [cyan]4[/cyan]  Start continuous monitor")
        rprint("  [cyan]5[/cyan]  View analysis history")
        rprint("  [cyan]6[/cyan]  View tracked entities")
        rprint("  [cyan]7[/cyan]  View pending alerts")
        rprint("  [cyan]8[/cyan]  Check Ollama status")
        rprint("  [cyan]9[/cyan]  View entity trends (all)")
        rprint("  [cyan]10[/cyan] View trend detail for one entity")
        rprint("  [cyan]11[/cyan] Diff two runs (by topic)")
        rprint("  [cyan]12[/cyan] Open TUI dashboard")
        rprint("  [cyan]q[/cyan]  Quit\n")

        choice = Prompt.ask("Choice", default="1")

        if choice == "1":
            n = int(Prompt.ask("How many top topics?", default="3"))
            analyze(top_n=n)
        elif choice == "2":
            q = Prompt.ask("Enter topic keyword")
            topic(query=q)
        elif choice == "3":
            u = Prompt.ask("Enter URL")
            url(target_url=u)
        elif choice == "4":
            iv = int(Prompt.ask("Poll interval (minutes)", default=str(MONITOR_INTERVAL_MIN)))
            monitor(interval=iv)
        elif choice == "5":
            t = Prompt.ask("Filter by topic (or Enter for all)", default="")
            history(topic_filter=t or None)
        elif choice == "6":
            et = Prompt.ask("Filter by type (COUNTRY/ORG/PERSON or Enter for all)", default="")
            entities(type_filter=et.upper() or None)
        elif choice == "7":
            alerts()
        elif choice == "8":
            check()
        elif choice == "9":
            trends()
        elif choice == "10":
            n = Prompt.ask("Enter entity name")
            trend_entity(name=n)
        elif choice == "11":
            t = Prompt.ask("Enter topic keyword")
            diff(topic_filter=t)
        elif choice == "12":
            dashboard()
        elif choice in ("q", "quit", "exit"):
            rprint("[dim]Exiting PIW.[/dim]")
            break
        else:
            rprint("[yellow]Unknown choice.[/yellow]")

        rprint()


# ── monitor ───────────────────────────────────────────────────────────────────

@cli.command()
def monitor(
    interval: int = typer.Option(MONITOR_INTERVAL_MIN, "--interval", "-i",
                                  help="Minutes between feed polls"),
    top_n: int = typer.Option(MONITOR_TOP_N, "--top", "-n",
                               help="Clusters to analyze per cycle"),
):
    """Continuous feed monitoring loop with alerts. Ctrl-C to stop."""
    init_db()
    ok, msg = check_ollama_connection()
    if not ok:
        rprint(f"[red]❌ {msg}[/red]")
        raise typer.Exit(1)
    from monitor.scheduler import run_monitor_loop
    run_monitor_loop(interval_min=interval, top_n=top_n)


# ── history ───────────────────────────────────────────────────────────────────

@cli.command()
def history(
    topic_filter: str = typer.Option(None, "--topic", "-t", help="Filter by topic keyword"),
    limit: int = typer.Option(20, "--limit", "-n"),
):
    """Show past analysis run history from memory."""
    init_db()
    runs = get_topic_history(topic=topic_filter, limit=limit)
    if not runs:
        rprint("[yellow]No history found.[/yellow]")
        return

    table = Table(title="PIW Analysis History", show_lines=True)
    table.add_column("Run ID", style="dim", width=22)
    table.add_column("Topic", style="bold")
    table.add_column("Articles", justify="right")
    table.add_column("Sources")
    table.add_column("Date", style="dim")

    for r in runs:
        import json as _json
        sources_list = _json.loads(r["sources"]) if r["sources"] else []
        table.add_row(
            r["run_id"][:22],
            r["topic"][:50],
            str(r["article_count"]),
            ", ".join(sources_list[:3]) + ("…" if len(sources_list) > 3 else ""),
            r["run_at"][:16],
        )
    console.print(table)


# ── entities ──────────────────────────────────────────────────────────────────

@cli.command()
def entities(
    type_filter: str = typer.Option(None, "--type", "-t",
                                     help="Filter: COUNTRY | ORG | PERSON | ECON | EVENT"),
    limit: int = typer.Option(30, "--limit", "-n"),
):
    """Show tracked entities ranked by mention count."""
    init_db()
    ents = get_top_entities(etype=type_filter, limit=limit)
    if not ents:
        rprint("[yellow]No entities tracked yet. Run an analysis first.[/yellow]")
        return
    rprint(format_entity_table(ents))


# ── entity-timeline ───────────────────────────────────────────────────────────

@cli.command(name="entity-timeline")
def entity_timeline(
    name: str = typer.Argument(..., help="Entity name to look up"),
):
    """Show all analysis runs where this entity appeared."""
    init_db()
    rprint(format_entity_timeline(name))


# ── alerts ────────────────────────────────────────────────────────────────────

@cli.command()
def alerts(
    ack: bool = typer.Option(False, "--ack", help="Acknowledge all alerts after viewing"),
):
    """Show pending alerts (new entities, topic spikes)."""
    init_db()
    pending = get_pending_alerts()
    if not pending:
        rprint("[green]✓ No pending alerts.[/green]")
        return

    rprint(f"\n[bold yellow]⚡ {len(pending)} pending alert(s)[/bold yellow]\n")
    for a in pending:
        icon = {"NEW_ENTITY": "🆕", "TOPIC_SPIKE": "📈", "TREND_CHANGE": "🔄"}.get(a["alert_type"], "⚠")
        rprint(f"  {icon} [bold]{a['alert_type']}[/bold]  {a['created_at'][:16]}")
        rprint(f"     {a['message']}\n")

    if ack:
        acknowledge_all_alerts()
        rprint("[dim]All alerts acknowledged.[/dim]")
    else:
        rprint("[dim]Run with --ack to dismiss all.[/dim]")


# ── memory-stats ──────────────────────────────────────────────────────────────

@cli.command(name="memory-stats")
def memory_stats():
    """Show memory database summary stats."""
    init_db()
    s = get_memory_stats()
    rprint("\n[bold]PIW Memory Stats[/bold]")
    rprint(f"  Articles seen   : [cyan]{s['articles_seen']}[/cyan]")
    rprint(f"  Analysis runs   : [cyan]{s['total_runs']}[/cyan]")
    rprint(f"  Entities known  : [cyan]{s['entities_known']}[/cyan]")
    rprint(f"  Pending alerts  : [yellow]{s['pending_alerts']}[/yellow]")
    rprint(f"  Tracking since  : [dim]{s['tracking_since'][:16] if s['tracking_since'] != 'never' else 'never'}[/dim]")


# ── trends (all entities) ─────────────────────────────────────────────────────

@cli.command()
def trends(
    type_filter: str  = typer.Option(None, "--type", "-t",
                                      help="Filter: COUNTRY | ORG | PERSON | ECON | EVENT"),
    window: int       = typer.Option(7,  "--window", "-w", help="Bucket size in days (1=daily, 7=weekly)"),
    lookback: int     = typer.Option(12, "--lookback", "-l", help="How many windows to look back"),
    min_mentions: int = typer.Option(2,  "--min", "-m", help="Min total mentions to include"),
):
    """Show trend state for all tracked entities (Rising/Surging/Declining/etc)."""
    init_db()
    from memory.trend_tracker import refresh_all_trends, format_trend_table, migrate_db
    migrate_db()
    rprint(f"\n[cyan]Computing trends (window={window}d, lookback={lookback} windows)...[/cyan]")
    results = refresh_all_trends(window_days=window, lookback_windows=lookback)

    if type_filter:
        results = [r for r in results if r["type"] == type_filter.upper()]

    rprint(format_trend_table(results, min_total=min_mentions))

    # Summary counts
    from collections import Counter
    state_counts = Counter(r["state"] for r in results if r["total"] >= min_mentions)
    if state_counts:
        rprint("\n[dim]Summary:[/dim]  " + "  ".join(
            f"{s}: {c}" for s, c in sorted(state_counts.items())
        ))


# ── trend-entity (single entity detail) ──────────────────────────────────────

@cli.command(name="trend-entity")
def trend_entity(
    name:     str = typer.Argument(..., help="Entity name to analyze"),
    window:   int = typer.Option(7,  "--window",   "-w", help="Bucket size in days"),
    lookback: int = typer.Option(12, "--lookback", "-l", help="Number of windows to look back"),
):
    """Show detailed trend breakdown + sparkline for a single entity."""
    init_db()
    from memory.trend_tracker import (
        format_entity_trend_detail, refresh_all_trends, migrate_db
    )
    migrate_db()
    rprint(format_entity_trend_detail(name, window_days=window, lookback_windows=lookback))



# ── diff ──────────────────────────────────────────────────────────────────────

@cli.command()
def diff(
    run_id_a:    str = typer.Argument(None, help="First run ID (older). Omit to use --topic auto-pick."),
    run_id_b:    str = typer.Argument(None, help="Second run ID (newer)."),
    topic_filter:str = typer.Option(None, "--topic", "-t",
                                    help="Auto-pick 2 most recent runs for this topic"),
    list_runs:   bool= typer.Option(False, "--list", "-l",
                                    help="List available run IDs for the topic"),
):
    """Compare two analysis runs — see what changed between them."""
    init_db()
    from output.diff import (
        diff_by_run_ids, diff_latest_by_topic,
        list_runs_for_topic, render_diff,
    )

    # List mode
    if list_runs:
        if not topic_filter:
            rprint("[red]--list requires --topic[/red]")
            raise typer.Exit(1)
        list_runs_for_topic(topic_filter)
        return

    # Auto-pick by topic
    if topic_filter and not run_id_a:
        result = diff_latest_by_topic(topic_filter)
        if result:
            render_diff(result)
        return

    # Explicit run IDs
    if run_id_a and run_id_b:
        result = diff_by_run_ids(run_id_a, run_id_b)
        if result:
            render_diff(result)
        return

    # Fallback: show last 2 runs overall
    from memory.memory import get_topic_history, get_report_by_run_id
    runs = get_topic_history(limit=2)
    if len(runs) < 2:
        rprint("[yellow]Not enough runs in memory. Run 'analyze' at least twice first.[/yellow]")
        raise typer.Exit(1)
    ra = get_report_by_run_id(runs[-1]["run_id"])
    rb = get_report_by_run_id(runs[0]["run_id"])
    if ra and rb:
        ra.setdefault("run_at", runs[-1]["run_at"])
        rb.setdefault("run_at", runs[0]["run_at"])
        from output.diff import diff_reports
        render_diff(diff_reports(ra, rb))



# ── dashboard (TUI) ───────────────────────────────────────────────────────────

@cli.command()
def dashboard():
    """Launch live TUI dashboard (requires: pip install textual)."""
    init_db()
    try:
        from tui import run_dashboard
        run_dashboard()
    except ImportError:
        rprint("[red]Textual not installed.[/red]")
        rprint("Run:  [cyan]pip install textual[/cyan]")
        raise typer.Exit(1)


# ── helpers ───────────────────────────────────────────────────────────────────

def _guess_topic(articles) -> str:
    from collections import Counter
    import re
    stopwords = {"the","a","an","in","on","at","to","for","of","and","or",
                 "is","are","was","were","says","said","new","over","after","as"}
    words = []
    for a in articles[:5]:
        words += [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", a.title)
                  if w.lower() not in stopwords]
    if not words:
        return articles[0].title[:50]
    return " ".join(w for w, _ in Counter(words).most_common(4)).title()


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()  # ensure DB is ready on every run
    if len(sys.argv) == 1:
        interactive()
    else:
        cli()