# file: tui.py
"""
PIW TUI Dashboard — live terminal UI using Textual.

Layout (6 panels):
  ┌─ Header: system status bar ──────────────────────────────────┐
  │ ┌── Trends (left) ──┬── Latest Report (center) ─────────────┤
  │ │  entity table     │  section tabs: Reality/Bias/etc        │
  │ │                   │                                        │
  │ ├── Alerts (left)  ─┼── Fact-Check summary (center-bottom)  │
  │ │  alert feed       │                                        │
  │ └───────────────────┴── Run History (right) ────────────────-┤
  │                         past runs table                      │
  └─ Footer: keybindings ────────────────────────────────────────┘

Keybindings:
  r   — Refresh all panels from DB
  a   — Acknowledge all alerts
  d   — Diff last two runs (launches in terminal)
  q   — Quit
  Tab — Cycle report section
  1-7 — Jump to report section directly

Run with:
  python piw.py dashboard
  python tui.py             (direct)
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
from datetime import datetime
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import (
    Header, Footer, Static, DataTable,
    Label, RichLog, TabbedContent, TabPane, Rule,
)
from textual import work, on
from textual.timer import Timer

# ── Safe imports from PIW ──────────────────────────────────────────────────────

def _safe_import():
    try:
        from memory.memory import (
            init_db, get_memory_stats, get_pending_alerts,
            acknowledge_all_alerts, get_topic_history,
            get_report_by_run_id,
        )
        from memory.trend_tracker import refresh_all_trends, migrate_db
        from agents.ollama_client import check_ollama_connection
        init_db()
        migrate_db()
        return True
    except Exception as e:
        return False

_READY = _safe_import()

def _get_stats() -> dict:
    try:
        from memory.memory import get_memory_stats
        return get_memory_stats()
    except Exception:
        return {"articles_seen": 0, "total_runs": 0, "entities_known": 0,
                "pending_alerts": 0, "tracking_since": "never"}

def _get_alerts(limit: int = 20) -> list[dict]:
    try:
        from memory.memory import get_pending_alerts
        return get_pending_alerts()[:limit]
    except Exception:
        return []

def _get_trends(limit: int = 30) -> list[dict]:
    try:
        from memory.trend_tracker import refresh_all_trends
        return refresh_all_trends(window_days=7, lookback_windows=12)
    except Exception:
        return []

def _get_history(limit: int = 15) -> list[dict]:
    try:
        from memory.memory import get_topic_history
        return get_topic_history(limit=limit)
    except Exception:
        return []

def _get_latest_report() -> dict | None:
    try:
        from memory.memory import get_topic_history, get_report_by_run_id
        runs = get_topic_history(limit=1)
        if runs:
            return get_report_by_run_id(runs[0]["run_id"])
    except Exception:
        pass
    return None

def _ollama_status() -> str:
    try:
        from agents.ollama_client import check_ollama_connection
        ok, msg = check_ollama_connection()
        return "✓ Ollama" if ok else "✗ Ollama"
    except Exception:
        return "✗ Ollama"

def _strip_rich(text: str) -> str:
    """Remove Rich markup tags for clean display in Textual."""
    return re.sub(r'\[/?[^\]]*\]', '', text)

def _sparkline(values: list[int], width: int = 10) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    if not values or max(values) == 0:
        return "▁" * min(len(values), width)
    mx = max(values)
    recent = values[-width:]
    return "".join(blocks[min(int(v / mx * 8), 8)] for v in recent)


# ── Widgets ───────────────────────────────────────────────────────────────────

class StatusBar(Static):
    """Top status bar showing system vitals."""

    DEFAULT_CSS = """
    StatusBar {
        background: $primary-darken-3;
        color: $text;
        padding: 0 2;
        height: 1;
    }
    """

    def update_status(self, stats: dict, ollama: str) -> None:
        alerts_str = f"⚡ {stats['pending_alerts']} alerts" if stats['pending_alerts'] > 0 else "✓ No alerts"
        self.update(
            f" PIW  |  {ollama}  |  "
            f"📊 {stats['entities_known']} entities  "
            f"🔄 {stats['total_runs']} runs  "
            f"📰 {stats['articles_seen']} articles  "
            f"{alerts_str}  "
            f"|  {datetime.now().strftime('%H:%M:%S')}"
        )


class TrendsPanel(Static):
    """Left panel: entity trend table."""

    DEFAULT_CSS = """
    TrendsPanel {
        border: solid $primary;
        padding: 0 1;
        height: 100%;
        overflow-y: auto;
    }
    """

    STATE_ICONS = {
        "RISING":    "📈",
        "SURGING":   "🚀",
        "PEAKED":    "🏔",
        "DECLINING": "📉",
        "DORMANT":   "💤",
        "STABLE":    "➡",
        "EMERGING":  "🌱",
    }

    def render_trends(self, trends: list[dict]) -> None:
        if not trends:
            self.update("[dim]No trend data.\nRun 'analyze' first.[/dim]")
            return

        lines = ["[bold cyan]📈 ENTITY TRENDS[/bold cyan]", ""]
        # Show top 18 by total mentions
        top = sorted(trends, key=lambda x: x.get("total", 0), reverse=True)[:18]
        for t in top:
            icon  = self.STATE_ICONS.get(t["state"], "?")
            name  = t["name"][:18]
            state = t["state"][:8]
            vel   = t.get("velocity", 0)
            vel_s = f"{vel:+.1f}" if vel != 0 else " 0.0"

            color = {
                "RISING": "green", "SURGING": "bright_green",
                "PEAKED": "yellow", "DECLINING": "red",
                "DORMANT": "dim", "STABLE": "cyan", "EMERGING": "bright_cyan",
            }.get(t["state"], "white")

            lines.append(
                f"[{color}]{icon} {name:<18} {state:<9} {vel_s}[/{color}]"
            )

        self.update("\n".join(lines))


class AlertsPanel(Static):
    """Left-bottom panel: pending alerts feed."""

    DEFAULT_CSS = """
    AlertsPanel {
        border: solid $warning;
        padding: 0 1;
        height: 100%;
        overflow-y: auto;
    }
    """

    ALERT_ICONS = {
        "NEW_ENTITY":   "🆕",
        "TOPIC_SPIKE":  "📈",
        "TREND_CHANGE": "🔄",
    }

    def render_alerts(self, alerts: list[dict]) -> None:
        if not alerts:
            self.update("[dim]⚡ ALERTS[/dim]\n\n[green]✓ No pending alerts[/green]")
            return

        lines = [f"[bold yellow]⚡ ALERTS ({len(alerts)})[/bold yellow]", ""]
        for a in alerts[:12]:
            icon = self.ALERT_ICONS.get(a["alert_type"], "⚠")
            ts   = a.get("created_at", "")[:16]
            msg  = a.get("message", "")[:55]
            lines.append(f"[yellow]{icon}[/yellow] [dim]{ts}[/dim]")
            lines.append(f"  {msg}")
            lines.append("")

        if len(alerts) > 12:
            lines.append(f"[dim]… and {len(alerts)-12} more. Press [a] to ack all.[/dim]")

        self.update("\n".join(lines))


class ReportPanel(Static):
    """Center panel: latest report sections as tabs."""

    DEFAULT_CSS = """
    ReportPanel {
        border: solid $success;
        padding: 0 1;
        height: 100%;
        overflow-y: auto;
    }
    """

    SECTIONS = [
        ("reality",    "🧩 Reality"),
        ("factcheck",  "🔍 Fact-Check"),
        ("bias",       "⚖ Bias"),
        ("missing",    "🕳 Missing"),
        ("incentives", "💰 Incentives"),
        ("trends",     "📈 Trends"),
        ("scenarios",  "🔮 Scenarios"),
        ("personal",   "🎯 Personal"),
    ]

    _current_section: int = 0
    _report: dict | None = None

    def render_report(self, report: dict | None) -> None:
        self._report = report
        self._draw()

    def next_section(self) -> None:
        self._current_section = (self._current_section + 1) % len(self.SECTIONS)
        self._draw()

    def goto_section(self, idx: int) -> None:
        self._current_section = idx % len(self.SECTIONS)
        self._draw()

    def _draw(self) -> None:
        if not self._report:
            self.update(
                "[bold green]🔍 LATEST REPORT[/bold green]\n\n"
                "[dim]No reports in memory yet.\n"
                "Run:  python piw.py analyze[/dim]"
            )
            return

        key, label = self.SECTIONS[self._current_section]
        topic = self._report.get("topic", "Unknown")
        ts    = self._report.get("generated_at", self._report.get("run_at",""))[:16]
        srcs  = ", ".join(self._report.get("sources_used", [])[:3])

        # Section tab bar
        tab_parts = []
        for i, (_, lbl) in enumerate(self.SECTIONS):
            if i == self._current_section:
                tab_parts.append(f"[reverse bold]{lbl}[/reverse bold]")
            else:
                tab_parts.append(f"[dim]{lbl}[/dim]")
        tabs = "  ".join(tab_parts)

        lines = [
            f"[bold green]{topic}[/bold green]  [dim]{ts} | {srcs}[/dim]",
            f"[dim]{tabs}[/dim]",
            "[dim]Tab=next section  1-8=jump[/dim]",
            "─" * 50,
            "",
        ]

        # Get section content
        if key == "factcheck":
            fc = self._report.get("factcheck")
            if fc:
                lines.append(self._format_factcheck(fc))
            else:
                lines.append("[dim]No fact-check data in this report.[/dim]")
        else:
            agents = self._report.get("agents", {})
            sec    = agents.get(key, {})
            text   = sec.get("output", "") if sec else ""
            if text:
                # Show first ~40 lines
                shown = "\n".join(text.splitlines()[:40])
                lines.append(shown)
            else:
                lines.append(f"[dim]No {label} data in this report.[/dim]")

        self.update("\n".join(lines))

    def _format_factcheck(self, fc: dict) -> str:
        lines = [
            f"[green]✅ {fc.get('confirmed',0)} confirmed[/green]  "
            f"[red]❌ {fc.get('disputed',0)} disputed[/red]  "
            f"[yellow]⚠ {fc.get('misleading',0)} misleading[/yellow]  "
            f"[cyan]🔵 {fc.get('single',0)} single[/cyan]  "
            f"[dim]❓ {fc.get('unverifiable',0)} unverifiable[/dim]",
            f"Confidence: [bold]{fc.get('confidence','?')}[/bold]",
            "",
        ]
        verdict_order = ["DISPUTED","MISLEADING","CONFIRMED","SINGLE","UNVERIFIABLE"]
        claims = sorted(
            fc.get("claims", []),
            key=lambda c: verdict_order.index(c["verdict"])
                          if c["verdict"] in verdict_order else 99
        )
        icons = {"CONFIRMED":"✅","DISPUTED":"❌","MISLEADING":"⚠","SINGLE":"🔵","UNVERIFIABLE":"❓"}
        for c in claims[:12]:
            icon = icons.get(c["verdict"], "?")
            lines.append(f"{icon} [{c.get('source','')}] {c.get('text','')[:70]}")
            lines.append(f"   [dim]{c.get('explanation','')[:60]}[/dim]")
        return "\n".join(lines)


class HistoryPanel(Static):
    """Right panel: run history table."""

    DEFAULT_CSS = """
    HistoryPanel {
        border: solid $secondary;
        padding: 0 1;
        height: 100%;
        overflow-y: auto;
    }
    """

    def render_history(self, runs: list[dict]) -> None:
        if not runs:
            self.update("[dim]📋 RUN HISTORY\n\nNo runs yet.[/dim]")
            return

        lines = ["[bold]📋 RUN HISTORY[/bold]", ""]
        for r in runs[:15]:
            ts    = r.get("run_at", "")[:16]
            topic = r.get("topic", "?")[:28]
            arts  = r.get("article_count", 0)
            lines.append(f"[cyan]{ts}[/cyan]")
            lines.append(f"  {topic}")
            lines.append(f"  [dim]{arts} articles[/dim]")
            lines.append("")

        self.update("\n".join(lines))


# ── Main App ──────────────────────────────────────────────────────────────────

class PIWDashboard(App):
    """PIW TUI Dashboard."""

    TITLE = "PIW — Personal Intelligence Workstation"
    SUB_TITLE = "Live Dashboard"

    CSS = """
    Screen {
        layout: vertical;
    }

    #status-bar {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        padding: 0 1;
    }

    #main-row {
        layout: horizontal;
        height: 1fr;
    }

    #left-col {
        layout: vertical;
        width: 28;
        min-width: 24;
    }

    #center-col {
        layout: vertical;
        width: 1fr;
    }

    #right-col {
        layout: vertical;
        width: 30;
        min-width: 26;
    }

    TrendsPanel {
        height: 2fr;
        border: solid $primary;
        padding: 0 1;
        overflow-y: auto;
    }

    AlertsPanel {
        height: 1fr;
        border: solid $warning;
        padding: 0 1;
        overflow-y: auto;
    }

    ReportPanel {
        height: 1fr;
        border: solid $success;
        padding: 0 1;
        overflow-y: auto;
    }

    HistoryPanel {
        height: 1fr;
        border: solid $secondary;
        padding: 0 1;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        Binding("r",   "refresh",   "Refresh",         show=True),
        Binding("a",   "ack_alerts","Ack Alerts",       show=True),
        Binding("tab", "next_tab",  "Next Section",     show=True),
        Binding("1",   "section_1", "Reality",          show=False),
        Binding("2",   "section_2", "Fact-Check",       show=False),
        Binding("3",   "section_3", "Bias",             show=False),
        Binding("4",   "section_4", "Missing",          show=False),
        Binding("5",   "section_5", "Incentives",       show=False),
        Binding("6",   "section_6", "Trends",           show=False),
        Binding("7",   "section_7", "Scenarios",        show=False),
        Binding("8",   "section_8", "Personal",         show=False),
        Binding("q",   "quit",      "Quit",             show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield StatusBar("Loading…", id="status-bar")

        with Horizontal(id="main-row"):
            with Vertical(id="left-col"):
                yield TrendsPanel(id="trends-panel")
                yield AlertsPanel(id="alerts-panel")

            with Vertical(id="center-col"):
                yield ReportPanel(id="report-panel")

            with Vertical(id="right-col"):
                yield HistoryPanel(id="history-panel")

        yield Footer()

    def on_mount(self) -> None:
        """Initial data load + start auto-refresh timer."""
        self.refresh_data()
        self._timer: Timer = self.set_interval(30, self.refresh_data)

    # ── Data loading ──────────────────────────────────────────────────────────

    @work(thread=True)
    def refresh_data(self) -> None:
        """Load all data from DB in a background thread."""
        stats   = _get_stats()
        alerts  = _get_alerts()
        trends  = _get_trends()
        history = _get_history()
        report  = _get_latest_report()
        ollama  = _ollama_status()
        self.call_from_thread(self._update_ui, stats, alerts, trends, history, report, ollama)

    def _update_ui(self, stats, alerts, trends, history, report, ollama) -> None:
        """Apply loaded data to all widgets (called on main thread)."""
        self.query_one("#status-bar", StatusBar).update_status(stats, ollama)
        self.query_one("#trends-panel", TrendsPanel).render_trends(trends)
        self.query_one("#alerts-panel", AlertsPanel).render_alerts(alerts)
        self.query_one("#report-panel", ReportPanel).render_report(report)
        self.query_one("#history-panel", HistoryPanel).render_history(history)

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_refresh(self) -> None:
        self.query_one("#status-bar", StatusBar).update("Refreshing…")
        self.refresh_data()

    def action_ack_alerts(self) -> None:
        try:
            from memory.memory import acknowledge_all_alerts
            acknowledge_all_alerts()
        except Exception:
            pass
        self.refresh_data()

    def action_next_tab(self) -> None:
        self.query_one("#report-panel", ReportPanel).next_section()

    def action_section_1(self) -> None:
        self.query_one("#report-panel", ReportPanel).goto_section(0)

    def action_section_2(self) -> None:
        self.query_one("#report-panel", ReportPanel).goto_section(1)

    def action_section_3(self) -> None:
        self.query_one("#report-panel", ReportPanel).goto_section(2)

    def action_section_4(self) -> None:
        self.query_one("#report-panel", ReportPanel).goto_section(3)

    def action_section_5(self) -> None:
        self.query_one("#report-panel", ReportPanel).goto_section(4)

    def action_section_6(self) -> None:
        self.query_one("#report-panel", ReportPanel).goto_section(5)

    def action_section_7(self) -> None:
        self.query_one("#report-panel", ReportPanel).goto_section(6)

    def action_section_8(self) -> None:
        self.query_one("#report-panel", ReportPanel).goto_section(7)


# ── Entry ─────────────────────────────────────────────────────────────────────

def run_dashboard() -> None:
    """Launch the TUI dashboard. Called from piw.py or directly."""
    app = PIWDashboard()
    app.run()


if __name__ == "__main__":
    run_dashboard()