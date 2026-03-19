# file: output/renderer.py
"""
Renders IntelligenceReport to terminal (Rich) and optionally saves to disk.
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich import box

try:
    from config import OUTPUT_DIR, SAVE_REPORTS
except ImportError:
    OUTPUT_DIR  = "output"
    SAVE_REPORTS = True

console = Console(width=100)

SECTION_ICONS = {
    "reality":    ("🧩", "REALITY", "cyan"),
    "factcheck":  ("🔍", "FACT-CHECK", "bright_cyan"),
    "bias":       ("⚖️ ", "BIAS & MANIPULATION", "yellow"),
    "missing":    ("🕳️ ", "MISSING INFORMATION", "magenta"),
    "incentives": ("💰", "INCENTIVES", "red"),
    "trends":     ("📈", "TRENDS & PATTERNS", "green"),
    "scenarios":  ("🔮", "SCENARIOS", "blue"),
    "personal":   ("🎯", "PERSONAL IMPACT", "bright_white"),
}

def _section(icon: str, title: str, color: str, content: str, error: bool = False):
    style = f"bold {color}"
    header = Text(f" {icon}  {title} ", style=style)
    if error:
        content = f"[red]⚠ Agent error:[/red] {content}"
    console.print()
    console.print(Panel(content, title=header, border_style=color,
                        padding=(0, 2), expand=True))

def render_report(report) -> None:
    """Print full intelligence report to terminal."""
    console.print()
    console.rule(f"[bold white]  PIW INTELLIGENCE REPORT  ", style="white")
    console.print(
        f"[dim]Topic:[/dim] [bold]{report.topic}[/bold]  "
        f"[dim]|  Articles: {report.article_count}  "
        f"|  Sources: {', '.join(report.sources_used)}  "
        f"|  {datetime.now().strftime('%Y-%m-%d %H:%M')}[/dim]"
    )
    console.rule(style="dim white")

    for key, (icon, title, color) in SECTION_ICONS.items():
        if key == "factcheck":
            fc = getattr(report, "factcheck", None)
            if fc is not None:
                try:
                    from agents.fact_checker import format_fact_check_report
                    _section(icon, title, color, format_fact_check_report(fc))
                except Exception:
                    pass
            continue
        agent_out = getattr(report, key, None)
        if agent_out is None:
            continue
        _section(icon, title, color, agent_out.output, error=agent_out.error)

    console.print()
    console.rule("[dim]END OF REPORT[/dim]", style="dim white")
    console.print()


def _serialize_factcheck(fc) -> dict | None:
    """Serialize FactCheckReport to plain dict for JSON storage."""
    if fc is None:
        return None
    try:
        return {
            "total": fc.total, "confirmed": fc.confirmed,
            "disputed": fc.disputed, "misleading": fc.misleading,
            "single": fc.single, "unverifiable": fc.unverifiable,
            "confidence": fc.confidence,
            "claims": [
                {
                    "text": c.text, "source": c.source, "verdict": c.verdict,
                    "explanation": c.explanation,
                    "supporting": c.supporting, "contradicting": c.contradicting,
                }
                for c in fc.claims
            ],
        }
    except Exception:
        return None


def save_report(report, label: str = "") -> str:
    """Save report to output/ as both .json and .txt. Returns saved path."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = label.replace(" ", "_").lower()[:40] if label else "report"
    base = Path(OUTPUT_DIR) / f"{ts}_{slug}"

    # JSON
    data = {
        "topic":         report.topic,
        "generated_at":  datetime.now().isoformat(),
        "sources":       report.sources_used,
        "article_count": report.article_count,
        "agents": {
            key: {"output": getattr(report, key).output,
                  "error":  getattr(report, key).error}
            for key in ["reality","bias","missing","incentives","trends","scenarios","personal"]
            if getattr(report, key) is not None
        },
        "factcheck": _serialize_factcheck(getattr(report, "factcheck", None)),
        "raw_articles": report.raw_articles,
    }
    json_path = str(base) + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Plain text
    txt_path = str(base) + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"PIW INTELLIGENCE REPORT\n")
        f.write(f"Topic: {report.topic}\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Sources: {', '.join(report.sources_used)}\n")
        f.write("=" * 80 + "\n\n")
        for key, (icon, title, _) in SECTION_ICONS.items():
            ao = getattr(report, key, None)
            if ao:
                f.write(f"\n{icon}  {title}\n{'─'*60}\n{ao.output}\n")

    # Fact-check plain text
    fc = getattr(report, "factcheck", None)
    if fc is not None:
        try:
            from agents.fact_checker import format_fact_check_report
            import re as _re
            plain = _re.sub(r'\[/?[a-zA-Z_ ]+\]', '', format_fact_check_report(fc))
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write(f"\n🔍  FACT-CHECK\n{'─'*60}\n{plain}\n")
        except Exception:
            pass

    return json_path