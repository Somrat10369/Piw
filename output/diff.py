# file: output/diff.py
"""
Report Diff Engine — compare two PIW intelligence reports.

Shows per-section what changed, what's new, what disappeared,
plus fact-check verdict flips and entity changes.

Usage:
  python piw.py diff <run_id_1> <run_id_2>
  python piw.py diff --topic "IMF"          # auto-picks 2 most recent
  python piw.py diff --topic "IMF" --list   # list available run IDs
"""

from __future__ import annotations
import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.columns import Columns
from rich import box

console = Console(width=110)

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class SectionDiff:
    name:        str
    similarity:  float        # 0.0 – 1.0
    added:       list[str]    # bullet points new in run B
    removed:     list[str]    # bullet points gone from run A
    unchanged:   list[str]    # identical or near-identical points
    summary:     str          # one-line change summary

@dataclass
class FactCheckDiff:
    flipped:     list[dict]   # {claim, from_verdict, to_verdict}
    new_claims:  list[dict]   # claims in B not in A
    gone_claims: list[dict]   # claims in A not in B
    confidence_changed: tuple[str,str] | None  # (old, new) or None

@dataclass
class ReportDiff:
    run_a:       dict          # metadata from run A
    run_b:       dict          # metadata from run B
    sections:    list[SectionDiff]
    factcheck:   FactCheckDiff | None
    new_sources: list[str]     # sources in B not in A
    gone_sources: list[str]    # sources in A not in B
    overall_change: str        # MAJOR | MODERATE | MINOR | IDENTICAL


# ── Text utilities ────────────────────────────────────────────────────────────

def _extract_bullets(text: str) -> list[str]:
    """
    Pull out every bullet/numbered point from a markdown-style agent output.
    Falls back to splitting on double-newline paragraphs if no bullets found.
    """
    lines = text.splitlines()
    bullets = []
    current = []

    for line in lines:
        stripped = line.strip()
        # Start of a new bullet (-, *, •, or numbered)
        if re.match(r'^[-*•]\s+', stripped) or re.match(r'^\d+[\.\)]\s+', stripped):
            if current:
                bullets.append(" ".join(current))
            current = [re.sub(r'^[-*•\d\.\)]+\s*', '', stripped)]
        elif stripped.startswith("##"):
            if current:
                bullets.append(" ".join(current))
                current = []
        elif stripped and current:
            current.append(stripped)

    if current:
        bullets.append(" ".join(current))

    # Fallback: paragraph chunks
    if not bullets and text.strip():
        chunks = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        bullets = [c[:200] for c in chunks[:20]]

    return [b.strip() for b in bullets if len(b.strip()) > 10]


def _similarity(a: str, b: str) -> float:
    """Character-level similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _find_matches(items_a: list[str],
                  items_b: list[str],
                  threshold: float = 0.55) -> tuple[list,list,list,list]:
    """
    Match bullets between two lists.
    Returns: (matched_a_idx, matched_b_idx, added_bullets, removed_bullets)
    """
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    pairs: list[tuple[int,int,float]] = []

    for i, a in enumerate(items_a):
        for j, b in enumerate(items_b):
            sim = _similarity(a, b)
            if sim >= threshold:
                pairs.append((i, j, sim))

    # Greedy best-match assignment
    pairs.sort(key=lambda x: -x[2])
    for i, j, sim in pairs:
        if i not in matched_a and j not in matched_b:
            matched_a.add(i)
            matched_b.add(j)

    unchanged = [items_a[i] for i in matched_a]
    removed   = [items_a[i] for i in range(len(items_a)) if i not in matched_a]
    added     = [items_b[j] for j in range(len(items_b)) if j not in matched_b]

    return matched_a, matched_b, added, removed, unchanged


def _section_similarity(text_a: str, text_b: str) -> float:
    """Overall similarity between two section texts."""
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0
    return _similarity(text_a[:2000], text_b[:2000])


# ── Section diff ──────────────────────────────────────────────────────────────

_SECTIONS = [
    ("reality",    "🧩 Reality"),
    ("bias",       "⚖️  Bias"),
    ("missing",    "🕳️  Missing Info"),
    ("incentives", "💰 Incentives"),
    ("trends",     "📈 Trends"),
    ("scenarios",  "🔮 Scenarios"),
    ("personal",   "🎯 Personal Impact"),
]

def _diff_section(name: str, label: str,
                  agents_a: dict, agents_b: dict) -> SectionDiff | None:
    """Diff a single agent section between two reports."""
    sec_a = agents_a.get(name, {})
    sec_b = agents_b.get(name, {})

    if not sec_a and not sec_b:
        return None

    text_a = sec_a.get("output", "") if sec_a else ""
    text_b = sec_b.get("output", "") if sec_b else ""

    sim = _section_similarity(text_a, text_b)

    bullets_a = _extract_bullets(text_a)
    bullets_b = _extract_bullets(text_b)

    _, _, added, removed, unchanged = _find_matches(bullets_a, bullets_b)

    # Build summary
    if sim >= 0.92:
        summary = "No significant change"
    elif not added and not removed:
        summary = f"Rephrased ({sim:.0%} similar)"
    else:
        parts = []
        if added:   parts.append(f"+{len(added)} new point{'s' if len(added)>1 else ''}")
        if removed: parts.append(f"-{len(removed)} removed")
        summary = ", ".join(parts) + f"  ({sim:.0%} similar)"

    return SectionDiff(
        name=label, similarity=sim,
        added=added, removed=removed,
        unchanged=unchanged, summary=summary
    )


# ── Fact-check diff ───────────────────────────────────────────────────────────

def _diff_factcheck(fc_a: dict | None, fc_b: dict | None) -> FactCheckDiff | None:
    """Diff two serialized FactCheckReport dicts."""
    if not fc_a and not fc_b:
        return None

    fc_a = fc_a or {}
    fc_b = fc_b or {}

    claims_a = {c["text"][:80]: c for c in fc_a.get("claims", [])}
    claims_b = {c["text"][:80]: c for c in fc_b.get("claims", [])}

    flipped = []
    for key, claim_b in claims_b.items():
        # Find closest matching claim in A
        best_match = None
        best_sim = 0.0
        for key_a, claim_a in claims_a.items():
            s = _similarity(key, key_a)
            if s > best_sim:
                best_sim = s
                best_match = claim_a

        if best_match and best_sim >= 0.6:
            if best_match["verdict"] != claim_b["verdict"]:
                flipped.append({
                    "claim":        claim_b["text"][:120],
                    "from_verdict": best_match["verdict"],
                    "to_verdict":   claim_b["verdict"],
                    "source":       claim_b.get("source", ""),
                })

    # New / gone claims (no close match)
    matched_b_texts = set()
    matched_a_texts = set()
    for key_b in claims_b:
        for key_a in claims_a:
            if _similarity(key_b, key_a) >= 0.6:
                matched_b_texts.add(key_b)
                matched_a_texts.add(key_a)
                break

    new_claims  = [claims_b[k] for k in claims_b if k not in matched_b_texts]
    gone_claims = [claims_a[k] for k in claims_a if k not in matched_a_texts]

    conf_a = fc_a.get("confidence")
    conf_b = fc_b.get("confidence")
    conf_changed = (conf_a, conf_b) if conf_a and conf_b and conf_a != conf_b else None

    return FactCheckDiff(
        flipped=flipped,
        new_claims=new_claims,
        gone_claims=gone_claims,
        confidence_changed=conf_changed,
    )


# ── Overall change magnitude ──────────────────────────────────────────────────

def _overall_change(sections: list[SectionDiff]) -> str:
    if not sections:
        return "IDENTICAL"
    avg_sim = sum(s.similarity for s in sections) / len(sections)
    total_changes = sum(len(s.added) + len(s.removed) for s in sections)
    if avg_sim >= 0.90 and total_changes <= 2:
        return "MINOR"
    if avg_sim >= 0.70 or total_changes <= 8:
        return "MODERATE"
    if avg_sim >= 0.50:
        return "MAJOR"
    return "MAJOR"


# ── Main diff function ────────────────────────────────────────────────────────

def diff_reports(report_a: dict, report_b: dict) -> ReportDiff:
    """
    Produce a ReportDiff between two serialized report dicts
    (as returned by get_report_by_run_id).
    """
    agents_a = report_a.get("agents", {})
    agents_b = report_b.get("agents", {})

    sections = []
    for key, label in _SECTIONS:
        sd = _diff_section(key, label, agents_a, agents_b)
        if sd:
            sections.append(sd)

    fc_diff = _diff_factcheck(
        report_a.get("factcheck"),
        report_b.get("factcheck"),
    )

    sources_a = set(report_a.get("sources_used", []))
    sources_b = set(report_b.get("sources_used", []))

    return ReportDiff(
        run_a         = report_a,
        run_b         = report_b,
        sections      = sections,
        factcheck     = fc_diff,
        new_sources   = sorted(sources_b - sources_a),
        gone_sources  = sorted(sources_a - sources_b),
        overall_change= _overall_change(sections),
    )


# ── Rendering ─────────────────────────────────────────────────────────────────

_CHANGE_COLORS = {
    "MAJOR":    "bold red",
    "MODERATE": "bold yellow",
    "MINOR":    "cyan",
    "IDENTICAL":"dim",
}

_VERDICT_COLORS = {
    "CONFIRMED":    "green",
    "SINGLE":       "cyan",
    "DISPUTED":     "bold red",
    "MISLEADING":   "yellow",
    "UNVERIFIABLE": "dim",
}

def render_diff(diff: ReportDiff) -> None:
    """Print a full diff report to the terminal."""
    ra = diff.run_a
    rb = diff.run_b

    console.print()
    console.rule("[bold white]  PIW REPORT DIFF  ", style="white")

    # Header
    change_color = _CHANGE_COLORS.get(diff.overall_change, "white")
    console.print(
        f"\n  [dim]Run A:[/dim] [bold]{ra.get('topic','?')}[/bold]  "
        f"[dim]{_run_date(ra)}  |  {ra.get('article_count',0)} articles[/dim]\n"
        f"  [dim]Run B:[/dim] [bold]{rb.get('topic','?')}[/bold]  "
        f"[dim]{_run_date(rb)}  |  {rb.get('article_count',0)} articles[/dim]\n"
        f"\n  Overall change: [{change_color}]{diff.overall_change}[/{change_color}]"
    )

    # Source changes
    if diff.new_sources or diff.gone_sources:
        console.print()
        if diff.new_sources:
            console.print(f"  [green]▲ New sources:[/green] {', '.join(diff.new_sources)}")
        if diff.gone_sources:
            console.print(f"  [red]▼ Gone sources:[/red] {', '.join(diff.gone_sources)}")

    console.rule(style="dim white")

    # Section diffs
    for sd in diff.sections:
        sim_color = "green" if sd.similarity >= 0.8 else "yellow" if sd.similarity >= 0.5 else "red"
        sim_bar   = _sim_bar(sd.similarity)

        lines = [f"[dim]{sim_bar}[/dim]  [bold]{sd.summary}[/bold]"]

        if sd.added:
            lines.append("")
            lines.append("[green]  ▲ NEW / CHANGED:[/green]")
            for b in sd.added[:6]:
                lines.append(f"[green]    + {b[:100]}[/green]")
            if len(sd.added) > 6:
                lines.append(f"[dim]    … and {len(sd.added)-6} more[/dim]")

        if sd.removed:
            lines.append("")
            lines.append("[red]  ▼ REMOVED / GONE:[/red]")
            for b in sd.removed[:6]:
                lines.append(f"[red]    - {b[:100]}[/red]")
            if len(sd.removed) > 6:
                lines.append(f"[dim]    … and {len(sd.removed)-6} more[/dim]")

        if not sd.added and not sd.removed:
            lines.append("[dim]  No structural changes.[/dim]")

        console.print()
        console.print(Panel(
            "\n".join(lines),
            title=Text(f" {sd.name} ", style=f"bold {sim_color}"),
            border_style=sim_color,
            padding=(0, 2),
        ))

    # Fact-check diff
    if diff.factcheck:
        fc = diff.factcheck
        fc_lines = []

        if fc.confidence_changed:
            old_c, new_c = fc.confidence_changed
            old_col = _VERDICT_COLORS.get(old_c, "white")
            new_col = _VERDICT_COLORS.get(new_c, "white")
            fc_lines.append(
                f"  Confidence: [{old_col}]{old_c}[/{old_col}] → [{new_col}]{new_c}[/{new_col}]"
            )

        if fc.flipped:
            fc_lines.append("\n  [bold]Verdict flips:[/bold]")
            for f in fc.flipped:
                fc_col = _VERDICT_COLORS.get(f["to_verdict"], "white")
                from_col = _VERDICT_COLORS.get(f["from_verdict"], "white")
                arrow = "🔴" if f["to_verdict"] == "DISPUTED" else "🟢" if f["to_verdict"] == "CONFIRMED" else "🟡"
                fc_lines.append(
                    f"  {arrow} [{from_col}]{f['from_verdict']}[/{from_col}] → "
                    f"[{fc_col}]{f['to_verdict']}[/{fc_col}]  [dim]{f['claim'][:90]}[/dim]"
                )

        if fc.new_claims:
            fc_lines.append(f"\n  [green]▲ {len(fc.new_claims)} new claim(s) in Run B[/green]")
            for c in fc.new_claims[:3]:
                fc_lines.append(f"  [green]  + [{c.get('verdict','?')}] {c.get('text','')[:90]}[/green]")

        if fc.gone_claims:
            fc_lines.append(f"\n  [red]▼ {len(fc.gone_claims)} claim(s) dropped from Run A[/red]")
            for c in fc.gone_claims[:3]:
                fc_lines.append(f"  [red]  - [{c.get('verdict','?')}] {c.get('text','')[:90]}[/red]")

        if not fc_lines:
            fc_lines = ["  [dim]No fact-check changes.[/dim]"]

        console.print()
        console.print(Panel(
            "\n".join(fc_lines),
            title=Text(" 🔍 FACT-CHECK DIFF ", style="bold bright_cyan"),
            border_style="bright_cyan",
            padding=(0, 2),
        ))

    console.print()
    console.rule("[dim]END OF DIFF[/dim]", style="dim white")
    console.print()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_date(report: dict) -> str:
    ts = report.get("generated_at") or report.get("run_at", "")
    return ts[:16] if ts else "unknown date"

def _sim_bar(sim: float, width: int = 20) -> str:
    """Render a small similarity bar e.g. '████░░░░░░░░░░░░░░░░ 42%'"""
    filled = round(sim * width)
    return "█" * filled + "░" * (width - filled) + f" {sim:.0%}"


# ── Convenience: load + diff by run IDs or topic ──────────────────────────────

def diff_by_run_ids(run_id_a: str, run_id_b: str) -> ReportDiff | None:
    """Load two reports from memory and diff them."""
    from memory.memory import get_report_by_run_id
    ra = get_report_by_run_id(run_id_a)
    rb = get_report_by_run_id(run_id_b)
    if not ra:
        console.print(f"[red]Run ID not found: {run_id_a}[/red]")
        return None
    if not rb:
        console.print(f"[red]Run ID not found: {run_id_b}[/red]")
        return None
    return diff_reports(ra, rb)


def diff_latest_by_topic(topic: str, n: int = 2) -> ReportDiff | None:
    """Auto-pick the N most recent runs matching a topic and diff them."""
    from memory.memory import get_topic_history, get_report_by_run_id
    runs = get_topic_history(topic=topic, limit=n)
    if len(runs) < 2:
        console.print(
            f"[yellow]Need at least 2 runs for topic '{topic}'. "
            f"Found {len(runs)}. Run more analyses first.[/yellow]"
        )
        return None
    # Oldest first so diff reads A→B chronologically
    run_a = get_report_by_run_id(runs[-1]["run_id"])
    run_b = get_report_by_run_id(runs[0]["run_id"])
    # Attach metadata
    if run_a: run_a.setdefault("run_at", runs[-1]["run_at"])
    if run_b: run_b.setdefault("run_at", runs[0]["run_at"])
    return diff_reports(run_a, run_b)


def list_runs_for_topic(topic: str, limit: int = 10) -> None:
    """Print available run IDs for a topic so user can pick two."""
    from memory.memory import get_topic_history
    runs = get_topic_history(topic=topic, limit=limit)
    if not runs:
        console.print(f"[yellow]No runs found for topic '{topic}'.[/yellow]")
        return
    console.print(f"\n[bold]Runs matching '{topic}':[/bold]")
    for r in runs:
        console.print(f"  [cyan]{r['run_id']}[/cyan]  {r['run_at'][:16]}  "
                      f"[dim]{r['topic'][:50]}  ({r['article_count']} articles)[/dim]")
    console.print(f"\n[dim]Use: python piw.py diff <run_id_1> <run_id_2>[/dim]")