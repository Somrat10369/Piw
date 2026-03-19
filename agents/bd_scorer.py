# file: agents/bd_scorer.py
"""
Bangladesh Relevance Scorer

Scores each article cluster for Bangladesh relevance and injects
rich BD-specific context into the Personal Strategist agent prompt
when the score exceeds BD_SCORE_THRESHOLD.

Score = source_bonus + keyword_hits (capped)
Range: 0.0 (no relevance) → 1.0 (max relevance)
"""

from __future__ import annotations
import re
from typing import TYPE_CHECKING

try:
    from config import (
        BD_LOCAL_SOURCES, BD_RELEVANCE_KEYWORDS,
        BD_SOURCE_WEIGHT, BD_KEYWORD_WEIGHT, BD_SCORE_THRESHOLD,
    )
except ImportError:
    BD_LOCAL_SOURCES       = {"Daily Star", "Prothom Alo EN", "bdnews24"}
    BD_RELEVANCE_KEYWORDS  = ["bangladesh", "dhaka", "taka", "imf", "garment"]
    BD_SOURCE_WEIGHT       = 0.4
    BD_KEYWORD_WEIGHT      = 0.1
    BD_SCORE_THRESHOLD     = 0.3

if TYPE_CHECKING:
    from collector.collector import Article


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_cluster(articles: list["Article"]) -> float:
    """
    Return Bangladesh relevance score [0.0, 1.0] for a cluster.
    """
    score = 0.0
    combined_text = " ".join(
        (a.title + " " + a.summary + " " + a.body[:500]).lower()
        for a in articles
    )

    # Source bonus (any BD source in cluster)
    if any(a.source in BD_LOCAL_SOURCES for a in articles):
        score += BD_SOURCE_WEIGHT

    # Keyword hits (capped at 0.6)
    keyword_hits = sum(
        1 for kw in BD_RELEVANCE_KEYWORDS
        if re.search(r"\b" + re.escape(kw) + r"\b", combined_text)
    )
    score += min(keyword_hits * BD_KEYWORD_WEIGHT, 0.6)

    return min(score, 1.0)


def get_bd_context_block(articles: list["Article"], score: float) -> str:
    """
    Build a Bangladesh-specific context string to inject into
    the Personal Strategist agent prompt.
    Returns empty string if score < threshold.
    """
    if score < BD_SCORE_THRESHOLD:
        return ""

    sources_present = [a.source for a in articles if a.source in BD_LOCAL_SOURCES]

    # Identify which BD-relevant keywords hit
    combined = " ".join(
        (a.title + " " + a.summary).lower() for a in articles
    )
    hit_keywords = [
        kw for kw in BD_RELEVANCE_KEYWORDS
        if re.search(r"\b" + re.escape(kw) + r"\b", combined)
    ]

    context_lines = [
        f"BANGLADESH RELEVANCE SCORE: {score:.2f}",
        "",
        "CONTEXT FOR PERSONAL ANALYSIS:",
        "The user is a student/developer in Bangladesh with the following profile:",
        "  - Age: ~16-18, SSC/HSC level, targeting SWE or ML/AI career",
        "  - Location: Dhaka Division, Bangladesh",
        "  - Economic context: middle-income Bangladeshi household",
        "  - Skills: Python, PyQt6, Fabric modding, desktop tooling",
        "  - Goal: FAANG-track career (remote or relocation)",
        "  - Financial constraints: limited budget, prefers free/open-source tools",
        "",
    ]

    if sources_present:
        context_lines.append(f"LOCAL SOURCES IN CLUSTER: {', '.join(set(sources_present))}")
    if hit_keywords:
        context_lines.append(f"RELEVANT KEYWORDS DETECTED: {', '.join(hit_keywords[:10])}")

    context_lines += [
        "",
        "When writing the Personal Impact section, specifically address:",
        "  1. How this affects Bangladesh's economy or job market",
        "  2. Implications for someone entering tech in Bangladesh",
        "  3. Any remittance, currency (BDT/USD), or cost-of-living impacts",
        "  4. Regional risks (India-Bangladesh relations, Myanmar, Bay of Bengal)",
        "  5. Concrete actions a student developer in Bangladesh should take",
        "  6. Free/low-cost resources or tools that remain accessible",
    ]

    return "\n".join(context_lines)


def annotate_report_with_bd_score(report, score: float) -> None:
    """Attach BD score to report object for display."""
    report.bd_relevance_score = score


# ── Summary label ─────────────────────────────────────────────────────────────

def bd_relevance_label(score: float) -> str:
    if score >= 0.7:
        return f"🇧🇩 HIGH ({score:.2f})"
    elif score >= 0.3:
        return f"🇧🇩 MEDIUM ({score:.2f})"
    else:
        return f"⬜ LOW ({score:.2f})"