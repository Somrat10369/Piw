# file: memory/entity_tracker.py
"""
Entity Tracker — extracts named entities from agent outputs and persists them.

Uses Ollama (same model) to identify:
  COUNTRY, ORG (organization/company), PERSON, ECON (economic indicator/currency),
  EVENT (named events), OTHER

Extracted entities feed into the memory layer for cross-run tracking.
"""

from __future__ import annotations
import json
import re
from typing import TYPE_CHECKING

from agents.ollama_client import call_ollama, extract_json
from memory.memory import upsert_entity, create_alert, get_top_entities, get_entity_timeline

if TYPE_CHECKING:
    from agents.agents import IntelligenceReport

# ── Extraction prompt ─────────────────────────────────────────────────────────

_ENTITY_SYSTEM = """You are a named entity extractor. Output ONLY valid JSON. No prose, no markdown."""

_ENTITY_PROMPT_TEMPLATE = """Extract all named entities from the following text.

Return a JSON array. Each item must have exactly these fields:
  "name": string (canonical name, e.g. "United States" not "US" or "America")
  "type": one of COUNTRY | ORG | PERSON | ECON | EVENT | OTHER
  "context": string (10-20 word excerpt showing how this entity is used)

Rules:
- Normalize names (full forms preferred)
- Skip generic words (government, official, company)
- Skip pronouns
- Max 25 entities per call
- If no entities, return []

TEXT:
{text}"""


def extract_entities_from_text(text: str, run_id: str = "") -> list[dict]:
    """
    Call Ollama to extract entities from a text block.
    Returns list of {name, type, context} dicts.
    Persists them to memory automatically.
    """
    if not text or len(text) < 50:
        return []

    prompt = _ENTITY_PROMPT_TEMPLATE.format(text=text[:3000])
    raw = call_ollama(prompt=prompt, system=_ENTITY_SYSTEM, temperature=0.0)

    entities = extract_json(raw)
    if not isinstance(entities, list):
        # Try to salvage partial output
        entities = _fallback_extract(text)

    valid = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        etype = str(e.get("type", "OTHER")).upper()
        context = str(e.get("context", ""))
        if name and len(name) > 1:
            eid = upsert_entity(name, etype, context, run_id)
            # Record trend snapshot for this entity
            try:
                from memory.trend_tracker import record_snapshot, migrate_db
                migrate_db()
                record_snapshot(eid, window_days=7)
            except Exception:
                pass
            valid.append({"name": name, "type": etype, "context": context})

    return valid


def extract_entities_from_report(report: "IntelligenceReport") -> list[dict]:
    """
    Extract entities from the reality + incentives sections of a report.
    These are the most fact-dense, least noisy sections.
    """
    run_id = report.topic[:40].replace(" ", "_")
    text_parts = []
    for field in ["reality", "incentives", "trends"]:
        ao = getattr(report, field, None)
        if ao and not ao.error:
            text_parts.append(ao.output[:1500])

    combined = "\n\n".join(text_parts)
    return extract_entities_from_text(combined, run_id=run_id)


# ── Alert logic ───────────────────────────────────────────────────────────────

def check_and_alert_new_entities(extracted: list[dict], known_before: set[str]) -> None:
    """
    Compare freshly extracted entities against pre-extraction known set.
    Fire alert for brand-new high-importance entities.
    """
    important_types = {"COUNTRY", "ORG", "PERSON"}
    for e in extracted:
        if e["name"].lower() not in known_before and e["type"] in important_types:
            create_alert(
                "NEW_ENTITY",
                f"New entity detected: [{e['type']}] {e['name']} — {e['context'][:80]}"
            )


def check_topic_spike(topic: str, source_count: int, threshold: int = 3) -> None:
    """Alert if a topic appears across N+ sources in one cycle."""
    if source_count >= threshold:
        create_alert(
            "TOPIC_SPIKE",
            f"Topic spike: '{topic}' appeared in {source_count} sources simultaneously."
        )


# ── Fallback extractor (regex, no LLM) ───────────────────────────────────────

_COUNTRY_LIST = {
    "bangladesh","india","pakistan","china","usa","united states","russia","ukraine",
    "myanmar","iran","israel","saudi arabia","uk","france","germany","japan",
    "south korea","north korea","turkey","egypt","brazil","indonesia","vietnam",
}

_ORG_PATTERNS = [
    r"\b(IMF|World Bank|UN|NATO|EU|ASEAN|SAARC|WTO|WHO|OPEC|G7|G20|BRI)\b",
    r"\b([A-Z][a-z]+ (?:Bank|Fund|Corp|Inc|Ltd|Group|Agency|Ministry|Department))\b",
]

def _fallback_extract(text: str) -> list[dict]:
    """Regex-based fallback when LLM JSON fails."""
    entities = []
    text_lower = text.lower()

    for country in _COUNTRY_LIST:
        if country in text_lower:
            entities.append({"name": country.title(), "type": "COUNTRY", "context": ""})

    for pat in _ORG_PATTERNS:
        for m in re.finditer(pat, text):
            entities.append({"name": m.group(), "type": "ORG", "context": ""})

    return entities[:25]


# ── Display helpers ───────────────────────────────────────────────────────────

def format_entity_table(entities: list[dict]) -> str:
    """Format entity list for terminal display."""
    if not entities:
        return "No entities tracked yet."
    lines = [f"{'NAME':<30} {'TYPE':<10} {'MENTIONS':>8}  LAST SEEN"]
    lines.append("─" * 70)
    for e in entities:
        lines.append(
            f"{e['name']:<30} {e['type']:<10} {e.get('mention_count',0):>8}  {e.get('last_seen','')[:10]}"
        )
    return "\n".join(lines)


def format_entity_timeline(entity_name: str) -> str:
    """Format an entity's appearance history."""
    events = get_entity_timeline(entity_name)
    if not events:
        return f"No history found for '{entity_name}'."
    lines = [f"Timeline for: {entity_name}", "─" * 50]
    for ev in events:
        lines.append(f"  {ev['run_at'][:16]}  →  {ev['topic']}")
    return "\n".join(lines)