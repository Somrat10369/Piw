# file: agents/agents.py
"""
Multi-Agent Analysis Pipeline — 7 sequential agents.
Each agent receives the article cluster text and prior agent outputs.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from agents.ollama_client import call_ollama

# ── Shared system prompt ──────────────────────────────────────────────────────

_BASE_SYSTEM = """You are a component of a local intelligence analysis system.
Your role is strictly analytical. Output only factual, structured analysis.
NEVER use emotional, manipulative, or persuasive language.
ALWAYS flag uncertainty explicitly. Use markers like [UNCERTAIN], [UNVERIFIED], [CONFLICTING].
Be concise and precise. No filler sentences. No moral judgments unless analyzing incentives."""

# ── Agent output model ────────────────────────────────────────────────────────

@dataclass
class AgentOutput:
    agent: str
    output: str
    error: bool = False

@dataclass
class IntelligenceReport:
    topic:         str
    sources_used:  list[str]
    article_count: int
    reality:       AgentOutput | None = None
    bias:          AgentOutput | None = None
    missing:       AgentOutput | None = None
    incentives:    AgentOutput | None = None
    trends:        AgentOutput | None = None
    scenarios:     AgentOutput | None = None
    personal:      AgentOutput | None = None
    raw_articles:  list[dict] = field(default_factory=list)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_articles(articles) -> str:
    """Format article list into a compact prompt-ready block."""
    parts = []
    for i, a in enumerate(articles, 1):
        text = a.full_text[:1500]
        parts.append(
            f"--- ARTICLE {i} | SOURCE: {a.source} | TITLE: {a.title} ---\n{text}"
        )
    return "\n\n".join(parts)

def _run_agent(name: str, prompt: str, verbose: bool = True) -> AgentOutput:
    if verbose:
        print(f"  ▸ Running Agent: {name}...", end=" ", flush=True)
    result = call_ollama(prompt=prompt, system=_BASE_SYSTEM, temperature=0.15)
    error = result.startswith("[ERROR]")
    if verbose:
        status = "✗ ERROR" if error else "✓ done"
        print(status)
    return AgentOutput(agent=name, output=result, error=error)

# ── 7 Agents ──────────────────────────────────────────────────────────────────

def agent_reality(articles_text: str, verbose=True) -> AgentOutput:
    prompt = f"""TASK: Reality Extraction
You are Agent 1 — Reality Extractor.

Given the following news articles about the same topic, extract ONLY what actually happened.

Instructions:
1. Build a factual timeline of events (date → event, if dates available).
2. List only confirmed facts present in at least one source.
3. Separate facts from claims — label claims with [CLAIM: source].
4. Remove all emotional framing, adjectives, and narrative spin.
5. If sources contradict each other on a fact, note: [CONFLICTING: source A says X, source B says Y].

Output format:
## Timeline
- [date/time if known] Event description

## Confirmed Facts
- Fact 1
- Fact 2

## Unresolved Contradictions
- [CONFLICTING: ...]

ARTICLES:
{articles_text}"""
    return _run_agent("Reality Extractor", prompt, verbose)


def agent_bias(articles_text: str, verbose=True) -> AgentOutput:
    prompt = f"""TASK: Bias and Manipulation Detection
You are Agent 2 — Bias Detector.

Analyze each source for bias, framing, and manipulation tactics.

For each source, identify:
1. Framing direction: Pro-[X] / Anti-[X] / Neutral
2. Emotional language used (list specific words/phrases)
3. Manipulation tactics present (choose from: fear, urgency, enemy framing, 
   oversimplification, appeal to authority, omission, strawman)
4. Overall bias score: Low / Medium / High

Output format:
## Per-Source Analysis
### [Source Name]
- Framing: ...
- Emotional language: ...
- Tactics: ...
- Bias level: ...

## Cross-Source Pattern
(What framing do most sources share? What does each uniquely push?)

ARTICLES:
{articles_text}"""
    return _run_agent("Bias Detector", prompt, verbose)


def agent_missing(articles_text: str, reality_output: str, verbose=True) -> AgentOutput:
    prompt = f"""TASK: Missing Information Detection
You are Agent 3 — Missing Information Finder.

Given the established facts and the articles, identify what is NOT being reported.

Look for:
1. Missing stakeholders — who is affected but not mentioned?
2. Missing context — what historical background is absent?
3. Missing consequences — what downstream effects are not discussed?
4. Missing data — what numbers, statistics, or evidence are absent?
5. Questions that should have been asked but weren't.

Output format:
## Missing Stakeholders
- ...

## Missing Context
- ...

## Missing Consequences
- ...

## Missing Data / Evidence
- ...

## Key Questions Not Asked
- ...

ESTABLISHED FACTS:
{reality_output[:800]}

ARTICLES:
{articles_text}"""
    return _run_agent("Missing Info Finder", prompt, verbose)


def agent_incentives(articles_text: str, reality_output: str, verbose=True) -> AgentOutput:
    prompt = f"""TASK: Incentive and Motivation Analysis
You are Agent 4 — Incentive Analyzer.

Analyze who benefits and who loses from this event and its coverage.

For each identified actor/group:
1. What do they gain from this event or its current framing?
2. What do they lose?
3. What economic/political/social motivation drives their behavior?
4. Are their public statements consistent with their incentives?

Output format:
## Actor Analysis
### [Actor/Group Name]
- Gains: ...
- Loses: ...
- Likely motivation: ...
- Statement vs incentive alignment: Consistent / Inconsistent / Unclear

## Hidden Dynamics
(What incentive structures explain why this story is being told this way?)

ESTABLISHED FACTS:
{reality_output[:800]}

ARTICLES:
{articles_text}"""
    return _run_agent("Incentive Analyzer", prompt, verbose)


def agent_trends(articles_text: str, reality_output: str, verbose=True) -> AgentOutput:
    prompt = f"""TASK: Trend and Pattern Analysis
You are Agent 5 — Trend Analyzer.

Analyze this event in the context of larger patterns and trends.

1. Is this event part of a larger trend? Describe the trend.
2. Is this an anomaly or expected continuation?
3. What historical parallels exist? (be specific)
4. What early signals or weak signals are present?
5. What is the rate of change — is this accelerating, stable, or reversing?

Output format:
## Trend Classification
- Pattern / Anomaly / New signal: ...
- Trend description: ...
- Rate of change: ...

## Historical Parallels
- [Event] in [Year]: similarity is [...]

## Early / Weak Signals
- Signal: ...  Strength: Low/Medium/High

## Confidence
- Overall confidence in trend analysis: Low / Medium / High
  Reason: ...

ESTABLISHED FACTS:
{reality_output[:800]}

ARTICLES:
{articles_text}"""
    return _run_agent("Trend Analyzer", prompt, verbose)


def agent_scenarios(reality_output: str, trends_output: str,
                    incentives_output: str, verbose=True) -> AgentOutput:
    prompt = f"""TASK: Scenario Building
You are Agent 6 — Scenario Builder.

Using the facts, trends, and incentives, build three forward-looking scenarios.

For each scenario:
1. Name it (e.g., "Escalation", "Status Quo", "Resolution")
2. Describe the conditions that lead to it
3. Estimate rough probability: Low / Medium / High
4. Key indicators to watch (what signals confirm this scenario is unfolding?)

Output format:
## Scenario 1: [Name] — Probability: [Low/Medium/High]
- Path: ...
- Conditions: ...
- Indicators to watch: ...

## Scenario 2: [Name] — Probability: [Low/Medium/High]
- Path: ...
- Conditions: ...
- Indicators to watch: ...

## Scenario 3: [Name] — Probability: [Low/Medium/High]
- Path: ...
- Conditions: ...
- Indicators to watch: ...

## Most Likely Outcome
...

FACTS:
{reality_output[:600]}

TRENDS:
{trends_output[:600]}

INCENTIVES:
{incentives_output[:600]}"""
    return _run_agent("Scenario Builder", prompt, verbose)


def agent_personal(reality_output: str, scenarios_output: str,
                   missing_output: str,
                   bd_context: str = "",
                   memory_context: str = "",
                   verbose: bool = True) -> AgentOutput:
    bd_block = f"\n\n{bd_context}" if bd_context else \
        "\nUser context: student/developer in Bangladesh, targeting SWE/ML career."

    mem_block = f"\n\nPAST CONTEXT FROM MEMORY:\n{memory_context}" if memory_context else ""

    prompt = f"""TASK: Personal Impact and Action Analysis
You are Agent 7 — Personal Strategist.

Translate the analysis into actionable personal guidance.
{bd_block}{mem_block}

1. Risks (short-term: 0-6 months, long-term: 1-3 years)
2. Opportunities created by this event
3. What to do now (concrete actions, not vague advice)
4. What to ignore (noise that doesn't require action)
5. What to monitor (specific metrics or events to track)

Output format:
## Risks
### Short-term (0-6 months)
- ...
### Long-term (1-3 years)
- ...

## Opportunities
- ...

## Actions to Take Now
- ...

## What to Ignore
- ...

## What to Monitor
- Metric/Event: [description] | How to track: [source/method]

FACTS:
{reality_output[:500]}

SCENARIOS:
{scenarios_output[:600]}

GAPS:
{missing_output[:400]}"""
    return _run_agent("Personal Strategist", prompt, verbose)


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

def run_pipeline(articles, topic: str = "unknown",
                 verbose: bool = True,
                 use_memory: bool = True,
                 use_bd_scoring: bool = True) -> "IntelligenceReport":
    """
    Run all 7 agents sequentially on a cluster of articles.
    Phase 2: integrates BD scoring and memory context.
    Returns a complete IntelligenceReport.
    """
    articles_text = _format_articles(articles)
    sources = list({a.source for a in articles})

    report = IntelligenceReport(
        topic=topic,
        sources_used=sources,
        article_count=len(articles),
        raw_articles=[a.to_dict() for a in articles],
    )
    report.bd_relevance_score = 0.0

    if verbose:
        print(f"\n[Pipeline] Topic: '{topic}' | {len(articles)} articles | {len(sources)} sources")

    # ── BD scoring ────────────────────────────────────────────────────────────
    bd_context = ""
    if use_bd_scoring:
        try:
            from agents.bd_scorer import score_cluster, get_bd_context_block, bd_relevance_label
            bd_score = score_cluster(articles)
            report.bd_relevance_score = bd_score
            bd_context = get_bd_context_block(articles, bd_score)
            if verbose:
                print(f"  BD Relevance: {bd_relevance_label(bd_score)}")
        except Exception as e:
            if verbose:
                print(f"  [BD Scorer] skipped: {e}")

    # ── Memory context for personal agent ─────────────────────────────────────
    memory_context = ""
    if use_memory:
        try:
            from memory.memory import init_db, get_topic_history
            init_db()
            past_runs = get_topic_history(topic=topic, limit=3)
            if past_runs:
                lines = [f"- [{r['run_at'][:10]}] {r['topic']} ({r['article_count']} articles)"
                         for r in past_runs]
                memory_context = "Previous analyses of similar topics:\n" + "\n".join(lines)
                if verbose:
                    print(f"  Memory: {len(past_runs)} past run(s) found for similar topics")
        except Exception as e:
            if verbose:
                print(f"  [Memory] skipped: {e}")

    # ── Run agents ────────────────────────────────────────────────────────────
    report.reality    = agent_reality(articles_text, verbose)
    report.bias       = agent_bias(articles_text, verbose)
    report.missing    = agent_missing(articles_text, report.reality.output, verbose)
    report.incentives = agent_incentives(articles_text, report.reality.output, verbose)
    report.trends     = agent_trends(articles_text, report.reality.output, verbose)
    report.scenarios  = agent_scenarios(
        report.reality.output, report.trends.output, report.incentives.output, verbose
    )
    report.personal   = agent_personal(
        report.reality.output, report.scenarios.output, report.missing.output,
        bd_context=bd_context, memory_context=memory_context, verbose=verbose,
    )

    # ── Persist to memory ─────────────────────────────────────────────────────
    if use_memory:
        try:
            from memory.memory import save_report_to_memory
            save_report_to_memory(report)
        except Exception:
            pass

    if verbose:
        print(f"[Pipeline] Complete ✓\n")

    return report