"""Tool layer for LLM skills.

A "tool" is a concrete capability a skill may call. There is exactly ONE external tool —
`web_search` (Tavily) — plus a few thin wrappers over deterministic helpers so a skill can pull
data-derived facts by name. Skills are handed tools by name (see advisors/skills.py); this keeps
the skill (its prompt/goal) separate from the tools it uses, so tools can be added/swapped without
touching the skill.

No LLM import here (this is the tool plane). CML-safe: Tavily is imported lazily and every tool
degrades to a plain string on failure so a tool call can never crash a run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str          # one line: what it does + its args (shown to the LLM)
    func: Callable

    def __call__(self, **kwargs):
        return self.func(**kwargs)


# ── the one external tool ────────────────────────────────────────────────────
def _web_search(query: str = "", **_) -> str:
    """Search the web (Tavily) for banking-ML best practices. Returns titles + URLs + snippets."""
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        resp = client.search(query=query, max_results=5, search_depth="advanced")
        rows = resp.get("results", [])
        if not rows:
            return "(web_search: no results)"
        return "\n".join(f"- {r.get('title', '')} — {r.get('url', '')}: "
                         f"{str(r.get('content', ''))[:200]}" for r in rows)
    except Exception as e:  # missing key / SSL / proxy — never crash the run
        return f"(web_search unavailable: {type(e).__name__})"


# ── deterministic helpers exposed as tools ───────────────────────────────────
def _imbalance_tier(pos_ratio: float = 0.0, **_) -> str:
    """Return the imbalance tier (balanced/mild/moderate/severe/extreme) for a positive rate."""
    from mlkit.metrics import assign_tier

    return assign_tier(float(pos_ratio))


def _positive_rate_drift(pos_ratio_per_snapshot: dict | None = None, **_) -> str:
    """Describe positive-rate drift across monthly snapshots (stable / drift + note)."""
    from mlkit.stats import pos_rate_drift

    _flagged, _med, note = pos_rate_drift(pos_ratio_per_snapshot or {})
    return note


TOOLS: dict[str, Tool] = {
    "web_search": Tool(
        "web_search",
        "web_search(query: str) — search the web for banking-ML best practices, algorithms and "
        "feature-engineering guidance; returns titles, URLs and snippets.",
        _web_search,
    ),
    "imbalance_tier": Tool(
        "imbalance_tier",
        "imbalance_tier(pos_ratio: float) — return the imbalance tier for a positive rate.",
        _imbalance_tier,
    ),
    "positive_rate_drift": Tool(
        "positive_rate_drift",
        "positive_rate_drift(pos_ratio_per_snapshot: dict) — describe positive-rate drift across "
        "snapshots so you can pick a stable observation window / OOT period.",
        _positive_rate_drift,
    ),
}


def get_tool(name: str) -> Tool | None:
    return TOOLS.get(name)


def list_tools() -> list[Tool]:
    return list(TOOLS.values())


def tools_spec(names) -> str:
    """Render the description block for the given tool names (for the skill prompt)."""
    lines = [f"- {TOOLS[n].description}" for n in names if n in TOOLS]
    return "\n".join(lines)
