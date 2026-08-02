"""Narrative advisor: turn a finished run into a human model write-up.

Deterministic template by default (persistence already writes model_card.md); an LLM can be
supplied to produce richer prose. Pure presentation — never feeds back into the pipeline.
"""
from __future__ import annotations

from contracts import RunResult


def _template(result: RunResult) -> str:
    best = next((r for r in result.leaderboard if r.get("algorithm") == result.best_algorithm), {})
    metrics = ", ".join(f"{k}={v:.4f}" for k, v in best.get("oot_metrics", {}).items())
    return (
        f"Best model **{result.best_algorithm}** (trial {result.best_trial_id}) selected by OOT "
        f"{result.primary_metric} across {result.n_trials_run} trials. OOT metrics: {metrics}. "
        f"F1-optimal threshold {result.optimal_threshold:.3f}."
    )


def summarize(result: RunResult, cached_llm=None) -> str:
    base = _template(result)
    if cached_llm is None:
        return base
    try:
        from advisors.skills import Skill

        narrative_skill = Skill(
            "narrative",
            "Write a concise (<=120 word) model summary for a banking model-risk reviewer. "
            "Be factual; do not invent numbers; use only the facts provided.",
        )
        out = narrative_skill.run(f"Facts: {base}", cached_llm, want_json=False)
        return (out or base).strip()
    except Exception:
        return base
