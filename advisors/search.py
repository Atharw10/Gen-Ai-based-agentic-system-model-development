"""Search strategist (optional LLM steering on top of deterministic Optuna HPO).

Given the trial history so far, suggest a revised algorithm shortlist to spend the remaining
budget on (e.g. drop a clearly-losing family). It returns a validated SearchSpace; if the LLM
is absent or returns nonsense, it returns the original space unchanged. The LLM never emits raw
hyperparameters — Optuna owns the within-family tuning. This keeps steering auditable.
"""
from __future__ import annotations

from contracts import SearchSpace, TrialResult


def revise_search_space(
    history: list[TrialResult],
    current: SearchSpace,
    primary_oot_key: str,
    cached_llm=None,
) -> tuple[SearchSpace, str]:
    """Return (possibly-revised SearchSpace, note). Deterministic no-op without an LLM."""
    if cached_llm is None or not history:
        return current, "no_llm:unchanged"

    # Summarise best OOT score per algorithm for the LLM.
    best_by_algo: dict[str, float] = {}
    for t in history:
        s = t.oot_metrics.get(primary_oot_key, 0.0)
        best_by_algo[t.algorithm] = max(best_by_algo.get(t.algorithm, float("-inf")), s)
    summary = ", ".join(f"{a}={s:.4f}" for a, s in best_by_algo.items())

    try:
        from advisors.skills import Skill

        strategist_skill = Skill(
            "strategist",
            "You are a banking-ML search strategist. Given best OOT scores per algorithm, decide "
            "which 1-3 algorithms to keep spending the tuning budget on. "
            'Return ONLY JSON: {"algorithms": ["..."], "reason": "..."}',
        )
        task = (f"Best OOT scores so far by algorithm: {summary}. Available: {current.algorithms}. "
                "Which 1-3 should we keep spending budget on?")
        j = strategist_skill.run(task, cached_llm) or {}
        keep = [a for a in j.get("algorithms", []) if a in current.algorithms]
        if not keep:
            return current, "llm_returned_empty:unchanged"
        revised = current.model_copy(update={"algorithms": keep})
        return revised, f"llm:{j.get('reason', '')[:160]}"
    except Exception as e:
        return current, f"llm_failed:{e}"


def should_continue(
    improvement: float,
    rounds_done: int,
    max_rounds: int,
    leaderboard_summary: str,
    cached_llm=None,
    min_improve: float = 0.005,
) -> tuple[bool, str]:
    """Decide whether to run another search round. (Feature (b): the agent stops itself.)

    Guardrails first, then judgement:
      1. HARD CAP: never exceed max_rounds (reproducibility + cost ceiling).
      2. OFFLINE (no LLM): a single round — extra rounds would only re-run the same deterministic
         Optuna study and reproduce identical trials, so there is nothing to gain.
      3. WITH LLM: the model decides continue/stop from the leaderboard; on any error it falls
         back to the deterministic rule (continue only if the last round improved >= min_improve).
    """
    if rounds_done >= max_rounds:
        return False, f"stop: hard cap reached ({rounds_done}/{max_rounds} rounds)"
    if cached_llm is None:
        return False, "stop: offline (no strategy engine; extra rounds would repeat identically)"

    det = improvement >= min_improve  # deterministic fallback decision
    imp_txt = "first round" if improvement == float("inf") else f"{improvement:+.4f}"
    try:
        from advisors.skills import Skill

        stop_skill = Skill(
            "stop_decision",
            "You are a banking-ML search controller optimising the OOT primary metric. Decide "
            "whether another tuning round is worthwhile given the leaderboard and the last "
            'improvement. Return ONLY JSON: {"continue": true/false, "reason": "..."}',
        )
        task = (f"Best-per-algorithm so far: {leaderboard_summary}. Improvement vs previous round: "
                f"{imp_txt}. Rounds used: {rounds_done}/{max_rounds}. Run ANOTHER round or stop?")
        j = stop_skill.run(task, cached_llm) or {}
        cont = bool(j["continue"])
        return cont, f"llm: {'continue' if cont else 'stop'} — {j.get('reason', '')[:160]}"
    except Exception as e:
        return det, f"{'continue' if det else 'stop'}: llm_failed ({e}); deterministic improvement={imp_txt}"
