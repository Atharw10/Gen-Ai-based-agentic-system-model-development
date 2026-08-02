"""Linear orchestrator (driver #1).

Composes the shared nodes in `app/nodes.py` in sequence, inserting HITLGate checkpoints between
them. Runs start-to-finish in one call and works fully offline (cached_llm=None). For the
pause/resume LangGraph driver, see `app/graph.py` — both call the SAME node functions.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from app import nodes
from runtime.cost import CostTracker
from runtime.hitl import HITLGate, apply_overrides_to_config
from runtime.run_logger import RunLogger


def _validate_columns(df: pd.DataFrame, target_col: str, date_col: str, group_col: str | None):
    cols = list(df.columns)
    missing = []
    for name, val in [("target_col", target_col), ("date_col", date_col)]:
        if val not in cols:
            close = [c for c in cols if c.lower() == val.lower()]
            hint = f" (did you mean: {close[0]!r}?)" if close else f" (available columns: {cols[:10]}...)"
            missing.append(f"  {name}={val!r} not found{hint}")
    if group_col and group_col not in cols:
        close = [c for c in cols if c.lower() == group_col.lower()]
        hint = f" (did you mean: {close[0]!r}?)" if close else ""
        missing.append(f"  group_col={group_col!r} not found{hint}")
    if missing:
        raise ValueError("Column(s) not found in DataFrame:\n" + "\n".join(missing))


def run(
    df: pd.DataFrame,
    business_objective: str,
    target_col: str,
    date_col: str,
    group_col: str | None = None,
    cached_llm=None,
    hitl_mode="auto",
    artifacts_root: str = "artifacts/runs",
    source_path: str | None = None,
    search_rounds: int = 2,
    alias_columns: bool = True,
    verbose: bool = True,
    model_type: str | None = None,
):
    """Full agentic run. Returns (RunResult, best_model, prepared, summary_text)."""
    _validate_columns(df, target_col, date_col, group_col)
    run_dir = str(Path(artifacts_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    gate = HITLGate(run_dir=run_dir, mode=hitl_mode)
    logger = RunLogger(run_dir=run_dir, verbose=verbose)
    # token/cost accounting — attach to the LLM so every call is metered (offline → $0 summary)
    cost = CostTracker(run_dir=run_dir, model_id=getattr(cached_llm, "model_id", "none"), verbose=verbose)
    if cached_llm is not None and hasattr(cached_llm, "tracker"):
        cached_llm.tracker = cost

    state: dict = {
        "df": df, "business_objective": business_objective, "target_col": target_col,
        "date_col": date_col, "group_col": group_col, "cached_llm": cached_llm,
        "run_dir": run_dir, "source_path": source_path, "search_rounds": search_rounds,
        "alias_columns": alias_columns, "logger": logger, "all_trials": [], "round_idx": 0,
        "model_type": model_type,
    }

    # 0. ALIAS columns at input (privacy shield) — LLM sees only aliases hereafter
    with logger.step("alias columns (privacy shield)"):
        state.update(nodes.node_alias(state))

    # 1. PROFILE
    with logger.step("profile data"):
        state.update(nodes.node_profile(state))
    gate.gate("profiling", "Data profiled.", {
        "n_rows": state["profile"].n_rows, "pos_ratio": state["profile"].target.pos_ratio,
        "n_snapshots": state["profile"].temporal.n_snapshots,
        "pos_ratio_per_snapshot": state["profile"].temporal.pos_ratio_per_snapshot,
    })

    # 2. RESEARCH → feature-family FE → 3. DESIGN (with regenerate loop)
    with logger.step("research (models / windows / HPO / features)"):
        state.update(nodes.node_research(state))
    with logger.step("domain feature engineering (research families)"):
        state.update(nodes.node_domain_fe(state))
    for _ in range(3):  # bounded regenerate loop
        with logger.step("model design"):
            state.update(nodes.node_design(state))
        action = gate.gate("model_design", state["config"].rationale or "Proposed run config.",
                           nodes.design_decisions(state["config"]))
        if action.get("action") == "override":
            state["config"] = apply_overrides_to_config(state["config"], action.get("overrides", {}))
            break
        if action.get("action") == "regenerate":
            state["business_objective"] += f"\n[user]: {action.get('instructions', '')}"
            continue
        break  # accept

    # 4. PREPARE (split → clean → select → encode; leakage-safe)
    with logger.step("prepare data (split/clean/select/encode)"):
        state.update(nodes.node_prepare(state))
    gate.gate("feature_selection", "Features selected.", {
        "n_in": state["prepared"].selection_report.n_in,
        "n_selected": state["prepared"].selection_report.n_selected,
        "suspicious_high_iv": state["prepared"].selection_report.suspicious_high_iv,
    })

    # 5. SEARCH LOOP — run a round, let the agent decide whether to continue (feature b)
    while True:
        with logger.step(f"search round {state.get('round_idx', 0) + 1} (HPO)"):
            state.update(nodes.node_search_round(state))
        with logger.step("assess: continue or stop?"):
            state.update(nodes.node_assess(state))
        logger.reason("search/stop_decision", "continue" if state["should_continue"] else "stop",
                      state.get("assess_note", ""), "agent (feature b)")
        if not state["should_continue"]:
            break
        with logger.step("strategist: revise algorithm budget"):
            state.update(nodes.node_strategist(state))
        gate.gate(f"search_round_{state['round_idx']}", state.get("assess_note", ""),
                  {"algorithms": state["config"].search.algorithms,
                   "best_oot": round(state["best_oot_so_far"], 5)})

    # 6. FINALIZE (refit best, evaluate on OOT, persist)
    with logger.step("finalize (refit best + OOT eval + persist)"):
        state.update(nodes.node_finalize(state))
    gate.gate("evaluation", "Best model selected on OOT.", {
        "best_algorithm": state["result"].best_algorithm,
        "oot": state["result"].leaderboard[0]["oot_metrics"],
    })

    # 7. NARRATIVE
    with logger.step("write model summary"):
        state.update(nodes.node_narrate(state))
    t = cost.totals()
    logger.info(f"LLM cost: {t['api_calls']} API calls, {t['total_tokens']} tokens, "
                f"~${t['est_cost_usd']} (see cost_summary.json)")
    logger.info(f"Done → {run_dir}")
    return state["result"], state["best_model"], state["prepared"], state["summary"]
