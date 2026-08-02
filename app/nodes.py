"""Shared orchestration nodes — driver-agnostic, HITL-free.

Both drivers compose these identical functions:
  - app/orchestrator.py  (linear)  — inserts HITLGate.gate() calls between nodes
  - app/graph.py         (LangGraph) — wraps them as graph nodes with interrupt() gates

Each node takes the run state dict and returns a dict of UPDATES to merge into it. No node
calls an LLM directly except via the advisor functions, and none performs HITL — that keeps
the nodes pure enough to reuse and to re-run safely under LangGraph.
"""
from __future__ import annotations

import json
from pathlib import Path

from advisors import design as design_advisor
from advisors import narrative as narrative_advisor
from advisors import research as research_advisor
from advisors import search as search_advisor
from dataio import aliasing
from pipeline import feature_engineer, profiling, training
from pipeline.preparation import prepare_data
from pipeline.runner import finalize_run


def _log(state):
    return state.get("logger")


def node_alias(state: dict) -> dict:
    """Privacy shield: rename all columns (incl. target/date/group) to neutral aliases at input,
    and write the real<->alias mapping to its own artifact file. Everything downstream — and the
    LLM — sees only aliases. Set alias_columns=False to disable."""
    if not state.get("alias_columns", True):
        return {}
    amap = aliasing.build_alias_map(
        list(state["df"].columns), state["target_col"], state.get("date_col"), state.get("group_col")
    )
    df_aliased = aliasing.apply_aliases(state["df"], amap)
    if state.get("run_dir"):
        aliasing.save_alias_map(amap, state["run_dir"])
    return {
        "df": df_aliased,
        "target_col": amap.target,
        "date_col": amap.date,
        "group_col": amap.group,
        "alias_map": amap.to_dict(),
    }


def node_profile(state: dict) -> dict:
    profile = profiling.profile_dataset(
        state["df"], state["target_col"], state["date_col"],
        state.get("group_col"), state["business_objective"],
    )
    log = _log(state)
    pmr = profile.temporal.pos_ratio_per_snapshot
    if log and pmr:
        from mlkit import stats as _stats
        # tabular (one row per snapshot) so it stays readable with many months
        rps = profile.temporal.rows_per_snapshot or {}
        log.info("Positive rate per snapshot:")
        log.info(f"    {'snapshot':<10} {'rows':>12}   pos_rate")
        for k, v in pmr.items():
            log.info(f"    {k:<10} {rps.get(k, 0):>12,}   {v:.2%}")
        flagged, _med, note = _stats.pos_rate_drift(pmr)
        log.reason("profiling/drift", "drift detected" if flagged else "stable", note,
                   "deterministic (per-snapshot positive rate)")
    # persist the full profile (incl. per-month positive rate) as its own artifact
    if state.get("run_dir"):
        Path(state["run_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(state["run_dir"]) / "data_profile.json").write_text(
            json.dumps(profile.model_dump(), indent=2, default=str), encoding="utf-8")
    return {"profile": profile}


def node_research(state: dict) -> dict:
    plan = research_advisor.research(
        state["business_objective"], state["profile"], cached_llm=state.get("cached_llm"),
        model_type=state.get("model_type"),
    )
    log = _log(state)
    if log:
        m, w, h = plan.get("models", {}), plan.get("windows", {}), plan.get("hpo", {})
        log.reason("research/models", f"Recommended models: {m.get('recommended')}",
                   m.get("reason", ""), m.get("source", "?"))
        log.reason("research/windows",
                   f"observation={w.get('observation_window_months')}m, OOT={w.get('oot_months')} snapshot(s)",
                   w.get("reason", ""), w.get("source", "?"))
        log.reason("research/hpo",
                   f"HPO ranges proposed for: {list((h.get('spaces') or {}).keys()) or 'none (use defaults)'}",
                   h.get("reason", ""), h.get("source", "?"))
    return {"findings": plan}


def node_domain_fe(state: dict) -> dict:
    """Use research-suggested feature families: match them in the data ('feature store') and build
    the derived features that resolve; otherwise normal FE. Runs on real names via the alias map."""
    log = _log(state)
    fam = (state.get("findings") or {}).get("feature_families", {})
    families = [f.get("name") for f in fam.get("families", [])]
    reverse = (state.get("alias_map") or {}).get("reverse", {})  # alias -> real, for matching
    df2, created, cmap = feature_engineer.engineer(state["df"], name_lookup=reverse)
    matched = [k for k, v in cmap.items() if v]
    if log:
        log.reason("feature_engineering/families", f"Suggested families: {families}",
                   fam.get("reason", "standard families"), fam.get("source", "?"))
        log.info(f"Feature-store match: resolved {len(matched)} concepts {matched}; "
                 f"built {len(created)} derived features {created}"
                 + ("" if matched else " → none matched, using normal FE/selection"))
    return {"df": df2, "domain_fe_done": True, "domain_features": created, "matched_concepts": matched}


def node_design(state: dict) -> dict:
    config, errors = design_advisor.design_config(
        state["profile"], state["business_objective"],
        cached_llm=state.get("cached_llm"), research=state.get("findings"),
        model_type=state.get("model_type"),
    )
    log = _log(state)
    if log:
        # Echo the RESEARCH plan's own reason + source for each applied decision — never invent a
        # justification here. (The only design-owned choice is the metric policy, labelled as such.)
        f = state.get("findings") or {}
        m, w, h = f.get("models", {}), f.get("windows", {}), f.get("hpo", {})
        log.reason("model_design/algorithms", f"Shortlist applied: {config.search.algorithms}",
                   m.get("reason", "(no reason provided)"), m.get("source", "?"))
        log.reason("model_design/windows",
                   f"train={config.windows.snapshots_for_training}, OOT={config.windows.oot_snapshots}",
                   w.get("reason", "(no reason provided)"), w.get("source", "?"))
        log.reason("model_design/metric", f"Primary metric: {config.primary_metric}",
                   f"deterministic metric policy for objective + tier '{config.data_strategy.imbalance_tier}'",
                   "deterministic policy")
        if config.search.hpo_overrides:
            log.reason("model_design/hpo", f"Using research HPO ranges for {list(config.search.hpo_overrides)}",
                       h.get("reason", "(no reason provided)"), h.get("source", "?"))
    return {"config": config, "design_errors": errors}


def node_prepare(state: dict) -> dict:
    # domain FE already done (with research families) before aliasing -> don't repeat it here
    prepared = prepare_data(state["df"], state["config"], state["profile"],
                            enable_domain_features=not state.get("domain_fe_done", False))
    log = _log(state)
    if log:
        sr = prepared.selection_report
        log.info(f"Feature selection: {sr.n_in} → {sr.n_selected} "
                 f"(suspicious/leakage flags: {len(sr.suspicious_high_iv)})")
    return {"prepared": prepared}


def node_search_round(state: dict) -> dict:
    """Run one round of search for the CURRENT config; accumulate trials and bump the counter."""
    trials, _ = training.run_search(state["prepared"], state["config"])
    all_trials = list(state.get("all_trials", [])) + trials
    log = _log(state)
    if log:
        ov = state["config"].search.hpo_overrides
        log.info(f"Round {state.get('round_idx', 0) + 1}: ran {len(trials)} trials over "
                 f"{state['config'].search.algorithms}"
                 + (f"; HPO ranges from research for {list(ov)}" if ov else "; default HPO ranges"))
    return {"all_trials": all_trials, "round_idx": state.get("round_idx", 0) + 1}


def node_strategist(state: dict) -> dict:
    """LLM strategist revises which algorithms get the next round's budget."""
    oot_key = training._oot_key(state["config"].primary_metric)
    revised, note = search_advisor.revise_search_space(
        state["all_trials"], state["config"].search, oot_key, cached_llm=state.get("cached_llm")
    )
    new_config = state["config"].model_copy(update={"search": revised})
    return {"config": new_config, "strategist_note": note}


def _round_summary(trials, oot_key: str) -> str:
    best: dict[str, float] = {}
    for t in trials:
        s = t.oot_metrics.get(oot_key, float("-inf"))
        best[t.algorithm] = max(best.get(t.algorithm, float("-inf")), s)
    return ", ".join(f"{a}={s:.4f}" for a, s in best.items())


def node_assess(state: dict) -> dict:
    """Feature (b): after a round, decide whether to keep searching (LLM, with guardrails)."""
    cfg = state["config"]
    oot_key = training._oot_key(cfg.primary_metric)
    trials = state["all_trials"]
    current_best = max((t.oot_metrics.get(oot_key, float("-inf")) for t in trials), default=float("-inf"))
    prev = state.get("best_oot_so_far", float("-inf"))
    improvement = float("inf") if prev == float("-inf") else current_best - prev

    cont, note = search_advisor.should_continue(
        improvement, state.get("round_idx", 0), state.get("search_rounds", 1),
        leaderboard_summary=_round_summary(trials, oot_key), cached_llm=state.get("cached_llm"),
    )
    log = list(state.get("search_log", []))
    log.append({
        "round": state.get("round_idx"), "best_oot": round(current_best, 5),
        "improvement": None if improvement == float("inf") else round(improvement, 5),
        "continue": cont, "note": note,
    })
    if state.get("run_dir"):
        Path(state["run_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(state["run_dir"]) / "search_log.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    return {"should_continue": cont, "assess_note": note, "best_oot_so_far": current_best, "search_log": log}


def node_finalize(state: dict) -> dict:
    oot_key = training._oot_key(state["config"].primary_metric)
    best = max(state["all_trials"], key=lambda t: t.oot_metrics.get(oot_key, float("-inf")))
    result, best_model, prepared = finalize_run(
        state["prepared"], state["config"], state["all_trials"], best,
        run_dir=state.get("run_dir"), source_path=state.get("source_path"),
        alias_map=state.get("alias_map"),
    )
    return {"best": best, "result": result, "best_model": best_model, "prepared": prepared}


def node_narrate(state: dict) -> dict:
    summary = narrative_advisor.summarize(state["result"], cached_llm=state.get("cached_llm"))
    if state.get("run_dir"):
        (Path(state["run_dir"]) / "summary.md").write_text(summary, encoding="utf-8")
    return {"summary": summary}


# --- routing helpers (shared by both drivers) ----------------------------------------------
def needs_another_round(state: dict) -> bool:
    """True if the search loop should run another round (LLM present, budget left, >1 algo).

    Accepts either the LLM object in state (linear driver) or a serializable `use_llm` flag
    (graph driver, where the LLM object lives outside the checkpointed state).
    """
    has_llm = state.get("cached_llm") is not None or state.get("use_llm", False)
    return (
        has_llm
        and state.get("round_idx", 0) < state.get("search_rounds", 1)
        and len(state["config"].search.algorithms) > 1
    )


def design_decisions(config) -> dict:
    """The small decision payload shown at the model_design HITL gate."""
    return {
        "primary_metric": config.primary_metric,
        "algorithms": config.search.algorithms,
        "imbalance_tier": config.data_strategy.imbalance_tier,
        "oot_snapshots": config.windows.oot_snapshots,
        "correlation_threshold": config.data_strategy.correlation_threshold,
    }
