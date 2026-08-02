"""LangGraph driver (driver #2) — true pause/resume HITL via interrupt() + a checkpointer.

Same node functions as the linear orchestrator (app/nodes.py); LangGraph adds the explicit graph
with conditional edges for the two loops (design-regenerate, search-rounds) and first-class HITL
pause points.

HITL model: each gate node calls `interrupt(payload)`. The graph pauses and returns control; the
caller inspects the payload, collects the user's decision, and resumes with `Command(resume=action)`.
`run_graph(...)` is a convenience driver that auto-resolves gates (accept by default, or via a
resolver callback) so the same graph serves interactive, headless, and test use.

IMPORTANT — what lives where:
The checkpointer SERIALISES graph state at every step, so heavy/non-serialisable objects (the
DataFrame, fitted models, the LLM client) must NOT go into state — you would never checkpoint a
500k-row frame anyway. Those live in a process-local REGISTRY keyed by thread_id; the graph state
carries only small serialisable data (config, reports, leaderboard, scalars). This is the correct
production pattern; for durable cross-process resume, back the registry with disk paths.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app import nodes
from app.orchestrator import _validate_columns
from runtime.hitl import apply_overrides_to_config

# process-local store for non-serialisable objects, keyed by thread_id
_REGISTRY: dict[str, dict] = {}
# keys that must never enter the checkpointed state
HEAVY_KEYS = ("df", "cached_llm", "prepared", "best_model", "logger")


class AMDState(TypedDict, total=False):
    # serialisable inputs / working state only (heavy objects live in _REGISTRY)
    business_objective: str
    model_type: Optional[str]
    target_col: str
    date_col: str
    group_col: Optional[str]
    run_dir: str
    source_path: Optional[str]
    search_rounds: int
    use_llm: bool
    alias_columns: bool
    alias_map: dict
    domain_fe_done: bool
    domain_features: list
    matched_concepts: list
    profile: Any
    findings: Any
    config: Any
    design_errors: Any
    all_trials: list
    round_idx: int
    strategist_note: str
    should_continue: bool
    assess_note: str
    best_oot_so_far: float
    search_log: list
    hitl_design_action: dict
    best: Any
    result: Any
    summary: str


def _reg(config) -> dict:
    return _REGISTRY.setdefault(config["configurable"]["thread_id"], {})


def _wrap(fn, name: str | None = None):
    """Adapt a shared (state-dict -> updates) node to a graph node, bridging the registry.

    Injects heavy objects from the registry before the call; routes any heavy objects in the
    update back into the registry instead of the checkpointed state. Emits live step logging
    when a RunLogger is present.
    """

    def node(state: AMDState, config) -> dict:
        reg = _reg(config)
        work = dict(state)
        for k in HEAVY_KEYS:
            if k in reg:
                work[k] = reg[k]
        log = reg.get("logger")
        if log is not None and name:
            with log.step(name):
                update = fn(work)
        else:
            update = fn(work)
        out = {}
        for k, v in update.items():
            if k in HEAVY_KEYS:
                reg[k] = v
            else:
                out[k] = v
        return out

    return node


# --- gate nodes (interrupt-based) -----------------------------------------------------------
def _gate(stage: str, decisions: dict) -> dict:
    action = interrupt({"stage": stage, "decisions": decisions})
    return action or {"action": "accept", "overrides": {}, "instructions": ""}


def hitl_profiling(state: AMDState) -> dict:
    p = state["profile"]
    _gate("profiling", {"n_rows": p.n_rows, "pos_ratio": p.target.pos_ratio,
                        "n_snapshots": p.temporal.n_snapshots,
                        "pos_ratio_per_snapshot": p.temporal.pos_ratio_per_snapshot})
    return {}


def hitl_design(state: AMDState) -> dict:
    action = _gate("model_design", nodes.design_decisions(state["config"]))
    update: dict = {"hitl_design_action": action}
    if action.get("action") == "override":
        update["config"] = apply_overrides_to_config(state["config"], action.get("overrides", {}))
    elif action.get("action") == "regenerate":
        update["business_objective"] = (
            state["business_objective"] + f"\n[user]: {action.get('instructions', '')}"
        )
    return update


def hitl_features(state: AMDState, config) -> dict:
    sr = _reg(config)["prepared"].selection_report
    _gate("feature_selection", {"n_in": sr.n_in, "n_selected": sr.n_selected,
                                "suspicious_high_iv": sr.suspicious_high_iv})
    return {}


def hitl_evaluation(state: AMDState) -> dict:
    _gate("evaluation", {"best_algorithm": state["result"].best_algorithm,
                         "oot": state["result"].leaderboard[0]["oot_metrics"]})
    return {}


# --- conditional routers --------------------------------------------------------------------
def route_after_design(state: AMDState) -> str:
    return "design" if state.get("hitl_design_action", {}).get("action") == "regenerate" else "prepare"


def route_after_assess(state: AMDState) -> str:
    return "strategist" if state.get("should_continue") else "finalize"


# --- graph assembly -------------------------------------------------------------------------
def build_graph(checkpointer=None):
    g = StateGraph(AMDState)
    g.add_node("alias", _wrap(nodes.node_alias, "alias columns (privacy shield)"))
    g.add_node("profile", _wrap(nodes.node_profile, "profile data"))
    g.add_node("hitl_profiling", hitl_profiling)
    g.add_node("research", _wrap(nodes.node_research, "research (models / windows / HPO / features)"))
    g.add_node("domain_fe", _wrap(nodes.node_domain_fe, "domain feature engineering (research families)"))
    g.add_node("design", _wrap(nodes.node_design, "model design"))
    g.add_node("hitl_design", hitl_design)
    g.add_node("prepare", _wrap(nodes.node_prepare, "prepare data (split/clean/select/encode)"))
    g.add_node("hitl_features", hitl_features)
    g.add_node("search", _wrap(nodes.node_search_round, "search round (HPO)"))
    g.add_node("assess", _wrap(nodes.node_assess, "assess: continue or stop?"))
    g.add_node("strategist", _wrap(nodes.node_strategist, "strategist: revise algorithm budget"))
    g.add_node("finalize", _wrap(nodes.node_finalize, "finalize (refit + OOT eval + persist)"))
    g.add_node("hitl_evaluation", hitl_evaluation)
    g.add_node("narrate", _wrap(nodes.node_narrate, "write model summary"))

    g.add_edge(START, "alias")
    g.add_edge("alias", "profile")
    g.add_edge("profile", "hitl_profiling")
    g.add_edge("hitl_profiling", "research")
    g.add_edge("research", "domain_fe")
    g.add_edge("domain_fe", "design")
    g.add_edge("design", "hitl_design")
    g.add_conditional_edges("hitl_design", route_after_design, {"design": "design", "prepare": "prepare"})
    g.add_edge("prepare", "hitl_features")
    g.add_edge("hitl_features", "search")
    g.add_edge("search", "assess")
    g.add_conditional_edges("assess", route_after_assess, {"strategist": "strategist", "finalize": "finalize"})
    g.add_edge("strategist", "search")
    g.add_edge("finalize", "hitl_evaluation")
    g.add_edge("hitl_evaluation", "narrate")
    g.add_edge("narrate", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


def _auto_accept(payload: dict) -> dict:
    return {"action": "accept", "overrides": {}, "instructions": ""}


def run_graph(
    df,
    business_objective: str,
    target_col: str,
    date_col: str,
    group_col: Optional[str] = None,
    cached_llm=None,
    artifacts_root: str = "artifacts/runs",
    source_path: Optional[str] = None,
    search_rounds: int = 2,
    alias_columns: bool = True,
    verbose: bool = True,
    resolver=None,
    thread_id: Optional[str] = None,
    model_type: Optional[str] = None,
):
    """Drive the graph to completion, auto-resolving HITL interrupts.

    `resolver(payload) -> action` decides each gate (default: accept). For a real UI/service, use
    build_graph() directly and handle interrupts yourself: the payload is on
    result['__interrupt__'][0].value; resume with Command(resume=action) on the same thread_id.
    """
    resolver = resolver or _auto_accept
    _validate_columns(df, target_col, date_col, group_col)
    run_dir = str(Path(artifacts_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    tid = thread_id or run_dir
    from runtime.cost import CostTracker
    from runtime.run_logger import RunLogger

    cost = CostTracker(run_dir=run_dir, model_id=getattr(cached_llm, "model_id", "none"), verbose=verbose)
    if cached_llm is not None and hasattr(cached_llm, "tracker"):
        cached_llm.tracker = cost
    # heavy / non-serialisable objects live out of the checkpointed state
    _REGISTRY[tid] = {"df": df, "cached_llm": cached_llm,
                      "logger": RunLogger(run_dir=run_dir, verbose=verbose), "cost": cost}

    graph = build_graph()
    cfg = {"configurable": {"thread_id": tid}, "recursion_limit": 100}
    init: dict = {
        "business_objective": business_objective, "model_type": model_type,
        "target_col": target_col, "date_col": date_col,
        "group_col": group_col, "run_dir": run_dir, "source_path": source_path,
        "search_rounds": search_rounds, "use_llm": cached_llm is not None,
        "alias_columns": alias_columns, "all_trials": [], "round_idx": 0,
    }

    payloads: list[dict] = []
    step_in: Any = init
    try:
        while True:
            out = graph.invoke(step_in, config=cfg)
            if "__interrupt__" in out:
                payload = out["__interrupt__"][0].value
                payloads.append(payload)
                step_in = Command(resume=resolver(payload))
                continue
            break

        final = graph.get_state(cfg).values
        reg = _REGISTRY.get(tid, {})
        return {
            "result": final["result"],
            "best_model": reg.get("best_model"),
            "prepared": reg.get("prepared"),
            "summary": final["summary"],
            "run_dir": run_dir,
            "gates_seen": [p["stage"] for p in payloads],
            "cost": cost.totals(),
        }
    finally:
        _REGISTRY.pop(tid, None)  # free heavy objects
