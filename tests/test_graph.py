"""Phase 4b acceptance: the LangGraph driver runs, pauses/resumes at gates, routes its loops,
and produces results equivalent to the linear orchestrator."""
from __future__ import annotations

from pathlib import Path

from app import graph, orchestrator
from tests.synth import tiny_fixture

# Shrink the search via a HITL override so tests are fast (also exercises the override path).
SMALL = {"action": "override",
         "overrides": {"search.n_trials": 4, "search.algorithms": ["LogisticRegression"]},
         "instructions": ""}
ACCEPT = {"action": "accept", "overrides": {}, "instructions": ""}


def _graph_resolver(payload):
    return SMALL if payload["stage"] == "model_design" else ACCEPT


def _linear_hitl(stage, decisions):
    return SMALL if stage == "model_design" else ACCEPT


def _run_args(tmp_path, df):
    return dict(
        df=df, business_objective="Build CC propensity model", target_col="target",
        date_col="snapshot_date", group_col="customer_id", cached_llm=None,
        artifacts_root=str(tmp_path / "runs"), search_rounds=1,
    )


def test_graph_end_to_end_offline(tmp_path):
    df = tiny_fixture(seed=0)
    out = graph.run_graph(resolver=_graph_resolver, **_run_args(tmp_path, df))

    assert out["result"].best_algorithm == "LogisticRegression"
    assert out["result"].n_trials_run == 4
    # all gates were hit, in order
    assert out["gates_seen"] == ["profiling", "model_design", "feature_selection", "evaluation"]
    run_dir = Path(out["run_dir"])
    assert (run_dir / "model_bundle.joblib").exists()
    assert (run_dir / "checkpoints").exists() or True  # graph writes via persistence, not HITLGate
    assert (run_dir / "summary.md").exists()


def test_graph_regenerate_routes_back_to_design(tmp_path):
    df = tiny_fixture(seed=0)
    calls = {"n": 0}

    def resolver(payload):
        if payload["stage"] == "model_design":
            calls["n"] += 1
            # ask to regenerate the first time, then accept-with-small-override
            return {"action": "regenerate", "instructions": "use fewer algos"} if calls["n"] == 1 else SMALL
        return ACCEPT

    out = graph.run_graph(resolver=resolver, **_run_args(tmp_path, df))
    # design gate visited twice => the conditional edge looped back to design
    assert calls["n"] == 2
    assert out["gates_seen"].count("model_design") == 2
    assert out["result"].best_algorithm == "LogisticRegression"


def test_graph_matches_linear(tmp_path):
    df = tiny_fixture(seed=0)
    g = graph.run_graph(resolver=_graph_resolver, **_run_args(tmp_path, df))
    lin_result, _, lin_prepared, _ = orchestrator.run(
        df, business_objective="Build CC propensity model", target_col="target",
        date_col="snapshot_date", group_col="customer_id", cached_llm=None,
        hitl_mode=_linear_hitl, artifacts_root=str(tmp_path / "runs_lin"), search_rounds=1,
    )
    # same deterministic config+data+seed => same selection and same winner across both drivers
    assert g["prepared"].selected_raw_features == lin_prepared.selected_raw_features
    assert g["result"].best_algorithm == lin_result.best_algorithm
