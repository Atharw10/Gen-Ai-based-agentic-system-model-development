"""Phase 4 acceptance: the orchestrator runs end-to-end offline, with HITL + checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

from app import orchestrator
from runtime.hitl import HITLGate, apply_overrides_to_config
from tests.synth import tiny_fixture


def test_full_orchestrated_run_offline(tmp_path):
    df = tiny_fixture(seed=0)
    seen_stages = []

    def hitl(stage, decisions):
        seen_stages.append(stage)
        return {"action": "accept", "overrides": {}, "instructions": ""}

    result, model, prepared, summary = orchestrator.run(
        df,
        business_objective="Build CC propensity model",
        target_col="target",
        date_col="snapshot_date",
        group_col="customer_id",
        cached_llm=None,  # fully deterministic, no API key needed
        hitl_mode=hitl,
        artifacts_root=str(tmp_path / "runs"),
        search_rounds=1,
    )

    assert result.n_trials_run >= 1
    assert "profiling" in seen_stages and "model_design" in seen_stages
    assert "feature_selection" in seen_stages and "evaluation" in seen_stages
    # artifacts persisted
    run_dir = Path(result.run_dir)
    assert (run_dir / "model_bundle.joblib").exists()
    assert (run_dir / "run_config.json").exists()
    assert (run_dir / "checkpoints" / "model_design.json").exists()
    assert (run_dir / "summary.md").exists()
    assert result.best_algorithm  # a winning algorithm was chosen
    assert isinstance(summary, str) and len(summary) > 0


def test_hitl_override_applies_to_config():
    from contracts import (
        DataStrategy,
        FeatureSelection,
        RunConfig,
        SearchSpace,
        Windows,
    )

    cfg = RunConfig(
        business_objective="x", target_col="target", date_col="snapshot_date",
        primary_metric="PR_AUC",
        windows=Windows(observation_window_months=3, performance_window_months=1,
                        snapshots_for_training=["2025-01", "2025-02"], oot_snapshots=["2025-03"]),
        data_strategy=DataStrategy(imbalance_tier="moderate", correlation_threshold=0.85),
        feature_selection=FeatureSelection(),
        search=SearchSpace(algorithms=["LightGBM"], n_trials=4),
    )
    cfg2 = apply_overrides_to_config(cfg, {"data_strategy.correlation_threshold": 0.95})
    assert cfg2.data_strategy.correlation_threshold == 0.95
    assert cfg.data_strategy.correlation_threshold == 0.85  # original untouched


def test_approval_file_mode(tmp_path):
    gate = HITLGate(run_dir=str(tmp_path), mode="approval_file")
    # no approve file -> accept
    assert gate.gate("s1", "summary", {"k": 1})["action"] == "accept"
    # write an override approval, then a gate should pick it up
    cp = Path(tmp_path) / "checkpoints"
    cp.mkdir(exist_ok=True)
    (cp / "s2.approve.json").write_text(json.dumps({"action": "override", "overrides": {"a": 2}}))
    act = gate.gate("s2", "summary", {"a": 1})
    assert act["action"] == "override" and act["overrides"] == {"a": 2}
