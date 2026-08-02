"""Phase 0 acceptance: RunConfig validates, round-trips, and enforces its invariants."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from contracts import DataStrategy, FeatureSelection, RunConfig, SearchSpace, Windows


def _good_config(**overrides) -> RunConfig:
    base = dict(
        business_objective="Build CC propensity model",
        target_col="target",
        date_col="snapshot_date",
        primary_metric="PR_AUC",
        windows=Windows(
            observation_window_months=3,
            performance_window_months=1,
            snapshots_for_training=["2025-01", "2025-02", "2025-03", "2025-04"],
            oot_snapshots=["2025-05", "2025-06"],
        ),
        data_strategy=DataStrategy(group_col="customer_id", imbalance_tier="severe"),
        feature_selection=FeatureSelection(),
        search=SearchSpace(algorithms=["LightGBM", "XGBoost"], n_trials=10),
    )
    base.update(overrides)
    return RunConfig(**base)


def test_roundtrip_json():
    cfg = _good_config()
    blob = cfg.model_dump_json()
    cfg2 = RunConfig.model_validate_json(blob)
    assert cfg2 == cfg
    # ensure it's plain-JSON serialisable too
    json.loads(blob)


def test_train_oot_overlap_rejected():
    with pytest.raises(ValidationError):
        Windows(
            observation_window_months=3,
            performance_window_months=1,
            snapshots_for_training=["2025-01", "2025-05"],
            oot_snapshots=["2025-05", "2025-06"],
        )


def test_oot_must_be_after_training():
    with pytest.raises(ValidationError):
        Windows(
            observation_window_months=3,
            performance_window_months=1,
            snapshots_for_training=["2025-03", "2025-04"],
            oot_snapshots=["2025-01"],
        )


def test_scaling_autocorrected_for_tree_only_shortlist():
    cfg = _good_config(
        data_strategy=DataStrategy(imbalance_tier="severe", scaling_required=True),
        search=SearchSpace(algorithms=["LightGBM", "XGBoost"], n_trials=5),
    )
    assert cfg.data_strategy.scaling_required is False  # auto-corrected


def test_scaling_kept_when_linear_model_present():
    cfg = _good_config(
        data_strategy=DataStrategy(imbalance_tier="mild", scaling_required=True),
        search=SearchSpace(algorithms=["LogisticRegression", "LightGBM"], n_trials=5),
    )
    assert cfg.data_strategy.scaling_required is True
