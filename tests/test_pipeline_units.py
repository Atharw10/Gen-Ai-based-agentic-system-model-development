"""Phase 2 acceptance: pipeline building blocks behave correctly."""
from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import DataStrategy, FeatureSelection, RunConfig, SearchSpace, Windows
from mlkit import selection
from pipeline import profiling, splitting
from tests.synth import tiny_fixture


def _cfg(df, **kw):
    snaps = sorted(df["snapshot_date"].dt.to_period("M").astype(str).unique())
    base = dict(
        business_objective="cc propensity", target_col="target", date_col="snapshot_date",
        primary_metric="PR_AUC",
        windows=Windows(observation_window_months=3, performance_window_months=1,
                        snapshots_for_training=snaps[:-2], oot_snapshots=snaps[-2:]),
        data_strategy=DataStrategy(group_col="customer_id", imbalance_tier="moderate"),
        feature_selection=FeatureSelection(top_k_by_importance=20, sample_rows_for_selection=6000),
        search=SearchSpace(algorithms=["LogisticRegression"], n_trials=2),
    )
    base.update(kw)
    return RunConfig(**base)


def test_profile_detects_snapshots_and_tier_inputs():
    df = tiny_fixture(seed=2)
    p = profiling.profile_dataset(df, "target", "snapshot_date", "customer_id", "cc")
    assert p.temporal.n_snapshots == 6
    assert p.target.is_binary and 0 < p.target.pos_ratio < 0.3
    assert "customer_id" not in p.numeric_cols  # group col excluded from features
    assert "const_col" in p.constant_cols


def test_temporal_split_is_disjoint_and_chronological():
    df = tiny_fixture(seed=2)
    snaps = sorted(df["snapshot_date"].dt.to_period("M").astype(str).unique())
    tr, oot, rep = splitting.temporal_split(df, "snapshot_date", snaps[:-2], snaps[-2:], "target")
    tr_p = set(pd.to_datetime(tr["snapshot_date"]).dt.to_period("M").astype(str))
    oot_p = set(pd.to_datetime(oot["snapshot_date"]).dt.to_period("M").astype(str))
    assert tr_p.isdisjoint(oot_p)
    assert max(tr_p) < min(oot_p)
    assert rep.train_rows + rep.oot_rows == len(df)


def test_cluster_prune_drops_redundant():
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    df = pd.DataFrame({
        "a": a,
        "a_copy": a + rng.normal(scale=1e-3, size=2000),  # ~identical to a
        "b": rng.normal(size=2000),
    })
    rank = {"a": 0.5, "a_copy": 0.1, "b": 0.3}
    out = selection.cluster_prune(df, ["a", "a_copy", "b"], rank, threshold=0.9, sample_rows=2000)
    assert "a" in out["kept"] and "b" in out["kept"]
    assert "a_copy" in out["dropped_redundant"]  # the lower-ranked twin is dropped


def test_hygiene_filter_drops_constant_and_high_missing():
    df = pd.DataFrame({
        "const": np.ones(100),
        "mostly_null": [np.nan] * 90 + list(range(10)),
        "good": np.arange(100),
    })
    out = selection.hygiene_filter(df, ["const", "mostly_null", "good"], max_missing_pct=0.7)
    assert out["kept"] == ["good"]
    assert "const" in out["dropped_constant"]
    assert "mostly_null" in out["dropped_high_missing"]


def test_full_run_smoke(tmp_path):
    from pipeline.runner import run_from_config

    df = tiny_fixture(seed=6)
    cfg = _cfg(df, search=SearchSpace(algorithms=["LogisticRegression", "HistGradientBoosting"], n_trials=4))
    result, model, prepared = run_from_config(df, cfg, run_dir=str(tmp_path / "run"))
    assert result.n_trials_run == 4
    assert result.best_algorithm in {"LogisticRegression", "HistGradientBoosting"}
    assert (tmp_path / "run" / "model_bundle.joblib").exists()
    assert (tmp_path / "run" / "model_card.md").exists()
    # OOT primary metric should beat the base rate clearly (signal is learnable)
    assert result.leaderboard[0]["oot_metrics"]["pr_auc"] > prepared.y_oot.mean()
