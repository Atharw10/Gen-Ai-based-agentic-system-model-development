"""Charts are generated into <run_dir>/charts and never break a run (best-effort)."""
from __future__ import annotations

from pathlib import Path

from app import orchestrator
from tests.synth import tiny_fixture

EXPECTED = [
    "gains_lift.png", "roc_pr.png", "ks_curve.png", "calibration.png",
    "score_psi.png", "csi_per_feature.png", "woe_trends.png", "iv_selected.png",
    "feature_importance.png", "leaderboard.png", "oot_metrics.png",
    "posrate_per_snapshot.png",
]


def test_charts_written_to_artifacts(tmp_path):
    df = tiny_fixture(seed=0)
    res, model, prepared, _ = orchestrator.run(
        df, business_objective="x", target_col="target", date_col="snapshot_date",
        group_col="customer_id", cached_llm=None, search_rounds=1,
        artifacts_root=str(tmp_path), source_path=None, model_type="propensity", verbose=False,
    )
    charts = Path(res.run_dir) / "charts"
    assert charts.exists()
    produced = {p.name for p in charts.glob("*.png")}
    # the core validation curves must all be present
    for name in EXPECTED:
        assert name in produced, f"missing chart: {name}"
    # no chart errors recorded
    assert not (charts / "_errors.txt").exists(), (charts / "_errors.txt").read_text()


def test_save_all_charts_is_best_effort(tmp_path):
    # a broken model (no predict_proba/decision_function/predict) must not raise
    from pipeline import charts

    class _Dummy:
        pass
    df = tiny_fixture(seed=1)
    res, model, prepared, _ = orchestrator.run(
        df, business_objective="x", target_col="target", date_col="snapshot_date",
        group_col="customer_id", cached_llm=None, search_rounds=1,
        artifacts_root=str(tmp_path), source_path=None, verbose=False,
    )
    # calling with a broken model should not raise (errors captured, not thrown)
    saved = charts.save_all_charts(res.run_dir, prepared, _Dummy(), res)
    assert isinstance(saved, list)
