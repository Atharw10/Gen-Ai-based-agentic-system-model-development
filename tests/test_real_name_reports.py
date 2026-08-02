"""Human-facing reports are translated back to real column names (LLM saw only aliases)."""
from __future__ import annotations

import json
from pathlib import Path

from dataio import aliasing
from tests.synth import tiny_fixture


def test_real_selection_report_helper():
    df = tiny_fixture(seed=0)
    amap = aliasing.build_alias_map(list(df.columns), "target", "snapshot_date", "customer_id")
    # fake a tiny selection report-like object
    class SR:
        n_in = 3
        n_selected = 1
        selected_features = ["col_0001"]
        suspicious_high_iv = ["col_0002"]
        dropped_redundant = []
        dropped_low_iv = ["col_0003"]
        dropped_constant = []
        dropped_high_missing = []
        dropped_high_cardinality = []
        iv_table = [{"feature": "col_0001", "IV": 0.4, "strength": "strong"}]

    out = aliasing.real_selection_report(SR(), amap.reverse)
    assert out["selected"][0]["real_name"] == amap.reverse["col_0001"]
    assert out["iv_table"][0]["real_name"] == amap.reverse["col_0001"]


def test_orchestrated_run_writes_real_name_artifacts(tmp_path):
    from app import orchestrator

    df = tiny_fixture(seed=0)

    def hitl(stage, decisions):
        if stage == "model_design":
            return {"action": "override",
                    "overrides": {"search.n_trials": 4, "search.algorithms": ["LogisticRegression"]},
                    "instructions": ""}
        return {"action": "accept", "overrides": {}, "instructions": ""}

    result, _, prepared, _ = orchestrator.run(
        df, business_objective="Build CC propensity model", target_col="target",
        date_col="snapshot_date", group_col="customer_id", cached_llm=None,
        hitl_mode=hitl, artifacts_root=str(tmp_path / "runs"), search_rounds=1,
    )

    run_dir = Path(result.run_dir)
    # alias map carried on the result for downstream translation
    assert result.column_alias_map and "reverse" in result.column_alias_map

    # full real-name selection report written
    real_report = json.loads((run_dir / "selection_report_real.json").read_text(encoding="utf-8"))
    aliases = {p["alias"] for p in real_report["selected"]}
    reals = {p["real_name"] for p in real_report["selected"]}
    assert aliases and reals and aliases != reals  # actually translated

    # model card lists top features by REAL synthetic name (feat_*/leak_*), not aliases
    card = (run_dir / "model_card.md").read_text(encoding="utf-8")
    assert "Top selected" in card
    assert ("feat_" in card) or ("leak_target_peek" in card)
    assert "col_0001" not in card  # no aliases leaked into the human-facing card
    # the real-name mapping file exists too
    assert (run_dir / "column_alias_map.csv").exists()
