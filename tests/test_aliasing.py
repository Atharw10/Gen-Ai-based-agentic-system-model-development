"""Privacy shield: column aliasing is reversible, covers target/date/group, and the LLM-facing
profile contains zero real column names."""
from __future__ import annotations

import json
from pathlib import Path

from dataio import aliasing
from pipeline import profiling
from tests.synth import tiny_fixture

REAL_SENSITIVE = ["feat_inf_0", "txn_ref", "leak_target_peek", "customer_id", "target", "high_missing_col"]


def test_alias_map_covers_roles_and_is_reversible():
    df = tiny_fixture(seed=0)
    amap = aliasing.build_alias_map(list(df.columns), "target", "snapshot_date", "customer_id")
    assert amap.target == "target" and amap.date == "snapshot_date" and amap.group == "entity_id"
    # every column mapped, all aliases unique, fully reversible
    assert len(amap.forward) == len(df.columns)
    assert len(set(amap.forward.values())) == len(amap.forward)
    for real in df.columns:
        assert amap.reverse[amap.forward[real]] == real
    # features get col_NNNN
    assert amap.forward["feat_inf_0"].startswith("col_")


def test_apply_and_unalias_roundtrip():
    df = tiny_fixture(seed=0)
    amap = aliasing.build_alias_map(list(df.columns), "target", "snapshot_date", "customer_id")
    da = aliasing.apply_aliases(df, amap)
    assert "feat_inf_0" not in da.columns and "txn_ref" not in da.columns
    assert "target" in da.columns and "entity_id" in da.columns
    # values unchanged, just renamed
    assert da["target"].sum() == df["target"].sum()
    assert amap.unalias(["col_0001", "target"]) == [amap.reverse["col_0001"], "target"]


def test_llm_profile_has_no_real_names():
    df = tiny_fixture(seed=0)
    amap = aliasing.build_alias_map(list(df.columns), "target", "snapshot_date", "customer_id")
    da = aliasing.apply_aliases(df, amap)
    profile = profiling.profile_dataset(da, amap.target, amap.date, amap.group, "cc")
    blob = json.dumps(profile.model_dump(), default=str)
    # the exact text that would be sent to the LLM must contain none of the real sensitive names
    for name in ["feat_inf_0", "txn_ref", "leak_target_peek", "high_missing_col", "customer_id"]:
        assert name not in blob


def test_save_alias_map_writes_files(tmp_path):
    df = tiny_fixture(seed=0)
    amap = aliasing.build_alias_map(list(df.columns), "target", "snapshot_date", "customer_id")
    aliasing.save_alias_map(amap, str(tmp_path))
    assert (tmp_path / "column_alias_map.json").exists()
    assert (tmp_path / "column_alias_map.csv").exists()
    data = json.loads((tmp_path / "column_alias_map.json").read_text())
    assert data["forward"]["customer_id"] == "entity_id"
