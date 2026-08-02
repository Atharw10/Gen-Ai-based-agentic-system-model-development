"""Phase 1 acceptance: single tier function, consistent across callers; VIF no longer crashes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlkit import metrics, stats


def test_assign_tier_boundaries():
    assert metrics.assign_tier(0.50) == "balanced"
    assert metrics.assign_tier(0.40) == "balanced"  # inclusive lower bound
    assert metrics.assign_tier(0.25) == "mild"
    assert metrics.assign_tier(0.05) == "moderate"  # the historic 5% boundary case
    assert metrics.assign_tier(0.0499) == "severe"
    assert metrics.assign_tier(0.01) == "severe"
    assert metrics.assign_tier(0.005) == "extreme"
    assert metrics.assign_tier(None) == "unknown"


def test_detect_imbalance_uses_assign_tier():
    y = np.array([0] * 95 + [1] * 5)  # 5% -> moderate per the single cutoffs
    info = metrics.detect_imbalance_tier(y)
    assert info["tier"] == metrics.assign_tier(info["pos_ratio"]) == "moderate"


def test_research_and_design_share_tier_logic():
    # tools/* must now agree with mlkit at the historically-inconsistent 5% boundary
    from tools import research as research_tool

    profile = {"target": {"pos_ratio": 0.05}}
    _, tier = research_tool._detect_tier(profile)
    assert tier == metrics.assign_tier(0.05) == "moderate"


def test_compute_vif_runs():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    df = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(scale=0.01, size=500), "c": rng.normal(size=500)})
    out = stats.compute_vif(df, ["a", "b", "c"])  # used to crash on select_types typo
    assert set(out["feature"]) == {"a", "b", "c"}
    # a and b are nearly collinear -> high VIF
    assert out.set_index("feature").loc["a", "VIF"] > 10
