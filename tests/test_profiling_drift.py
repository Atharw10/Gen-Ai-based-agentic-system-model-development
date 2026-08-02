"""Per-snapshot positive rate + drift detection (feeds window/OOT selection in research)."""
from __future__ import annotations

import pandas as pd

from mlkit.stats import pos_rate_drift
from pipeline import profiling


def _panel(rates):
    """Build a tiny panel: one month per (month, rate) with 100 rows each."""
    rows = []
    for month, rate in rates:
        npos = int(round(rate * 100))
        for i in range(100):
            rows.append({"snapshot_date": f"{month}-01",
                         "target": 1 if i < npos else 0,
                         "f1": float(i), "customer_id": i})
    return pd.DataFrame(rows)


def test_profiling_computes_pos_rate_per_snapshot():
    df = _panel([("2025-01", 0.10), ("2025-02", 0.10), ("2025-03", 0.50)])
    p = profiling.profile_dataset(df, "target", "snapshot_date", "customer_id", "x")
    pmr = p.temporal.pos_ratio_per_snapshot
    assert set(pmr) == {"2025-01", "2025-02", "2025-03"}
    assert abs(pmr["2025-01"] - 0.10) < 1e-6
    assert abs(pmr["2025-03"] - 0.50) < 1e-6


def test_drift_flags_outlier_month():
    df = _panel([("2025-01", 0.10), ("2025-02", 0.10), ("2025-03", 0.50)])
    p = profiling.profile_dataset(df, "target", "snapshot_date", "customer_id", "x")
    flagged, med, note = pos_rate_drift(p.temporal.pos_ratio_per_snapshot)
    assert "2025-03" in flagged
    assert abs(med - 0.10) < 1e-6
    assert "DRIFT" in note


def test_drift_quiet_when_stable():
    df = _panel([("2025-01", 0.10), ("2025-02", 0.105), ("2025-03", 0.095)])
    p = profiling.profile_dataset(df, "target", "snapshot_date", "customer_id", "x")
    flagged, _med, note = pos_rate_drift(p.temporal.pos_ratio_per_snapshot)
    assert flagged == []
    assert "stable" in note
