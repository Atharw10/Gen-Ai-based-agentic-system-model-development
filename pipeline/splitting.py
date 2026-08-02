"""Stage 3 - temporal split. Runs FIRST, before any fit/statistic, to prevent OOT leakage."""
from __future__ import annotations

import pandas as pd

from contracts import SplitReport


def temporal_split(
    df: pd.DataFrame,
    date_col: str,
    train_snapshots: list[str],
    oot_snapshots: list[str],
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitReport]:
    """Split by YYYY-MM snapshot membership. Rows outside both sets are dropped (with a note).

    Returns (df_train, df_oot, report). No imputation/selection/scaling happens here — those
    must be fit on df_train AFTER this split.
    """
    period = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").astype(str)
    train_mask = period.isin(set(train_snapshots))
    oot_mask = period.isin(set(oot_snapshots))

    df_train = df.loc[train_mask].copy()
    df_oot = df.loc[oot_mask].copy()

    def _pos_ratio(d):
        if target_col and target_col in d.columns and len(d):
            return round(float((d[target_col] == 1).mean()), 4)
        return None

    report = SplitReport(
        train_rows=len(df_train),
        oot_rows=len(df_oot),
        train_range=[min(train_snapshots), max(train_snapshots)] if train_snapshots else [],
        oot_range=[min(oot_snapshots), max(oot_snapshots)] if oot_snapshots else [],
        train_pos_ratio=_pos_ratio(df_train),
        oot_pos_ratio=_pos_ratio(df_oot),
    )
    return df_train, df_oot, report
