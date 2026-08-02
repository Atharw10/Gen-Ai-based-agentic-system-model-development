"""Stage 1 - profiling. Pure observation, no decisions, no LLM. Emits a typed ProfileResult."""
from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import ProfileResult, TargetProfile, TemporalProfile


def profile_dataset(
    df: pd.DataFrame,
    target_col: str,
    date_col: str | None = None,
    group_col: str | None = None,
    business_objective: str = "",
) -> ProfileResult:
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    memory_mb = round(df.memory_usage(deep=True).sum() / 1e6, 2)

    # --- target ---
    y = df[target_col].dropna()
    uniq, counts = np.unique(y, return_counts=True)
    dist = {str(k): int(v) for k, v in zip(uniq, counts)}
    is_binary = len(uniq) == 2
    pos_ratio = None
    if is_binary:
        pos = int((y == 1).sum())
        tot = int((y == 0).sum()) + pos
        pos_ratio = round(pos / max(tot, 1), 4)
    target = TargetProfile(
        name=target_col,
        n_unique=int(y.nunique()),
        is_binary=is_binary,
        pos_ratio=pos_ratio,
        null_pct=round(df[target_col].isna().mean() * 100, 2),
        distribution=dist,
    )

    # --- temporal ---
    if date_col and date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce")
        valid = d.dropna()
        periods = valid.dt.to_period("M")
        # positive (target==1) rate per snapshot month — aligned to rows with a valid date
        pos_per_snapshot: dict[str, float] = {}
        if is_binary and len(valid):
            ymonth = pd.DataFrame({
                "p": periods.astype(str),
                "y": (df.loc[valid.index, target_col] == 1).astype(int).to_numpy(),
            })
            per = ymonth.groupby("p")["y"].mean()
            pos_per_snapshot = {str(k): round(float(v), 4) for k, v in per.sort_index().items()}
        temporal = TemporalProfile(
            date_col=date_col,
            min_date=str(valid.min()) if len(valid) else None,
            max_date=str(valid.max()) if len(valid) else None,
            n_snapshots=int(periods.nunique()),
            snapshots=[str(s) for s in sorted(periods.unique())],
            rows_per_snapshot={str(k): int(v) for k, v in periods.value_counts().sort_index().items()},
            pos_ratio_per_snapshot=pos_per_snapshot,
        )
    else:
        temporal = TemporalProfile(date_col=None)

    # --- columns ---
    ignore = {target_col, date_col, group_col}
    feat_cols = [c for c in df.columns if c not in ignore]
    num_cols = df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feat_cols if c not in num_cols]
    high_card = [c for c in cat_cols if df[c].nunique(dropna=True) > 50]
    constant = [c for c in feat_cols if df[c].nunique(dropna=True) <= 1]

    return ProfileResult(
        business_objective=business_objective,
        n_rows=n_rows,
        n_cols=n_cols,
        memory_mb=memory_mb,
        target=target,
        temporal=temporal,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        high_cardinality_cols=high_card,
        constant_cols=constant,
        group_col=group_col,
        overall_missing_pct=round(df[feat_cols].isna().mean().mean() * 100, 2) if feat_cols else 0.0,
    )
