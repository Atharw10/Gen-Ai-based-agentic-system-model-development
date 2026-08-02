"""Leakage-safe cleaning primitives: every transform is FIT on train, APPLIED to both.

The fitted bounds/values are returned so they can be persisted and re-applied at scoring time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_outlier_bounds(
    df: pd.DataFrame, cols: list[str], strategy: str = "winsorize_99"
) -> dict[str, tuple[float, float]]:
    """Learn per-column clip bounds on TRAIN only. Skips low-cardinality (binary/flag) cols."""
    bounds: dict[str, tuple[float, float]] = {}
    if strategy == "none":
        return bounds
    for c in cols:
        s = df[c].dropna()
        if s.nunique() <= 10:  # don't crush binary/flag features
            continue
        if strategy == "winsorize_99":
            lo, hi = s.quantile(0.01), s.quantile(0.99)
        elif strategy == "iqr_safe":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            continue
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            bounds[c] = (float(lo), float(hi))
    return bounds


def apply_outlier_bounds(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, (lo, hi) in bounds.items():
        if c in out.columns:
            out[c] = out[c].clip(lo, hi)
    return out


def fit_impute_values(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> dict[str, object]:
    """Learn imputation fill values on TRAIN only."""
    fills: dict[str, object] = {}
    for c in numeric_cols:
        s = df[c].dropna()
        if s.empty:
            fills[c] = 0.0
        elif numeric_strategy == "median":
            fills[c] = float(s.median())
        elif numeric_strategy == "mean":
            fills[c] = float(s.mean())
        else:  # zero
            fills[c] = 0.0
    for c in categorical_cols:
        if categorical_strategy == "mode":
            m = df[c].mode(dropna=True)
            fills[c] = m.iloc[0] if len(m) else "_missing_"
        else:
            fills[c] = "_missing_"
    return fills


def apply_impute_values(df: pd.DataFrame, fills: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for c, v in fills.items():
        if c in out.columns:
            out[c] = out[c].fillna(v)
    return out
