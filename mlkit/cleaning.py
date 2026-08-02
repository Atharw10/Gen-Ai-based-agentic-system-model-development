"""Cleaning utilities - IQR (safe), winsorization, smart imputation."""
from __future__ import annotations
import numpy as np
import pandas as pd


def safe_iqr_cap(series: pd.Series, min_unique: int = 10, k: float = 1.5):
    """IQR-cap but ONLY for truly continuous features. Returns (capped_series, lo, hi)."""
    if series.nunique(dropna=True) <= min_unique or not pd.api.types.is_numeric_dtype(series):
        return series, None, None
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr <= 0:
        return series, None, None
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return series.clip(lo, hi), float(lo), float(hi)


def winsorize(series: pd.Series, lo_pct: float = 0.01, hi_pct: float = 0.99):
    """Cap at percentiles - better than IQR for heavy-tailed transaction data."""
    if not pd.api.types.is_numeric_dtype(series):
        return series, None, None
    lo, hi = series.quantile([lo_pct, hi_pct])
    return series.clip(lo, hi), float(lo), float(hi)


def smart_impute(df: pd.DataFrame, num_strategy: str = "median",
                 cat_strategy: str = "mode", target_col: str = None):
    """Impute missing values learned from this df. Returns (df_out, imputation_map)."""
    out = df.copy()
    imp_map = {}
    num_cols = [c for c in out.select_dtypes(include=[np.number]).columns if c != target_col]
    cat_cols = [c for c in out.select_dtypes(exclude=[np.number]).columns if c != target_col]

    for c in num_cols:
        if not out[c].isna().any(): continue
        v = (out[c].median() if num_strategy == "median"
             else out[c].mean() if num_strategy == "mean"
             else 0)
        out[c] = out[c].fillna(v); imp_map[c] = float(v)

    for c in cat_cols:
        if not out[c].isna().any(): continue
        mode = out[c].mode(dropna=True)
        v = mode.iloc[0] if len(mode) > 0 else "unknown"
        out[c] = out[c].fillna(v); imp_map[c] = str(v)
    return out, imp_map


def apply_imputation(df: pd.DataFrame, imp_map: dict):
    """Apply a learned imputation map (from smart_impute) to a new dataframe."""
    out = df.copy()
    for c, v in imp_map.items():
        if c in out.columns and out[c].isna().any():
            out[c] = out[c].fillna(v)
    return out