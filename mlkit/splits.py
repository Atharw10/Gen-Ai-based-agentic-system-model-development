"""Temporal splits - NEVER use random splits for propensity models."""
from __future__ import annotations
import pandas as pd
import numpy as np


def temporal_split(df: pd.DataFrame, date_col: str, oot_start: str,
                   train_end: str = None):
    """Carve out an OOT (Out-of-Time) test set; everything before oot_start is train.

    Args:
        df: DataFrame containing a date column
        date_col: name of date column (must be datetime)
        oot_start: 'YYYY-MM-DD' - first day of OOT period
        train_end: optional 'YYYY-MM-DD' - last day of training; defaults to day before oot_start

    Returns:
        (df_train, df_oot, split_info_dict)
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    oot_start_ts = pd.Timestamp(oot_start)
    if train_end is None:
        train_end_ts = oot_start_ts - pd.Timedelta(days=1)
    else:
        train_end_ts = pd.Timestamp(train_end)

    df_train = df[df[date_col] <= train_end_ts].copy()
    df_oot = df[df[date_col] >= oot_start_ts].copy()

    info = {
        "split_type": "temporal_OOT",
        "date_col": date_col,
        "train_range": (str(df_train[date_col].min()), str(df_train[date_col].max())),
        "oot_range": (str(df_oot[date_col].min()), str(df_oot[date_col].max())),
        "train_rows": len(df_train), "oot_rows": len(df_oot),
    }
    return df_train, df_oot, info


def monthly_snapshots(df: pd.DataFrame, date_col: str):
    """List the unique monthly snapshots present in the data."""
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.assign(**{date_col: pd.to_datetime(df[date_col])})
    return sorted(df[date_col].dt.to_period("M").astype(str).unique())