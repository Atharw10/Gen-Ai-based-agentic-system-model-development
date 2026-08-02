"""Leakage-safe data preparation: split FIRST, then fit everything on train only.

This is computed ONCE per RunConfig and reused across all HPO trials (hyperparameters don't
change the data prep). Returns a PreparedData bundle with X/y matrices + fitted transformers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from contracts import ProfileResult, RunConfig, SelectionReport, SplitReport
from pipeline import cleaning, feature_engineer, feature_select, splitting


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_oot: np.ndarray
    y_oot: np.ndarray
    groups_train: np.ndarray | None
    feature_names: list[str]
    selected_raw_features: list[str]
    cleaning_artifacts: object
    preprocessor: object
    split_report: SplitReport
    selection_report: SelectionReport
    domain_features: list[str] = field(default_factory=list)
    column_map: dict = field(default_factory=dict)


def prepare_data(
    df: pd.DataFrame,
    config: RunConfig,
    profile: ProfileResult,
    enable_domain_features: bool = True,
) -> PreparedData:
    target, date, group = config.target_col, config.date_col, config.data_strategy.group_col

    # 1. SPLIT FIRST (before any fit/statistic) -------------------------------------------------
    df_train, df_oot, split_report = splitting.temporal_split(
        df, date, config.windows.snapshots_for_training, config.windows.oot_snapshots, target
    )
    if len(df_train) == 0 or len(df_oot) == 0:
        raise ValueError(f"empty split: train={len(df_train)} oot={len(df_oot)} — check snapshots")

    # 2. optional domain FE: regex map learned on TRAIN, same transform on OOT ------------------
    domain_created: list[str] = []
    column_map: dict = {}
    if enable_domain_features:
        df_train, domain_created, column_map = feature_engineer.engineer(df_train)
        df_oot, _, _ = feature_engineer.engineer(df_oot, column_map=column_map)

    ignore = {target, date, group}
    feat_cols = [c for c in df_train.columns if c not in ignore]
    numeric_cols = df_train[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feat_cols if c not in numeric_cols]

    # 3. CLEANING fit on TRAIN, applied to both -------------------------------------------------
    clean_art = cleaning.fit_clean(
        df_train, numeric_cols, categorical_cols, config.data_strategy, config.feature_selection.max_missing_pct
    )
    df_train = clean_art.transform(df_train)
    df_oot = clean_art.transform(df_oot)

    feat_cols = [c for c in df_train.columns if c not in ignore]
    numeric_cols = df_train[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feat_cols if c not in numeric_cols]

    # 4. FEATURE SELECTION on TRAIN ONLY --------------------------------------------------------
    sel_report = feature_select.select_features(
        df_train, feat_cols, target, config.feature_selection, seed=config.random_seed
    )
    selected = sel_report.selected_features
    sel_numeric = [c for c in selected if c in numeric_cols]
    sel_categorical = [c for c in selected if c in categorical_cols]

    # 5. ENCODE: fit preprocessor on TRAIN, transform both --------------------------------------
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_steps = []
    if config.data_strategy.scaling_required:
        num_steps.append(("sc", StandardScaler()))
    numeric_pipe = Pipeline(num_steps) if num_steps else "passthrough"
    cat_pipe = Pipeline(
        [("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50, min_frequency=10))]
    )
    transformers = []
    if sel_numeric:
        transformers.append(("num", numeric_pipe, sel_numeric))
    if sel_categorical:
        transformers.append(("cat", cat_pipe, sel_categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

    preprocessor.fit(df_train[sel_numeric + sel_categorical])
    X_train = np.asarray(preprocessor.transform(df_train[sel_numeric + sel_categorical]), dtype=np.float32)
    X_oot = np.asarray(preprocessor.transform(df_oot[sel_numeric + sel_categorical]), dtype=np.float32)
    feature_names = list(preprocessor.get_feature_names_out())

    y_train = df_train[target].astype(int).to_numpy()
    y_oot = df_oot[target].astype(int).to_numpy()
    groups_train = df_train[group].to_numpy() if group and group in df_train.columns else None

    # contract checks
    assert X_train.dtype != object and X_oot.dtype != object
    assert X_train.shape[1] == X_oot.shape[1] == len(feature_names)
    assert len(y_train) == X_train.shape[0]

    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_oot=X_oot,
        y_oot=y_oot,
        groups_train=groups_train,
        feature_names=feature_names,
        selected_raw_features=selected,
        cleaning_artifacts=clean_art,
        preprocessor=preprocessor,
        split_report=split_report,
        selection_report=sel_report,
        domain_features=domain_created,
        column_map=column_map,
    )
