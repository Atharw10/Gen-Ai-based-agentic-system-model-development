"""Statistical functions for feature analysis and stability checks."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def pos_rate_drift(pos_ratio_per_snapshot: dict, rel_tol: float = 0.5, abs_tol: float = 0.01):
    """Flag snapshot months whose positive rate drifts strongly from the median.

    Returns (flagged_months, median, note). A month is flagged when it deviates from the median
    positive rate by more than `rel_tol` (relative) AND `abs_tol` (absolute) — so tiny wiggles are
    ignored. Used to warn before choosing OOT / observation windows over an unstable period.
    No LLM here (deterministic plane).
    """
    pmr = {k: float(v) for k, v in (pos_ratio_per_snapshot or {}).items() if v is not None}
    if len(pmr) < 2:
        return [], (next(iter(pmr.values()), None)), "too few snapshots to assess drift"
    import statistics
    med = statistics.median(pmr.values())
    flagged = [m for m, v in pmr.items()
               if med > 0 and abs(v - med) / med > rel_tol and abs(v - med) > abs_tol]
    if flagged:
        note = (f"positive-rate DRIFT in {flagged}: they differ sharply from the median "
                f"{med:.2%} — prefer a recent window with a stable positive rate for the "
                f"observation window and OOT.")
    else:
        note = f"positive rate is stable across snapshots (median {med:.2%})."
    return flagged, med, note


def _safe_bin(series: pd.Series, n_bins: int = 10) -> pd.Series:
    """Bin a series safely. Uses raw values for low-cardinality, qcut+cut otherwise."""
    s = series.dropna()
    if s.nunique() <= n_bins or not pd.api.types.is_numeric_dtype(s):
        return s.astype(str)
    try:
        b = pd.qcut(s, q=n_bins, duplicates="drop")
        if b.nunique() < 3:
            b = pd.cut(s, bins=min(5, s.nunique()), duplicates="drop")
        return b
    except Exception:
        return pd.cut(s, bins=5, duplicates="drop")


def compute_iv(feature: pd.Series, target: pd.Series, n_bins: int = 10) -> float:
    """Information Value of a single feature against a binary target."""
    df = pd.DataFrame({"x": feature, "y": target}).dropna()
    if df.empty or df["y"].nunique() < 2:
        return 0.0

    df["bin"] = _safe_bin(df["x"], n_bins)
    if df["bin"].nunique() < 2:
        return 0.0

    total_good = max((df["y"] == 0).sum(), 1)
    total_bad = max((df["y"] == 1).sum(), 1)
    grp = df.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    grp["good"] = grp["count"] - grp["sum"]
    grp["bad"] = grp["sum"]
    n = len(grp)
    grp["pg"] = (grp["good"] + 0.5) / (total_good + 0.5 * n)
    grp["pb"] = (grp["bad"] + 0.5) / (total_bad + 0.5 * n)
    grp["woe"] = np.log(grp["pg"] / grp["pb"])
    grp["iv"] = (grp["pg"] - grp["pb"]) * grp["woe"]
    return float(grp["iv"].sum())


def compute_iv_table(
    df: pd.DataFrame, target_col: str, feature_cols=None
) -> pd.DataFrame:
    """Compute IV for every feature in one call. Returns sorted DataFrame."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    rows = [
        {"feature": c, "IV": compute_iv(df[c], df[target_col])}
        for c in feature_cols
    ]
    out = pd.DataFrame(rows).sort_values("IV", ascending=False).reset_index(drop=True)

    def cat(iv):
        if iv < 0.02: return "useless"
        if iv < 0.10: return "weak"
        if iv < 0.30: return "medium"
        if iv < 0.50: return "strong"
        return "suspicious"

    out["strength"] = out["IV"].apply(cat)
    return out


def compute_woe_map(feature: pd.Series, target: pd.Series, n_bins: int = 10) -> dict:
    """Return {bin -> WoE} mapping for transforming a feature to WoE space."""
    df = pd.DataFrame({"x": feature, "y": target}).dropna()
    df["bin"] = _safe_bin(df["x"], n_bins)
    tg = max((df["y"] == 0).sum(), 1); tb = max((df["y"] == 1).sum(), 1)
    grp = df.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    grp["good"] = grp["count"] - grp["sum"]; grp["bad"] = grp["sum"]
    n = len(grp)
    grp["woe"] = np.log(((grp["good"]+0.5)/(tg+0.5*n)) / ((grp["bad"]+0.5)/(tb+0.5*n)))
    return {str(k): float(v) for k, v in grp["woe"].items()}


def woe_transform(df: pd.DataFrame, feature_cols, target_col: str, n_bins: int = 10):
    """Transform multiple features to WoE values. Returns (df_woe, woe_maps_dict)."""
    out = pd.DataFrame(index=df.index)
    maps = {}
    for c in feature_cols:
        m = compute_woe_map(df[c], df[target_col], n_bins)
        maps[c] = m
        bins = _safe_bin(df[c], n_bins).astype(str)
        out[c + "_woe"] = bins.map(m).fillna(0.0)
    return out, maps


def compute_psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """Population Stability Index between two distributions of the same feature."""
    expected = expected.dropna(); actual = actual.dropna()
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    if expected.nunique() <= n_bins or not pd.api.types.is_numeric_dtype(expected):
        # Discrete distribution comparison
        e = expected.value_counts(normalize=True).to_dict()
        a = actual.value_counts(normalize=True).to_dict()
        keys = set(e) | set(a)
        psi = 0.0
        for k in keys:
            pe = e.get(k, 1e-4); pa = a.get(k, 1e-4)
            psi += (pa - pe) * np.log(pa / pe)
        return float(psi)

    # Continuous - bin both using expected's quantiles
    try:
        cuts = pd.qcut(expected, q=n_bins, retbins=True, duplicates="drop")[1]
        cuts[0] = -np.inf; cuts[-1] = np.inf
        e_freq = pd.cut(expected, bins=cuts).value_counts(normalize=True).sort_index()
        a_freq = pd.cut(actual, bins=cuts).value_counts(normalize=True).sort_index()
        a_freq = a_freq.reindex(e_freq.index, fill_value=1e-4)
        e_freq = e_freq.replace(0, 1e-4)
        psi = ((a_freq - e_freq) * np.log(a_freq / e_freq)).sum()
        return float(psi)
    except Exception:
        return np.nan


def compute_psi_table(df: pd.DataFrame, time_col: str, feature_cols, target_col=None) -> pd.DataFrame:
    """PSI per feature across earliest vs latest time-buckets in the data."""
    df = df.sort_values(time_col)
    months = sorted(df[time_col].dt.to_period("M").unique() if pd.api.types.is_datetime64_any_dtype(df[time_col])
                    else df[time_col].unique())
    
    if len(months) < 2:
        return pd.DataFrame()
    
    earliest, latest = months[0], months[-1]
    mask_e = df[time_col].dt.to_period("M") == earliest if pd.api.types.is_datetime64_any_dtype(df[time_col]) \
             else df[time_col] == earliest
    mask_l = df[time_col].dt.to_period("M") == latest if pd.api.types.is_datetime64_any_dtype(df[time_col]) \
             else df[time_col] == latest

    feats = [c for c in feature_cols if c != target_col and c != time_col]
    rows = []
    for c in feats:
        psi = compute_psi(df.loc[mask_e, c], df.loc[mask_l, c])
        rows.append({"feature": c, "PSI": psi})
    out = pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)

    def status(p):
        if pd.isna(p): return "unknown"
        if p < 0.10: return "stable"
        if p < 0.25: return "minor_drift"
        return "significant_drift"
    out["status"] = out["PSI"].apply(status)
    return out


def compute_vif(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Variance Inflation Factor for multicollinearity diagnosis. Numeric features only."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = df[feature_cols].select_dtypes(include=[np.number]).dropna()
    rows = []
    for i, c in enumerate(X.columns):
        try:
            v = variance_inflation_factor(X.values, i)
            rows.append({"feature": c, "VIF": float(v)})
        except Exception:
            rows.append({"feature": c, "VIF": np.nan})
    out = pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    return out


def ks_statistic(y_true, y_score) -> float:
    """KS statistic - bank-standard separation metric."""
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]; neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(ks_2samp(pos, neg).statistic)


def gini_coefficient(y_true, y_score) -> float:
    """Gini = 2*AUC - 1, bank-standard reporting metric."""
    from sklearn.metrics import roc_auc_score
    try:
        return float(2 * roc_auc_score(y_true, y_score) - 1)
    except Exception:
        return np.nan