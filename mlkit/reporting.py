"""Reporting helpers - lift table, IV chart, PSI chart."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def lift_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Decile lift table - bank-standard for propensity reports."""
    df = pd.DataFrame({"score": np.asarray(y_score), "actual": np.asarray(y_true)})
    df["decile"] = pd.qcut(df["score"].rank(method="first"),
                          n_bins, labels=list(range(n_bins, 0, -1)))
    
    g = df.groupby("decile", observed=True).agg(
        n=("actual", "size"),
        positives=("actual", "sum"),
        avg_score=("score", "mean")
    ).sort_index(ascending=False)
    
    base = df["actual"].mean()
    g["response_rate"] = g["positives"] / g["n"]
    g["lift"] = g["response_rate"] / max(base, 1e-9)
    g["cum_positives"] = g["positives"].cumsum()
    g["cum_capture_rate"] = g["cum_positives"] / g["positives"].sum()
    return g.round(4)


def iv_chart(iv_df: pd.DataFrame, ax=None):
    """Horizontal bar chart of IVs with strength color-coding."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(iv_df))))
        
    colors = {"useless": "#d62728", "weak": "#ff7f0e",
              "medium": "#2ca02c", "strong": "#1f77b4",
              "suspicious": "#9467bd"}
              
    cs = [colors.get(s, "#999999") for s in iv_df["strength"][::-1]]
    ax.barh(iv_df["feature"][::-1], iv_df["IV"][::-1], color=cs)
    
    for x, label in [(0.02, "useless"), (0.10, "weak"), (0.30, "medium"), (0.50, "strong")]:
        ax.axvline(x, color="black", linestyle="--", alpha=0.3)
        
    ax.set_xlabel("Information Value")
    ax.set_title("IV per Feature")
    plt.tight_layout()
    return ax


def psi_chart(psi_df: pd.DataFrame, ax=None):
    """Bar chart of PSI per feature."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(psi_df))))
        
    colors = {"stable": "#2ca02c", "minor_drift": "#ff7f0e",
              "significant_drift": "#d62728", "unknown": "#999999"}
              
    cs = [colors.get(s, "#999999") for s in psi_df["status"][::-1]]
    ax.barh(psi_df["feature"][::-1], psi_df["PSI"][::-1], color=cs)
    
    ax.axvline(0.10, color="orange", linestyle="--", alpha=0.5)
    ax.axvline(0.25, color="red", linestyle="--", alpha=0.5)
    
    ax.set_xlabel("PSI")
    ax.set_title("Feature Stability (PSI)")
    plt.tight_layout()
    return ax