"""Profiling - pure-pandas data observation. Output feeds Model Design."""
from __future__ import annotations
import pandas as pd
import numpy as np

def profile_dataset(
    df: pd.DataFrame,
    target_col: str,
    date_col: str = None,
    business_objective: str = "",
) -> dict:
    """Produce a comprehensive profile of the dataset. NO decisions, NO cleaning.
    
    Returns a JSON-ready dict for Model Design to consume.
    """
    profile = {
        "business_objective": business_objective,
        "shape": {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])},
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }
    
    # Target analysis
    if target_col in df.columns:
        y = df[target_col].dropna()
        uniq, counts = np.unique(y, return_counts=True)
        target_dist = {str(k): int(v) for k, v in zip(uniq, counts)}
        profile["target"] = {
            "name": target_col,
            "dtype": str(df[target_col].dtype),
            "n_unique": int(y.nunique()),
            "distribution": target_dist,
            "null_pct": round(df[target_col].isna().mean() * 100, 2),
        }
        if len(uniq) == 2:
            pos = int((y == 1).sum())
            tot = int((y == 0).sum()) + pos
            profile["target"]["pos_ratio"] = round(pos / max(tot, 1), 4)
            profile["target"]["is_binary"] = True
        else:
            profile["target"]["is_binary"] = False
    else:
        profile["target"] = {"name": target_col, "error": "target_col not in df"}
        
    # Temporal analysis
    if date_col and date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce")
        valid = d.dropna()
        profile["temporal"] = {
            "date_col": date_col,
            "min_date": str(valid.min()) if len(valid) else None,
            "max_date": str(valid.max()) if len(valid) else None,
            "n_snapshots": int(valid.dt.to_period("M").nunique()),
            "snapshots": [str(s) for s in sorted(valid.dt.to_period("M").unique())],
            "rows_per_snapshot": {
                str(k): int(v) for k, v in
                valid.dt.to_period("M").value_counts().sort_index().items()
            },
            "null_pct": round(df[date_col].isna().mean() * 100, 2),
        }
    else:
        profile["temporal"] = {"date_col": None, "note": "no date column declared"}
        
    # Column types
    feat_cols = [c for c in df.columns if c not in (target_col, date_col)]
    num_cols = df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feat_cols if c not in num_cols]
    high_card = [
        c for c in cat_cols
        if df[c].nunique(dropna=True) > 50
    ]
    
    profile["columns"] = {
        "n_features": len(feat_cols),
        "numeric": num_cols,
        "categorical": cat_cols,
        "high_cardinality": high_card,
        "constant": [c for c in feat_cols if df[c].nunique(dropna=True) <= 1],
    }
    
    # Missingness
    miss_pct = (df[feat_cols].isna().mean() * 100).round(2).sort_values(ascending=False)
    profile["missingness"] = {
        "overall_pct": round(df[feat_cols].isna().mean().mean() * 100, 2),
        "top10_columns": miss_pct.head(10).to_dict(),
        "n_cols_over_50pct": int((miss_pct > 50).sum()),
    }
    
    # Quick numeric stats
    if num_cols:
        desc = df[num_cols].describe().T[["mean", "std", "min", "max"]].round(2)
        profile["numeric_summary_sample"] = desc.head(15).to_dict("index")
        
    # Quick categorical stats
    if cat_cols:
        cat_card = {c: int(df[c].nunique(dropna=True)) for c in cat_cols[:15]}
        profile["categorical_cardinality_sample"] = cat_card
        
    return profile


def print_profile(profile: dict):
    """Pretty-print profile to console (for HITL display)."""
    print("=" * 60)
    print(f"DATA PROFILE - {profile.get('business_objective', 'n/a')}")
    print("=" * 60)
    
    sh = profile["shape"]
    print(f"Shape            : {sh['n_rows']:,} rows x {sh['n_cols']} cols ({profile['memory_mb']} MB)")
    
    tgt = profile.get("target", {})
    print(f"\nTarget           : {tgt.get('name')} ({tgt.get('dtype')})")
    if "pos_ratio" in tgt:
        print(f"  Distribution   : {tgt['distribution']} -> pos_ratio = {tgt['pos_ratio']}")
    print(f"  Null %         : {tgt.get('null_pct')}%")
    
    tm = profile.get("temporal", {})
    if tm.get("date_col"):
        print(f"\nTime range       : {tm['min_date']} -> {tm['max_date']}")
        print(f"  Snapshots ({tm['n_snapshots']}): {tm['snapshots']}")
    else:
        print("\nTime column      : NONE - model design may struggle with temporal split")
        
    cols = profile["columns"]
    print(f"\nFeatures         : {cols['n_features']} ({len(cols['numeric'])} numeric, {len(cols['categorical'])} categorical)")
    if cols["high_cardinality"]:
        print(f"  High-card      : {cols['high_cardinality']}")
    if cols["constant"]:
        print(f"  Constant       : {cols['constant']}")
        
    miss = profile["missingness"]
    print(f"\nMissingness      : {miss['overall_pct']}% overall, {miss['n_cols_over_50pct']} cols >50%")
    if miss["top10_columns"]:
        print(f"  Worst cols     : {dict(list(miss['top10_columns'].items())[:5])}")
    print("=" * 60)