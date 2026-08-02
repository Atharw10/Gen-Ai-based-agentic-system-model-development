"""Schema validator - catches column-name typos before they crash downstream code.

Use it in Cell 4 right after loading data:
    from mlkit.schema_validator import resolve_column, validate_schema
    target_col = resolve_column(df, "ordered") # auto-corrects "Ordered" / "ordered " / typos
"""

from __future__ import annotations
import difflib
import pandas as pd
from typing import Optional, List


def resolve_column(df: pd.DataFrame, name: str, raise_if_missing: bool = True) -> Optional[str]:
    """Resolve a user-supplied column name against df.columns using:
        1. Exact match
        2. Strip whitespace match
        3. Case-insensitive match
        4. Fuzzy match (suggests options if none found)
        
    Args:
        df: dataframe to search
        name: user-supplied column name
        raise_if_missing: if True, raises KeyError with suggestions; else returns None
        
    Returns:
        The resolved column name (str) or None.
    """
    cols = list(df.columns)
    name = (name or "").strip()
    
    # 1. exact
    if name in cols:
        return name
        
    # 2. strip whitespace from columns
    stripped_map = {c.strip(): c for c in cols}
    if name in stripped_map:
        return stripped_map[name]
        
    # 3. case-insensitive
    lower_map = {c.lower().strip(): c for c in cols}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
        
    # 4. fuzzy
    near = difflib.get_close_matches(name, cols, n=3, cutoff=0.6)
    msg = (f"Column '{name}' not found in dataframe.\n"
           f"Closest matches: {near}\n"
           f"All columns: {cols[:30]}{'...' if len(cols) > 30 else ''}")
    if raise_if_missing:
        raise KeyError(msg)
    print(f"⚠️ {msg}")
    return None


def validate_schema(df: pd.DataFrame, required: List[str], optional: List[str] = None):
    """Validate that required columns exist. Auto-resolves typos.
    
    Returns:
        dict {requested_name: resolved_name_or_None}
    """
    optional = optional or []
    resolved = {}
    missing_required = []
    for r in required:
        try:
            resolved[r] = resolve_column(df, r, raise_if_missing=False)
            if resolved[r] is None:
                missing_required.append(r)
        except KeyError:
            missing_required.append(r)
            resolved[r] = None
            
    for o in optional:
        resolved[o] = resolve_column(df, o, raise_if_missing=False)
        
    if missing_required:
        raise KeyError(
            f"Required columns not found (after fuzzy matching): {missing_required}\n"
            f"Available columns: {list(df.columns)[:30]}"
        )
    return resolved


def diff_schemas(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """Compare two dataframes' columns. Returns:
        {only_in_train: [...], only_in_test: [...], common: [...]}
    """
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)
    return {
        "only_in_train": sorted(train_cols - test_cols),
        "only_in_test": sorted(test_cols - train_cols),
        "common": sorted(train_cols & test_cols),
        "train_count": len(train_cols),
        "test_count": len(test_cols),
    }


