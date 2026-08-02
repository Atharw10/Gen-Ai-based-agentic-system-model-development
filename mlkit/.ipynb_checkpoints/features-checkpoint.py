"""Feature engineering for propensity models.

Three-layer column resolution:
1. User-provided column_map (explicit)
2. LLM semantic mapping (if llm provided)
3. Regex pattern fallback (always works)

Adaptive: only creates features whose source columns are resolvable.
"""

from __future__ import annotations
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# ==========================================
# FEATURE CONCEPT REGISTRY
# ==========================================
# Declares the semantic concepts FE needs, with regex hints for fallback.
FEATURE_CONCEPTS = {
    "credit_3m": {
        "regex": [r"^(credit|cr|inflow).*3.?m", r"3.?m.*(credit|cr|inflow)"],
        "desc": "Total credit amount in last 3 months",
    },
    "credit_6m": {
        "regex": [r"^(credit|cr|inflow).*6.?m", r"6.?m.*(credit|cr|inflow)"],
        "desc": "Total credit amount in last 6 months",
    },
    "debit_3m": {
        "regex": [r"^(debit|dr|outflow|spend).*3.?m", r"3.?m.*(debit|dr|outflow|spend)"],
        "desc": "Total debit amount in last 3 months",
    },
    "debit_6m": {
        "regex": [r"^(debit|dr|outflow|spend).*6.?m", r"6.?m.*(debit|dr|outflow|spend)"],
        "desc": "Total debit amount in last 6 months",
    },
    "txn_count_3m": {
        "regex": [
            r"(txn|trans|tran).*(count|cnt|num|n).*3.?m",
            r"(count|cnt|num).*(txn|trans).*3.?m",
        ],
        "desc": "Number of transactions in last 3 months",
    },
    "txn_count_6m": {
        "regex": [
            r"(txn|trans|tran).*(count|cnt|num|n).*6.?m",
            r"(count|cnt|num).*(txn|trans).*6.?m",
        ],
        "desc": "Number of transactions in last 6 months",
    },
    "total_amount_3m": {
        "regex": [r"(total|sum).*(amt|amount).*3.?m", r"amt.*3.?m"],
        "desc": "Total transaction amount in last 3 months",
    },
    "bounce_count_3m": {
        "regex": [r"bounc.*3.?m", r"3.?m.*bounc"],
        "desc": "Number of bounced transactions in last 3 months",
    },
    "mandate_pay_amount_3m": {
        "regex": [r"mandate.*3.?m", r"standing.*instr.*3.?m"],
        "desc": "Amount paid via mandate/standing instruction in last 3 months",
    },
    "mcc_dining_3m": {
        "regex": [r"(dining|restaurant|food).*3.?m", r"mcc.*dining"],
        "desc": "Spend in dining MCC in last 3 months",
    },
    "mcc_entertainment_3m": {
        "regex": [r"(entertain|movie|leisure).*3.?m", r"mcc.*entertain"],
        "desc": "Spend in entertainment MCC in last 3 months",
    },
    "mcc_travel_3m": {
        "regex": [r"(travel|hotel|airline).*3.?m", r"mcc.*travel"],
        "desc": "Spend in travel MCC in last 3 months",
    },
    "mcc_ecommerce_3m": {
        "regex": [r"(ecom|ecommerce|online).*3.?m", r"mcc.*ecom"],
        "desc": "Spend in e-commerce MCC in last 3 months",
    },
}

# ==========================================
# LAYER 1 - LLM SEMANTIC MAPPING
# ==========================================
def _llm_infer_column_map(df: pd.DataFrame, llm) -> Dict[str, Optional[str]]:
    """Ask the LLM to map each FEATURE_CONCEPT to an actual column."""
    from langchain_core.messages import HumanMessage

    cols = df.columns.tolist()
    sample_rows = df.head(3).to_dict(orient="records")
    concepts_desc = "\n".join(
        [f"- {k}: {v['desc']}" for k, v in FEATURE_CONCEPTS.items()]
    )

    prompt = f"""You are a banking ML expert. Map each propensity-modeling CONCEPT to the best-matching column in the user's dataset, or null if absent.

=== AVAILABLE COLUMNS ===
{cols}

=== SAMPLE ROWS (first 3) ===
{json.dumps(sample_rows, indent=2, default=str)}

=== CONCEPTS TO MAP ===
{concepts_desc}

Rules:
- If a concept has NO matching column in the dataset, return null for that key.
- Be conservative: if unsure, return null. Do NOT guess wildly.
- A column can match at most ONE concept.
- Column names in the dataset may be abbreviated (e.g., "cr_amt_13m" = credit_3m).

Return ONLY a JSON object with each concept name as a key and the actual column name (or null) as the value. No prose, no markdown fences.

Example output format:
{{
    "credit_3m": "cr_amt_13m",
    "debit_3m": "dr_amt_13m",
    "mcc_dining_3m": null,
    ...
}}"""

    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
            
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            raise ValueError("No JSON object in LLM response")
        mapping = json.loads(text[start : end + 1])

        # Validate: every value must be None or an actual column
        for k, v in list(mapping.items()):
            if v is not None and v not in cols:
                mapping[k] = None
        return mapping

    except Exception as e:
        print(f"⚠️ LLM column mapping failed ({e}); falling back to regex.")
        return _regex_infer_column_map(df)


# ==========================================
# LAYER 2 - REGEX FALLBACK MAPPING
# ==========================================
def _regex_infer_column_map(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Match concepts to columns using regex patterns (case-insensitive)."""
    mapping = {}
    cols = df.columns.tolist()
    taken = set()

    for concept, info in FEATURE_CONCEPTS.items():
        found = None
        for col in cols:
            if col in taken:
                continue
            for pat in info["regex"]:
                if re.search(pat, col, re.IGNORECASE):
                    found = col
                    break
            if found:
                break
        mapping[concept] = found
        if found:
            taken.add(found)
            
    return mapping


# ==========================================
# LAYER 3 - UNIFIED RESOLVER
# ==========================================
def resolve_column_map(
    df: pd.DataFrame,
    column_map: Optional[Dict] = None,
    llm = None,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """Resolve concept->column mapping using the best available method.

    Priority:
    1. User-provided column_map (validates each value exists in df)
    2. LLM semantic mapping (if llm provided)
    3. Regex fallback
    """
    cols = set(df.columns)

    if column_map is not None:
        # Validate user-provided map
        cleaned = {}
        for concept, colname in column_map.items():
            if colname is None or colname in cols:
                cleaned[concept] = colname
            else:
                if verbose:
                    print(f"⚠️ User-provided mapping '{concept}={colname}' - column not in df, set to None")
                cleaned[concept] = None
                
        # Fill missing concepts with None
        for concept in FEATURE_CONCEPTS:
            if concept not in cleaned:
                cleaned[concept] = None
        method = "user_provided"
        
    elif llm is not None:
        cleaned = _llm_infer_column_map(df, llm)
        method = "llm_inferred"
    else:
        cleaned = _regex_infer_column_map(df)
        method = "regex_fallback"

    if verbose:
        resolved = {k: v for k, v in cleaned.items() if v is not None}
        unresolved = [k for k, v in cleaned.items() if v is None]
        print(f"📋 Column mapping ({method}): resolved {len(resolved)}/{len(FEATURE_CONCEPTS)} concepts")
        if resolved:
            print("   Resolved:")
            for k, v in resolved.items():
                print(f"     {k:25s} -> {v}")
        if unresolved:
            print(f"   Unresolved (will skip): {unresolved}")
            
    return cleaned

# =====================================================================
# MAIN - ENGINEER PROPENSITY FEATURES
# =====================================================================
def engineer_propensity_features(
    df: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None,
    llm: Any = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Optional[str]]]:
    """Generates derived, domain-specific banking metrics over adaptive source variables."""
    cmap = resolve_column_map(df, column_map=column_map, llm=llm, verbose=verbose)
    out = df.copy()
    created = []

    def col(concept: str) -> Optional[str]:
        return cmap.get(concept)

    def has(*concepts: str) -> bool:
        return all(col(c) is not None for c in concepts)

    # --- Velocity (Rate of Change) ---
    if has("credit_3m", "credit_6m"):
        out["credit_velocity_3m_6m"] = out[col("credit_3m")] / out[col("credit_6m")].replace(0, np.nan)
        created.append("credit_velocity_3m_6m")

    if has("debit_3m", "debit_6m"):
        out["debit_velocity_3m_6m"] = out[col("debit_3m")] / out[col("debit_6m")].replace(0, np.nan)
        created.append("debit_velocity_3m_6m")

    # --- Behavioral Ratios ---
    if has("credit_3m", "debit_3m"):
        out["credit_debit_ratio_3m"] = out[col("credit_3m")] / out[col("debit_3m")].replace(0, np.nan)
        created.append("credit_debit_ratio_3m")

    if has("total_amount_3m", "txn_count_3m"):
        out["avg_txn_size_3m"] = out[col("total_amount_3m")] / out[col("txn_count_3m")].replace(0, np.nan)
        created.append("avg_txn_size_3m")

    if has("mandate_pay_amount_3m", "debit_3m"):
        out["mandate_share_3m"] = out[col("mandate_pay_amount_3m")] / out[col("debit_3m")].replace(0, np.nan)
        created.append("mandate_share_3m")

    if has("bounce_count_3m", "txn_count_3m"):
        out["bounce_rate_3m"] = out[col("bounce_count_3m")] / out[col("txn_count_3m")].replace(0, np.nan)
        created.append("bounce_rate_3m")

    # --- MCC-Based Discretionary Signals ---
    disc_concepts = ["mcc_entertainment_3m", "mcc_dining_3m", "mcc_travel_3m"]
    disc_cols = [col(c) for c in disc_concepts if col(c) is not None]
    
    if disc_cols and col("total_amount_3m") is not None:
        out["discretionary_share_3m"] = out[disc_cols].sum(axis=1) / out[col("total_amount_3m")].replace(0, np.nan)
        created.append("discretionary_share_3m")

    if has("mcc_ecommerce_3m", "total_amount_3m"):
        out["ecommerce_share_3m"] = out[col("mcc_ecommerce_3m")] / out[col("total_amount_3m")].replace(0, np.nan)
        created.append("ecommerce_share_3m")

    # --- Stability (Volatility Across Monthly Inflows) ---
    monthly_credits = [c for c in out.columns if re.search(r"credit.*m[1-9]", c, re.IGNORECASE)]
    if len(monthly_credits) >= 3:
        out["credit_volatility"] = out[monthly_credits].std(axis=1)
        created.append("credit_volatility")

    # --- Engagement ---
    monthly_txns = [c for c in out.columns if re.search(r"(txn|trans).*count.*m[1-9]", c, re.IGNORECASE)]
    if len(monthly_txns) >= 3:
        out["active_months"] = (out[monthly_txns] > 0).sum(axis=1)
        created.append("active_months")

    # Clean infinities resulting from division by zero calculations
    out = out.replace([np.inf, -np.inf], np.nan)

    if verbose:
        if created:
            print(f"\n✨ Created {len(created)} derived features: {created}")
        else:
            print("\nℹ️ No derived features created - required source columns absent.")

    return out, created, cmap


# =====================================================================
# STANDARD UTILITIES (Maintained Pipeline Cleanliness Functions)
# =====================================================================
def drop_high_cardinality(
    df: pd.DataFrame, max_unique: int = 50, exclude: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[Tuple[str, int]]]:
    """Drops ID-like or raw string high-variance categorical features automatically."""
    exclude_set = set(exclude or [])
    dropped = []
    
    for c in df.select_dtypes(include=["object", "category"]).columns:
        if c in exclude_set:
            continue
        nu = int(df[c].nunique())
        if nu > max_unique:
            dropped.append((c, nu))
            
    return df.drop(columns=[d[0] for d in dropped]), dropped


def drop_correlated_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: Optional[str] = None,
    threshold: float = 0.85,
) -> Tuple[List[str], List[Tuple[str, str, float, str, str]]]:
    """Drops collinear variable iterations keeping the one strongest against objective outcomes."""
    numeric_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_feats) < 2:
        return feature_cols, []

    corr = df[numeric_feats].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    target_corr = {}
    if target_col is not None and pd.api.types.is_numeric_dtype(df[target_col]):
        for c in numeric_feats:
            try:
                target_corr[c] = abs(df[c].corr(df[target_col]))
            except Exception:
                target_corr[c] = 0.0

    drop_set = set()
    pairs_dropped = []
    
    for col_name in upper.columns:
        for row_name in upper.index:
            v = upper.loc[row_name, col_name]
            if pd.notna(v) and v > threshold:
                # Retain whichever element holds greater relationship weights to label constraints
                keep = max(row_name, col_name, key=lambda x: target_corr.get(x, 0.0))
                drop = row_name if keep == col_name else col_name
                if drop not in drop_set:
                    drop_set.add(drop)
                    pairs_dropped.append((row_name, col_name, float(v), drop, keep))

    kept = [c for c in feature_cols if c not in drop_set]
    return kept, pairs_dropped


def one_hot_safe(df: pd.DataFrame, cols: List[str], max_categories: int = 20) -> pd.DataFrame:
    """Safely handles sparse encoding arrays by bundling small clusters into an internal backup category."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        top = out[c].value_counts().nlargest(max_categories).index
        out[c] = out[c].where(out[c].isin(top), other="_other_")
        
    return pd.get_dummies(out, columns=cols, drop_first=False, dummy_na=False)