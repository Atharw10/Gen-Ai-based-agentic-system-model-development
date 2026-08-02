"""Adaptive metric selection - choose the right metric for the problem type."""
from __future__ import annotations
import numpy as np


def select_primary_metric(business_objective: str, imbalance_tier: str,
                          is_binary: bool, problem_type: str = "classification"):
    """Pick the primary ranking metric based on business context + imbalance.

    Returns: dict(primary, secondary, rationale)
    """
    bo = (business_objective or "").lower()

    # Regression
    if problem_type == "regression":
        return {"primary": "R2", "secondary": "RMSE",
                "rationale": "Default regression metrics"}

    # Multi-class
    if not is_binary:
        return {"primary": "F1", "secondary": "Accuracy",
                "rationale": "Multi-class - use macro-F1"}

    # Business-objective priority
    if any(k in bo for k in ["fraud", "anomaly", "default", "churn"]):
        return {"primary": "Recall", "secondary": "PR_AUC",
                "rationale": "Recall-first: high cost of false negatives in fraud/default/churn"}
    if any(k in bo for k in ["credit", "risk", "scoring"]):
        return {"primary": "KS", "secondary": "Gini",
                "rationale": "KS/Gini: bank-standard discrimination metrics for credit scoring"}
    if any(k in bo for k in ["propensity", "marketing", "response", "conversion", "upsell"]):
        return {"primary": "PR_AUC", "secondary": "F1",
                "rationale": "PR-AUC + lift: marketing/propensity cares about top-decile precision"}
    if any(k in bo for k in ["precision", "spam"]):
        return {"primary": "Precision", "secondary": "F1",
                "rationale": "Precision-first: false positives expensive"}

    # Fall back on imbalance tier
    fallback = {
        "balanced": ("ROC_AUC", "Symmetric classes"),
        "mild": ("F1", "Mild imbalance, F1 balances P/R"),
        "moderate": ("PR_AUC", "PR-AUC honest for imbalance"),
        "severe": ("PR_AUC", "PR-AUC + recall priority for rare positives"),
        "extreme": ("Recall", "Catch every rare positive"),
    }
    p, r = fallback.get(imbalance_tier, ("F1", "Default"))
    return {"primary": p, "secondary": "F1", "rationale": r}


def detect_imbalance_tier(y) -> dict:
    """Determine imbalance tier from y values."""
    y = np.asarray(y)
    uniq, counts = np.unique(y, return_counts=True)
    if len(uniq) < 2:
        return {"tier": "degenerate", "is_binary": False}
    if len(uniq) > 2:
        return {"tier": "multiclass", "is_binary": False, "n_classes": int(len(uniq))}
    
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    pr = pos / (pos + neg)
    tier = ("balanced" if pr >= 0.40 else "mild" if pr >= 0.20 else
            "moderate" if pr >= 0.05 else "severe" if pr >= 0.01 else "extreme")
    
    return {
        "is_binary": True, "n_classes": 2,
        "pos": pos, "neg": neg, "pos_ratio": float(pr),
        "tier": tier, "scale_pos_weight": float(neg / max(pos, 1))
    }