"""Research tool: data-aware recommendations for banking ML.

Modes:
  - 'tavily' : real Tavily API (requires TAVILY_API_KEY)
  - 'gemini' : Gemini grounding (uses GOOGLE_API_KEY)
  - 'stub'   : canned banking knowledge (default fallback)

KEY FEATURE: All modes now receive the data profile and adapt
recommendations based on use case + imbalance level.
"""

from __future__ import annotations
import os
import json
from typing import Dict, Optional


# ==============================================================================
# TIER DETECTION (shared helper)
# ==============================================================================
def _detect_tier(profile: Optional[dict]) -> tuple:
    """Extract pos_ratio and tier from profile."""
    if not profile:
        return None, "unknown"
    
    pr = profile.get("target", {}).get("pos_ratio")
    if pr is None:
        return None, "unknown"
    # Single source of truth — no local tier cutoffs (was the "5% boundary" inconsistency).
    from mlkit.metrics import assign_tier
    return pr, assign_tier(pr)


# ==============================================================================
# USE-CASE BASE KNOWLEDGE
# ==============================================================================
_USE_CASE_KNOWLEDGE = {
    "credit card propensity": {
        "summary": (
            "CC propensity models predict which customers are likely to apply for "
            "or accept a credit card offer. Built on monthly customer snapshots with "
            "transaction behavior, demographics, and product holdings as features."
        ),
        "key_features": [
            "credit/debit velocity ratios (3m vs 6m)",
            "MCC-based discretionary spend (dining, travel, ecommerce)",
            "transaction recency/frequency/monetary (RFM)",
            "bounce rate trends",
            "mandate/standing instruction share of debits",
            "digital channel engagement",
        ],
        "sources": [
            "Banking ML best practices - propensity modeling",
            "Industry benchmarks: ICICI/HDFC/SBI propensity playbooks",
        ]
    },
    "churn": {
        "summary": (
            "Customer churn prediction identifies customers likely to leave. "
            "Requires longer observation windows (6+ months) to capture declining engagement. "
            "Recall is critical - missing a churning customer is expensive."
        ),
        "key_features": [
            "declining transaction frequency",
            "reduced balance trends",
            "customer service complaint frequency",
            "competitor product usage signals",
        ],
        "sources": ["Telco/banking churn prediction benchmarks"],
    },
    "fraud detection": {
        "summary": (
            "Fraud detection operates on extreme imbalance (<1% positives). "
            "Requires real-time or near-real-time scoring. "
            "Every missed fraud is direct financial loss."
        ),
        "key_features": [
            "transaction amount deviation from customer average",
            "geographic anomalies",
            "velocity checks (transactions per hour)",
            "merchant category anomalies",
        ],
        "sources": ["Visa/Mastercard fraud detection papers"],
    },
    "default": {
        "summary": "Generic banking classification task with tabular data.",
        "key_features": ["transaction aggregates", "velocity ratios", "recency"],
        "sources": ["Banking ML best practices"],
    },
}


# ==============================================================================
# TIER-SPECIFIC RECOMMENDATIONS (the core adaptive logic)
# ==============================================================================
_TIER_RECOMMENDATIONS = {
    "balanced": {
        "recommended_algorithms": ["LogisticRegression", "RandomForest", "LightGBM"],
        "algorithm_reasoning": (
            "Data is balanced - simpler models perform well. "
            "LogisticRegression: strong interpretable baseline; coefficients directly explain "
            "feature importance; no class rebalancing needed; fast to train and deploy. "
            "RandomForest: captures non-linear interactions; robust to noise; built-in feature "
            "importance via impurity reduction. "
            "LightGBM: included as gradient boosting comparison but unlikely to significantly "
            "outperform simpler models on balanced data. "
            "NOT recommended: XGBoost (overkill for balanced); Neural nets (need more data, "
            "less interpretable); IsolationForest (designed for anomaly/extreme imbalance)."
        ),
        "metrics": ["ROC_AUC", "Accuracy", "F1"],
        "metric_reasoning": (
            "With balanced classes, ROC-AUC is the appropriate primary metric - it honestly "
            "measures ranking ability without the misleading inflation seen in imbalanced data. "
            "Accuracy is meaningful (unlike imbalanced data where predicting all-negative gives "
            "high accuracy). F1 provides precision-recall balance. "
            "PR-AUC is less critical since baseline PR-AUC is already ~0.50."
        ),
        "class_weight_strategy": "Not needed - classes are roughly equal",
        "smote_recommended": False,
    },
    "mild": {
        "recommended_algorithms": ["LightGBM", "LogisticRegression", "XGBoost"],
        "algorithm_reasoning": (
            "Mild imbalance - mix of simple and complex models works best. "
            "LightGBM: handles mild imbalance with is_unbalance=True; fast; good default. "
            "LogisticRegression: with class_weight='balanced', adjusts decision boundary; "
            "still interpretable; strong baseline for banking. "
            "XGBoost: alternative GBM; scale_pos_weight provides mild correction. "
            "NOT recommended: IsolationForest (for extreme imbalance only); "
            "RandomForest (slower than GBMs, less accurate on tabular data)."
        ),
        "metrics": ["F1", "ROC_AUC", "PR_AUC"],
        "metric_reasoning": (
            "F1 balances precision and recall - appropriate when imbalance is noticeable but "
            "not extreme. ROC-AUC still meaningful at this level. PR-AUC as secondary check "
            "to ensure the model doesn't just predict the majority class."
        ),
        "class_weight_strategy": "class_weight='balanced' for LogReg; is_unbalance=True for LightGBM",
        "smote_recommended": False,
    },
    "moderate": {
        "recommended_algorithms": ["LightGBM", "XGBoost"],
        "algorithm_reasoning": (
            "Moderate imbalance requires models that handle class skew natively. "
            "LightGBM: scale_pos_weight or is_unbalance=True handles the skew; "
            "fast histogram-based training on large datasets; num_leaves controls complexity. "
            "XGBoost: scale_pos_weight parameter auto-rebalances; robust tree-based model; "
            "eval_metric='aucpr' focuses optimization on the minority class. "
            "NOT recommended: LogisticRegression (linear boundaries often insufficient for "
            "moderate imbalance unless extensive WoE feature engineering is applied); "
            "RandomForest (slower, less precise than GBMs at this imbalance level)."
        ),
        "metrics": ["PR_AUC", "KS", "Gini", "F1"],
        "metric_reasoning": (
            "PR-AUC is the primary metric because ROC-AUC starts becoming misleadingly "
            "optimistic at this imbalance level - a mediocre model can still show ROC-AUC > 0.80. "
            "PR-AUC directly measures; 'among customers I target, what fraction convert?' "
            "KS (Kolmogorov-Smirnov) and Gini (2*AUC-1) are banking-standard secondary metrics "
            "for measuring score distribution separation. "
            "Accuracy is becoming meaningless - predicting all negatives gives high accuracy."
        ),
        "class_weight_strategy": "scale_pos_weight = neg_count / pos_count",
        "smote_recommended": False,
    },
    "severe": {
        "recommended_algorithms": ["LightGBM", "XGBoost"],
        "algorithm_reasoning": (
            "Severe imbalance - gradient boosting with strong class rebalancing is critical. "
            "LightGBM: with scale_pos_weight or is_unbalance=True; handles rare events well; "
            "histogram binning is robust to the sparse positive class. "
            "XGBoost: scale_pos_weight provides aggressive upweighting of minority class; "
            "tree-based splitting naturally finds the rare positive patterns. "
            "SMOTE oversampling may be applied before training to create synthetic positives. "
            "NOT recommended: LogisticRegression (linear models struggle with severe imbalance "
            "unless WoE transformation is applied); RandomForest (bootstrap sampling may miss "
            "rare positives entirely in some trees)."
        ),
        "metrics": ["PR_AUC", "Recall", "KS", "Gini"],
        "metric_reasoning": (
            "PR-AUC is essential - ROC-AUC will be misleadingly high even for bad models. "
            "Recall is secondary because missing a rare positive is expensive (e.g., missed "
            "marketing opportunity, missed risk signal). "
            "KS and Gini for banking-standard reporting. "
            "Accuracy is meaningless - predicting all negatives gives very high accuracy."
        ),
        "class_weight_strategy": "scale_pos_weight = neg_count / pos_count (typically 20-100)",
        "smote_recommended": True,
    },
    "extreme": {
        "recommended_algorithms": ["XGBoost", "LightGBM", "IsolationForest"],
        "algorithm_reasoning": (
            "Extreme imbalance - standard supervised learning alone may not suffice. "
            "XGBoost: with very high scale_pos_weight; aggressive minority upweighting. "
            "LightGBM: similar capability; try both and compare. "
            "IsolationForest: unsupervised anomaly detector - catches patterns that "
            "supervised models miss when positives are extremely rare; doesn't need labels. "
            "SMOTE/ADASYN oversampling strongly recommended before supervised training. "
            "Consider cost-sensitive learning or custom loss functions."
        ),
        "metrics": ["Recall", "PR_AUC"],
        "metric_reasoning": (
            "Recall is the primary metric - every missed positive is critical "
            "(missed fraud = financial loss, missed default = credit risk). "
            "PR-AUC as secondary. "
            "ROC-AUC, Accuracy, F1 are all meaningless at this imbalance level."
        ),
        "class_weight_strategy": "scale_pos_weight = neg/pos (can be 500-1000); consider focal loss",
        "smote_recommended": True,
    },
    "unknown": {
        "recommended_algorithms": ["LightGBM", "XGBoost"],
        "algorithm_reasoning": "Unknown imbalance - defaulting to GBMs as safe choice.",
        "metrics": ["PR_AUC", "F1"],
        "metric_reasoning": "Unknown imbalance - PR-AUC and F1 cover most scenarios.",
        "class_weight_strategy": "Use class_weight='balanced' as safe default",
        "smote_recommended": False,
    }
}


# ==============================================================================
# WINDOW RECOMMENDATIONS (data-aware)
# ==============================================================================
def _recommend_windows(profile: Optional[dict]) -> dict:
    """Recommend observation/performance/OOT windows based on actual data."""
    if not profile:
        return {
            "observation_window_months": 3,
            "performance_window_months": 1,
            "oot_months": 1,
            "window_reasoning": "No profile available - using safe defaults (3m obs, 1m perf, 1m OOT).",
        }
    
    n_snapshots = profile.get("temporal", {}).get("n_snapshots", 0)
    snapshots = profile.get("temporal", {}).get("snapshots", [])
    
    if n_snapshots <= 2:
        return {
            "observation_window_months": 1,
            "performance_window_months": 1,
            "oot_months": 1,
            "window_reasoning": (
                f"Only {n_snapshots} monthly snapshots available. With so little data, "
                "using 1-month observation, 1-month performance, and 1-month OOT. "
                "This is the minimum viable configuration - model quality will be limited."
            ),
        }
        
    if n_snapshots <= 4:
        obs = 2
        oot = 1
    elif n_snapshots <= 6:
        obs = 3
        oot = 1
    elif n_snapshots <= 9:
        obs = 3
        oot = 2
    elif n_snapshots <= 12:
        obs = 3
        oot = 2
    else:
        obs = 6
        oot = 3
        
    perf = 1
    train_months = n_snapshots - oot
    
    reasoning = (
        f"Data has {n_snapshots} monthly snapshots ({snapshots[0]} to {snapshots[-1]}). "
        f"\n\nObservation window = {obs} months: "
        f"This means features are computed using {obs} months of history per customer. "
    )
    
    if obs == 3:
        reasoning += (
            "3 months captures recent behavioral trends (spending velocity, transaction "
            "patterns) without going too far back where behavior may be stale. "
            "Shorter windows (1-2m) miss seasonal patterns; longer windows (6m+) dilute "
            "recent signals and require more historical data. "
        )
    elif obs == 6:
        reasoning += (
            "6 months captures seasonal patterns and longer-term trends. Appropriate when "
            "12+ months of data is available to support both training and OOT needs."
        )
    else:
        reasoning += (
            f"{obs} months is the maximum feasible given {n_snapshots} total snapshots "
            "while preserving enough data for training and OOT."
        )
        
    reasoning += (
        f"\n\nPerformance window = {perf} month: "
        "This answers 'did the event happen within 1 month after observation?' "
        "1 month is standard for propensity - shorter misses slow converters, "
        "longer reduces marketing actionability."
    )
    
    reasoning += (
        f"\n\nOOT = {oot} months (last {oot} snapshots): "
        f"Training uses {train_months} months, OOT uses {oot} months. "
        "OOT must be the LATEST snapshots to simulate real deployment "
        "(model trains on past, predicts on future)."
    )
    
    # Explain why NOT more or fewer OOT months
    if n_snapshots <= 6:
        reasoning += (
            f"With only {n_snapshots} snapshots, we can't afford more than {oot} month(s) "
            "for OOT - training needs enough data volume for stable cross-validation."
        )
    else:
        reasoning += (
            f"Using {oot} OOT months provides enough future data to test generalization "
            f"while preserving {train_months} months for training. "
            f"More OOT (e.g., {oot+2}) would starve the training set."
        )
        
    return {
        "observation_window_months": obs,
        "performance_window_months": perf,
        "oot_months": oot,
        "window_reasoning": reasoning
    }


# ==============================================================================
# STUB RESEARCH (fully data-aware)
# ==============================================================================
def _stub_research(query: str, profile: dict = None) -> Dict:
    """Return canned knowledge adapted to use case + imbalance."""
    q = query.lower()
    
    # # 1. Find use-case base knowledge
    base = None
    matched_use_case = "default"
    for k in _USE_CASE_KNOWLEDGE:
        if k in q:
            base = dict(_USE_CASE_KNOWLEDGE[k])
            matched_use_case = k
            break
            
    if base is None:
        base = dict(_USE_CASE_KNOWLEDGE["default"])
        
    # # 2. Detect imbalance tier
    pr, tier = _detect_tier(profile)
    
    # # 3. Get tier-specific recommendations
    tier_recs = dict(_TIER_RECOMMENDATIONS.get(tier, _TIER_RECOMMENDATIONS["unknown"]))
    
    # # 4. Get data-aware window recommendations
    windows = _recommend_windows(profile)
    
    # # 5. Combine everything
    result = {
        "mode": "stub",
        "query": query,
        "matched_use_case": matched_use_case,
        "detected_pos_ratio": pr,
        "detected_tier": tier,
        "summary": base["summary"],
        "recommended_algorithms": tier_recs["recommended_algorithms"],
        "algorithm_reasoning": tier_recs["algorithm_reasoning"],
        "windows": {
            "observation_window_months": windows["observation_window_months"],
            "performance_window_months": windows["performance_window_months"],
            "oot_months": windows["oot_months"],
        },
        "window_reasoning": windows["window_reasoning"],
        "metrics": tier_recs["metrics"],
        "metric_reasoning": tier_recs["metric_reasoning"],
        "class_weight_strategy": tier_recs.get("class_weight_strategy", ""),
        "smote_recommended": tier_recs.get("smote_recommended", False),
        "key_features": base.get("key_features", []),
        "sources": base.get("sources", []),
    }
    return result


# ==============================================================================
# GEMINI RESEARCH (data-aware prompt)
# ==============================================================================
def _gemini_research(query: str, llm, profile: dict = None) -> Dict:
    """Use Gemini to answer - with full data context for adaptive recommendations."""
    from langchain_core.messages import HumanMessage
    
    pr, tier = _detect_tier(profile)
    n_rows = profile.get("shape", {}).get("n_rows", "unknown") if profile else "unknown"
    n_cols = profile.get("shape", {}).get("n_cols", "unknown") if profile else "unknown"
    n_snaps = profile.get("temporal", {}).get("n_snapshots", "unknown") if profile else "unknown"
    snapshots = profile.get("temporal", {}).get("snapshots", []) if profile else []
    
    prompt = f"""You are a senior banking ML research assistant. Answer this query about banking ML best practices. CRITICALLY IMPORTANT: adapt ALL recommendations to the specific data context below.

Query: {query}

=== DATA CONTEXT (you MUST use this) ===
- Rows: {n_rows}
- Columns: {n_cols}
- Positive rate: {pr} -> Imbalance tier: {tier}
- Monthly snapshots available: {n_snaps} ({snapshots})

=== RULES FOR RECOMMENDATIONS ===
1. ALGORITHMS must be appropriate for the imbalance tier:
   - balanced (>40%): prefer LogisticRegression, RandomForest
   - mild (20-40%): mix of LogReg + GBMs
   - moderate (7-20%): GBMs (LightGBM, XGBoost)
   - severe (1-7%): GBMs with scale_pos_weight
   - extreme (<1%): GBMs + IsolationForest, consider SMOTE

2. WINDOWS must be feasible given {n_snaps} snapshots:
   - observation + OOT cannot exceed total snapshots
   - OOT should be 1-3 months depending on data size
   - observation typically 3 months (or less if few snapshots)

3. METRICS must match the imbalance:
   - balanced: ROC-AUC, Accuracy
   - moderate: PR-AUC, KS, Gini
   - severe/extreme: Recall, PR-AUC

Respond in JSON only:
{{
  "summary": "<2-3 sentences adapted to THIS data>",
  "recommended_algorithms": ["<algo1>", "<algo2>"],
  "algorithm_reasoning": "<WHY these algorithms for pos_ratio={{pr}} and {{n_rows}} rows>",
  "windows": {{"observation_window_months": N, "performance_window_months": N, "oot_months": N}},
  "window_reasoning": "<WHY these windows given {{n_snaps}} snapshots available>",
  "key_features": ["<feature1>", "<feature2>"],
  "metrics": ["<metric1>", "<metric2>"],
  "metric_reasoning": "<WHY these metrics for tier={{tier}}>",
  "rationale": "<overall reasoning>"
}}"""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)]).content
        text = resp.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
            
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            raise ValueError("No JSON object found")
            
        data = json.loads(text[start:end + 1])
        data["mode"] = "gemini"
        data["query"] = query
        data["detected_pos_ratio"] = pr
        data["detected_tier"] = tier
        data["sources"] = ["Gemini Flash Lite - data-aware recommendation"]
        return data
    except Exception as e:
        result = _stub_research(query, profile)
        result["mode"] = f"stub (Gemini failed: {e})"
        return result


# ==============================================================================
# TAVILY RESEARCH
# ==============================================================================
def _tavily_research(query: str, profile: dict = None) -> Dict:
    """Real web search via Tavily API."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        result = _stub_research(query, profile)
        result["mode"] = "stub (no TAVILY_API_KEY)"
        return result
        
    try:
        from tavily import TavilyClient
        pr, tier = _detect_tier(profile)
        enriched_query = f"{query} imbalance ratio {pr} tier {tier}" if pr else query
        client = TavilyClient(api_key=api_key)
        resp = client.search(query=enriched_query, max_results=5, search_depth="advanced")
        return {
            "mode": "tavily",
            "query": enriched_query,
            "detected_pos_ratio": pr,
            "detected_tier": tier,
            "summary": resp.get("answer", ""),
            "results": [
                {"title": r["title"], "url": r["url"], "snippet": r["content"][:300]}
                for r in resp.get("results", [])
            ],
            "sources": [r["url"] for r in resp.get("results", [])],
        }
    except Exception as e:
        result = _stub_research(query, profile)
        result["mode"] = f"stub (Tavily failed: {e})"
        return result


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def research(query: str, mode: str = "auto", llm=None, profile: dict = None) -> Dict:
    """Main entry. Auto-falls back. Now accepts profile for data-aware results."""
    mode = mode.lower()
    if mode == "auto":
        if os.environ.get("TAVILY_API_KEY"):
            mode = "tavily"
        elif llm is not None:
            mode = "gemini"
        else:
            mode = "stub"
            
    if mode == "tavily":
        return _tavily_research(query, profile)
    if mode == "gemini":
        if llm is None:
            return _stub_research(query, profile)
        return _gemini_research(query, llm, profile)
    return _stub_research(query, profile)