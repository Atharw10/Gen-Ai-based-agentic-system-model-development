"""Model Design tool - produces the modeling strategy from profile + research.

Inputs:  profile (from profiler.py) + research output + business objective
Output: model_design.json (windows, OOT, algos, imputation, etc.)
"""

from __future__ import annotations
import json
from typing import Dict
from . import research as research_tool

MODEL_DESIGN_PROMPT = """You are a SENIOR banking ML strategist. Design a modeling strategy
for the user's propensity problem using the data profile and research findings below.
You must produce a COMPLETE, VALID JSON object - no prose, no markdown fences.

=== USER OBJECTIVE ===
{business_objective}

=== DATA PROFILE (what the data actually looks like) ===
{profile_json}

=== RESEARCH FINDINGS (banking industry best practices) ===
{research_json}

=== YOUR TASK ===
Decide the modeling strategy and return ONE JSON object matching this exact schema:

{{
  "problem_framing": "<1 sentence problem statement>",
  "windows": {{
    "observation_window_months": <int>,
    "performance_window_months": <int>,
    "snapshots_for_training": [<list of YYYY-MM strings from available_snapshots>],
    "oot_snapshots":           [<list of YYYY-MM strings - typically the latest 1-2>],
    "oot_rationale": "<why these snapshots are OOT>"
  }},
  "data_strategy": {{
    "split_method": "temporal_OOT_by_snapshot",
    "imputation_numeric": "<median|mean|zero|knn>",
    "imputation_categorical": "<mode|constant>",
    "outlier_strategy": "<winsorize_99|iqr_safe|none>",
    "scaling_required": <true|false>,
    "woe_transformation": <true|false>,
    "correlation_threshold": <float between 0.80 and 0.95>,
    "psi_threshold": <float, default 0.25>,
    "imbalance_tier": "<balanced|mild|moderate|severe|extreme>"
  }},
  "algorithm_shortlist": [<list of 1-3 algorithm names from: LogisticRegression, RandomForest, XGBoost, LightGBM, GradientBoosting>],
  "rationale": "<2-3 sentence explanation of WHY these choices given the data + research>",
  "expected_metric_targets": {{
    "primary_metric": "<PR_AUC|ROC_AUC|KS|F1|Recall>",
    "target_value": "<numeric target, e.g., 0.20 for PR_AUC>"
  }},
  "research_sources": [<list of source strings from research findings>]
}}"""


def _validate_design(design: Dict, profile: Dict):
    """Sanity-check the LLM's design JSON. Raises on hard violations."""
    required = ["problem_framing", "windows", "data_strategy",
                "algorithm_shortlist", "rationale", "expected_metric_targets"]
    for k in required:
        if k not in design:
            raise KeyError(f"Missing required field: {k}")
            
    if not isinstance(design["algorithm_shortlist"], list) or not design["algorithm_shortlist"]:
        raise ValueError("algorithm_shortlist must be a non-empty list")
        
    train_snapshots = set(design["windows"].get("snapshots_for_training", []))
    oot_snapshots = set(design["windows"].get("oot_snapshots", []))
    if train_snapshots & oot_snapshots:
        raise ValueError(f"OOT snapshots overlap with training: {train_snapshots & oot_snapshots}")


def _fallback_design(profile: Dict, business_objective: str, research: Dict) -> Dict:
    """Construct a sensible default design when LLM keeps failing."""
    snapshots = profile.get("temporal", {}).get("snapshots", [])
    train_snapshots = snapshots[:-1] if len(snapshots) > 1 else snapshots
    oot_snapshots = snapshots[-1:] if len(snapshots) > 1 else []
    
    from mlkit.metrics import assign_tier
    pr = profile.get("target", {}).get("pos_ratio", 0.10)
    tier = assign_tier(pr)  # single source of truth

    return {
        "problem_framing": f"binary propensity for: {business_objective}",
        "windows": {
            "observation_window_months": 3,
            "performance_window_months": 1,
            "snapshots_for_training": train_snapshots,
            "oot_snapshots": oot_snapshots,
            "oot_rationale": "Latest snapshot held out as OOT (fallback default)",
        },
        "data_strategy": {
            "split_method": "temporal_OOT_by_snapshot",
            "imputation_numeric": "median",
            "imputation_categorical": "mode",
            "outlier_strategy": "winsorize_99",
            "scaling_required": False,
            "woe_transformation": False,
            "correlation_threshold": 0.85,
            "psi_threshold": 0.25,
            "imbalance_tier": tier,
        },
        "algorithm_shortlist": ["LightGBM", "XGBoost"],
        "rationale": (
            "Fallback design - research suggested GBMs; data tier indicates "
            f"'{tier}' imbalance which GBMs handle via scale_pos_weight."
        ),
        "expected_metric_targets": {"primary_metric": "PR_AUC", "target_value": 0.20},
        "research_sources": research.get("sources", []),
    }


def design_model(
    profile: Dict,
    business_objective: str,
    llm,
    research_mode: str = "auto",
) -> Dict:
    """Run the Model Design stage.
    
    Returns:
        dict with keys: design (the JSON), research (research findings used), errors (list)
    """
    # CONSTRAINTS (strict):
    # - observation_window_months + performance_window_months must fit within available snapshots
    # - snapshots_for_training and oot_snapshots must be DISJOINT
    # - oot_snapshots must come AFTER snapshots_for_training (no future leakage)
    # - algorithm_shortlist MUST have at least 1 and at most 3 entries
    # - if pos_ratio < 0.05: imbalance_tier must be 'severe' or 'extreme'
    # - if pos_ratio in [0.05, 0.20): imbalance_tier must be 'moderate'
    # - if the data is mostly binary 0/1 flag features: scaling_required = false
    # - algorithms must match imbalance handling capability (e.g., GBMs for moderate+ imbalance)
    #
    # Return ONLY the JSON object. No explanation text.

    from langchain_core.messages import HumanMessage  # lazy: only needed when actually calling the LLM

    # Step 1: research
    query = f"banking ML best practices for: {business_objective}"
    research_out = research_tool.research(query, mode=research_mode, llm=llm, profile=profile)
    
    # Step 2: ask LLM for design
    prompt = MODEL_DESIGN_PROMPT.format(
        business_objective=business_objective,
        profile_json=json.dumps(profile, indent=2, default=str),
        research_json=json.dumps(research_out, indent=2, default=str),
    )
    
    errors = []
    design = None
    for attempt in range(1, 4):
        try:
            raw = llm.invoke([HumanMessage(content=prompt)]).content
            # Strip fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
                
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end < 0:
                raise ValueError("No JSON object in LLM response")
            design = json.loads(text[start : end + 1])
            _validate_design(design, profile)
            break
        except Exception as e:
            errors.append(f"attempt {attempt}: {e}")
            if attempt == 3:
                # Fallback: produce a sensible default design
                design = _fallback_design(profile, business_objective, research_out)
                errors.append("FALLBACK_USED")
                
    return {"design": design, "research": research_out, "errors": errors}