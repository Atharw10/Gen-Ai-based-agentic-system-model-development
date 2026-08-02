"""Research advisor: produces a structured research PLAN that drives real decisions.

Every section carries its own reasoning + source so the orchestrator can log WHY a choice was
made and WHERE it came from (web via the web_search tool / Gemini / deterministic default).

Plan shape:
{
  "mode", "summary", "detected_tier", "sources": [...],
  "models":           {"recommended": [...], "reason": str, "source": str},
  "windows":          {"observation_window_months": int, "oot_months": int, "reason": str, "source": str},
  "hpo":              {"spaces": {algo: {param: [lo, hi]}}, "reason": str, "source": str},
  "feature_families": {"families": [{"name", "reason"}], "reason": str, "source": str},
  "recommended_algorithms": [...],   # back-compat alias of models.recommended
}

With no LLM, returns a deterministic plan (source='deterministic default (no LLM)'). With a key,
research runs as a SKILL: Gemini fills the plan and may call tools (web_search, imbalance_tier,
positive_rate_drift) inside a bounded reason-act loop before returning the final plan.
"""
from __future__ import annotations

from contracts import ProfileResult
from mlkit.metrics import assign_tier

_USE_CASE = {
    "credit card": "CC propensity: predict acceptance of a card offer from monthly customer "
    "snapshots (transaction behaviour, demographics, product holdings).",
    "churn": "Churn: identify customers likely to leave; longer observation windows; recall matters.",
    "fraud": "Fraud: extreme imbalance (<1%); near-real-time scoring; every miss is a direct loss.",
    "default": "Generic banking tabular classification.",
}
_TIER_ALGO = {
    "balanced": ["LogisticRegression", "RandomForest", "LightGBM"],
    "mild": ["LightGBM", "LogisticRegression"],
    "moderate": ["LightGBM", "XGBoost"],
    "severe": ["LightGBM", "XGBoost"],
    "extreme": ["XGBoost", "LightGBM"],
    "unknown": ["LightGBM", "LogisticRegression"],
}
_FAMILIES = {
    "credit card": ["transaction RFM (recency/frequency/monetary)", "credit/debit velocity ratios",
                    "MCC discretionary spend shares", "bounce/mandate behaviour", "digital engagement"],
    "churn": ["declining transaction frequency", "balance trend", "complaint frequency"],
    "fraud": ["amount deviation from customer mean", "velocity per hour", "geo/merchant anomalies"],
    "credit risk": ["repayment history", "utilisation ratios", "delinquency buckets", "exposure trend"],
    "default": ["transaction aggregates", "velocity ratios", "recency"],
}

# Explicit model_type (dropdown) -> internal use-case key. Bypasses fuzzy text matching when given.
_MODEL_TYPE_USECASE = {
    "propensity": "credit card",
    "cross_sell": "credit card",
    "up_sell": "credit card",
    "churn": "churn",
    "fraud": "fraud",
    "credit_risk": "credit risk",
    "generic": "default",
}

_USE_CASE["credit risk"] = (
    "Credit risk: predict probability of default/delinquency from monthly snapshots; "
    "PR-AUC/KS matter; longer observation windows."
)


def _use_case(objective: str, model_type: str | None = None) -> str:
    # Explicit model_type wins (reliable); otherwise fall back to the legacy substring match.
    if model_type and model_type in _MODEL_TYPE_USECASE:
        return _MODEL_TYPE_USECASE[model_type]
    o = (objective or "").lower()
    return next((k for k in _USE_CASE if k in o), "default")


def _default_windows(profile: ProfileResult) -> tuple[int, int, str]:
    from mlkit.stats import pos_rate_drift

    n = profile.temporal.n_snapshots or 0
    oot = 2 if n >= 6 else 1
    obs = 3 if n >= 4 else max(1, n - oot)
    reason = (f"{n} monthly snapshots available → observation window {obs}m captures recent "
              f"behaviour; OOT held out = last {oot} snapshot(s) to simulate future deployment.")
    # factor in per-snapshot positive-rate stability (OOT is always the most recent months, but we
    # surface drift so a wider OOT/shorter look-back can be chosen and the human is warned).
    _flagged, _med, drift_note = pos_rate_drift(profile.temporal.pos_ratio_per_snapshot)
    if profile.temporal.pos_ratio_per_snapshot:
        reason += f" Positive-rate check: {drift_note}"
    return obs, oot, reason


def _stub_plan(objective: str, profile: ProfileResult, model_type: str | None = None) -> dict:
    uc = _use_case(objective, model_type)
    tier = assign_tier(profile.target.pos_ratio)
    models = _TIER_ALGO.get(tier, _TIER_ALGO["unknown"])
    obs, oot, wreason = _default_windows(profile)
    src = "deterministic default (no LLM)"
    return {
        "mode": "stub",
        "summary": _USE_CASE[uc],
        "detected_tier": tier,
        "sources": [src],
        "models": {"recommended": models,
                   "reason": f"Tier '{tier}' (pos_ratio={profile.target.pos_ratio}): gradient boosting "
                             "handles class skew natively; linear baseline kept when classes are closer.",
                   "source": src},
        "windows": {"observation_window_months": obs, "oot_months": oot, "reason": wreason, "source": src},
        "hpo": {"spaces": {}, "reason": "No research HPO plan; using built-in default ranges.", "source": src},
        "feature_families": {"families": [{"name": f, "reason": "standard for this use case"}
                                          for f in _FAMILIES.get(uc, _FAMILIES["default"])],
                             "reason": "Canonical feature families for the detected use case.", "source": src},
        "recommended_algorithms": models,
    }


# ── research SKILL: role + JSON schema, plus the tools it may call ────────────
_RESEARCH_SYSTEM = (
    "You are a senior banking ML research assistant. Produce a RESEARCH PLAN as JSON for the given "
    "problem. You may use web_search to ground your choices in current best practice, and "
    "imbalance_tier / positive_rate_drift to reason about the data. When choosing "
    "observation_window_months and oot_months, prefer a recent window whose positive rate is "
    "stable; avoid spanning months whose positive rate differs sharply from the recent norm, and "
    "note in 'windows.reason' if the OOT period looks unusual.\n"
    "The FINAL result must be ONLY this JSON (concise reasons; valid algorithm names from "
    "LogisticRegression/RandomForest/XGBoost/LightGBM/GradientBoosting/HistGradientBoosting):\n"
    '{"summary": "...",'
    ' "models": {"recommended": ["..."], "reason": "..."},'
    ' "windows": {"observation_window_months": N, "oot_months": N, "reason": "..."},'
    ' "hpo": {"spaces": {"LightGBM": {"learning_rate": [0.01,0.2], "num_leaves": [15,255]}}, "reason": "..."},'
    ' "feature_families": {"families": [{"name":"...","reason":"..."}], "reason": "..."}}'
)


def _research_skill():
    from advisors.skills import Skill

    return Skill("research", _RESEARCH_SYSTEM,
                 tool_names=("web_search", "imbalance_tier", "positive_rate_drift"))


def _llm_plan(objective: str, profile: ProfileResult, cached_llm, model_type: str | None = None) -> dict:
    from mlkit.stats import pos_rate_drift

    tier = assign_tier(profile.target.pos_ratio)
    pmr = profile.temporal.pos_ratio_per_snapshot
    _flagged, _med, drift_note = pos_rate_drift(pmr)
    # The profile summary + positive rate (+ per-snapshot rate + drift) go to the LLM as the task,
    # so the plan is grounded in THIS data.
    task = (
        f"Objective: {objective}. Positive rate: {profile.target.pos_ratio} (imbalance tier: {tier}). "
        f"Monthly snapshots: {profile.temporal.n_snapshots}."
        + (f" Model type: {model_type}." if model_type else "")
        + (f" Positive rate per snapshot: {pmr}. Drift check: {drift_note}" if pmr else "")
    )
    skill = _research_skill()
    j = skill.run(task, cached_llm)  # bounded tool loop -> final plan JSON
    if not isinstance(j, dict) or not j:
        raise ValueError("research skill returned no usable plan")
    used_web = skill.used_tool("web_search")
    src = "Gemini (skill)" + (" + web" if used_web else "")
    plan = _stub_plan(objective, profile, model_type)  # valid defaults; overlay ONLY genuine LLM sections
    plan["mode"] = "llm"
    plan["summary"] = j.get("summary", plan["summary"])

    # INTEGRITY RULE: a section is marked Source=Gemini ONLY if the LLM actually supplied that
    # section WITH its own non-empty reason. Otherwise the deterministic section (and its honest
    # 'deterministic default' source) is kept — we never stamp a hardcoded reason as the model's.
    used_llm = False
    for key in ("models", "windows", "hpo", "feature_families"):
        sec = j.get(key)
        if isinstance(sec, dict) and str(sec.get("reason", "")).strip():
            plan[key] = {**plan[key], **sec, "source": src}
            used_llm = True
    plan["sources"] = [src] if used_llm else plan["sources"]
    if isinstance(plan["models"].get("recommended"), list) and plan["models"]["recommended"]:
        plan["recommended_algorithms"] = plan["models"]["recommended"]
    return plan


def research(objective: str, profile: ProfileResult, cached_llm=None, mode: str = "auto",
             model_type: str | None = None) -> dict:
    """Return a structured research plan. Falls back to a deterministic plan on any failure.

    `model_type` (optional dropdown hint) selects the use-case template reliably; when None, the
    legacy substring match on `objective` is used (fully backward compatible).
    """
    if cached_llm is None or mode == "stub":
        return _stub_plan(objective, profile, model_type)
    try:
        return _llm_plan(objective, profile, cached_llm, model_type)
    except Exception as e:
        plan = _stub_plan(objective, profile, model_type)
        plan["mode"] = f"stub (llm failed: {e})"
        return plan
