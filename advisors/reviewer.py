"""LLM 'second opinion' for the validation agent (advisory, grounded).

Given the DETERMINISTIC findings + the actual metrics + the model's own narrative, the LLM writes a
short plain-English assessment, explicitly cross-checks the narrative for hallucinated numbers, and
lists up to a few concerns. It is told NOT to invent problems — it may only comment on the evidence
shown. Deterministic rules remain the source of truth for the verdict; this layer adds readability
and catches subtle issues (and primary-agent hallucinations).

Returns a dict; degrades gracefully (no LLM / any error -> a 'skipped' note, never raises).
"""
from __future__ import annotations


def review_validation(det_report: dict, result, summary: str = "", cached_llm=None) -> dict:
    if cached_llm is None:
        return {"mode": "skipped (no LLM)", "assessment": "", "hallucination_check": "",
                "concerns": [], "trust": det_report.get("verdict", "").lower()}
    try:
        from advisors.skills import Skill

        oot = result.leaderboard[0].get("oot_metrics", {}) if result.leaderboard else {}
        findings = [{"level": f["level"], "id": f["id"], "message": f["message"]}
                    for f in det_report.get("findings", [])]
        review_skill = Skill(
            "validation_review",
            "You are an ML model-validation reviewer. You are given (a) automated deterministic check "
            "results, (b) the actual out-of-time (OOT) metrics, and (c) the model's own written "
            "summary. Tasks: (1) Give a 2-3 sentence second opinion on whether these results look "
            "trustworthy. (2) HALLUCINATION CHECK: does the summary state any number that disagrees "
            "with the actual OOT metrics? Quote the mismatch if so, else say 'none'. (3) List up to 3 "
            "concerns. Do NOT invent problems that the data does not support. "
            'Return ONLY JSON: {"assessment": "...", "hallucination_check": "...", '
            '"concerns": ["..."], "trust": "green|amber|red"}',
        )
        task = (f"Automated checks: {findings}\nActual OOT metrics: {oot}\n"
                f"Best algorithm: {result.best_algorithm}; primary metric: {result.primary_metric}\n"
                f"Model summary text: \"\"\"{(summary or '')[:1200]}\"\"\"")
        j = review_skill.run(task, cached_llm) or {}
        trust = str(j.get("trust", det_report.get("verdict", ""))).lower()
        return {"mode": "llm", "assessment": j.get("assessment", ""),
                "hallucination_check": j.get("hallucination_check", ""),
                "concerns": j.get("concerns", []) or [], "trust": trust}
    except Exception as e:
        return {"mode": f"llm_failed: {e}", "assessment": "", "hallucination_check": "",
                "concerns": [], "trust": det_report.get("verdict", "").lower()}
