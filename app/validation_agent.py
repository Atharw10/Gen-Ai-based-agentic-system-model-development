"""Double-validation agent — run AFTER training+evaluation complete.

Combines:
  * deterministic checks  (pipeline/validation.py)  — reliable bug/anomaly detection, the source of truth
  * an optional LLM 'second opinion' (advisors/reviewer.py) — grounded, advisory, anti-hallucination

Saves `validation_report.json` + `validation_report.md` into the run folder and prints a summary.
WARN-only: it never raises / never blocks the run.
"""
from __future__ import annotations

import json
from pathlib import Path

_ICON = {"FAIL": "✗", "WARN": "!", "INFO": "i", "OK": "✓"}
_LIGHT = {"RED": "🔴 RED", "AMBER": "🟡 AMBER", "GREEN": "🟢 GREEN"}


def validate_run(run_dir, result, prepared=None, model=None, summary="",
                 cached_llm=None, context=None, verbose=True) -> dict:
    """Validate a finished run. Returns the report dict; also saves + prints it."""
    from pipeline import validation
    from advisors import reviewer

    det = validation.run_checks(result, prepared, model, run_dir=run_dir, context=context)
    rev = reviewer.review_validation(det, result, summary=summary, cached_llm=cached_llm)
    report = {"verdict": det["verdict"], "deterministic": det, "llm_review": rev}

    if run_dir:
        rd = Path(run_dir)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "validation_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        (rd / "validation_report.md").write_text(_to_md(report), encoding="utf-8")

    if verbose:
        _print(report)
    return report


def _to_md(report: dict) -> str:
    det = report["deterministic"]; rev = report["llm_review"]
    out = [f"# Validation report — verdict: {_LIGHT.get(report['verdict'], report['verdict'])}", ""]
    out.append("## Deterministic checks")
    for f in det["findings"]:
        out.append(f"- **[{f['level']}] {f['id']}** — {f['message']}")
        if f.get("evidence"):
            out.append(f"  - evidence: `{f['evidence']}`")
    out += ["", "## LLM second opinion"]
    if rev.get("mode", "").startswith("llm"):
        out += [f"- **Assessment:** {rev.get('assessment','')}",
                f"- **Hallucination check:** {rev.get('hallucination_check','')}",
                f"- **LLM trust:** {rev.get('trust','')}"]
        if rev.get("concerns"):
            out.append("- **Concerns:**")
            out += [f"  - {c}" for c in rev["concerns"]]
    else:
        out.append(f"- _{rev.get('mode', 'skipped')}_ (deterministic verdict stands)")
    return "\n".join(out)


def _print(report: dict):
    det = report["deterministic"]; rev = report["llm_review"]; W = 64
    print("\n" + "═" * W)
    print(f"  VALIDATION AGENT  —  verdict: {_LIGHT.get(report['verdict'], report['verdict'])}")
    print("─" * W)
    c = det["counts"]
    print(f"  checks: {c['FAIL']} FAIL · {c['WARN']} WARN · {c['OK']} OK")
    for f in det["findings"]:
        if f["level"] == "OK":
            continue
        print(f"   [{_ICON.get(f['level'], '?')}] {f['level']:<4} {f['id']}: {f['message']}")
    if rev.get("mode", "").startswith("llm"):
        print("─" * W)
        print(f"  LLM 2nd opinion ({rev.get('trust','')}): {rev.get('assessment','')}")
        if rev.get("hallucination_check"):
            print(f"  hallucination check: {rev.get('hallucination_check')}")
        for con in rev.get("concerns", []):
            print(f"   · concern: {con}")
    else:
        print(f"  LLM review: {rev.get('mode','skipped')}")
    print("═" * W)
