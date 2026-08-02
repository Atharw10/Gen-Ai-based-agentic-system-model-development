"""RunLogger — live step progress (notebook/CLI) + a reasoning audit log (log1_cot.md).

Two jobs:
  1. step(name): prints "▶ name ... ✓ name (Xs)" so you see what's executing live in a cell.
  2. reason(stage, decision, why, source): records WHY a decision was made and WHERE it came from
     (Tavily URL / Gemini / deterministic default) into log1_cot.md + log1_reasons.json — the audit
     trail your team needs.

Non-serialisable (holds open state), so in the graph driver it lives in the registry, not in the
checkpointed state.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path


_BAR = "═" * 64
_THIN = "─" * 64


class RunLogger:
    def __init__(self, run_dir: str | None = None, verbose: bool = True):
        self.run_dir = Path(run_dir) if run_dir else None
        self.verbose = verbose
        self.events: list[dict] = []
        self.reasons: list[dict] = []
        self._step_n = 0
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def step(self, name: str, detail: str = ""):
        self._step_n += 1
        n = self._step_n
        t0 = time.time()
        if self.verbose:
            print(f"\n{_BAR}", flush=True)
            label = f"  STEP {n}  ▶  {name}"
            if detail:
                label += f"  [{detail}]"
            print(label, flush=True)
            print(_THIN, flush=True)
        try:
            yield self
        finally:
            dur = round(time.time() - t0, 2)
            self.events.append({"step": name, "seconds": dur})
            if self.verbose:
                print(f"  ✓  Completed in {dur}s", flush=True)
            self._flush()

    def info(self, msg: str):
        if self.verbose:
            print(f"     · {msg}", flush=True)

    def reason(self, stage: str, decision: str, why: str, source: str):
        """Record a decision with its justification and provenance."""
        rec = {"stage": stage, "decision": decision, "why": why, "source": source}
        self.reasons.append(rec)
        if self.verbose:
            print(f"     ↳  [{stage}]", flush=True)
            print(f"        Decision : {decision}", flush=True)
            if why:
                # wrap long why text
                why_lines = [why[i:i+55] for i in range(0, len(why), 55)]
                print(f"        Why      : {why_lines[0]}", flush=True)
                for line in why_lines[1:]:
                    print(f"                   {line}", flush=True)
            print(f"        Source   : {source}", flush=True)
        self._flush()

    def _flush(self):
        if not self.run_dir:
            return
        out = ["# Run reasoning log — `log1_cot.md`", "", "## Steps"]
        out += [f"- `{e['step']}` — {e['seconds']}s" for e in self.events]
        out += ["", "## Decisions & reasoning (with sources)"]
        for r in self.reasons:
            out += [
                f"### [{r['stage']}] {r['decision']}",
                f"- **Why:** {r['why']}",
                f"- **Source:** {r['source']}",
                "",
            ]
        (self.run_dir / "log1_cot.md").write_text("\n".join(out), encoding="utf-8")
        (self.run_dir / "log1_reasons.json").write_text(json.dumps(self.reasons, indent=2, default=str), encoding="utf-8")
