"""Per-run LLM token + cost accounting.

Every LLM call goes through CachedLLM.invoke (advisors/llm.py), which reports token usage here.
We record one row per call (step label, model, input/output tokens, cache hit, estimated $) to
`cost_log.jsonl` and roll up `cost_summary.json` in the run folder.

Pricing is a configurable table (USD per 1,000 tokens). ⚠️ VERIFY against current Google pricing —
these are placeholders; override via the `pricing` arg or the PRICING table / env vars. On the free
AI-Studio tier real cost is $0; this estimates what a paid/Vertex run would cost.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# USD per 1,000 tokens. Placeholders — confirm at https://ai.google.dev/pricing
PRICING = {
    "gemini-2.5-flash-lite": {"input": 0.00010, "output": 0.00040},
    "gemini-2.5-flash": {"input": 0.00030, "output": 0.00250},
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01000},
    "default": {"input": 0.00010, "output": 0.00040},
}


def price_for(model: str | None, override: dict | None = None) -> dict:
    if override:
        return override
    env_in, env_out = os.environ.get("GEMINI_PRICE_INPUT_PER_1K"), os.environ.get("GEMINI_PRICE_OUTPUT_PER_1K")
    if env_in and env_out:
        return {"input": float(env_in), "output": float(env_out)}
    for k, v in PRICING.items():
        if k != "default" and model and k in model:
            return v
    return PRICING["default"]


class CostTracker:
    def __init__(self, run_dir: str | None = None, model_id: str = "none",
                 verbose: bool = False, pricing: dict | None = None):
        self.run_dir = Path(run_dir) if run_dir else None
        self.model_id = model_id
        self.verbose = verbose
        self.pricing = price_for(model_id, pricing)
        self.events: list[dict] = []
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self._flush()  # always leave a summary, even for a $0/offline run

    def record(self, label: str, input_tokens: int = 0, output_tokens: int = 0,
               cached: bool = False, estimated: bool = False, model: str | None = None,
               cached_tokens: int = 0, thinking_tokens: int = 0) -> dict:
        p = price_for(model or self.model_id) if model else self.pricing
        it, ot = int(input_tokens or 0), int(output_tokens or 0)
        ct, tt = int(cached_tokens or 0), int(thinking_tokens or 0)
        cost = 0.0 if cached else (it / 1000 * p["input"] + ot / 1000 * p["output"])
        total = it + ct + ot + tt
        ev = {"label": label, "model": model or self.model_id,
              "input_tokens": it, "cached_tokens": ct,
              "output_tokens": ot, "thinking_tokens": tt,
              "total_tokens": total, "cached": cached,
              "tokens_estimated": estimated, "est_cost_usd": round(cost, 6)}
        self.events.append(ev)
        if self.verbose:
            tag = "(cache hit, $0)" if cached else (f"~${cost:.6f}" + (" est" if estimated else ""))
            print(f"   $ {label}: in={it} cached={ct} out={ot} think={tt} {tag}", flush=True)
        self._flush()
        return ev

    def totals(self) -> dict:
        api = [e for e in self.events if not e["cached"]]
        return {
            "model": self.model_id,
            "pricing_per_1k_usd": self.pricing,
            "calls": len(self.events),
            "api_calls": len(api),
            "cache_hits": sum(1 for e in self.events if e["cached"]),
            "input_tokens": sum(e["input_tokens"] for e in self.events),
            "cached_tokens": sum(e.get("cached_tokens", 0) for e in self.events),
            "output_tokens": sum(e["output_tokens"] for e in self.events),
            "thinking_tokens": sum(e.get("thinking_tokens", 0) for e in self.events),
            "total_tokens": sum(e["total_tokens"] for e in self.events),
            "est_cost_usd": round(sum(e["est_cost_usd"] for e in self.events), 6),
            "note": "estimate only; $0 on free AI-Studio tier. Verify pricing table.",
        }

    def _flush(self):
        if not self.run_dir:
            return
        with open(self.run_dir / "cost_log.jsonl", "w") as f:
            for e in self.events:
                f.write(json.dumps(e) + "\n")
        (self.run_dir / "cost_summary.json").write_text(json.dumps(self.totals(), indent=2), encoding="utf-8")
