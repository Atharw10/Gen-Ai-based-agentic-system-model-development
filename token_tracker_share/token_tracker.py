"""Standalone LLM token + cost tracker — drop into any project that calls Gemini.

Zero third-party dependencies (Python standard library only). Works with:
  * langchain-google-genai  (ChatGoogleGenerativeAI -> AIMessage with .usage_metadata)
  * the raw google-generativeai SDK (response.usage_metadata.prompt_token_count, ...)
  * any provider — fall back to record_tokens(...) with explicit numbers, or let it
    estimate from text length when no usage data is available.

The token counts are READ FROM THE PROVIDER'S RESPONSE (the provider counts them, that's what
it bills on). Only when the provider returns no usage data does this estimate as len(text)//4
(~4 chars per token) and mark the row estimated=True.

Quick use
---------
    from token_tracker import TokenTracker

    tracker = TokenTracker(use_case="Churn model", model="gemini-2.5-flash-lite")

    msg = llm.invoke([HumanMessage(content=prompt)])          # your existing call
    tracker.record(msg, prompt=prompt, resp=msg.content, label="research")

    tracker.print_summary()
    tracker.to_csv("cost_log.csv")        # creates the file if it doesn't exist; appends a row
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

# CSV column order — keep identical across every project so all rows line up in one sheet.
CSV_COLUMNS = [
    "Use Case", "Model", "Iteration ID",
    "Prompt Tokens", "Cached Tokens", "Output Tokens", "Thinking Tokens", "Total Tokens",
]

# USD per 1,000 tokens. Placeholders — verify at https://ai.google.dev/pricing.
# Override per-instance via TokenTracker(pricing={"input": .., "output": ..}) or env vars
# GEMINI_PRICE_INPUT_PER_1K / GEMINI_PRICE_OUTPUT_PER_1K.
PRICING = {
    "gemini-2.5-flash-lite": {"input": 0.00010, "output": 0.00040},
    "gemini-2.5-flash":      {"input": 0.00030, "output": 0.00250},
    "gemini-2.5-pro":        {"input": 0.00125, "output": 0.01000},
    "default":               {"input": 0.00010, "output": 0.00040},
}


def _price_for(model: str | None, override: dict | None = None) -> dict:
    if override:
        return override
    env_in = os.environ.get("GEMINI_PRICE_INPUT_PER_1K")
    env_out = os.environ.get("GEMINI_PRICE_OUTPUT_PER_1K")
    if env_in and env_out:
        return {"input": float(env_in), "output": float(env_out)}
    for k, v in PRICING.items():
        if k != "default" and model and k in model:
            return v
    return PRICING["default"]


def extract_usage(msg, prompt: str = "", resp: str = "") -> dict:
    """Pull token counts out of an LLM response object, with a length-based fallback.

    Handles LangChain AIMessage (.usage_metadata / .response_metadata) AND the raw
    google-generativeai response (.usage_metadata.prompt_token_count ...). Returns a dict with
    input_tokens / cached_tokens / output_tokens / thinking_tokens / estimated.
    """
    u, meta = {}, {}

    # LangChain AIMessage
    if hasattr(msg, "usage_metadata") and isinstance(getattr(msg, "usage_metadata"), dict):
        u = msg.usage_metadata or {}
    if hasattr(msg, "response_metadata"):
        rm = msg.response_metadata or {}
        meta = rm.get("usage_metadata") or rm.get("token_usage") or {}

    # Raw google-generativeai response: msg.usage_metadata is an object with *_token_count attrs
    raw = getattr(msg, "usage_metadata", None)
    if raw is not None and not isinstance(raw, dict):
        meta = {
            "prompt_token_count": getattr(raw, "prompt_token_count", None),
            "candidates_token_count": getattr(raw, "candidates_token_count", None),
            "cached_content_token_count": getattr(raw, "cached_content_token_count", None),
            "thoughts_token_count": getattr(raw, "thoughts_token_count", None),
        }

    it = (u.get("input_tokens") or u.get("prompt_tokens")
          or meta.get("prompt_token_count") or meta.get("input_tokens"))
    ot = (u.get("output_tokens") or u.get("candidates_tokens") or u.get("completion_tokens")
          or meta.get("candidates_token_count") or meta.get("output_tokens"))

    itd = u.get("input_token_details") or {}
    ct = (itd.get("cache_read") or itd.get("cached_tokens")
          or meta.get("cached_content_token_count") or 0)

    otd = u.get("output_token_details") or {}
    # LangChain's standard key is "reasoning" (OutputTokenDetails); the *_tokens variants and
    # thoughts_token_count are fallbacks for the raw SDK / older shapes.
    tt = (otd.get("reasoning") or otd.get("reasoning_tokens") or otd.get("thinking_tokens")
          or meta.get("thoughts_token_count") or 0)
    if tt and ot and ot >= tt:          # some providers fold thinking into output; don't double count
        ot = ot - tt

    if not it and not ot:               # provider gave nothing -> estimate (~4 chars / token)
        return {"input_tokens": len(prompt) // 4, "cached_tokens": 0,
                "output_tokens": len(resp) // 4, "thinking_tokens": 0, "estimated": True}
    return {"input_tokens": int(it or 0), "cached_tokens": int(ct or 0),
            "output_tokens": int(ot or 0), "thinking_tokens": int(tt or 0), "estimated": False}


class TokenTracker:
    """Accumulates per-call token usage for one run and writes a summary CSV row."""

    def __init__(self, use_case: str, model: str = "gemini-2.5-flash-lite",
                 iteration_id: str | None = None, verbose: bool = True, pricing: dict | None = None):
        self.use_case = use_case
        self.model = model
        # default Iteration ID = a timestamp; override with your own run id if you have one
        from datetime import datetime
        self.iteration_id = iteration_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.verbose = verbose
        self.pricing = _price_for(model, pricing)
        self.events: list[dict] = []

    # ---- recording -------------------------------------------------------------------------
    def record(self, msg, prompt: str = "", resp: str = "", label: str = "llm") -> dict:
        """Record one LLM call from its response object. Call this right after every invoke()."""
        usage = extract_usage(msg, prompt=prompt, resp=resp or getattr(msg, "content", ""))
        return self._add(usage, label)

    def record_tokens(self, input_tokens=0, output_tokens=0, cached_tokens=0,
                      thinking_tokens=0, label: str = "llm") -> dict:
        """Record one call from explicit numbers (use when you have counts but no response object)."""
        usage = {"input_tokens": int(input_tokens), "cached_tokens": int(cached_tokens),
                 "output_tokens": int(output_tokens), "thinking_tokens": int(thinking_tokens),
                 "estimated": False}
        return self._add(usage, label)

    def _add(self, usage: dict, label: str) -> dict:
        it, ct = usage["input_tokens"], usage["cached_tokens"]
        ot, tt = usage["output_tokens"], usage["thinking_tokens"]
        cost = it / 1000 * self.pricing["input"] + ot / 1000 * self.pricing["output"]
        ev = {"label": label, "input_tokens": it, "cached_tokens": ct, "output_tokens": ot,
              "thinking_tokens": tt, "total_tokens": it + ct + ot + tt,
              "estimated": usage.get("estimated", False), "est_cost_usd": round(cost, 6)}
        self.events.append(ev)
        if self.verbose:
            est = " (estimated)" if ev["estimated"] else ""
            print(f"   $ {label}: in={it} cached={ct} out={ot} think={tt} "
                  f"~${cost:.6f}{est}", flush=True)
        return ev

    # ---- summary ---------------------------------------------------------------------------
    def totals(self) -> dict:
        s = lambda k: sum(e[k] for e in self.events)
        return {
            "Use Case": self.use_case,
            "Model": self.model,
            "Iteration ID": self.iteration_id,
            "Prompt Tokens": s("input_tokens"),
            "Cached Tokens": s("cached_tokens"),
            "Output Tokens": s("output_tokens"),
            "Thinking Tokens": s("thinking_tokens"),
            "Total Tokens": s("total_tokens"),
        }

    def print_summary(self):
        t = self.totals()
        bar = "=" * 60
        print(f"\n{bar}\n  TOKEN USAGE SUMMARY\n" + "-" * 60)
        for k, v in t.items():
            print(f"  {k:<18}  {v}")
        print(f"  {'Est. cost (USD)':<18}  ~${sum(e['est_cost_usd'] for e in self.events):.6f}")
        print(bar)

    def to_csv(self, path: str = "cost_log.csv") -> str:
        """Append this run's summary row to `path`. CREATES the file (with header) if it does not
        exist — you do NOT need to have a CSV beforehand. Returns the file path."""
        p = Path(path)
        write_header = not p.exists()
        if p.parent and str(p.parent) not in ("", "."):
            p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if write_header:
                w.writeheader()
            w.writerow(self.totals())
        if self.verbose:
            print(f"  [OK] Row {'created' if write_header else 'appended'} -> {p}")
        return str(p)
