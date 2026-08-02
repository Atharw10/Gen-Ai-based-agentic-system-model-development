"""LLM token + cost accounting: CachedLLM meters every call; cache hits cost $0."""
from __future__ import annotations

import json
from pathlib import Path

from advisors.llm import CachedLLM
from runtime.cost import CostTracker, price_for


class _FakeMsg:
    def __init__(self, content, usage):
        self.content = content
        self.usage_metadata = usage
        self.response_metadata = {}


class _FakeChat:
    """Stand-in for ChatGoogleGenerativeAI returning LangChain-style usage_metadata."""

    def __init__(self, content="ok", usage=None):
        self.content = content
        self.usage = usage if usage is not None else {"input_tokens": 100, "output_tokens": 25}
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return _FakeMsg(self.content, self.usage)


def test_cost_math():
    t = CostTracker(model_id="gemini-2.5-flash-lite")
    p = price_for("gemini-2.5-flash-lite")
    ev = t.record("research", 1000, 1000)
    assert ev["est_cost_usd"] == round(p["input"] + p["output"], 6)
    assert t.totals()["total_tokens"] == 2000


def test_cached_llm_meters_tokens_and_cost(tmp_path):
    chat = _FakeChat(usage={"input_tokens": 100, "output_tokens": 25})
    tracker = CostTracker(run_dir=str(tmp_path), model_id="gemini-2.5-flash-lite")
    llm = CachedLLM(chat, cache_dir=str(tmp_path / "cache"), model_id="gemini-2.5-flash-lite", tracker=tracker)

    llm.invoke("hello world", label="research")
    assert chat.calls == 1
    ev = tracker.events[0]
    assert ev["label"] == "research" and ev["input_tokens"] == 100 and ev["output_tokens"] == 25
    assert ev["cached"] is False and ev["est_cost_usd"] > 0

    # same prompt again -> cache hit -> no API call, $0
    llm.invoke("hello world", label="research")
    assert chat.calls == 1  # not called again
    assert tracker.events[1]["cached"] is True and tracker.events[1]["est_cost_usd"] == 0.0

    # per-run files written
    assert (tmp_path / "cost_log.jsonl").exists()
    summ = json.loads((tmp_path / "cost_summary.json").read_text())
    assert summ["api_calls"] == 1 and summ["cache_hits"] == 1 and summ["total_tokens"] == 250


def test_missing_metadata_falls_back_to_estimate(tmp_path):
    chat = _FakeChat(content="some response text", usage={})  # no usage metadata
    tracker = CostTracker(run_dir=str(tmp_path), model_id="gemini-2.5-flash-lite")
    llm = CachedLLM(chat, cache_dir=str(tmp_path / "c"), model_id="gemini-2.5-flash-lite", tracker=tracker)
    llm.invoke("a fairly long prompt " * 5, label="design")
    ev = tracker.events[0]
    assert ev["tokens_estimated"] is True and ev["total_tokens"] > 0


def test_offline_writes_zero_cost_summary(tmp_path):
    CostTracker(run_dir=str(tmp_path), model_id="none")
    summ = json.loads((tmp_path / "cost_summary.json").read_text())
    assert summ["calls"] == 0 and summ["est_cost_usd"] == 0.0
