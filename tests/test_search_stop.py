"""Feature (b): the agent decides when to stop searching, with guardrails."""
from __future__ import annotations

from advisors import search as search_advisor


class _FakeLLM:
    """Minimal cached-LLM stand-in: .invoke(prompt) -> str."""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def invoke(self, prompt: str, label: str = "llm") -> str:
        self.calls += 1
        return self.response


def test_hard_cap_always_stops():
    cont, note = search_advisor.should_continue(
        improvement=1.0, rounds_done=2, max_rounds=2, leaderboard_summary="x",
        cached_llm=_FakeLLM('{"continue": true}'),
    )
    assert cont is False and "hard cap" in note


def test_offline_is_single_round():
    cont, note = search_advisor.should_continue(
        improvement=float("inf"), rounds_done=1, max_rounds=3, leaderboard_summary="x", cached_llm=None,
    )
    assert cont is False and "offline" in note


def test_llm_can_stop_early():
    llm = _FakeLLM('{"continue": false, "reason": "plateaued"}')
    cont, note = search_advisor.should_continue(
        improvement=0.001, rounds_done=1, max_rounds=4, leaderboard_summary="LGBM=0.29", cached_llm=llm,
    )
    assert cont is False and "plateaued" in note and llm.calls == 1


def test_llm_can_continue():
    llm = _FakeLLM('{"continue": true, "reason": "still improving"}')
    cont, note = search_advisor.should_continue(
        improvement=0.03, rounds_done=1, max_rounds=4, leaderboard_summary="LGBM=0.29", cached_llm=llm,
    )
    assert cont is True and "still improving" in note


def test_llm_failure_falls_back_to_improvement_rule():
    llm = _FakeLLM("not json at all")
    # improvement above threshold -> deterministic fallback says continue
    cont, note = search_advisor.should_continue(
        improvement=0.05, rounds_done=1, max_rounds=4, leaderboard_summary="x", cached_llm=llm, min_improve=0.005,
    )
    assert cont is True and "llm_failed" in note
