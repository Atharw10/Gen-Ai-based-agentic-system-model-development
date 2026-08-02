"""Skill layer.

A "skill" is a capability = a system prompt (the role/goal) + a set of tools it may use + a
bounded reason-act loop. It runs over the existing CachedLLM text interface, so it works with the
Vertex wrapper, stays prompt-hash cached (reproducible) and CML-safe. No native function-calling
(bind_tools) is required.

Protocol (the model replies with ONE JSON object per step):
    to use a tool:  {"action": "tool", "tool": "<name>", "args": {...}}
    when finished:  {"action": "final", "result": <answer in the required schema>}

A skill with NO tools does a single LLM call (identical behaviour to a plain prompt). Degrades
gracefully: with no LLM it returns None so the caller falls back to its deterministic stub.
"""
from __future__ import annotations

from advisors import tools as toolmod
from advisors.llm import extract_json

MAX_TOOL_CALLS = 3  # reproducibility/cost ceiling: a skill calls tools at most this many times


class Skill:
    def __init__(self, name: str, system_prompt: str, tool_names=(), max_tool_calls: int = MAX_TOOL_CALLS):
        self.name = name
        self.system_prompt = system_prompt
        self.tool_names = list(tool_names)
        self.max_tool_calls = max_tool_calls
        self.last_trace: list[tuple[str, str]] = []  # [(tool, observation)] from the most recent run

    # ── prompt assembly ──────────────────────────────────────────────────────
    def _compose(self, task: str, with_tools: bool, scratch, force_final: bool) -> str:
        parts = [self.system_prompt.strip()]
        if with_tools:
            parts.append("\nYou may use these tools:\n" + toolmod.tools_spec(self.tool_names))
            parts.append(
                '\nTo CALL a tool reply with ONLY this JSON: '
                '{"action":"tool","tool":"<name>","args":{...}}\n'
                'When you have enough information reply with ONLY this JSON: '
                '{"action":"final","result": <the answer in the required schema>}'
            )
        parts.append("\nTASK:\n" + task)
        if scratch:
            obs = "\n".join(f"Observation from {t}: {o}" for t, o in scratch)
            parts.append("\nTool results so far:\n" + obs)
        if force_final:
            parts.append('\nYou have used the maximum number of tool calls. Reply now with ONLY '
                         '{"action":"final","result": ...}. Do not call any more tools.')
        return "\n".join(parts)

    # ── run ──────────────────────────────────────────────────────────────────
    def run(self, task: str, cached_llm=None, want_json: bool = True):
        """Run the skill. Returns the parsed result (or raw text if want_json=False), or None if
        there is no LLM (caller then uses its deterministic fallback)."""
        self.last_trace = []
        if cached_llm is None:
            return None

        # No tools -> single call (same as a plain prompt).
        if not self.tool_names:
            raw = cached_llm.invoke(self._compose(task, False, [], False), label=self.name)
            return extract_json(raw) if want_json else raw

        # Tools -> bounded reason-act loop.
        scratch: list[tuple[str, str]] = []
        for step in range(self.max_tool_calls + 1):
            force_final = step == self.max_tool_calls
            label = self.name if step == 0 else f"{self.name}/step{step}"
            raw = cached_llm.invoke(self._compose(task, True, scratch, force_final), label=label)
            try:
                action = extract_json(raw)
            except Exception:
                return raw if not want_json else {}  # unparseable -> let caller fall back
            if not force_final and action.get("action") == "tool":
                tool = toolmod.get_tool(action.get("tool", ""))
                args = action.get("args") or {}
                try:
                    obs = str(tool(**args)) if tool else f"(unknown tool: {action.get('tool')})"
                except Exception as e:
                    obs = f"(tool error: {type(e).__name__}: {e})"
                scratch.append((action.get("tool", "?"), obs[:800]))
                self.last_trace.append((action.get("tool", "?"), obs[:800]))
                continue
            # final (or forced): unwrap {"action":"final","result":...} or accept a bare object
            result = action.get("result", action) if isinstance(action, dict) else action
            return result if want_json else raw
        return None

    def used_tool(self, name: str) -> bool:
        return any(t == name for t, _ in self.last_trace)
