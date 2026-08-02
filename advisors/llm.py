"""LLM client factory + a deterministic, cached invoke wrapper.

Determinism matters: temperature=0, a fixed model id, and a prompt-hash response cache so that
re-running a design produces the same config (and doesn't re-bill the API). The cache lives on
disk under the run dir so an audit can see exactly which prompt produced which response.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def build_llm(model: str = "gemini-2.5-flash", temperature: float = 0.0):
    """Construct the chat model. Imported lazily so the deterministic plane never needs the SDK."""
    import os

    import vertexai
    from google.oauth2 import service_account
    from vertexai.generative_models import GenerativeModel

    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-south1")

    credentials = service_account.Credentials.from_service_account_file(
        cred_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    vertexai.init(project=project_id, location=location, credentials=credentials)

    class _VertexWrapper:
        """Thin wrapper so CachedLLM.invoke() works unchanged."""
        def invoke(self, messages):
            from langchain_core.messages import AIMessage
            prompt = messages[0].content if hasattr(messages[0], "content") else str(messages[0])
            resp = GenerativeModel(model).generate_content(prompt)
            msg = AIMessage(content=resp.text)
            # Map Vertex usage into the raw-Google shape that CachedLLM._usage already reads
            # (via response_metadata["usage_metadata"]). No change to _usage needed.
            um = getattr(resp, "usage_metadata", None)
            if um is not None:
                msg.response_metadata = {"usage_metadata": {
                    "prompt_token_count": int(getattr(um, "prompt_token_count", 0) or 0),
                    "candidates_token_count": int(getattr(um, "candidates_token_count", 0) or 0),
                    "cached_content_token_count": int(getattr(um, "cached_content_token_count", 0) or 0),
                    "thoughts_token_count": int(getattr(um, "thoughts_token_count", 0) or 0),
                }}
            else:
                msg.usage_metadata = {}
            return msg

    return _VertexWrapper()


class CachedLLM:
    """Wraps a chat model with a disk cache keyed by sha256(model + prompt)."""

    def __init__(self, llm: Any, cache_dir: str | None = None, model_id: str = "unknown", tracker=None):
        self.llm = llm
        self.model_id = model_id
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.tracker = tracker  # CostTracker; set by the orchestrator once run_dir is known
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model_id}\x00{prompt}".encode()).hexdigest()

    @staticmethod
    def _usage(msg, prompt: str, resp: str) -> dict:
        """Extract token usage from a LangChain AIMessage, with a length-based fallback.

        Fields returned:
          input_tokens   — prompt tokens sent to the model (excluding cached portion)
          cached_tokens  — prompt tokens served from Gemini context cache (not re-billed)
          output_tokens  — completion tokens (excluding thinking/reasoning)
          thinking_tokens — reasoning/thinking tokens (Gemini 2.5 models)
          estimated      — True when the provider gave no usage data
        """
        u = getattr(msg, "usage_metadata", None) or {}
        rm = getattr(msg, "response_metadata", {}) or {}
        meta = rm.get("usage_metadata") or rm.get("token_usage") or {}

        # ── input / prompt tokens ────────────────────────────────────────────
        it = (u.get("input_tokens") or u.get("prompt_tokens")
              or meta.get("prompt_token_count") or meta.get("input_tokens"))

        # ── cached tokens (Gemini context cache) ─────────────────────────────
        # LangChain ≥0.3: usage_metadata["input_token_details"]["cache_read"]
        # Older / raw Google API: response_metadata["usage_metadata"]["cached_content_token_count"]
        itd = u.get("input_token_details") or {}
        ct = (itd.get("cache_read") or itd.get("cached_tokens")
              or meta.get("cached_content_token_count") or 0)

        # ── output / completion tokens ────────────────────────────────────────
        ot = (u.get("output_tokens") or u.get("candidates_tokens") or u.get("completion_tokens")
              or meta.get("candidates_token_count") or meta.get("output_tokens"))

        # ── thinking / reasoning tokens (Gemini 2.5) ─────────────────────────
        otd = u.get("output_token_details") or {}
        # LangChain's standard key is "reasoning" (OutputTokenDetails); the *_tokens variants and
        # thoughts_token_count are fallbacks for raw-API / older shapes.
        tt = (otd.get("reasoning") or otd.get("reasoning_tokens") or otd.get("thinking_tokens")
              or meta.get("thoughts_token_count") or 0)
        # If output_tokens already includes thinking, subtract so we don't double-count
        if tt and ot and ot >= tt:
            ot = ot - tt

        if not it and not ot:  # provider gave nothing -> rough estimate (~4 chars/token)
            return {"input_tokens": len(prompt) // 4, "cached_tokens": 0,
                    "output_tokens": len(resp) // 4, "thinking_tokens": 0, "estimated": True}
        return {"input_tokens": int(it or 0), "cached_tokens": int(ct or 0),
                "output_tokens": int(ot or 0), "thinking_tokens": int(tt or 0), "estimated": False}

    def invoke(self, prompt: str, label: str = "llm") -> str:
        key = self._key(prompt)
        if self.cache_dir:
            hit = self.cache_dir / f"{key}.json"
            if hit.exists():
                data = json.loads(hit.read_text(encoding="utf-8"))
                if self.tracker:
                    u = data.get("usage", {})
                    self.tracker.record(label, u.get("input_tokens", 0), u.get("output_tokens", 0),
                                        cached=True, cached_tokens=u.get("cached_tokens", 0),
                                        thinking_tokens=u.get("thinking_tokens", 0))
                return data["response"]
        from langchain_core.messages import HumanMessage

        msg = self.llm.invoke([HumanMessage(content=prompt)])
        resp = msg.content
        usage = self._usage(msg, prompt, resp)
        if self.tracker:
            self.tracker.record(label, usage["input_tokens"], usage["output_tokens"],
                                cached=False, estimated=usage.get("estimated", False),
                                cached_tokens=usage.get("cached_tokens", 0),
                                thinking_tokens=usage.get("thinking_tokens", 0))
        if self.cache_dir:
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps({"prompt": prompt, "response": resp, "usage": usage}, indent=2),
                encoding="utf-8",
            )
        return resp


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response, tolerating ``` fences."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object found in LLM response")
    return json.loads(t[start : end + 1])
