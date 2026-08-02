"""Resumable, auditable HITL gate (replaces v1's blocking input()).

Each gate writes a checkpoint JSON and resolves an action via the configured mode:
  - "auto"          : accept everything (headless / CI).
  - "approval_file" : read <run_dir>/checkpoints/<stage>.approve.json if present, else accept.
  - callable        : a function(stage, decisions) -> dict for tests / custom UIs / widgets.

Action dict shape: {"action": "accept|override|regenerate", "overrides": {...}, "instructions": ""}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

Action = dict


def _accept() -> Action:
    return {"action": "accept", "overrides": {}, "instructions": ""}


class HITLGate:
    def __init__(self, run_dir: str | None = None, mode="auto"):
        self.run_dir = Path(run_dir) if run_dir else None
        self.mode = mode
        self.log: list[dict] = []
        if self.run_dir:
            (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def gate(self, stage: str, summary: str, decisions: dict) -> Action:
        checkpoint = {"stage": stage, "summary": summary, "decisions": decisions}
        if self.run_dir:
            (self.run_dir / "checkpoints" / f"{stage}.json").write_text(
                json.dumps(checkpoint, indent=2, default=str)
            )

        action = self._resolve(stage, decisions)
        self.log.append({"stage": stage, "action": action})
        if self.run_dir:
            (self.run_dir / "checkpoints" / f"{stage}.resolved.json").write_text(
                json.dumps(action, indent=2, default=str)
            )
        return action

    def _resolve(self, stage: str, decisions: dict) -> Action:
        if callable(self.mode):
            return self.mode(stage, decisions)
        if self.mode == "approval_file" and self.run_dir:
            f = self.run_dir / "checkpoints" / f"{stage}.approve.json"
            if f.exists():
                data = json.loads(f.read_text())
                return {"action": data.get("action", "accept"),
                        "overrides": data.get("overrides", {}),
                        "instructions": data.get("instructions", "")}
        return _accept()


def apply_overrides_to_config(config, overrides: dict):
    """Apply HITL overrides to a RunConfig, returning a new validated config.

    Supports dotted keys for nested fields, e.g. 'data_strategy.correlation_threshold'.
    Re-validates via pydantic, so an illegal override is rejected loudly.
    """
    if not overrides:
        return config
    data = config.model_dump()
    for key, value in overrides.items():
        target = data
        parts = key.split(".")
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value
    return config.__class__(**data)
