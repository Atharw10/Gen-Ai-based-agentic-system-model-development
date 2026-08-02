from __future__ import annotations
import os
import json
import time
from pathlib import Path
from datetime import datetime

"""3-stream logger:
    log1_cot.md      : structured chain-of-thought (markdown)
    log2_code.py     : all generated code blocks (python)
    log3_tools.jsonl : tool I/O records with full communication template (jsonl)
"""

class RunLogger:
    def __init__(self, run_root: str = "artifacts/runs"):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_id = ts
        self.run_dir = Path(run_root) / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.cot_path = self.run_dir / "log1_cot.md"
        self.code_path = self.run_dir / "log2_code.py"
        self.tools_path = self.run_dir / "log3_tools.jsonl"
        
        # initialize files
        self.cot_path.write_text(f"# Run {ts}\n\n", encoding="utf-8")
        self.code_path.write_text(f"# Generated code - run {ts}\n\n", encoding="utf-8")
        self.tools_path.write_text("", encoding="utf-8")
        print(f"📁 Run directory: {self.run_dir}")

    # =========================================================================
    # Log 1: structured chain-of-thought
    # =========================================================================
    def log_cot(self, stage: str, summary: str, assumptions: list = None,
                decisions: dict = None, inputs_from: list = None,
                outputs_to: list = None, ns_keys_read: list = None,
                ns_keys_written: list = None):
        """Write a structured chain-of-thought block to log1_cot.md."""
        with self.cot_path.open("a", encoding="utf-8") as f:
            f.write(f"\n## Stage: `{stage}`\n")
            f.write(f"_{datetime.now().isoformat(timespec='seconds')}_\n\n")
            
            if inputs_from:
                f.write("**Inputs from previous stages:**\n")
                for src in inputs_from:
                    f.write(f"- {src}\n")
                f.write("\n")
                
            if ns_keys_read:
                f.write(f"**NS keys read:** {', '.join(f'`{k}`' for k in ns_keys_read)}\n\n")
                
            f.write(f"**What this stage did:**\n{summary}\n\n")
            
            if decisions:
                f.write("**Decisions made:**\n")
                for k, v in decisions.items():
                    f.write(f"- `{k}` = {v}\n")
                f.write("\n")
                
            if assumptions:
                f.write("**Assumptions:**\n")
                for a in assumptions:
                    f.write(f"- {a}\n")
                f.write("\n")
                
            if outputs_to:
                f.write("**Outputs flowing to next stages:**\n")
                for tgt in outputs_to:
                    f.write(f"- {tgt}\n")
                f.write("\n")
                
            if ns_keys_written:
                f.write(f"**NS keys written:** {', '.join(f'`{k}`' for k in ns_keys_written)}\n\n")
                
            f.write("---\n")

    # =========================================================================
    # Log 2: generated code
    # =========================================================================
    def log_code(self, stage: str, code: str, attempt: int = 1):
        """Append a code block to log2_code.py."""
        with self.code_path.open("a", encoding="utf-8") as f:
            f.write(f"\n# ================= {stage} (attempt {attempt}) =================\n")
            f.write(code)
            f.write("\n\n")

    # =========================================================================
    # Log 3: tool I/O with full communication template
    # =========================================================================
    def log_tool(self, tool_name: str, inputs: dict, outputs: dict,
                 duration_sec: float = None, errors: list = None,
                 hitl_decisions: list = None, contract: dict = None):
        """Append a structured tool I/O record (one JSONL line).
        
        Schema:
        {
            run_id, ts, tool, duration_sec,
            contract: {
                reads_from_ns: [...],    # NS keys this tool reads
                writes_to_ns:  [...],    # NS keys this tool writes
                depends_on:    [...],    # previous tools/stages that must finish first
            },
            inputs: {
                from_user_overrides: {...},
                from_previous_stages: [...],
                ns_keys_read:        [...],
                parameters:          {...}
            },
            outputs: {
                key_decisions:      {...},
                new_ns_keys:         [...],
                artifacts_produced: [...],
                stdout_preview:     "..."
            },
            hitl_decisions: [...],
            errors: [...]
        }
        """
        record = {
            "run_id":         self.run_id,
            "ts":             datetime.now().isoformat(timespec="seconds"),
            "tool":           tool_name,
            "duration_sec":   duration_sec,
            "contract":       contract or {},
            "inputs":         _safe(inputs),
            "outputs":        _safe(outputs),
            "hitl_decisions": hitl_decisions or [],
            "errors":         errors or [],
        }
        with self.tools_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


# Convenience: time a tool call
class Timer:
    def __init__(self):
        self.t0 = time.time()
        
    def stop(self):
        return round(time.time() - self.t0, 2)


def _safe(obj):
    """Make any object JSON-serializable (truncate large ones)."""
    try:
        s = json.dumps(obj, default=str)
        if len(s) > 50_000:
            return {"_truncated": True, "preview": s[:1000]}
        return obj
    except Exception:
        return {"_error_serializing": str(obj)[:500]}