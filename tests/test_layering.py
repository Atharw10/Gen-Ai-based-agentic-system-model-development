"""Layering guard: the deterministic plane must never import an LLM.

This is what *enforces* reproducibility/MRM-compliance, rather than merely hoping for it.
`pipeline/` and `mlkit/` may not import any LLM SDK or the `advisors/` package.
"""
from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
FORBIDDEN_PREFIXES = (
    "advisors",
    "langchain",
    "langchain_core",
    "langchain_google_genai",
    "langgraph",
    "openai",
    "google.generativeai",
    "tavily",
)
GUARDED_PACKAGES = ("pipeline", "mlkit", "contracts", "dataio")

# Known v1 legacy violations, emptied as the refactor lands. Now empty — the deterministic
# plane is fully LLM-free (the v1 LLM column-mapper was removed from mlkit/features.py).
LEGACY_ALLOWLIST: set[str] = set()


def _imports(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def test_deterministic_plane_has_no_llm_imports():
    violations: list[str] = []
    for pkg in GUARDED_PACKAGES:
        pkg_dir = ROOT / pkg
        if not pkg_dir.exists():
            continue
        for py in pkg_dir.rglob("*.py"):
            rel = str(py.relative_to(ROOT))
            if ".ipynb_checkpoints" in py.parts or rel in LEGACY_ALLOWLIST:
                continue
            for mod in _imports(py):
                if any(mod == p or mod.startswith(p + ".") for p in FORBIDDEN_PREFIXES):
                    violations.append(f"{py.relative_to(ROOT)} imports forbidden '{mod}'")
    assert not violations, "Layering violations:\n" + "\n".join(violations)
