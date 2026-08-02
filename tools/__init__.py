"""Agentic tools layer (v1). Submodules import independently to avoid forcing optional
deps (langchain, IPython) when only one tool is needed. Import explicitly, e.g.:

    from tools import research          # no langchain/IPython needed
    from tools import model_design      # pulls langchain lazily inside its functions
"""

__all__ = ["profiler", "research", "hitl", "model_design", "logger"]
