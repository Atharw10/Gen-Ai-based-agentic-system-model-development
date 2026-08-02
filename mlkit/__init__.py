"""MLKit - deterministic ML functions for the pipeline plane.

Submodules are imported lazily/independently so that using a lean module (e.g. `mlkit.metrics`)
does NOT drag in heavy optional deps like matplotlib (reporting) or any LLM client.
Import what you need explicitly, e.g.:

    from mlkit import metrics, stats          # lean, no plotting/LLM
    from mlkit import reporting               # pulls matplotlib only when you ask for it
"""

__all__ = [
    "features",
    "stats",
    "cleaning",
    "splits",
    "metrics",
    "reporting",
    "schema_validator",
    "selection",
    "cv",
]
__version__ = "2.0.0"
