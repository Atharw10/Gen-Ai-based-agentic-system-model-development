"""MLKit - Standard ML library for the agentic banking propensity pipeline.

All standard logic lives here so prompts don't have to regenerate it.
Prompts call these functions by name; LLM only handles orchestration.

Usage in prompts:
    from mlkit import features, stats, cleaning, splits, metrics, reporting, schema_validator
"""

from . import features, stats, cleaning, splits, metrics, reporting, schema_validator

__all__ = ["features", "stats", "cleaning", "splits", "metrics", "reporting", "schema_validator"]
__version__ = "1.1.0"