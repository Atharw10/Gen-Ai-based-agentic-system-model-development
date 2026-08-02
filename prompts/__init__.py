"""Prompts package - one file per stage prompt + shared system rules.

Usage:
    from prompts import NODE_PROMPT_TEMPLATES
    NODE_PROMPT_TEMPLATES['feature_engineering']  # -> the FE prompt string
"""

from .system_rules import SYSTEM_RULES
from .eda import EDA_PROMPT
from .cleaning import CLEANING_PROMPT
from .feature_engineering import FE_PROMPT
from .training import TRAINING_PROMPT
from .evaluation import EVALUATION_PROMPT
from .output import OUTPUT_PROMPT

NODE_PROMPT_TEMPLATES = {
    'eda':                 EDA_PROMPT,
    'cleaning':            CLEANING_PROMPT,
    'feature_engineering': FE_PROMPT,
    'training':            TRAINING_PROMPT,
    'evaluation':          EVALUATION_PROMPT,
    'output':              OUTPUT_PROMPT,
}

__all__ = [
    "SYSTEM_RULES",
    "NODE_PROMPT_TEMPLATES",
    "EDA_PROMPT", "CLEANING_PROMPT", "FE_PROMPT",
    "TRAINING_PROMPT", "EVALUATION_PROMPT", "OUTPUT_PROMPT",
]