"""RSNA 2026 knee MRI abnormality detection pipeline."""

from .constants import DUAL_STREAMS, N_STREAMS, N_TARGETS, SUBMISSION_COLUMNS, TARGETS, TARGET_SLUGS

__all__ = [
    "TARGETS",
    "TARGET_SLUGS",
    "SUBMISSION_COLUMNS",
    "N_TARGETS",
    "DUAL_STREAMS",
    "N_STREAMS",
]
__version__ = "0.3.0"
