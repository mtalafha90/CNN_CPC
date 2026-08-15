"""Clean interface for the active CNN-based knee MRI model."""

from .architecture import CURRENT_MODEL, TARGETS, load_current_model

__all__ = ["CURRENT_MODEL", "TARGETS", "load_current_model"]
