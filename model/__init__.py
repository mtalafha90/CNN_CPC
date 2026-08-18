"""The working CNN model for knee MRI abnormality detection."""

from .architecture import TARGETS, WORKING_MODEL, build, describe, load

__all__ = ["TARGETS", "WORKING_MODEL", "build", "describe", "load"]
