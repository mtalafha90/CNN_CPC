"""Clean local-notebook interface to the frozen DINOv2 training implementation.

This subpackage deliberately leaves the existing model, data and trainer source
files unchanged, including their source-digest and checkpoint contracts.
"""

from .session import LocalRun

__all__ = ["LocalRun"]
