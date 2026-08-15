"""Dataset objects used by the current B20 working model.

The research implementation is preserved under `developments/src/rsna_knee`.
This module exposes only the pieces needed by the clean current-model pipeline.
"""

from __future__ import annotations

from model.bootstrap import ensure_developments_source

ensure_developments_source()

from rsna_knee.b12_variable_series import (  # noqa: E402
    build_variable_series_index,
    collate_variable_series,
)
from rsna_knee.b20_crop_focus import CropFocusedVariableSeriesKneeDataset  # noqa: E402

__all__ = [
    "CropFocusedVariableSeriesKneeDataset",
    "build_variable_series_index",
    "collate_variable_series",
]
