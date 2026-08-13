"""Dataset routing for the matched B20-v2/B21 crop-order experiment."""
from __future__ import annotations

import numpy as np

from .b12_variable_series import VariableSeriesKneeDataset
from .b20_crop_focus import CropFocusedVariableSeriesKneeDataset
from .crop_focus import CROP_FOCUS_VERSION
from .preresize_crop import center_crop_raw_volume, validate_pre_resize_crop_fraction


class PreResizeCropVariableSeriesKneeDataset(VariableSeriesKneeDataset):
    """Crop raw [D,H,W] MRI before the unchanged triplet resize."""

    def __init__(self, *args, crop_fraction: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_fraction = validate_pre_resize_crop_fraction(crop_fraction)

    def _read_volume(self, path, stream_name: str) -> np.ndarray:
        raw = super()._read_volume(path, stream_name)
        return center_crop_raw_volume(raw, self.crop_fraction)


def make_matched_crop_dataset(
    mode: str,
    uids,
    variable_index,
    dataset_config,
    *,
    crop_fraction: float,
    targets=None,
    weights=None,
    train: bool,
):
    """Return historical B20 post-resize control or B21 pre-resize candidate."""
    fraction = validate_pre_resize_crop_fraction(crop_fraction)
    common = dict(targets=targets, weights=weights, train=train)
    if mode == "control":
        return CropFocusedVariableSeriesKneeDataset(
            uids,
            variable_index,
            dataset_config,
            crop_focus_policy={
                "version": CROP_FOCUS_VERSION,
                "crop_fraction": fraction,
            },
            **common,
        )
    if mode == "preresize":
        return PreResizeCropVariableSeriesKneeDataset(
            uids,
            variable_index,
            dataset_config,
            crop_fraction=fraction,
            **common,
        )
    raise ValueError("mode must be 'control' or 'preresize'")
