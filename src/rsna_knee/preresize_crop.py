from __future__ import annotations

import numpy as np

PRE_RESIZE_CROP_VERSION = "native_center_crop_before_resize_v1"
DEFAULT_PRE_RESIZE_CROP_FRACTION = 0.90


def validate_pre_resize_crop_fraction(value: float) -> float:
    fraction = float(value)
    if fraction < 0.70 or fraction > 1.0:
        raise ValueError("crop fraction must be in [0.70, 1.0]")
    return fraction


def center_crop_raw_volume(volume: np.ndarray, crop_fraction: float = DEFAULT_PRE_RESIZE_CROP_FRACTION) -> np.ndarray:
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"expected [D,H,W], got {array.shape}")
    _, height, width = array.shape
    fraction = validate_pre_resize_crop_fraction(crop_fraction)
    crop_h = max(2, min(height, int(round(height * fraction))))
    crop_w = max(2, min(width, int(round(width * fraction))))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return array[:, top : top + crop_h, left : left + crop_w]
