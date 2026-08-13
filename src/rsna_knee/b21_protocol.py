from __future__ import annotations

import numpy as np

from .preresize_crop import DEFAULT_PRE_RESIZE_CROP_FRACTION, validate_pre_resize_crop_fraction

B20_V2_CONTROL_VARIANT = "b20_postresize_crop_weak_v2_fixed_e2_control_v1"
B20_V2_CONTROL_EXPERIMENT = "B20_v2_fixed_e2_matched_control"
B21_VARIANT = "b21_preresize_crop_weak_v2_fixed_e2_v1"
B21_EXPERIMENT = "B21_preresize_crop_fixed_e2"
B21_FIXED_EPOCHS = 2
B21_WEAK_TRAIN_STUDIES = 2497
B21_WEAK_HOLDOUT_STUDIES = 623
B21_EXPECTED_BATCHES = 1249


def mode_identity(mode: str) -> tuple[str, str, str]:
    if mode == "control":
        return B20_V2_CONTROL_VARIANT, B20_V2_CONTROL_EXPERIMENT, "post_resize_224"
    if mode == "preresize":
        return B21_VARIANT, B21_EXPERIMENT, "native_array_pre_resize"
    raise ValueError("mode must be control or preresize")


def require_b21_crop_fraction(config: dict) -> float:
    fraction = validate_pre_resize_crop_fraction(
        float(config.get("b21_crop_fraction", DEFAULT_PRE_RESIZE_CROP_FRACTION))
    )
    if not np.isclose(fraction, 0.90, atol=1e-12, rtol=0):
        raise ValueError("first B21 run freezes crop fraction at 0.90")
    return fraction
