"""Crop-only knee focus used by B20.

B20 intentionally removes the B19 cosine/vignette mask. It keeps only a
centered field-of-view crop followed by resize back to the model resolution, so
no artificial dark border or deterministic intensity transition is introduced.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

CROP_FOCUS_VERSION = "joint_focus_center_crop_only_v1"
DEFAULT_CROP_FOCUS_POLICY = {
    "version": CROP_FOCUS_VERSION,
    "crop_fraction": 0.90,
}


def validate_crop_focus_policy(policy: dict) -> dict:
    if not isinstance(policy, dict):
        raise ValueError("crop focus policy must be a dictionary")
    version = str(policy.get("version", CROP_FOCUS_VERSION))
    if version != CROP_FOCUS_VERSION:
        raise ValueError(
            f"unsupported crop focus version {version!r}; expected {CROP_FOCUS_VERSION!r}"
        )
    crop = float(policy.get("crop_fraction", 0.90))
    if not 0.70 <= crop <= 1.0:
        raise ValueError("crop focus crop_fraction must be in [0.70, 1.0]")
    return {"version": version, "crop_fraction": crop}


def apply_crop_focus(volume: torch.Tensor, policy: dict) -> torch.Tensor:
    """Apply centered crop + resize to [...,C,H,W] without an output mask."""
    policy = validate_crop_focus_policy(policy)
    if volume.ndim < 4:
        raise ValueError(
            f"crop focus expects [...,C,H,W] with at least 4 dims, got {tuple(volume.shape)}"
        )
    h, w = int(volume.shape[-2]), int(volume.shape[-1])
    if h < 2 or w < 2:
        raise ValueError("crop focus requires spatial dimensions >=2")

    original_shape = tuple(volume.shape)
    channels = int(volume.shape[-3])
    flat = volume.reshape(-1, channels, h, w)

    crop_fraction = float(policy["crop_fraction"])
    crop_h = max(2, min(h, int(round(h * crop_fraction))))
    crop_w = max(2, min(w, int(round(w * crop_fraction))))
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    cropped = flat[:, :, top : top + crop_h, left : left + crop_w]
    if crop_h != h or crop_w != w:
        cropped = F.interpolate(
            cropped,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
    return cropped.reshape(original_shape)
