"""Deterministic knee-joint spatial focus for MRI inputs.

The transform is deliberately conservative: a centered crop removes the most
peripheral field of view, then a smooth separable cosine window keeps the
central knee at full weight while tapering the remaining image border to zero.

It is not a learned segmentation and does not use pathology labels.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


JOINT_FOCUS_VERSION = "joint_focus_center_crop_cosine_v1"
DEFAULT_JOINT_FOCUS_POLICY = {
    "version": JOINT_FOCUS_VERSION,
    "crop_fraction": 0.90,
    "full_weight_fraction": 0.72,
    "outer_zero_fraction": 0.90,
}


def validate_joint_focus_policy(policy: dict) -> dict:
    if not isinstance(policy, dict):
        raise ValueError("joint focus policy must be a dictionary")
    version = str(policy.get("version", JOINT_FOCUS_VERSION))
    if version != JOINT_FOCUS_VERSION:
        raise ValueError(
            f"unsupported joint focus version {version!r}; expected {JOINT_FOCUS_VERSION!r}"
        )
    crop = float(policy.get("crop_fraction", 0.90))
    inner = float(policy.get("full_weight_fraction", 0.72))
    outer = float(policy.get("outer_zero_fraction", 0.90))
    if not 0.70 <= crop <= 1.0:
        raise ValueError("joint focus crop_fraction must be in [0.70, 1.0]")
    if not 0.40 <= inner < outer <= 1.0:
        raise ValueError(
            "joint focus fractions must satisfy 0.40 <= full_weight_fraction < "
            "outer_zero_fraction <= 1.0"
        )
    return {
        "version": version,
        "crop_fraction": crop,
        "full_weight_fraction": inner,
        "outer_zero_fraction": outer,
    }


def _cosine_axis(
    n: int,
    *,
    inner: float,
    outer: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n < 1:
        raise ValueError("joint focus axis length must be positive")
    coordinate = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype).abs()
    weight = torch.ones_like(coordinate)
    weight = torch.where(coordinate >= outer, torch.zeros_like(weight), weight)
    transition = (coordinate > inner) & (coordinate < outer)
    phase = (coordinate - inner) / max(outer - inner, 1e-6)
    tapered = 0.5 * (1.0 + torch.cos(math.pi * phase))
    weight = torch.where(transition, tapered, weight)
    return weight


def joint_focus_mask(
    height: int,
    width: int,
    policy: dict,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a [H,W] smooth center-prior mask."""
    policy = validate_joint_focus_policy(policy)
    device = torch.device(device)
    inner = float(policy["full_weight_fraction"])
    outer = float(policy["outer_zero_fraction"])
    wy = _cosine_axis(height, inner=inner, outer=outer, device=device, dtype=dtype)
    wx = _cosine_axis(width, inner=inner, outer=outer, device=device, dtype=dtype)
    return wy[:, None] * wx[None, :]


def apply_joint_focus(volume: torch.Tensor, policy: dict) -> torch.Tensor:
    """Apply centered crop + smooth border suppression to [...,C,H,W].

    Accepted examples include [S,C,H,W] and [V,S,C,H,W]. The output has the
    same shape as the input so the downstream B13/B18 architecture is unchanged.
    """
    policy = validate_joint_focus_policy(policy)
    if volume.ndim < 4:
        raise ValueError(
            f"joint focus expects [...,C,H,W] with at least 4 dims, got {tuple(volume.shape)}"
        )
    h, w = int(volume.shape[-2]), int(volume.shape[-1])
    if h < 2 or w < 2:
        raise ValueError("joint focus requires spatial dimensions >=2")

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

    mask = joint_focus_mask(
        h,
        w,
        policy,
        device=cropped.device,
        dtype=cropped.dtype,
    )[None, None]
    focused = cropped * mask
    return focused.reshape(original_shape)
