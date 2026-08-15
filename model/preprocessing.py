"""Current B20 image preprocessing exposed without historical experiment names."""

from __future__ import annotations

from .bootstrap import ensure_developments_source

CURRENT_CROP_POLICY = {
    "version": "joint_focus_center_crop_only_v1",
    "crop_fraction": 0.90,
}


def apply_current_crop(volumes):
    """Apply the active model's deterministic 90% center crop and resize."""
    ensure_developments_source()
    from rsna_knee.crop_focus import apply_crop_focus

    return apply_crop_focus(volumes, CURRENT_CROP_POLICY)
