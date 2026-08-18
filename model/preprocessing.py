"""Image preprocessing for the working model."""

from __future__ import annotations

from ._implementation import apply_crop, crop_policy

CROP_POLICY = {
    "version": "joint_focus_center_crop_only_v1",
    "crop_fraction": 0.90,
}


def resolve_crop_policy(config: dict) -> dict:
    """Return the crop contract a configuration declares, rejecting mismatches."""
    return crop_policy(config)


def crop_to_joint(volumes, policy: dict | None = None):
    """Apply the deterministic centred crop used by the working model.

    Slices are resized to 224, cropped to the central 90%, then resized back to
    224.  The crop is fixed rather than learned, so training and inference see
    exactly the same geometry.
    """
    return apply_crop(volumes, policy or CROP_POLICY)
