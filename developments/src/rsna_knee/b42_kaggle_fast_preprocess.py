"""Submission-only B42 preprocessing acceleration with unchanged view construction.

Historical B42 calls ``preprocess_dense_triplets_b42`` independently for each of
three centre offsets. That function normalizes the complete native DICOM volume
on every call, so the same deterministic 1st/99th-percentile normalization is
recomputed three times per series.

This module performs that full-volume normalization once and then runs the exact
historical centre selection, 90% native crop, constant-area isotropic resize and
reflection stride-padding separately for each of the same three offsets. Each
resize/pad call remains separate; only the redundant normalization is removed.
"""
from __future__ import annotations

import numpy as np
import torch

from .b35_target_spatial_residual import b35_centers
from .b37_highres_sparse_mil import _native_center_crop
from .b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectDataset,
    resize_triplets_constant_area,
)
from .dicom import _normalise_volume, find_series_dir

B42_FAST_TTA_OFFSETS = (-1, 0, 1)


def preprocess_three_offsets_b42_normalize_once(
    raw: np.ndarray,
    *,
    gap: int,
    crop_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct exact B42 three-view tensors after one volume normalization."""
    gap = int(gap)
    if gap < 1:
        raise ValueError("B42 2.5D gap must be positive")

    normalized = _normalise_volume(raw)
    triplet_offsets = np.asarray([-gap, 0, gap], dtype=np.int64)
    images: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []

    for center_offset in B42_FAST_TTA_OFFSETS:
        centers, position = b35_centers(
            len(normalized),
            gap=gap,
            center_offset=int(center_offset),
        )
        index = np.clip(
            centers[:, None] + triplet_offsets[None, :],
            0,
            len(normalized) - 1,
        )
        triplets = normalized[index].astype(np.float32, copy=False)
        cropped = _native_center_crop(triplets, float(crop_fraction))
        image = resize_triplets_constant_area(cropped)
        images.append(image)
        positions.append(torch.from_numpy(position))

    return torch.stack(images), torch.stack(positions)


class B42KaggleNormalizeOnceDataset(B42ConstantAreaAspectDataset):
    """B42 test dataset that removes only repeated full-volume normalization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if tuple(self.center_offsets) != B42_FAST_TTA_OFFSETS:
            raise ValueError(
                "B42 normalize-once dataset requires offsets [-1,0,1]"
            )

    def _load_b42(self, uid: str, series_uid: str, plane: str):
        path = find_series_dir(
            self.config.data_root,
            self.config.split,
            uid,
            str(series_uid),
        )
        if path is None:
            if self.config.strict_dicom:
                raise FileNotFoundError(f"missing series {uid}/{series_uid}")
            image, position = self._zero_b42()
            return image, position, 0.0

        try:
            raw = self._read_volume(path, plane.lower())
            images, positions = preprocess_three_offsets_b42_normalize_once(
                raw,
                gap=int(self.config.triplet_gap),
                crop_fraction=float(self.crop_focus_policy["crop_fraction"]),
            )
            return images, positions, 1.0
        except Exception:
            if self.config.strict_dicom:
                raise
            image, position = self._zero_b42()
            return image, position, 0.0


__all__ = [
    "B42_FAST_TTA_OFFSETS",
    "B42KaggleNormalizeOnceDataset",
    "preprocess_three_offsets_b42_normalize_once",
]
