"""Submission-only B39 preprocessing acceleration with unchanged view construction.

Historical B39 calls ``preprocess_dense_triplets_b37`` independently for each of
five centre offsets.  That function normalizes the complete native DICOM volume
on every call, so the same 1st/99th-percentile normalization is recomputed five
times per series.

This module performs that deterministic full-volume normalization once and then
runs the exact historical centre selection, 90% crop and per-view antialiased
448x448 resize separately for each of the same five offsets.  Keeping each
``F.interpolate`` call separate preserves the historical per-view execution
rather than introducing a new cross-view resize batch.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .b35_target_spatial_residual import b35_centers
from .b37_highres_sparse_mil import (
    B37_CROP_FRACTION,
    B37_IMAGE_SIZE,
    B37HighResSparseDataset,
    _native_center_crop,
)
from .dicom import _normalise_volume, find_series_dir

B39_FAST_TTA_OFFSETS = (-2, -1, 0, 1, 2)


def preprocess_five_offsets_b39_normalize_once(
    raw: np.ndarray,
    *,
    gap: int,
    crop_fraction: float = B37_CROP_FRACTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct exact B39 five-view tensors after one native-volume normalization."""
    gap = int(gap)
    if gap < 1:
        raise ValueError("B39 2.5D gap must be positive")

    normalized = _normalise_volume(raw)
    triplet_offsets = np.asarray([-gap, 0, gap], dtype=np.int64)
    images: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []

    for center_offset in B39_FAST_TTA_OFFSETS:
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
        tensor = torch.from_numpy(np.ascontiguousarray(cropped))
        resized = F.interpolate(
            tensor,
            size=(B37_IMAGE_SIZE, B37_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        images.append(resized)
        positions.append(torch.from_numpy(position))

    return torch.stack(images), torch.stack(positions)


class B39KaggleNormalizeOnceDataset(B37HighResSparseDataset):
    """B39 submission dataset that removes only repeated volume normalization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if tuple(self.center_offsets) != B39_FAST_TTA_OFFSETS:
            raise ValueError(
                "B39 normalize-once dataset requires offsets [-2,-1,0,1,2]"
            )

    def _load_b37(self, uid: str, series_uid: str, plane: str):
        path = find_series_dir(
            self.config.data_root,
            self.config.split,
            uid,
            str(series_uid),
        )
        if path is None:
            if self.config.strict_dicom:
                raise FileNotFoundError(f"missing series {uid}/{series_uid}")
            image, position = self._zero()
            return image, position, 0.0

        try:
            raw = self._read_volume(path, plane.lower())
            images, positions = preprocess_five_offsets_b39_normalize_once(
                raw,
                gap=int(self.config.triplet_gap),
                crop_fraction=float(self.crop_focus_policy["crop_fraction"]),
            )
            return images, positions, 1.0
        except Exception:
            if self.config.strict_dicom:
                raise
            image, position = self._zero()
            return image, position, 0.0


__all__ = [
    "B39_FAST_TTA_OFFSETS",
    "B39KaggleNormalizeOnceDataset",
    "preprocess_five_offsets_b39_normalize_once",
]
