"""B37: high-resolution native crop with one deterministic resize.

Scientific motivation
---------------------
Historical B20 executes

    full native volume normalization -> resize 224 -> center crop 90% -> resize 224

and B21/B22 already tested a raw pre-resize crop at 224, where the crop also
changed the support used by percentile normalization.  B37 is therefore *not*
another B21 crop-order experiment.

B37 preserves B20's full-native-volume percentile normalization, then crops the
central 90% while the sampled images are still at native in-plane resolution,
and performs one deterministic resize to 288x288.  Slice sampling, 2.5D
triplets, augmentation policy, series exposure, labels, architecture and the
fixed-E2 optimization protocol are left unchanged by the B37 dataset.

Training-time affine augmentation can of course interpolate the image; "one
resize" here means the deterministic anatomical preprocessing path before the
unchanged augmentation pipeline.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .b12_variable_series import VariableSeriesKneeDataset
from .crop_focus import CROP_FOCUS_VERSION, validate_crop_focus_policy
from .dicom import _centers, _normalise_volume

B37_VERSION = "b37_fullnorm_nativecrop_single_resize_288_v1"
B37_IMAGE_SIZE = 288
B37_CROP_FRACTION = 0.90
B37_RESIZE_COUNT = 1
B37_NORMALIZATION_STAGE = "full_native_volume_before_crop"
B37_CROP_STAGE = "normalized_native_pixels_before_resize"


def require_b37_preprocessing_contract(config: dict) -> dict:
    """Validate the one declared B37 preprocessing intervention.

    The broader B20/B34 training contract is validated separately by the B37
    training wrapper on a shadow config whose image size is restored to 224.
    """
    image_size = int(config.get("b7_image_size", B37_IMAGE_SIZE))
    if image_size != B37_IMAGE_SIZE:
        raise ValueError(f"B37 freezes b7_image_size={B37_IMAGE_SIZE}; got {image_size}")
    policy = validate_crop_focus_policy(
        {
            "version": str(config.get("b20_crop_focus_version", CROP_FOCUS_VERSION)),
            "crop_fraction": float(
                config.get("b20_crop_focus_crop_fraction", B37_CROP_FRACTION)
            ),
        }
    )
    if policy["version"] != CROP_FOCUS_VERSION:
        raise ValueError("B37 requires the historical B20 crop-policy version")
    if not np.isclose(
        float(policy["crop_fraction"]), B37_CROP_FRACTION, atol=1e-12, rtol=0
    ):
        raise ValueError(
            f"B37 freezes crop_fraction={B37_CROP_FRACTION}; "
            f"got {policy['crop_fraction']}"
        )
    return policy


def _native_center_crop_4d(
    triplets: np.ndarray,
    crop_fraction: float = B37_CROP_FRACTION,
) -> np.ndarray:
    """Center-crop [S,C,H,W] without interpolation."""
    x = np.asarray(triplets)
    if x.ndim != 4:
        raise ValueError(f"B37 expected [S,C,H,W], got {x.shape}")
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if h < 2 or w < 2:
        raise ValueError("B37 requires native spatial dimensions >=2")
    fraction = float(crop_fraction)
    crop_h = max(2, min(h, int(round(h * fraction))))
    crop_w = max(2, min(w, int(round(w * fraction))))
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return x[..., top : top + crop_h, left : left + crop_w]


def preprocess_triplets_b37(
    v: np.ndarray,
    n_slices: int = 16,
    image_size: int = B37_IMAGE_SIZE,
    gap: int = 1,
    *,
    center_offset: int = 0,
    jitter: int = 0,
    rng: np.random.Generator | None = None,
    crop_fraction: float = B37_CROP_FRACTION,
) -> torch.Tensor:
    """B37 triplets: full-volume normalize -> native crop -> one resize.

    Normalization deliberately happens before any crop.  This keeps the 1st/99th
    percentile support identical to historical B20 for a given source volume,
    isolating B37 from B21's crop-before-normalization intervention.
    """
    if gap < 1:
        raise ValueError("B37 2.5D gap must be >=1")
    if int(image_size) != B37_IMAGE_SIZE:
        raise ValueError(f"B37 output size must remain {B37_IMAGE_SIZE}")

    normalized = _normalise_volume(v)
    centers = _centers(
        len(normalized),
        int(n_slices),
        int(gap),
        center_offset=int(center_offset),
        jitter=int(jitter),
        rng=rng,
    )
    offsets = np.asarray([-int(gap), 0, int(gap)], dtype=int)
    idx = np.clip(
        centers[:, None] + offsets[None, :],
        0,
        len(normalized) - 1,
    )
    triplets = normalized[idx].astype(np.float32, copy=False)
    cropped = _native_center_crop_4d(triplets, crop_fraction=float(crop_fraction))
    tensor = torch.from_numpy(np.ascontiguousarray(cropped))
    return F.interpolate(
        tensor,
        size=(B37_IMAGE_SIZE, B37_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )


class B37HighResVariableSeriesKneeDataset(VariableSeriesKneeDataset):
    """B34 variable-series dataset with B37's 288 single-resize preprocessing."""

    def __init__(self, *args, crop_focus_policy: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_focus_policy = validate_crop_focus_policy(crop_focus_policy)
        if int(self.config.image_size) != B37_IMAGE_SIZE:
            raise ValueError(
                f"B37 dataset requires DatasetConfig.image_size={B37_IMAGE_SIZE}"
            )
        if not np.isclose(
            float(self.crop_focus_policy["crop_fraction"]),
            B37_CROP_FRACTION,
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("B37 dataset requires the fixed 90% center crop")

    def _training_view(self, raw: np.ndarray) -> torch.Tensor:
        choices = self.config.train_gap_choices
        gap = int(choices[int(torch.randint(len(choices), (1,)).item())])
        jitter_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = np.random.default_rng(jitter_seed)
        volume = preprocess_triplets_b37(
            raw,
            n_slices=self.config.n_slices,
            image_size=self.config.image_size,
            gap=gap,
            center_offset=0,
            jitter=self.config.center_jitter,
            rng=rng,
            crop_fraction=float(self.crop_focus_policy["crop_fraction"]),
        )
        return self._augment_mri(volume)

    def _evaluation_view(self, raw: np.ndarray) -> torch.Tensor:
        if self.config.tta_center_offsets:
            return torch.stack(
                [
                    preprocess_triplets_b37(
                        raw,
                        n_slices=self.config.n_slices,
                        image_size=self.config.image_size,
                        gap=self.config.triplet_gap,
                        center_offset=int(offset),
                        jitter=0,
                        crop_fraction=float(
                            self.crop_focus_policy["crop_fraction"]
                        ),
                    )
                    for offset in self.config.tta_center_offsets
                ],
                dim=0,
            )
        return preprocess_triplets_b37(
            raw,
            n_slices=self.config.n_slices,
            image_size=self.config.image_size,
            gap=self.config.triplet_gap,
            center_offset=self.config.center_offset,
            jitter=0,
            crop_fraction=float(self.crop_focus_policy["crop_fraction"]),
        )


def b37_preprocessing_state() -> dict:
    return {
        "version": B37_VERSION,
        "image_size": B37_IMAGE_SIZE,
        "crop_fraction": B37_CROP_FRACTION,
        "normalization_stage": B37_NORMALIZATION_STAGE,
        "crop_stage": B37_CROP_STAGE,
        "deterministic_resize_count": B37_RESIZE_COUNT,
        "deterministic_resize_mode": "bilinear_align_corners_false",
        "historical_b20_post_resize_crop_applied": False,
    }
