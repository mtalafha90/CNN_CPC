"""B41: aspect-preserving counterpart to the completed B37 sparse-MIL endpoint.

B37 applied its fixed 90% native centre crop and then resized every retained
matrix directly to a 448x448 square.  That is harmless for square matrices but
stretches genuinely rectangular acquisitions such as 640x1280.  B41 changes
only that last spatial operation:

    full native-volume percentile normalization
    -> fixed 90% native centre crop
    -> one antialiased resize-to-fit that preserves the crop aspect ratio
    -> symmetric zero padding into the fixed 448x448 model canvas

The sparse-MIL architecture, 32 deterministic 2.5D centres, 6x6 local grid,
top-k=8 pooling, optimizer contract, supervision, and two-epoch endpoint are
all inherited unchanged from B37.  B37 itself is deliberately not modified.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .b35_target_spatial_residual import B35_DENSE_SLICES, b35_centers
from .b37_highres_sparse_mil import (
    B37_CROP_FRACTION,
    B37_ENCODER_CHUNK_SIZE,
    B37_ENCODER_LR_SCALE,
    B37_ENCODER_TRAINABLE_STAGES,
    B37_GRID_SIZE,
    B37_IMAGE_SIZE,
    B37_LOCAL_AUX_WEIGHT,
    B37_RESIZE_MODE,
    B37_TEMPERATURE,
    B37_TOP_K,
    B37HighResSparseDataset,
    require_b37_sparse_contract,
)
from .crop_focus import validate_crop_focus_policy
from .dicom import _normalise_volume, find_series_dir

B41_VERSION = "b41_native_aspect_90crop_sparse_mil_v1"
B41_EXPERIMENT = "B41_native_aspect_preserving_90pct_crop_sparse_mil"
B41_NUMBERED_CONTAINER = "runs/076_Experiment_B41_native_aspect_90crop_sparse_mil"
B41_RUN_ROOT = f"{B41_NUMBERED_CONTAINER}/b41_highres_aspect_sparse_mil"
B41_EXPERT58_ROOT = f"{B41_RUN_ROOT}/expert58"

# B41 freezes every B37 model value and changes only this resize policy.
B41_IMAGE_SIZE = B37_IMAGE_SIZE
B41_CROP_FRACTION = B37_CROP_FRACTION
B41_GRID_SIZE = B37_GRID_SIZE
B41_TOP_K = B37_TOP_K
B41_TEMPERATURE = B37_TEMPERATURE
B41_LOCAL_AUX_WEIGHT = B37_LOCAL_AUX_WEIGHT
B41_ENCODER_TRAINABLE_STAGES = B37_ENCODER_TRAINABLE_STAGES
B41_ENCODER_LR_SCALE = B37_ENCODER_LR_SCALE
B41_ENCODER_CHUNK_SIZE = B37_ENCODER_CHUNK_SIZE
B41_RESIZE_MODE = B37_RESIZE_MODE
B41_RESIZE_POLICY = "aspect_preserving_pad"
B41_PAD_VALUE = 0.0


def require_b41_aspect_contract(config: dict) -> dict:
    """Freeze B37's controls and require B41's resize-to-fit policy."""
    # Reuse B37's exact high-resolution sparse-MIL and 90%-crop contract.
    crop = require_b37_sparse_contract(config)
    # Require an explicit policy field so a B41 run cannot silently square-stretch.
    policy = str(config.get("b41_resize_policy", ""))
    if policy != B41_RESIZE_POLICY:
        raise ValueError(
            f"B41 freezes b41_resize_policy={B41_RESIZE_POLICY!r}; got {policy!r}"
        )
    # Require black/zero padding after the normalized image is resized.
    pad_value = float(config.get("b41_pad_value", float("nan")))
    if not np.isclose(pad_value, B41_PAD_VALUE, atol=1e-12, rtol=0):
        raise ValueError(
            f"B41 freezes b41_pad_value={B41_PAD_VALUE}; got {pad_value}"
        )
    return crop


def _native_center_crop(triplets: np.ndarray, fraction: float) -> np.ndarray:
    """Apply B37's exact native-resolution centre crop before B41 resizing."""
    # Convert array-like inputs without changing their native values.
    x = np.asarray(triplets)
    # Require a batch of 2.5D triplets with spatial axes at the end.
    if x.ndim != 4:
        raise ValueError(f"B41 expected [S,C,H,W], got {x.shape}")
    # Read the original in-plane matrix before any resize or padding.
    height, width = int(x.shape[-2]), int(x.shape[-1])
    # Retain the requested fraction in each dimension, keeping at least two pixels.
    crop_h = max(2, min(height, int(round(height * float(fraction)))))
    crop_w = max(2, min(width, int(round(width * float(fraction)))))
    # Place the retained rectangle in the centre of the original field of view.
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    # Return the full retained crop with no in-plane stretch.
    return x[..., top : top + crop_h, left : left + crop_w]


def resize_triplets_aspect_preserving_pad(
    triplets: np.ndarray,
    *,
    image_size: int = B41_IMAGE_SIZE,
    pad_value: float = B41_PAD_VALUE,
) -> torch.Tensor:
    """Resize [S,C,H,W] once to fit a square canvas, then centre-pad it.

    The same scale is used for height and width.  Therefore a 576x1152 crop,
    for example, becomes 224x448 and is padded with 112 rows above and below;
    it is never stretched into a square.
    """
    # Convert to a predictable float image array while keeping the [S,C,H,W] layout.
    x = np.asarray(triplets, dtype=np.float32)
    # Reject malformed inputs before deriving a scale factor.
    if x.ndim != 4:
        raise ValueError(f"B41 expected [S,C,H,W], got {x.shape}")
    # Require a useful positive square canvas.
    target = int(image_size)
    if target < 2:
        raise ValueError("B41 image_size must be at least two pixels")
    # Require finite padding so black margins cannot introduce undefined values.
    if not np.isfinite(float(pad_value)):
        raise ValueError("B41 pad_value must be finite")
    # Read the already-cropped native matrix dimensions.
    height, width = int(x.shape[-2]), int(x.shape[-1])
    if height < 1 or width < 1:
        raise ValueError(f"B41 cannot resize an empty matrix {x.shape}")
    # Choose one scale that makes the long side exactly fit the square canvas.
    scale = min(float(target) / float(height), float(target) / float(width))
    # Round the fitted dimensions and clamp them to the canvas bounds.
    resized_h = max(1, min(target, int(round(height * scale))))
    resized_w = max(1, min(target, int(round(width * scale))))
    # Convert the contiguous triplet batch into the tensor format required by interpolate.
    tensor = torch.from_numpy(np.ascontiguousarray(x))
    # Perform B41's only resampling operation with B37's antialiased bilinear mode.
    resized = F.interpolate(
        tensor,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    # Allocate one fixed square canvas with the explicit normalized black pad value.
    output = resized.new_full(
        (resized.shape[0], resized.shape[1], target, target),
        float(pad_value),
    )
    # Split any leftover rows and columns as symmetrically as integer geometry permits.
    top = (target - resized_h) // 2
    left = (target - resized_w) // 2
    # Copy the full resized crop into the centre without another interpolation.
    output[..., top : top + resized_h, left : left + resized_w] = resized
    # Return a fixed-size model tensor while preserving the retained crop's aspect ratio.
    return output


def preprocess_dense_triplets_b41(
    raw: np.ndarray,
    *,
    image_size: int = B41_IMAGE_SIZE,
    gap: int = 1,
    center_offset: int = 0,
    crop_fraction: float = B41_CROP_FRACTION,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return 32 B37-style triplets with B41 aspect-preserving geometry."""
    # Keep B41's fixed output canvas identical to B37's 448-pixel representation.
    if int(image_size) != B41_IMAGE_SIZE:
        raise ValueError(f"B41 output size must remain {B41_IMAGE_SIZE}")
    # Preserve B37's immediate-neighbour 2.5D triplet definition.
    if int(gap) < 1:
        raise ValueError("B41 2.5D gap must be positive")
    # Normalize over the complete native volume before any crop or padding.
    normalized = _normalise_volume(raw)
    # Select the same deterministic 32 B35/B37 centres and through-plane positions.
    centers, position = b35_centers(
        len(normalized),
        gap=int(gap),
        center_offset=int(center_offset),
    )
    # Define the previous, central, and next image for every 2.5D triplet.
    offsets = np.asarray([-int(gap), 0, int(gap)], dtype=np.int64)
    # Clamp edge neighbours to valid native frames.
    index = np.clip(centers[:, None] + offsets[None, :], 0, len(normalized) - 1)
    # Gather every triplet before changing its in-plane geometry.
    triplets = normalized[index].astype(np.float32, copy=False)
    # Keep B37's exact central 90% crop in native image coordinates.
    cropped = _native_center_crop(triplets, float(crop_fraction))
    # Apply one aspect-preserving resize-to-fit followed by deterministic zero padding.
    images = resize_triplets_aspect_preserving_pad(
        cropped,
        image_size=B41_IMAGE_SIZE,
        pad_value=B41_PAD_VALUE,
    )
    # Return both image triplets and unchanged normalized slice positions.
    return images, position


class B41HighResAspectSparseDataset(B37HighResSparseDataset):
    """B37 dataset contract with only B41's native-aspect image transform."""

    def __init__(
        self,
        study_uids,
        series_records,
        config,
        *,
        crop_focus_policy: dict,
        center_offsets: tuple[int, ...] = (0,),
        targets=None,
        weights=None,
    ) -> None:
        # Retain B37's all-series, 32-centre, zero-series and metadata behavior.
        super().__init__(
            study_uids,
            series_records,
            config,
            crop_focus_policy=crop_focus_policy,
            center_offsets=center_offsets,
            targets=targets,
            weights=weights,
        )
        # Record B41's policy locally for explicit dataset-level validation.
        self.resize_policy = B41_RESIZE_POLICY
        # Revalidate the inherited crop policy under B41's stated contract.
        policy = validate_crop_focus_policy(crop_focus_policy)
        if not np.isclose(
            float(policy["crop_fraction"]), B41_CROP_FRACTION, atol=1e-12, rtol=0
        ):
            raise ValueError("B41 dataset requires the fixed 90% native crop")

    def _load_b37(self, uid: str, series_uid: str, plane: str):
        """Override B37's loader only where its direct-square resize occurs."""
        # Locate the original series directory using B37's strict/non-strict behavior.
        path = find_series_dir(
            self.config.data_root,
            self.config.split,
            uid,
            str(series_uid),
        )
        # Reuse B37's all-zero fallback for a missing non-strict series.
        if path is None:
            if self.config.strict_dicom:
                raise FileNotFoundError(f"missing series {uid}/{series_uid}")
            image, position = self._zero()
            return image, position, 0.0
        try:
            # Decode the full native DICOM volume once for every requested TTA offset.
            raw = self._read_volume(path, plane.lower())
            images, positions = [], []
            for offset in self.center_offsets:
                # Preserve all B37 triplet choices while using B41's resize policy.
                image, position = preprocess_dense_triplets_b41(
                    raw,
                    image_size=B41_IMAGE_SIZE,
                    gap=int(self.config.triplet_gap),
                    center_offset=int(offset),
                    crop_fraction=float(self.crop_focus_policy["crop_fraction"]),
                )
                images.append(image)
                positions.append(torch.from_numpy(position))
            # Keep B37's [views, centres, channels, height, width] return layout.
            return torch.stack(images), torch.stack(positions), 1.0
        except Exception:
            # Preserve B37's strict inference behavior while tolerating bad training DICOMs.
            if self.config.strict_dicom:
                raise
            image, position = self._zero()
            return image, position, 0.0


def b41_preprocessing_state() -> dict:
    """Return auditable, immutable B41 geometry choices for checkpoints and tests."""
    return {
        "normalization": "full native volume before crop",
        "crop_fraction": B41_CROP_FRACTION,
        "crop_stage": "native resolution before deterministic resize",
        "resize_policy": B41_RESIZE_POLICY,
        "preserves_in_plane_aspect_ratio": True,
        "image_size": B41_IMAGE_SIZE,
        "deterministic_resize_count": 1,
        "resize": "bilinear antialias=True align_corners=False",
        "padding": {
            "value": B41_PAD_VALUE,
            "placement": "symmetric centre padding after resize-to-fit",
        },
        "dense_slices": B35_DENSE_SLICES,
    }
