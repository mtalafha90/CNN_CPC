"""B42: constant-area native-aspect rectangular sparse MIL.

B42 keeps the B37/B41 90% native crop and all model/supervision controls, but
removes B41's large square zero bars.  Each retained crop is resized with one
isotropic scale so its anatomical pixel area is approximately 448**2, then only
thin reflection padding is added independently per axis to reach a multiple of
32.  Different series in one study remain ragged and are encoded separately.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .b35_target_spatial_residual import B35_BASE_SLICES, B35_DENSE_SLICES, b35_centers
from .b37_highres_sparse_mil import (
    B37_CROP_FRACTION,
    B37_ENCODER_CHUNK_SIZE,
    B37_GRID_SIZE,
    B37_IMAGE_SIZE,
    B37_LOCAL_AUX_WEIGHT,
    B37_TEMPERATURE,
    B37_TOP_K,
    B37Forward,
    B37HighResSparseDataset,
    B37HighResSparseMILResidual,
    _native_center_crop,
    require_b37_sparse_contract,
)
from .crop_focus import validate_crop_focus_policy
from .dicom import _normalise_volume, find_series_dir

B42_VERSION = "b42_constant_area_native_aspect_rectangular_sparse_mil_v1"
B42_EXPERIMENT = "B42_constant_area_native_aspect_rectangular_sparse_MIL"
B42_NUMBERED_CONTAINER = "runs/077_Experiment_B42_constant_area_aspect_sparse_mil"
B42_RUN_ROOT = f"{B42_NUMBERED_CONTAINER}/b42_constant_area_aspect_sparse_mil"
B42_EXPERT58_ROOT = f"{B42_RUN_ROOT}/expert58"

B42_REFERENCE_SIDE = B37_IMAGE_SIZE
B42_REFERENCE_AREA = B42_REFERENCE_SIDE * B42_REFERENCE_SIDE
B42_STRIDE_ALIGNMENT = 32
B42_PADDING_MODE = "reflect"
B42_RESIZE_POLICY = "constant_area_aspect_rectangular"
B42_EFFECTIVE_BATCH = 2


def require_b42_contract(config: dict) -> dict:
    """Require the fixed B42 geometry while inheriting every B37 model control."""
    crop = require_b37_sparse_contract(config)
    expected_int = {
        "b42_reference_area": B42_REFERENCE_AREA,
        "b42_stride_alignment": B42_STRIDE_ALIGNMENT,
        "b42_effective_batch": B42_EFFECTIVE_BATCH,
    }
    for key, expected in expected_int.items():
        value = int(config.get(key, expected))
        if value != expected:
            raise ValueError(f"B42 freezes {key}={expected}; got {value}")
    policy = str(config.get("b42_resize_policy", B42_RESIZE_POLICY))
    if policy != B42_RESIZE_POLICY:
        raise ValueError(
            f"B42 freezes b42_resize_policy={B42_RESIZE_POLICY!r}; got {policy!r}"
        )
    padding = str(config.get("b42_padding_mode", B42_PADDING_MODE))
    if padding != B42_PADDING_MODE:
        raise ValueError(
            f"B42 freezes b42_padding_mode={B42_PADDING_MODE!r}; got {padding!r}"
        )
    if int(config.get("b37_encoder_chunk_size", B37_ENCODER_CHUNK_SIZE)) != B37_ENCODER_CHUNK_SIZE:
        raise ValueError("B42 retains B37 encoder chunk size")
    return crop


def constant_area_shape(
    height: int,
    width: int,
    *,
    reference_area: int = B42_REFERENCE_AREA,
    alignment: int = B42_STRIDE_ALIGNMENT,
) -> dict[str, int | float]:
    """Return isotropic resized and minimally stride-aligned rectangular geometry."""
    h, w = int(height), int(width)
    area = int(reference_area)
    stride = int(alignment)
    if h < 1 or w < 1:
        raise ValueError("B42 requires non-empty in-plane dimensions")
    if area < 4:
        raise ValueError("B42 reference area must be positive")
    if stride < 1:
        raise ValueError("B42 alignment must be positive")
    scale = math.sqrt(float(area) / float(h * w))
    resized_h = max(2, int(round(h * scale)))
    resized_w = max(2, int(round(w * scale)))
    aligned_h = int(math.ceil(resized_h / stride) * stride)
    aligned_w = int(math.ceil(resized_w / stride) * stride)
    pad_h = aligned_h - resized_h
    pad_w = aligned_w - resized_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return {
        "source_height": h,
        "source_width": w,
        "scale": float(scale),
        "resized_height": resized_h,
        "resized_width": resized_w,
        "aligned_height": aligned_h,
        "aligned_width": aligned_w,
        "pad_top": top,
        "pad_bottom": bottom,
        "pad_left": left,
        "pad_right": right,
        "anatomical_pixels": int(resized_h * resized_w),
        "tensor_pixels": int(aligned_h * aligned_w),
    }


def resize_triplets_constant_area(
    triplets: np.ndarray,
    *,
    reference_area: int = B42_REFERENCE_AREA,
    alignment: int = B42_STRIDE_ALIGNMENT,
) -> torch.Tensor:
    """Resize [S,C,H,W] once with one scale and reflection-pad only to stride."""
    x = np.asarray(triplets, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"B42 expected [S,C,H,W], got {x.shape}")
    if int(x.shape[1]) != 3:
        raise ValueError("B42 requires three-channel 2.5D triplets")
    geometry = constant_area_shape(
        int(x.shape[-2]),
        int(x.shape[-1]),
        reference_area=int(reference_area),
        alignment=int(alignment),
    )
    tensor = torch.from_numpy(np.ascontiguousarray(x))
    resized = F.interpolate(
        tensor,
        size=(int(geometry["resized_height"]), int(geometry["resized_width"])),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    padding = (
        int(geometry["pad_left"]),
        int(geometry["pad_right"]),
        int(geometry["pad_top"]),
        int(geometry["pad_bottom"]),
    )
    if any(padding):
        # Reflection padding needs every per-side pad to be smaller than the input axis.
        if max(padding[:2]) >= resized.shape[-1] or max(padding[2:]) >= resized.shape[-2]:
            raise ValueError("B42 reflection padding is invalid for this resized geometry")
        resized = F.pad(resized, padding, mode=B42_PADDING_MODE)
    return resized


def preprocess_dense_triplets_b42(
    raw: np.ndarray,
    *,
    gap: int = 1,
    center_offset: int = 0,
    crop_fraction: float = B37_CROP_FRACTION,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return 32 native-aspect B42 triplets at approximately constant pixel area."""
    if int(gap) < 1:
        raise ValueError("B42 2.5D gap must be positive")
    normalized = _normalise_volume(raw)
    centers, position = b35_centers(
        len(normalized), gap=int(gap), center_offset=int(center_offset)
    )
    offsets = np.asarray([-int(gap), 0, int(gap)], dtype=np.int64)
    index = np.clip(centers[:, None] + offsets[None, :], 0, len(normalized) - 1)
    triplets = normalized[index].astype(np.float32, copy=False)
    cropped = _native_center_crop(triplets, float(crop_fraction))
    images = resize_triplets_constant_area(cropped)
    return images, position


def collate_b42(items):
    """Keep study items ragged instead of padding heterogeneous rectangles."""
    return list(items)


class B42ConstantAreaAspectDataset(B37HighResSparseDataset):
    """B37 all-series dataset that returns a list of rectangular series tensors."""

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
        super().__init__(
            study_uids,
            series_records,
            config,
            crop_focus_policy=crop_focus_policy,
            center_offsets=center_offsets,
            targets=targets,
            weights=weights,
        )
        policy = validate_crop_focus_policy(crop_focus_policy)
        if not np.isclose(float(policy["crop_fraction"]), B37_CROP_FRACTION, atol=1e-12, rtol=0):
            raise ValueError("B42 dataset requires B37's fixed 90% native crop")

    def _zero_b42(self) -> tuple[torch.Tensor, torch.Tensor]:
        views = len(self.center_offsets)
        image = torch.zeros(
            views,
            B35_DENSE_SLICES,
            3,
            B42_REFERENCE_SIDE,
            B42_REFERENCE_SIDE,
            dtype=torch.float32,
        )
        position = torch.zeros(views, B35_DENSE_SLICES, dtype=torch.float32)
        return image, position

    def _load_b42(self, uid: str, series_uid: str, plane: str):
        path = find_series_dir(
            self.config.data_root, self.config.split, uid, str(series_uid)
        )
        if path is None:
            if self.config.strict_dicom:
                raise FileNotFoundError(f"missing series {uid}/{series_uid}")
            image, position = self._zero_b42()
            return image, position, 0.0
        try:
            raw = self._read_volume(path, plane.lower())
            images, positions = [], []
            for offset in self.center_offsets:
                image, position = preprocess_dense_triplets_b42(
                    raw,
                    gap=int(self.config.triplet_gap),
                    center_offset=int(offset),
                    crop_fraction=float(self.crop_focus_policy["crop_fraction"]),
                )
                images.append(image)
                positions.append(torch.from_numpy(position))
            return torch.stack(images), torch.stack(positions), 1.0
        except Exception:
            if self.config.strict_dicom:
                raise
            image, position = self._zero_b42()
            return image, position, 0.0

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        records = self.series_records[uid]
        volumes, positions, present, meta, geometry = [], [], [], [], []
        for record in records:
            image, position, flag = self._load_b42(
                uid, record["series_uid"], str(record["plane"])
            )
            volumes.append(image)
            positions.append(position)
            present.append(flag)
            meta.append([record["plane_id"], record["fluid_id"], record["fat_id"]])
            geometry.append(
                {
                    "series_uid": str(record["series_uid"]),
                    "height": int(image.shape[-2]),
                    "width": int(image.shape[-1]),
                    "present": bool(flag > 0),
                }
            )
        position_tensor = torch.stack(positions)
        if len(self.center_offsets) == 1:
            volumes = [volume[0] for volume in volumes]
            position_tensor = position_tensor[:, 0]
        item = {
            "study_uid": uid,
            "volumes": volumes,
            "slice_position": position_tensor,
            "present": torch.tensor(present, dtype=torch.float32),
            "series_meta": torch.tensor(meta, dtype=torch.long),
            "geometry": geometry,
        }
        if self.targets is not None:
            item["target"] = torch.from_numpy(np.asarray(self.targets[idx], dtype=np.float32))
        if self.weights is not None:
            item["weight"] = torch.from_numpy(np.asarray(self.weights[idx], dtype=np.float32))
        return item


class B42ConstantAreaAspectSparseMILResidual(B37HighResSparseMILResidual):
    """B37 hierarchy/head with per-series rectangular ConvNeXt encoding."""

    def _encode_rect_group(self, group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if group.ndim != 4:
            raise ValueError("B42 group must be [G,3,H,W]")
        g, c, h, w = group.shape
        if c != 3 or h < B42_STRIDE_ALIGNMENT or w < B42_STRIDE_ALIGNMENT:
            raise ValueError("B42 group requires 3-channel stride-valid rectangles")
        if h % B42_STRIDE_ALIGNMENT or w % B42_STRIDE_ALIGNMENT:
            raise ValueError("B42 rectangular inputs must be stride aligned")
        global_blocks, spatial_blocks = [], []
        use_checkpoint = bool(self.training and self.encoder_trainable_stages > 0)
        for chunk in group.split(self.encoder_chunk_size, dim=0):
            if use_checkpoint:
                global_feature, spatial = checkpoint(
                    self._encode_chunk,
                    chunk,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                global_feature, spatial = self._encode_chunk(chunk)
            global_blocks.append(global_feature)
            spatial_blocks.append(spatial)
        return torch.cat(global_blocks, dim=0), torch.cat(spatial_blocks, dim=0)

    def _encode_ragged_study(
        self,
        volumes: list[torch.Tensor],
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(volumes, list) or not volumes:
            raise ValueError("B42 expects a non-empty list of series tensors")
        if present.ndim == 2:
            if int(present.shape[0]) != 1:
                raise ValueError("B42 forward processes exactly one study at a time")
            present_flat = present[0]
        elif present.ndim == 1:
            present_flat = present
        else:
            raise ValueError("B42 present mask must be [K] or [1,K]")
        if len(volumes) != int(present_flat.numel()):
            raise ValueError("B42 volumes/present series count mismatch")

        global_rows: list[torch.Tensor | None] = []
        spatial_rows: list[torch.Tensor | None] = []
        template_global = template_spatial = None
        for series_tensor, flag in zip(volumes, present_flat):
            if series_tensor.ndim != 4 or int(series_tensor.shape[0]) != B35_DENSE_SLICES:
                raise ValueError("B42 series must be [32,3,H,W]")
            if float(flag.detach().item()) <= 0:
                global_rows.append(None)
                spatial_rows.append(None)
                continue
            base_global, base_spatial = self._encode_rect_group(
                series_tensor[:B35_BASE_SLICES]
            )
            extra_global, extra_spatial = self._encode_rect_group(
                series_tensor[B35_BASE_SLICES:]
            )
            global_series = torch.cat((base_global, extra_global), dim=0)
            spatial_series = torch.cat((base_spatial, extra_spatial), dim=0)
            global_rows.append(global_series)
            spatial_rows.append(spatial_series)
            if template_global is None:
                template_global, template_spatial = global_series, spatial_series

        if template_global is None or template_spatial is None:
            raise RuntimeError("B42 study has no readable MRI series")
        for index in range(len(global_rows)):
            if global_rows[index] is None:
                global_rows[index] = torch.zeros_like(template_global)
                spatial_rows[index] = torch.zeros_like(template_spatial)
        global_feature = torch.stack([x for x in global_rows if x is not None], dim=0).unsqueeze(0)
        spatial = torch.stack([x for x in spatial_rows if x is not None], dim=0).unsqueeze(0)
        return global_feature, spatial

    def forward(
        self,
        volumes: list[torch.Tensor],
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> B37Forward:
        if present.ndim == 1:
            present = present.unsqueeze(0)
        if series_meta.ndim == 2:
            series_meta = series_meta.unsqueeze(0)
        if slice_position.ndim == 2:
            slice_position = slice_position.unsqueeze(0)
        global_feature, spatial = self._encode_ragged_study(volumes, present)
        base_logits = self._base_logits_from_global(global_feature, present, series_meta)
        local_logits, top_indices, top_values = self.head(
            spatial, present, series_meta, slice_position
        )
        gate = self.head.effective_gate().to(dtype=local_logits.dtype)
        logits = base_logits.float() + gate[None, :] * local_logits.float()
        return B37Forward(
            logits=logits,
            base_logits=base_logits,
            local_logits=local_logits,
            top_indices=top_indices,
            top_values=top_values,
        )

    def state(self) -> dict:
        state = super().state()
        state.update(
            {
                "version": B42_VERSION,
                "input_geometry": B42_RESIZE_POLICY,
                "reference_pixel_area": B42_REFERENCE_AREA,
                "stride_alignment": B42_STRIDE_ALIGNMENT,
                "padding_mode": B42_PADDING_MODE,
                "ragged_series_encoding": True,
            }
        )
        return state


def b42_preprocessing_state() -> dict:
    return {
        "normalization": "full native volume before crop",
        "crop_fraction": B37_CROP_FRACTION,
        "crop_stage": "native resolution before deterministic resize",
        "resize_policy": B42_RESIZE_POLICY,
        "preserves_in_plane_aspect_ratio": True,
        "reference_pixel_area": B42_REFERENCE_AREA,
        "reference_side": B42_REFERENCE_SIDE,
        "deterministic_resize_count": 1,
        "resize": "bilinear antialias=True align_corners=False",
        "stride_alignment": B42_STRIDE_ALIGNMENT,
        "padding": {
            "mode": B42_PADDING_MODE,
            "maximum_total_added_pixels_per_axis": B42_STRIDE_ALIGNMENT - 1,
            "square_padding": False,
        },
        "ragged_series_encoding": True,
        "dense_slices": B35_DENSE_SLICES,
    }


__all__ = [
    "B42_VERSION",
    "B42_EXPERIMENT",
    "B42_RUN_ROOT",
    "B42_EXPERT58_ROOT",
    "B42_REFERENCE_AREA",
    "B42_STRIDE_ALIGNMENT",
    "B42_RESIZE_POLICY",
    "B42ConstantAreaAspectDataset",
    "B42ConstantAreaAspectSparseMILResidual",
    "b42_preprocessing_state",
    "collate_b42",
    "constant_area_shape",
    "preprocess_dense_triplets_b42",
    "require_b42_contract",
    "resize_triplets_constant_area",
]
