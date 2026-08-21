"""B37: high-resolution B36 sparse MIL with limited encoder adaptation.

This is a prospective joint experiment motivated by two completed diagnostics:

* the native-resolution audit showed a median in-plane sampling of 0.3125 mm/pixel
  and a median matrix of 512x512, while matrix size and PixelSpacing vary widely;
* B36 proved that pathology-specific sparse top-k localization can learn, but the
  frozen 224 representation did not turn that localization into expert-AUC gain.

B37 therefore keeps the B36 sparse-MIL mechanism but gives it a substantially
richer image representation:

    full-native-volume normalization
    -> fixed 90% center crop at native resolution
    -> one antialiased bilinear resize to 448x448
    -> 32 deterministic 2.5D centres
    -> ConvNeXt local features on a 6x6 grid
    -> pathology-specific top-k=8 log-mean-exp MIL

The B34 non-encoder predictor stays frozen and in eval mode.  Only the final
ConvNeXt stage/output norm and the B36 sparse-MIL head learn.  This preserves the
B34 deployed aggregation function while allowing the representation itself to
adapt to the higher-resolution input.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .b12_variable_series import VariableSeriesKneeDataset
from .b17_training import freeze_encoder
from .b20_crop_focus import b20_crop_focus_policy
from .b35_target_spatial_residual import (
    B35_BASE_SLICES,
    B35_DENSE_SLICES,
    b35_centers,
    collate_b35,
)
from .b36_sparse_mil import B36SparseMILHead
from .crop_focus import validate_crop_focus_policy
from .dicom import _normalise_volume, find_series_dir
from .encoder_finetune import unfreeze_encoder_tail

B37_VERSION = "b37_highres448_b36_sparse_mil_encoder_tail_v1"
B37_NUMBERED_CONTAINER = "runs/071_Experiment_B37_highres_448_sparse_mil"
B37_RUN_ROOT = f"{B37_NUMBERED_CONTAINER}/b37_highres_sparse_mil"
B37_EXPERT58_ROOT = f"{B37_RUN_ROOT}/expert58"
B37_IMAGE_SIZE = 448
B37_CROP_FRACTION = 0.90
B37_GRID_SIZE = 6
B37_TOP_K = 8
B37_TEMPERATURE = 1.0
B37_LOCAL_AUX_WEIGHT = 1.0
B37_ENCODER_TRAINABLE_STAGES = 1
B37_ENCODER_LR_SCALE = 0.05
B37_ENCODER_CHUNK_SIZE = 4
B37_RESIZE_MODE = "bilinear_antialias_align_corners_false"


def require_b37_sparse_contract(config: dict) -> dict:
    """Require every prospectively frozen B37 high-resolution mechanism choice."""
    image_size = int(config.get("b7_image_size", B37_IMAGE_SIZE))
    grid_size = int(config.get("b37_grid_size", B37_GRID_SIZE))
    top_k = int(config.get("b37_top_k", B37_TOP_K))
    temperature = float(config.get("b37_temperature", B37_TEMPERATURE))
    aux = float(config.get("b37_local_aux_weight", B37_LOCAL_AUX_WEIGHT))
    stages = int(
        config.get("b37_encoder_trainable_stages", B37_ENCODER_TRAINABLE_STAGES)
    )
    scale = float(config.get("b37_encoder_lr_scale", B37_ENCODER_LR_SCALE))
    chunk = int(config.get("b37_encoder_chunk_size", B37_ENCODER_CHUNK_SIZE))

    expected = {
        "b7_image_size": (image_size, B37_IMAGE_SIZE),
        "b37_grid_size": (grid_size, B37_GRID_SIZE),
        "b37_top_k": (top_k, B37_TOP_K),
        "b37_encoder_trainable_stages": (
            stages,
            B37_ENCODER_TRAINABLE_STAGES,
        ),
        "b37_encoder_chunk_size": (chunk, B37_ENCODER_CHUNK_SIZE),
    }
    for key, (value, frozen) in expected.items():
        if value != frozen:
            raise ValueError(f"B37 freezes {key}={frozen}; got {value}")
    for key, value, frozen in (
        ("b37_temperature", temperature, B37_TEMPERATURE),
        ("b37_local_aux_weight", aux, B37_LOCAL_AUX_WEIGHT),
        ("b37_encoder_lr_scale", scale, B37_ENCODER_LR_SCALE),
    ):
        if not np.isclose(value, frozen, atol=1e-12, rtol=0):
            raise ValueError(f"B37 freezes {key}={frozen}; got {value}")

    crop = b20_crop_focus_policy({**config, "b7_image_size": 224})
    if not np.isclose(
        float(crop["crop_fraction"]), B37_CROP_FRACTION, atol=1e-12, rtol=0
    ):
        raise ValueError("B37 requires the historical fixed 90% center crop")
    return crop


def _native_center_crop(triplets: np.ndarray, fraction: float) -> np.ndarray:
    x = np.asarray(triplets)
    if x.ndim != 4:
        raise ValueError(f"B37 expected [S,C,H,W], got {x.shape}")
    h, w = int(x.shape[-2]), int(x.shape[-1])
    crop_h = max(2, min(h, int(round(h * float(fraction)))))
    crop_w = max(2, min(w, int(round(w * float(fraction)))))
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return x[..., top : top + crop_h, left : left + crop_w]


def preprocess_dense_triplets_b37(
    raw: np.ndarray,
    *,
    image_size: int = B37_IMAGE_SIZE,
    gap: int = 1,
    center_offset: int = 0,
    crop_fraction: float = B37_CROP_FRACTION,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return 32 B36-style triplets after one native-crop -> 448 resize.

    Percentile normalization intentionally uses the complete native volume before
    cropping, preserving the historical B20 normalization support while avoiding
    B20's resize->crop->resize sequence.
    """
    if int(image_size) != B37_IMAGE_SIZE:
        raise ValueError(f"B37 output size must remain {B37_IMAGE_SIZE}")
    if int(gap) < 1:
        raise ValueError("B37 2.5D gap must be positive")

    normalized = _normalise_volume(raw)
    centers, position = b35_centers(
        len(normalized),
        gap=int(gap),
        center_offset=int(center_offset),
    )
    offsets = np.asarray([-int(gap), 0, int(gap)], dtype=np.int64)
    index = np.clip(
        centers[:, None] + offsets[None, :],
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
    return resized, position


class B37HighResSparseDataset(VariableSeriesKneeDataset):
    """All-series 32-centre B37 dataset with native crop and one 448 resize."""

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
            targets=targets,
            weights=weights,
            train=False,
        )
        self.crop_focus_policy = validate_crop_focus_policy(crop_focus_policy)
        self.center_offsets = tuple(int(x) for x in center_offsets)
        if int(config.image_size) != B37_IMAGE_SIZE:
            raise ValueError("B37 dataset requires 448x448 output")
        if not self.center_offsets:
            raise ValueError("B37 requires at least one center offset")
        if not np.isclose(
            float(self.crop_focus_policy["crop_fraction"]),
            B37_CROP_FRACTION,
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("B37 dataset requires the fixed 90% crop")

    def _zero(self) -> tuple[torch.Tensor, torch.Tensor]:
        views = len(self.center_offsets)
        images = torch.zeros(
            views,
            B35_DENSE_SLICES,
            3,
            B37_IMAGE_SIZE,
            B37_IMAGE_SIZE,
            dtype=torch.float32,
        )
        positions = torch.zeros(
            views,
            B35_DENSE_SLICES,
            dtype=torch.float32,
        )
        return images, positions

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
            images, positions = [], []
            for offset in self.center_offsets:
                image, pos = preprocess_dense_triplets_b37(
                    raw,
                    image_size=B37_IMAGE_SIZE,
                    gap=int(self.config.triplet_gap),
                    center_offset=int(offset),
                    crop_fraction=float(self.crop_focus_policy["crop_fraction"]),
                )
                images.append(image)
                positions.append(torch.from_numpy(pos))
            return torch.stack(images), torch.stack(positions), 1.0
        except Exception:
            if self.config.strict_dicom:
                raise
            image, position = self._zero()
            return image, position, 0.0

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        records = self.series_records[uid]
        volumes, positions, present, meta = [], [], [], []
        for record in records:
            image, position, flag = self._load_b37(
                uid,
                record["series_uid"],
                str(record["plane"]),
            )
            volumes.append(image)
            positions.append(position)
            present.append(flag)
            meta.append(
                [record["plane_id"], record["fluid_id"], record["fat_id"]]
            )
        volume = torch.stack(volumes).permute(1, 0, 2, 3, 4, 5).contiguous()
        position = torch.stack(positions).permute(1, 0, 2).contiguous()
        item = {
            "study_uid": uid,
            "volumes": volume,
            "slice_position": position,
            "present": torch.tensor(present, dtype=torch.float32),
            "series_meta": torch.tensor(meta, dtype=torch.long),
        }
        if len(self.center_offsets) == 1:
            item["volumes"] = item["volumes"][0]
            item["slice_position"] = item["slice_position"][0]
        if self.targets is not None:
            item["target"] = torch.from_numpy(
                np.asarray(self.targets[idx], dtype=np.float32)
            )
        if self.weights is not None:
            item["weight"] = torch.from_numpy(
                np.asarray(self.weights[idx], dtype=np.float32)
            )
        return item


@dataclass(frozen=True)
class B37Forward:
    logits: torch.Tensor
    base_logits: torch.Tensor
    local_logits: torch.Tensor
    top_indices: torch.Tensor
    top_values: torch.Tensor


class B37HighResSparseMILResidual(nn.Module):
    """Frozen B34 aggregation + trainable encoder tail + B36 sparse local head."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        grid_size: int = B37_GRID_SIZE,
        top_k: int = B37_TOP_K,
        temperature: float = B37_TEMPERATURE,
        encoder_trainable_stages: int = B37_ENCODER_TRAINABLE_STAGES,
        encoder_chunk_size: int = B37_ENCODER_CHUNK_SIZE,
    ) -> None:
        super().__init__()
        self.base = base_model
        freeze_encoder(self.base)
        for name, parameter in self.base.named_parameters():
            if not name.startswith("encoder."):
                parameter.requires_grad_(False)
        self.finetune = unfreeze_encoder_tail(
            self.base,
            int(encoder_trainable_stages),
        )
        self.encoder_trainable_stages = int(encoder_trainable_stages)
        self.encoder_chunk_size = int(encoder_chunk_size)
        if self.encoder_chunk_size < 1:
            raise ValueError("B37 encoder chunk size must be positive")
        self.base.encoder_batch_size = self.encoder_chunk_size
        self.base.eval()
        if int(self.base.n_slices) != B35_BASE_SLICES:
            raise ValueError("B37 requires a 16-slice B34 base")

        dim = int(self.base.encoder.out_dim)
        self.head = B36SparseMILHead(
            dim,
            grid_size=int(grid_size),
            top_k=int(top_k),
            temperature=float(temperature),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # The deployed B34 hierarchy remains in its exact evaluation path; only
        # encoder gradients and the new sparse head are enabled.
        self.base.eval()
        self.base.encoder.eval()
        self.head.train(mode)
        return self

    def _encode_chunk(self, chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self.base.encoder
        normalized = encoder._normalize(chunk)
        fmap = encoder.features(normalized)
        global_feature = encoder.pre_classifier(encoder.avgpool(fmap)).reshape(
            chunk.shape[0], int(encoder.out_dim)
        )
        pooled = F.adaptive_avg_pool2d(
            fmap,
            (int(self.head.grid_size), int(self.head.grid_size)),
        )
        normalized_grid = encoder.pre_classifier[0](pooled)
        spatial = normalized_grid.permute(0, 2, 3, 1).reshape(
            chunk.shape[0], self.head.n_regions, int(encoder.out_dim)
        )
        return global_feature, spatial

    def _encode_active_group(
        self,
        active_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if active_group.ndim != 5:
            raise ValueError("B37 active group must be [N,G,3,H,W]")
        n, g, c, h, w = active_group.shape
        if c != 3 or h != B37_IMAGE_SIZE or w != B37_IMAGE_SIZE:
            raise ValueError("B37 active group requires 3x448x448 triplets")
        flat = active_group.reshape(n * g, c, h, w)
        global_blocks, spatial_blocks = [], []
        use_checkpoint = bool(self.training and self.encoder_trainable_stages > 0)
        for chunk in flat.split(self.encoder_chunk_size, dim=0):
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
        dim = int(self.base.encoder.out_dim)
        regions = int(self.head.n_regions)
        return (
            torch.cat(global_blocks, dim=0).reshape(n, g, dim),
            torch.cat(spatial_blocks, dim=0).reshape(n, g, regions, dim),
        )

    def _encode_combined(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if volumes.ndim != 6:
            raise ValueError("B37 expects [B,K,32,3,448,448]")
        b, k, s, c, h, w = volumes.shape
        if (
            s != B35_DENSE_SLICES
            or c != 3
            or h != B37_IMAGE_SIZE
            or w != B37_IMAGE_SIZE
        ):
            raise ValueError("B37 input shape does not match the frozen protocol")
        active_indices = torch.nonzero(
            present.reshape(-1) > 0,
            as_tuple=False,
        ).flatten()
        if active_indices.numel() == 0:
            raise RuntimeError("B37 batch has no readable MRI series")
        flat_series = volumes.reshape(b * k, s, c, h, w)
        active = flat_series.index_select(0, active_indices)

        # Preserve the B35/B36 first-16 / extra-16 grouping.  The first group has
        # the exact ordering used by a 16-centre B34 pass at the same resolution.
        base_global, base_spatial = self._encode_active_group(
            active[:, :B35_BASE_SLICES]
        )
        extra_global, extra_spatial = self._encode_active_group(
            active[:, B35_BASE_SLICES:]
        )
        global_active = torch.cat((base_global, extra_global), dim=1)
        spatial_active = torch.cat((base_spatial, extra_spatial), dim=1)

        dim = int(self.base.encoder.out_dim)
        regions = int(self.head.n_regions)
        all_global = global_active.new_zeros((b * k, s, dim)).index_copy(
            0,
            active_indices,
            global_active,
        )
        all_spatial = spatial_active.new_zeros(
            (b * k, s, regions, dim)
        ).index_copy(0, active_indices, spatial_active)
        return (
            all_global.reshape(b, k, s, dim),
            all_spatial.reshape(b, k, s, regions, dim),
        )

    def _base_logits_from_global(
        self,
        global_feature: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base
        x = global_feature[:, :, :B35_BASE_SLICES]
        plane = base.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
        fluid = base.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
        fat = base.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        metadata = plane + fluid + fat
        mask = present[:, :, None, None].to(x.dtype)
        x = (
            x
            + base.slice_position[None, None, :, :]
            + metadata[:, :, None, :]
        ) * mask
        tokens = base._pool_real_series_b31(x, present)
        padding = present <= 0
        empty = padding.all(dim=1)
        safe_padding = padding.clone()
        if empty.any():
            safe_padding[empty, 0] = False
            tokens = tokens.clone()
            tokens[empty, 0] = 0
        memory = base.context(tokens, src_key_padding_mask=safe_padding)
        memory = memory.masked_fill(padding[:, :, None], 0.0)
        queries = base.pathology_tokens[None, :, :].expand(memory.shape[0], -1, -1)
        queries = base.pathology_context(queries)
        attended, _ = base.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=safe_padding,
            need_weights=False,
        )
        queries = base.dropout(base.query_norm(queries + attended))
        logits = (
            queries * base.target_weight[None, :, :]
        ).sum(dim=-1) + base.target_bias
        return torch.where(empty[:, None], base.target_bias[None, :], logits)

    def forward(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> B37Forward:
        global_feature, spatial = self._encode_combined(volumes, present)
        base_logits = self._base_logits_from_global(
            global_feature,
            present,
            series_meta,
        )
        local_logits, top_indices, top_values = self.head(
            spatial,
            present,
            series_meta,
            slice_position,
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

    @torch.no_grad()
    def base_equivalence_error_448(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> float:
        """Guard only the reconstructed B34 function at the new 448 input size."""
        was_training = self.training
        self.eval()
        reference = self.base(
            volumes[:, :, :B35_BASE_SLICES],
            present,
            series_meta,
        )
        global_feature, _ = self._encode_combined(volumes, present)
        reconstructed = self._base_logits_from_global(
            global_feature,
            present,
            series_meta,
        )
        if was_training:
            self.train(True)
        return float(
            (reference.float() - reconstructed.float()).abs().max().item()
        )

    def state(self) -> dict:
        trainable_encoder = sum(
            p.numel() for p in self.base.encoder.parameters() if p.requires_grad
        )
        frozen_base = sum(
            p.numel() for name, p in self.base.named_parameters()
            if not name.startswith("encoder.") and not p.requires_grad
        )
        return {
            "version": B37_VERSION,
            "image_size": B37_IMAGE_SIZE,
            "crop_fraction": B37_CROP_FRACTION,
            "grid_size": int(self.head.grid_size),
            "regions_per_slice": int(self.head.n_regions),
            "dense_slices": B35_DENSE_SLICES,
            "base_slices": B35_BASE_SLICES,
            "top_k": int(self.head.top_k),
            "temperature": float(self.head.temperature),
            "encoder_chunk_size": self.encoder_chunk_size,
            "encoder_trainable_stages": self.encoder_trainable_stages,
            "encoder_trainable_parameters": int(trainable_encoder),
            "frozen_nonencoder_base_parameters": int(frozen_base),
            "head": self.head.state(),
        }


__all__ = [
    "B37_VERSION",
    "B37_NUMBERED_CONTAINER",
    "B37_RUN_ROOT",
    "B37_EXPERT58_ROOT",
    "B37_IMAGE_SIZE",
    "B37_GRID_SIZE",
    "B37_TOP_K",
    "B37_LOCAL_AUX_WEIGHT",
    "B37HighResSparseDataset",
    "B37HighResSparseMILResidual",
    "collate_b35",
    "preprocess_dense_triplets_b37",
    "require_b37_sparse_contract",
]
