"""B38: 448 native-crop global B34 ablation with one encoder tail stage free.

B38 is deliberately the smallest high-resolution counterpart to B37.  It keeps
the native-volume normalization, 90% native centre crop, one 448 resize, and
limited ConvNeXt-tail adaptation, while removing every B37 sparse-MIL mechanism:

* exactly the historical 16 deterministic 2.5D centres;
* the deployed B34 global aggregation only;
* no local feature grid, sparse head, residual gate, or local auxiliary loss.

This isolates whether B37's gain can be explained by the higher-resolution
global path and encoder adaptation alone.  It is not a replacement for B37 and
does not alter B37's frozen checkpoint or protocol.
"""
from __future__ import annotations

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
    b35_centers,
    collate_b35,
)
from .crop_focus import validate_crop_focus_policy
from .dicom import _normalise_volume, find_series_dir
from .encoder_finetune import unfreeze_encoder_tail

B38_VERSION = "b38_highres448_global16_encoder_tail_v1"
B38_NUMBERED_CONTAINER = "runs/073_Experiment_B38_highres_448_global_tail_ablation"
B38_RUN_ROOT = f"{B38_NUMBERED_CONTAINER}/b38_highres_global_tail"
B38_EXPERT58_ROOT = f"{B38_RUN_ROOT}/expert58"

B38_IMAGE_SIZE = 448
B38_CROP_FRACTION = 0.90
B38_N_SLICES = B35_BASE_SLICES
B38_ENCODER_TRAINABLE_STAGES = 1
B38_ENCODER_LR_SCALE = 0.05
B38_ENCODER_CHUNK_SIZE = 4
B38_TAIL_REFERENCE_LR = 1e-4
B38_RESIZE_MODE = "bilinear_antialias_align_corners_false"


def require_b38_global_contract(config: dict) -> dict:
    """Require every frozen B38 choice and reject sparse-MIL carry-over fields."""
    sparse_keys = (
        "b37_grid_size",
        "b37_top_k",
        "b37_temperature",
        "b37_local_aux_weight",
    )
    supplied_sparse = [key for key in sparse_keys if key in config]
    if supplied_sparse:
        raise ValueError(
            "B38 is global-only and must not receive B37 sparse-MIL fields: "
            + ", ".join(supplied_sparse)
        )

    image_size = int(config.get("b7_image_size", B38_IMAGE_SIZE))
    n_slices = int(config.get("b7_n_slices", B38_N_SLICES))
    stages = int(
        config.get("b38_encoder_trainable_stages", B38_ENCODER_TRAINABLE_STAGES)
    )
    scale = float(config.get("b38_encoder_lr_scale", B38_ENCODER_LR_SCALE))
    chunk = int(config.get("b38_encoder_chunk_size", B38_ENCODER_CHUNK_SIZE))
    reference_lr = float(
        config.get("b38_tail_reference_lr", B38_TAIL_REFERENCE_LR)
    )

    expected = {
        "b7_image_size": (image_size, B38_IMAGE_SIZE),
        "b7_n_slices": (n_slices, B38_N_SLICES),
        "b38_encoder_trainable_stages": (
            stages,
            B38_ENCODER_TRAINABLE_STAGES,
        ),
        "b38_encoder_chunk_size": (chunk, B38_ENCODER_CHUNK_SIZE),
    }
    for key, (value, frozen) in expected.items():
        if value != frozen:
            raise ValueError(f"B38 freezes {key}={frozen}; got {value}")
    for key, value, frozen in (
        ("b38_encoder_lr_scale", scale, B38_ENCODER_LR_SCALE),
        ("b38_tail_reference_lr", reference_lr, B38_TAIL_REFERENCE_LR),
    ):
        if not np.isclose(value, frozen, atol=1e-12, rtol=0):
            raise ValueError(f"B38 freezes {key}={frozen}; got {value}")

    # The B20 policy is historically defined at 224.  B38 applies the same
    # 90% geometrical crop before its single 448 resize, so validate its policy
    # through the historical contract while independently freezing the 448 size.
    crop = b20_crop_focus_policy({**config, "b7_image_size": 224})
    if not np.isclose(
        float(crop["crop_fraction"]), B38_CROP_FRACTION, atol=1e-12, rtol=0
    ):
        raise ValueError("B38 requires the historical fixed 90% centre crop")
    return crop


def _native_center_crop(triplets: np.ndarray, fraction: float) -> np.ndarray:
    """Crop equally on all in-plane sides before the sole deterministic resize."""
    x = np.asarray(triplets)
    if x.ndim != 4:
        raise ValueError(f"B38 expected [S,C,H,W], got {x.shape}")
    h, w = int(x.shape[-2]), int(x.shape[-1])
    crop_h = max(2, min(h, int(round(h * float(fraction)))))
    crop_w = max(2, min(w, int(round(w * float(fraction)))))
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return x[..., top : top + crop_h, left : left + crop_w]


def preprocess_global_triplets_b38(
    raw: np.ndarray,
    *,
    image_size: int = B38_IMAGE_SIZE,
    gap: int = 1,
    center_offset: int = 0,
    crop_fraction: float = B38_CROP_FRACTION,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return the historical 16 B34 centres after native crop -> 448 resize.

    Percentile normalization uses the complete native volume before cropping.
    The sixteen centre positions are exactly the B34/B35-base positions for the
    given volume and centre offset; only the in-plane preprocessing changes.
    """
    if int(image_size) != B38_IMAGE_SIZE:
        raise ValueError(f"B38 output size must remain {B38_IMAGE_SIZE}")
    if int(gap) < 1:
        raise ValueError("B38 2.5D gap must be positive")

    normalized = _normalise_volume(raw)
    centers, position = b35_centers(
        len(normalized),
        gap=int(gap),
        center_offset=int(center_offset),
        base_slices=B38_N_SLICES,
        dense_slices=B38_N_SLICES,
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
        size=(B38_IMAGE_SIZE, B38_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized, position


class B38HighResGlobalDataset(VariableSeriesKneeDataset):
    """All-series B38 dataset using 16 historical centres at 448x448."""

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
        if int(config.image_size) != B38_IMAGE_SIZE:
            raise ValueError("B38 dataset requires 448x448 output")
        if not self.center_offsets:
            raise ValueError("B38 requires at least one centre offset")
        if not np.isclose(
            float(self.crop_focus_policy["crop_fraction"]),
            B38_CROP_FRACTION,
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("B38 dataset requires the fixed 90% crop")

    def _zero(self) -> tuple[torch.Tensor, torch.Tensor]:
        views = len(self.center_offsets)
        images = torch.zeros(
            views,
            B38_N_SLICES,
            3,
            B38_IMAGE_SIZE,
            B38_IMAGE_SIZE,
            dtype=torch.float32,
        )
        positions = torch.zeros(
            views,
            B38_N_SLICES,
            dtype=torch.float32,
        )
        return images, positions

    def _load_b38(self, uid: str, series_uid: str, plane: str):
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
                image, pos = preprocess_global_triplets_b38(
                    raw,
                    image_size=B38_IMAGE_SIZE,
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
            image, position, flag = self._load_b38(
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
class B38Forward:
    """The frozen B34 global hierarchy evaluated from adapted 448 features."""

    logits: torch.Tensor


class B38HighResGlobalTail(nn.Module):
    """Frozen B34 aggregation with only the final ConvNeXt stage trainable."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        encoder_trainable_stages: int = B38_ENCODER_TRAINABLE_STAGES,
        encoder_chunk_size: int = B38_ENCODER_CHUNK_SIZE,
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
            raise ValueError("B38 encoder chunk size must be positive")
        self.base.encoder_batch_size = self.encoder_chunk_size
        self.base.eval()
        if int(self.base.n_slices) != B38_N_SLICES:
            raise ValueError("B38 requires a 16-slice B34 base")

    def train(self, mode: bool = True):
        super().train(mode)
        # Preserve the deployed B34 hierarchy exactly.  Eval mode does not block
        # gradients for the explicitly unfrozen ConvNeXt tail.
        self.base.eval()
        self.base.encoder.eval()
        return self

    def _encode_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        encoder = self.base.encoder
        normalized = encoder._normalize(chunk)
        fmap = encoder.features(normalized)
        return encoder.pre_classifier(encoder.avgpool(fmap)).reshape(
            chunk.shape[0],
            int(encoder.out_dim),
        )

    def _encode_global(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        if volumes.ndim != 6:
            raise ValueError("B38 expects [B,K,16,3,448,448]")
        b, k, s, c, h, w = volumes.shape
        if (
            s != B38_N_SLICES
            or c != 3
            or h != B38_IMAGE_SIZE
            or w != B38_IMAGE_SIZE
        ):
            raise ValueError("B38 input shape does not match the frozen protocol")

        active_indices = torch.nonzero(
            present.reshape(-1) > 0,
            as_tuple=False,
        ).flatten()
        if active_indices.numel() == 0:
            raise RuntimeError("B38 batch has no readable MRI series")
        flat_series = volumes.reshape(b * k, s, c, h, w)
        active = flat_series.index_select(0, active_indices)
        flat = active.reshape(-1, c, h, w)

        blocks = []
        use_checkpoint = bool(
            self.training and self.encoder_trainable_stages > 0
        )
        for chunk in flat.split(self.encoder_chunk_size, dim=0):
            if use_checkpoint:
                block = checkpoint(
                    self._encode_chunk,
                    chunk,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                block = self._encode_chunk(chunk)
            blocks.append(block)
        dim = int(self.base.encoder.out_dim)
        global_active = torch.cat(blocks, dim=0).reshape(
            active.shape[0],
            B38_N_SLICES,
            dim,
        )
        all_global = global_active.new_zeros(
            (b * k, B38_N_SLICES, dim)
        ).index_copy(0, active_indices, global_active)
        return all_global.reshape(b, k, B38_N_SLICES, dim)

    def _base_logits_from_global(
        self,
        global_feature: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base
        if global_feature.ndim != 4 or global_feature.shape[2] != B38_N_SLICES:
            raise ValueError("B38 global features must be [B,K,16,D]")
        plane = base.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
        fluid = base.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
        fat = base.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        metadata = plane + fluid + fat
        mask = present[:, :, None, None].to(global_feature.dtype)
        x = (
            global_feature
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
        queries = base.pathology_tokens[None, :, :].expand(
            memory.shape[0],
            -1,
            -1,
        )
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
    ) -> B38Forward:
        global_feature = self._encode_global(volumes, present)
        logits = self._base_logits_from_global(
            global_feature,
            present,
            series_meta,
        )
        return B38Forward(logits=logits)

    @torch.no_grad()
    def base_equivalence_error_448(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> float:
        """Guard that the reconstructed global path matches B34 at 448 input."""
        was_training = self.training
        self.eval()
        reference = self.base(volumes, present, series_meta)
        reconstructed = self(
            volumes,
            present,
            series_meta,
        ).logits
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
            p.numel()
            for name, p in self.base.named_parameters()
            if not name.startswith("encoder.") and not p.requires_grad
        )
        return {
            "version": B38_VERSION,
            "image_size": B38_IMAGE_SIZE,
            "crop_fraction": B38_CROP_FRACTION,
            "n_slices": B38_N_SLICES,
            "global_aggregation": "frozen_b34_only",
            "sparse_mil": False,
            "local_auxiliary_loss": False,
            "encoder_chunk_size": self.encoder_chunk_size,
            "encoder_trainable_stages": self.encoder_trainable_stages,
            "encoder_trainable_parameters": int(trainable_encoder),
            "frozen_nonencoder_base_parameters": int(frozen_base),
        }


__all__ = [
    "B38_VERSION",
    "B38_NUMBERED_CONTAINER",
    "B38_RUN_ROOT",
    "B38_EXPERT58_ROOT",
    "B38_IMAGE_SIZE",
    "B38_N_SLICES",
    "B38HighResGlobalDataset",
    "B38HighResGlobalTail",
    "collate_b35",
    "preprocess_global_triplets_b38",
    "require_b38_global_contract",
]
