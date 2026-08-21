"""B35 pathology-conditioned spatial residual over the current B34 model.

B35 is deliberately a *probe* of the suspected information bottleneck rather
than another wholesale replacement.  The complete B34 predictor is frozen and
kept in eval mode.  A new branch sees coarse spatial ConvNeXt features before
B34's global slice pooling and learns one pathology-specific attention summary
per target.  A target-wise tanh gate starts at exactly zero, therefore the B35
prediction is exactly the frozen B34 prediction at initialization.

The image encoder is shared with B34 and remains frozen in Phase A.  For each
real series, the first 16 sampled centres are exactly the historical B34 centres;
16 extra centres are interleaved deterministically from a 32-centre grid.  A
single ConvNeXt pass over those 32 centres supplies both:

* global vectors for the historical 16-centre B34 path, and
* a 3x3 spatial grid for all 32 centres for the new local-evidence path.

This makes the experiment substantially cheaper than running independent 16-
and 32-slice encoders and lets the code assert that the reconstructed B34 base
logits are numerically equivalent to the ordinary frozen B34 forward pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .b12_variable_series import VariableSeriesKneeDataset
from .constants import N_TARGETS
from .crop_focus import apply_crop_focus, validate_crop_focus_policy
from .dicom import _centers, _normalise_volume, find_series_dir

B35_VERSION = "b35_target_conditioned_spatial_residual_probe_v1"
B35_BASE_SLICES = 16
B35_DENSE_SLICES = 32
B35_GRID_SIZE = 3
B35_REGIONS = B35_GRID_SIZE * B35_GRID_SIZE
B35_POSITION_BASIS = 8
B35_TOKEN_DROPOUT = 0.05


def _extra_centers(base: np.ndarray, dense: np.ndarray, count: int) -> np.ndarray:
    """Pick deterministic dense-grid centres not already used by the base path."""
    base_list = [int(x) for x in np.asarray(base).reshape(-1)]
    dense_list = [int(x) for x in np.asarray(dense).reshape(-1)]
    used = set(base_list)
    extras = [x for x in dense_list if x not in used]
    if len(extras) < count:
        # Short series can quantize several requested centres to the same frame.
        # Repetition is preferable to inventing coordinates outside the scan.
        for x in dense_list:
            extras.append(x)
            if len(extras) >= count:
                break
    while len(extras) < count:
        extras.append(base_list[len(extras) % len(base_list)])
    return np.asarray(extras[:count], dtype=np.int64)


def b35_centers(
    n_frames: int,
    *,
    gap: int = 1,
    center_offset: int = 0,
    base_slices: int = B35_BASE_SLICES,
    dense_slices: int = B35_DENSE_SLICES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 32 centres whose first 16 exactly reproduce the B34 centres."""
    if dense_slices < base_slices:
        raise ValueError("dense_slices must be >= base_slices")
    base = _centers(
        n_frames,
        base_slices,
        gap,
        center_offset=center_offset,
        jitter=0,
    )
    dense = _centers(
        n_frames,
        dense_slices,
        gap,
        center_offset=center_offset,
        jitter=0,
    )
    extras = _extra_centers(base, dense, dense_slices - base_slices)
    combined = np.concatenate([base, extras]).astype(np.int64, copy=False)
    denom = float(max(n_frames - 1, 1))
    normalized_position = combined.astype(np.float32) / denom
    return combined, normalized_position


def _triplets_at_centers(
    raw: np.ndarray,
    centers: np.ndarray,
    *,
    image_size: int,
    gap: int,
) -> torch.Tensor:
    v = _normalise_volume(raw)
    offsets = np.asarray([-gap, 0, gap], dtype=np.int64)
    idx = np.clip(
        np.asarray(centers, dtype=np.int64)[:, None] + offsets[None, :],
        0,
        len(v) - 1,
    )
    tensor = torch.from_numpy(v[idx].astype(np.float32, copy=False))
    return F.interpolate(
        tensor,
        (int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
    )


class B35SpatialDataset(VariableSeriesKneeDataset):
    """All-series dataset returning shared 32-centre B35 image stacks.

    The first 16 centres in every view are the exact ordinary B34 centres for
    that same centre offset.  ``center_offsets=(-1,0,1)`` therefore reproduces
    the historical B34 TTA while also producing spatial evidence at 16 extra
    centres in each view.
    """

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
        if not self.center_offsets:
            raise ValueError("B35 requires at least one centre offset")
        if int(config.triplet_gap) < 1:
            raise ValueError("B35 triplet gap must be positive")

    def _zero_b35(self) -> tuple[torch.Tensor, torch.Tensor]:
        v = len(self.center_offsets)
        images = torch.zeros(
            v,
            B35_DENSE_SLICES,
            3,
            int(self.config.image_size),
            int(self.config.image_size),
            dtype=torch.float32,
        )
        pos = torch.zeros(v, B35_DENSE_SLICES, dtype=torch.float32)
        return images, pos

    def _load_b35(self, uid: str, series_uid: str, plane: str):
        path = find_series_dir(
            self.config.data_root,
            self.config.split,
            uid,
            str(series_uid),
        )
        if path is None:
            if self.config.strict_dicom:
                raise FileNotFoundError(f"missing series {uid}/{series_uid}")
            images, pos = self._zero_b35()
            return images, pos, 0.0
        try:
            raw = self._read_volume(path, plane.lower())
            views, positions = [], []
            for offset in self.center_offsets:
                centers, normalized = b35_centers(
                    len(raw),
                    gap=int(self.config.triplet_gap),
                    center_offset=int(offset),
                )
                image = _triplets_at_centers(
                    raw,
                    centers,
                    image_size=int(self.config.image_size),
                    gap=int(self.config.triplet_gap),
                )
                image = apply_crop_focus(image, self.crop_focus_policy)
                views.append(image)
                positions.append(torch.from_numpy(normalized))
            return torch.stack(views), torch.stack(positions), 1.0
        except Exception:
            if self.config.strict_dicom:
                raise
            images, pos = self._zero_b35()
            return images, pos, 0.0

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        records = self.series_records[uid]
        volumes, positions, present, meta = [], [], [], []
        for record in records:
            image, pos, flag = self._load_b35(
                uid,
                record["series_uid"],
                str(record["plane"]),
            )
            volumes.append(image)
            positions.append(pos)
            present.append(flag)
            meta.append(
                [record["plane_id"], record["fluid_id"], record["fat_id"]]
            )
        # [K,V,S,C,H,W] -> [V,K,S,C,H,W]
        volume = torch.stack(volumes).permute(1, 0, 2, 3, 4, 5).contiguous()
        pos = torch.stack(positions).permute(1, 0, 2).contiguous()
        item = {
            "study_uid": uid,
            "volumes": volume,
            "slice_position": pos,
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


def collate_b35(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("cannot collate an empty B35 batch")
    max_k = max(int(item["present"].shape[0]) for item in batch)
    first = batch[0]["volumes"]
    b = len(batch)
    if first.ndim == 5:  # [K,S,C,H,W]
        _, s, c, h, w = first.shape
        volumes = first.new_zeros((b, max_k, s, c, h, w))
        positions = torch.zeros((b, max_k, s), dtype=torch.float32)
        for i, item in enumerate(batch):
            k = item["volumes"].shape[0]
            volumes[i, :k] = item["volumes"]
            positions[i, :k] = item["slice_position"]
    elif first.ndim == 6:  # [V,K,S,C,H,W]
        v, _, s, c, h, w = first.shape
        volumes = first.new_zeros((b, v, max_k, s, c, h, w))
        positions = torch.zeros((b, v, max_k, s), dtype=torch.float32)
        for i, item in enumerate(batch):
            if item["volumes"].shape[0] != v:
                raise ValueError("B35 batch has inconsistent view counts")
            k = item["volumes"].shape[1]
            volumes[i, :, :k] = item["volumes"]
            positions[i, :, :k] = item["slice_position"]
    else:
        raise ValueError(f"unexpected B35 volume shape {tuple(first.shape)}")

    present = torch.zeros((b, max_k), dtype=torch.float32)
    meta = torch.zeros((b, max_k, 3), dtype=torch.long)
    for i, item in enumerate(batch):
        k = item["present"].shape[0]
        present[i, :k] = item["present"]
        meta[i, :k] = item["series_meta"]

    out = {
        "study_uid": [str(item["study_uid"]) for item in batch],
        "volumes": volumes,
        "slice_position": positions,
        "present": present,
        "series_meta": meta,
    }
    if all("target" in item for item in batch):
        out["target"] = torch.stack([item["target"] for item in batch])
    if all("weight" in item for item in batch):
        out["weight"] = torch.stack([item["weight"] for item in batch])
    return out


def _position_basis(position: torch.Tensor) -> torch.Tensor:
    """Deterministic 8-D continuous through-plane coordinate basis."""
    z = position.float().clamp(0.0, 1.0)
    return torch.stack(
        [
            z,
            z.square(),
            torch.sin(math.pi * z),
            torch.cos(math.pi * z),
            torch.sin(2.0 * math.pi * z),
            torch.cos(2.0 * math.pi * z),
            torch.sin(4.0 * math.pi * z),
            torch.cos(4.0 * math.pi * z),
        ],
        dim=-1,
    )


class TargetSpatialHead(nn.Module):
    """Pathology-specific attention over local spatial tokens.

    All acquisition/position embeddings start at zero.  The branch therefore
    begins by asking target-specific questions directly of the pretrained local
    ConvNeXt features rather than overwriting them with random geometry codes.
    """

    def __init__(
        self,
        dim: int = 768,
        *,
        grid_size: int = B35_GRID_SIZE,
        token_dropout: float = B35_TOKEN_DROPOUT,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.grid_size = int(grid_size)
        self.n_regions = self.grid_size * self.grid_size
        self.token_dropout = nn.Dropout(float(token_dropout))
        self.position_projection = nn.Linear(B35_POSITION_BASIS, self.dim, bias=False)
        self.region_embedding = nn.Parameter(torch.zeros(self.n_regions, self.dim))
        self.plane_embedding = nn.Embedding(4, self.dim, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, self.dim, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, self.dim, padding_idx=0)
        self.target_query = nn.Parameter(torch.randn(N_TARGETS, self.dim) * 0.02)
        self.target_weight = nn.Parameter(torch.empty(N_TARGETS, self.dim))
        self.target_bias = nn.Parameter(torch.zeros(N_TARGETS))
        self.gate = nn.Parameter(torch.zeros(N_TARGETS))
        nn.init.zeros_(self.position_projection.weight)
        nn.init.zeros_(self.plane_embedding.weight)
        nn.init.zeros_(self.fluid_embedding.weight)
        nn.init.zeros_(self.fat_embedding.weight)
        nn.init.xavier_uniform_(self.target_weight)

    def effective_gate(self) -> torch.Tensor:
        return torch.tanh(self.gate)

    def forward(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spatial.ndim != 5:
            raise ValueError("B35 spatial features must be [B,K,S,R,D]")
        b, k, s, r, d = spatial.shape
        if d != self.dim or r != self.n_regions:
            raise ValueError("B35 spatial feature shape does not match the head")
        if present.shape != (b, k):
            raise ValueError("B35 present mask shape mismatch")
        if series_meta.shape != (b, k, 3):
            raise ValueError("B35 series metadata shape mismatch")
        if slice_position.shape != (b, k, s):
            raise ValueError("B35 slice-position shape mismatch")

        x = F.layer_norm(spatial.float(), (d,)).to(dtype=spatial.dtype)
        pos = self.position_projection(_position_basis(slice_position).to(x.device)).to(
            dtype=x.dtype
        )
        region = self.region_embedding.to(dtype=x.dtype)[None, None, None, :, :]
        plane = self.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
        fluid = self.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
        fat = self.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        metadata = (plane + fluid + fat).to(dtype=x.dtype)
        x = (
            x
            + pos[:, :, :, None, :]
            + region
            + metadata[:, :, None, None, :]
        )
        x = self.token_dropout(x)
        tokens = x.reshape(b, k * s * r, d)
        invalid = (
            (present <= 0)[:, :, None, None]
            .expand(b, k, s, r)
            .reshape(b, k * s * r)
        )
        if invalid.all(dim=1).any():
            raise RuntimeError("B35 received a study with no readable MRI series")

        query = self.target_query.to(dtype=tokens.dtype)
        score = torch.einsum("bnd,td->btn", tokens, query) / math.sqrt(float(d))
        score = score.masked_fill(invalid[:, None, :], float("-inf"))
        attention = torch.softmax(score.float(), dim=-1).to(dtype=tokens.dtype)
        summary = torch.einsum("btn,bnd->btd", attention, tokens)
        local_logits = (
            summary * self.target_weight.to(dtype=summary.dtype)[None, :, :]
        ).sum(dim=-1) + self.target_bias
        return local_logits, attention

    def state(self) -> dict:
        raw = self.gate.detach().float().cpu()
        effective = torch.tanh(raw)
        return {
            "version": B35_VERSION,
            "grid_size": self.grid_size,
            "regions_per_slice": self.n_regions,
            "feature_dim": self.dim,
            "gate_raw": [float(x) for x in raw.tolist()],
            "gate_effective": [float(x) for x in effective.tolist()],
            "gate_effective_abs_mean": float(effective.abs().mean().item()),
            "gate_effective_abs_max": float(effective.abs().max().item()),
        }


@dataclass(frozen=True)
class B35Forward:
    logits: torch.Tensor
    base_logits: torch.Tensor
    local_logits: torch.Tensor
    attention: torch.Tensor


class B35TargetSpatialResidual(nn.Module):
    """Frozen B34 predictor plus a zero-gated target-conditioned spatial branch."""

    def __init__(self, base_model: nn.Module, *, grid_size: int = B35_GRID_SIZE) -> None:
        super().__init__()
        self.base = base_model
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        d = int(self.base.encoder.out_dim)
        self.head = TargetSpatialHead(d, grid_size=int(grid_size))
        if int(self.base.n_slices) != B35_BASE_SLICES:
            raise ValueError(
                f"B35 requires a {B35_BASE_SLICES}-slice B34 base; got {self.base.n_slices}"
            )

    def train(self, mode: bool = True):
        super().train(mode)
        # B34's local-context branch is a training-only scaffold.  The base must
        # remain the exact deployed inference function while the new branch learns.
        self.base.eval()
        return self

    @torch.no_grad()
    def _encode_combined(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if volumes.ndim != 6:
            raise ValueError("B35 expects [B,K,32,3,H,W]")
        b, k, s, c, h, w = volumes.shape
        if s != B35_DENSE_SLICES or c != 3:
            raise ValueError("B35 requires exactly 32 sampled 3-channel positions")
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            raise RuntimeError("B35 batch has no readable MRI series")
        flat_series = volumes.reshape(b * k, s, c, h, w)
        active = flat_series.index_select(0, active_indices)
        flat = active.reshape(-1, c, h, w)
        encoder = self.base.encoder
        global_blocks, spatial_blocks = [], []
        for chunk in flat.split(int(self.base.encoder_batch_size), dim=0):
            normalized = encoder._normalize(chunk)
            fmap = encoder.features(normalized)
            global_feature = encoder.pre_classifier(encoder.avgpool(fmap)).reshape(
                chunk.shape[0], int(encoder.out_dim)
            )
            pooled = F.adaptive_avg_pool2d(fmap, (self.head.grid_size, self.head.grid_size))
            normalized_grid = encoder.pre_classifier[0](pooled)
            spatial_feature = normalized_grid.permute(0, 2, 3, 1).reshape(
                chunk.shape[0], self.head.n_regions, int(encoder.out_dim)
            )
            global_blocks.append(global_feature)
            spatial_blocks.append(spatial_feature)
        global_active = torch.cat(global_blocks, dim=0).reshape(active.shape[0], s, -1)
        spatial_active = torch.cat(spatial_blocks, dim=0).reshape(
            active.shape[0], s, self.head.n_regions, -1
        )
        d = int(encoder.out_dim)
        all_global = global_active.new_zeros((b * k, s, d)).index_copy(
            0, active_indices, global_active
        )
        all_spatial = spatial_active.new_zeros(
            (b * k, s, self.head.n_regions, d)
        ).index_copy(0, active_indices, spatial_active)
        return all_global.reshape(b, k, s, d), all_spatial.reshape(
            b, k, s, self.head.n_regions, d
        )

    def _base_logits_from_global(
        self,
        global_feature: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> torch.Tensor:
        # Reconstruct B34 eval from the exact first-16 global slice vectors.
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
    ) -> B35Forward:
        with torch.no_grad():
            global_feature, spatial = self._encode_combined(volumes, present)
            base_logits = self._base_logits_from_global(
                global_feature, present, series_meta
            )
        local_logits, attention = self.head(
            spatial.detach(), present, series_meta, slice_position
        )
        gate = self.head.effective_gate().to(dtype=local_logits.dtype)
        logits = base_logits.detach() + gate[None, :] * local_logits
        return B35Forward(logits, base_logits.detach(), local_logits, attention)

    @torch.no_grad()
    def base_equivalence_error(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> float:
        """Maximum difference from ordinary B34 on the exact first 16 centres."""
        self.base.eval()
        reference = self.base(
            volumes[:, :, :B35_BASE_SLICES],
            present,
            series_meta,
        )
        global_feature, _ = self._encode_combined(volumes, present)
        reconstructed = self._base_logits_from_global(
            global_feature, present, series_meta
        )
        return float((reference.float() - reconstructed.float()).abs().max().item())