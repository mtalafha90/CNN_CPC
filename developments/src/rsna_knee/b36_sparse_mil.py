"""B36: pathology-specific sparse top-k spatial MIL over the frozen B34 model.

B35 established two useful facts:

1. the 32-centre/3x3 local feature path can reproduce the frozen B34 predictor
   exactly, and
2. dense softmax attention remained almost uniform (normalized entropy ~1), so
   the intended focal localization mechanism never emerged.

B36 keeps the validated B35 data/encoder path but replaces dense attention with
explicit sparse multiple-instance learning (MIL).  Every local ConvNeXt token is
scored independently for each pathology; only the strongest ``top_k`` evidence
locations contribute to that pathology's local logit via a smooth log-mean-exp
pool.  A direct auxiliary local loss (implemented in ``b36_training.py``) gives
this scorer full gradient even while the residual gate starts at exactly zero.

The frozen B34 inference path therefore remains bitwise/numerically identical at
initialization, while the local head is no longer gradient-starved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .b35_exact_batch import B35TargetSpatialResidualExactBatch
from .b35_target_spatial_residual import (
    B35_GRID_SIZE,
    B35_POSITION_BASIS,
    B35_TOKEN_DROPOUT,
    _position_basis,
)
from .constants import N_TARGETS

B36_VERSION = "b36_pathology_sparse_topk_mil_residual_v1"
B36_TOP_K = 8
B36_TEMPERATURE = 1.0
B36_LOCAL_AUX_WEIGHT = 1.0


class B36SparseMILHead(nn.Module):
    """Pathology-specific local evidence scorer with sparse top-k MIL pooling."""

    def __init__(
        self,
        dim: int = 768,
        *,
        grid_size: int = B35_GRID_SIZE,
        top_k: int = B36_TOP_K,
        temperature: float = B36_TEMPERATURE,
        token_dropout: float = B35_TOKEN_DROPOUT,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.grid_size = int(grid_size)
        self.n_regions = self.grid_size * self.grid_size
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        if self.top_k < 1:
            raise ValueError("B36 top_k must be >=1")
        if self.temperature <= 0:
            raise ValueError("B36 temperature must be positive")

        self.token_dropout = nn.Dropout(float(token_dropout))
        self.position_projection = nn.Linear(B35_POSITION_BASIS, self.dim, bias=False)
        self.region_embedding = nn.Parameter(torch.zeros(self.n_regions, self.dim))
        self.plane_embedding = nn.Embedding(4, self.dim, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, self.dim, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, self.dim, padding_idx=0)

        # These are evidence classifiers, not attention queries.  Their dot
        # products are token-level logits and are intentionally not divided by
        # sqrt(D), which would make the initial evidence surface nearly flat.
        self.evidence_weight = nn.Parameter(torch.empty(N_TARGETS, self.dim))
        self.evidence_bias = nn.Parameter(torch.zeros(N_TARGETS))
        self.gate = nn.Parameter(torch.zeros(N_TARGETS))

        nn.init.zeros_(self.position_projection.weight)
        nn.init.zeros_(self.plane_embedding.weight)
        nn.init.zeros_(self.fluid_embedding.weight)
        nn.init.zeros_(self.fat_embedding.weight)
        nn.init.normal_(self.evidence_weight, mean=0.0, std=0.02)

    def effective_gate(self) -> torch.Tensor:
        return torch.tanh(self.gate)

    def _tokens(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spatial.ndim != 5:
            raise ValueError("B36 spatial features must be [B,K,S,R,D]")
        b, k, s, r, d = spatial.shape
        if d != self.dim or r != self.n_regions:
            raise ValueError("B36 spatial feature shape does not match the head")
        if present.shape != (b, k):
            raise ValueError("B36 present mask shape mismatch")
        if series_meta.shape != (b, k, 3):
            raise ValueError("B36 series metadata shape mismatch")
        if slice_position.shape != (b, k, s):
            raise ValueError("B36 slice-position shape mismatch")

        # Parameter-free LN keeps the pretrained local representation on a
        # stable per-token scale before pathology-specific evidence scoring.
        x = F.layer_norm(spatial.float(), (d,)).to(dtype=spatial.dtype)
        pos = self.position_projection(_position_basis(slice_position).to(x.device)).to(
            dtype=x.dtype
        )
        region = self.region_embedding.to(dtype=x.dtype)[None, None, None, :, :]
        plane = self.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
        fluid = self.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
        fat = self.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        metadata = (plane + fluid + fat).to(dtype=x.dtype)
        x = x + pos[:, :, :, None, :] + region + metadata[:, :, None, None, :]
        x = self.token_dropout(x)
        tokens = x.reshape(b, k * s * r, d)
        invalid = (
            (present <= 0)[:, :, None, None]
            .expand(b, k, s, r)
            .reshape(b, k * s * r)
        )
        if invalid.all(dim=1).any():
            raise RuntimeError("B36 received a study with no readable MRI series")
        if int((~invalid).sum(dim=1).min().item()) < self.top_k:
            raise RuntimeError("B36 has fewer valid local tokens than top_k")
        return tokens, invalid

    def forward(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, invalid = self._tokens(
            spatial,
            present,
            series_meta,
            slice_position,
        )
        # [B,T,N] token-level pathology evidence logits.
        score = torch.einsum(
            "bnd,td->btn",
            tokens,
            self.evidence_weight.to(dtype=tokens.dtype),
        ) + self.evidence_bias.to(dtype=tokens.dtype)[None, :, None]
        score = score.masked_fill(invalid[:, None, :], float("-inf"))

        top_values, top_indices = torch.topk(
            score,
            k=self.top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )
        # Smooth max over only the explicitly selected evidence instances.
        # Subtracting log(k) keeps the pooled logit on the token-logit scale.
        tau = float(self.temperature)
        local_logits = tau * (
            torch.logsumexp(top_values.float() / tau, dim=-1)
            - math.log(float(self.top_k))
        )
        return local_logits, top_indices, top_values.float()

    def state(self) -> dict:
        raw = self.gate.detach().float().cpu()
        effective = torch.tanh(raw)
        return {
            "version": B36_VERSION,
            "grid_size": self.grid_size,
            "regions_per_slice": self.n_regions,
            "feature_dim": self.dim,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "gate_raw": [float(x) for x in raw.tolist()],
            "gate_effective": [float(x) for x in effective.tolist()],
            "gate_effective_abs_mean": float(effective.abs().mean().item()),
            "gate_effective_abs_max": float(effective.abs().max().item()),
        }


@dataclass(frozen=True)
class B36Forward:
    logits: torch.Tensor
    base_logits: torch.Tensor
    local_logits: torch.Tensor
    top_indices: torch.Tensor
    top_values: torch.Tensor


class B36SparseMILResidual(B35TargetSpatialResidualExactBatch):
    """Frozen B34 plus zero-gated, directly supervised sparse spatial MIL."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        grid_size: int = B35_GRID_SIZE,
        top_k: int = B36_TOP_K,
        temperature: float = B36_TEMPERATURE,
    ) -> None:
        # Parent establishes the frozen B34 contract and exact-batch encoder.
        super().__init__(base_model, grid_size=int(grid_size))
        self.head = B36SparseMILHead(
            int(self.base.encoder.out_dim),
            grid_size=int(grid_size),
            top_k=int(top_k),
            temperature=float(temperature),
        )

    def forward(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> B36Forward:
        with torch.no_grad():
            global_feature, spatial = self._encode_combined(volumes, present)
            base_logits = self._base_logits_from_global(
                global_feature,
                present,
                series_meta,
            )
        local_logits, top_indices, top_values = self.head(
            spatial.detach(),
            present,
            series_meta,
            slice_position,
        )
        gate = self.head.effective_gate().to(dtype=local_logits.dtype)
        logits = base_logits.detach().float() + gate[None, :] * local_logits.float()
        return B36Forward(
            logits=logits,
            base_logits=base_logits.detach(),
            local_logits=local_logits,
            top_indices=top_indices,
            top_values=top_values,
        )
