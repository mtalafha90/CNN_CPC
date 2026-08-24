"""B45 prospective plane-calibrated target-conditioned sparse MIL.

B43/B44 mechanistic diagnostics on the reused Expert-58 surface identified a
specific failure mode in frozen B42: global sparse top-k selection strongly
favours axial tokens even when another plane carries more discriminative signal
(e.g. ACL sagittal), while doubling deterministic centre coverage from 32 to 64
changes the weak targets negligibly.  B45 therefore keeps the complete B42 image
representation/training contract and changes only how local evidence is routed
across anatomical planes.

For every target, sparse top-k=8 log-mean-exp pooling is performed independently
inside each available plane.  Plane identity is deliberately removed from the
per-token evidence score and enters only through a learned target-specific
three-plane router.  The router is initialized with zero logits, giving uniform
weight over the planes actually present in a study.  No target-specific clinical
plane prior is hard coded and Expert-58 labels never enter training.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from .b35_target_spatial_residual import B35_POSITION_BASIS, _position_basis
from .b36_sparse_mil import B36SparseMILHead
from .b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectSparseMILResidual,
    B42_EFFECTIVE_BATCH,
    B42_EXPERT58_ROOT,
    B42_RUN_ROOT,
    require_b42_contract,
)
from .constants import N_TARGETS

B45_VERSION = "b45_plane_calibrated_target_conditioned_sparse_mil_v1"
B45_EXPERIMENT = "B45_plane_calibrated_target_conditioned_sparse_MIL"
B45_NUMBERED_CONTAINER = "runs/078_Experiment_B45_plane_calibrated_sparse_mil"
B45_RUN_ROOT = f"{B45_NUMBERED_CONTAINER}/b45_plane_calibrated_sparse_mil"
B45_EXPERT58_ROOT = f"{B45_RUN_ROOT}/expert58"
B45_PLANE_COUNT = 3
B45_PLANE_IDS = (1, 2, 3)
B45_PLANE_POOLING = "independent_plane_topk_lme_then_target_softmax"
B45_PLANE_ROUTER_INIT = "zero_logits_uniform_over_available_planes"
B45_PLANE_ROUTER_TEMPERATURE = 1.0
B45_REMOVE_PLANE_EMBEDDING_FROM_TOKEN_SCORE = True
B45_EFFECTIVE_BATCH = B42_EFFECTIVE_BATCH


def require_b45_contract(config: dict) -> dict:
    """Require B42 unchanged plus the single prospective B45 routing mechanism."""
    crop = require_b42_contract(config)
    pooling = str(config.get("b45_plane_pooling", B45_PLANE_POOLING))
    if pooling != B45_PLANE_POOLING:
        raise ValueError(f"B45 freezes b45_plane_pooling={B45_PLANE_POOLING!r}; got {pooling!r}")
    router_init = str(config.get("b45_plane_router_init", B45_PLANE_ROUTER_INIT))
    if router_init != B45_PLANE_ROUTER_INIT:
        raise ValueError(
            f"B45 freezes b45_plane_router_init={B45_PLANE_ROUTER_INIT!r}; got {router_init!r}"
        )
    temperature = float(
        config.get("b45_plane_router_temperature", B45_PLANE_ROUTER_TEMPERATURE)
    )
    if not math.isclose(
        temperature, B45_PLANE_ROUTER_TEMPERATURE, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "B45 freezes b45_plane_router_temperature="
            f"{B45_PLANE_ROUTER_TEMPERATURE}; got {temperature}"
        )
    remove_plane = bool(
        config.get(
            "b45_remove_plane_embedding_from_token_score",
            B45_REMOVE_PLANE_EMBEDDING_FROM_TOKEN_SCORE,
        )
    )
    if remove_plane is not B45_REMOVE_PLANE_EMBEDDING_FROM_TOKEN_SCORE:
        raise ValueError("B45 requires plane identity to be excluded from token evidence scores")
    if int(config.get("b45_plane_count", B45_PLANE_COUNT)) != B45_PLANE_COUNT:
        raise ValueError("B45 freezes three anatomical plane pools")
    if int(config.get("b45_top_k_per_plane", config.get("b37_top_k", 8))) != int(
        config.get("b37_top_k", 8)
    ):
        raise ValueError("B45 retains B42 top-k independently inside every available plane")
    return crop


@dataclass(frozen=True)
class B45HeadForward:
    local_logits: torch.Tensor
    plane_logits: torch.Tensor
    plane_weights: torch.Tensor
    plane_top_indices: torch.Tensor
    plane_top_values: torch.Tensor
    plane_available: torch.Tensor


class B45PlaneCalibratedSparseMILHead(B36SparseMILHead):
    """B42 sparse evidence scorer with explicit target-conditioned plane fusion."""

    def __init__(
        self,
        dim: int = 768,
        *,
        grid_size: int = 6,
        top_k: int = 8,
        temperature: float = 1.0,
        router_temperature: float = B45_PLANE_ROUTER_TEMPERATURE,
        initial_b42_head: B36SparseMILHead | None = None,
    ) -> None:
        super().__init__(
            dim,
            grid_size=int(grid_size),
            top_k=int(top_k),
            temperature=float(temperature),
        )
        self.router_temperature = float(router_temperature)
        if self.router_temperature <= 0:
            raise ValueError("B45 plane-router temperature must be positive")
        self.plane_router_logits = nn.Parameter(
            torch.zeros(N_TARGETS, B45_PLANE_COUNT, dtype=torch.float32)
        )

        # For a clean B42-vs-B45 construction, copy the complete initialized B42
        # sparse head before changing only the routing semantics.  The plane
        # embedding remains in the state for checkpoint transparency but is
        # frozen at its inherited initialization and is never added to tokens.
        if initial_b42_head is not None:
            inherited = initial_b42_head.state_dict()
            missing, unexpected = self.load_state_dict(inherited, strict=False)
            if set(missing) != {"plane_router_logits"} or unexpected:
                raise RuntimeError(
                    "B45 could not inherit exact B42 sparse-head initialization: "
                    f"missing={missing} unexpected={unexpected}"
                )
            with torch.no_grad():
                self.plane_router_logits.zero_()
        self.plane_embedding.weight.requires_grad_(False)

    def _tokens_without_plane(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if spatial.ndim != 5:
            raise ValueError("B45 spatial features must be [B,K,S,R,D]")
        b, k, s, r, d = spatial.shape
        if d != self.dim or r != self.n_regions:
            raise ValueError("B45 spatial feature shape does not match the head")
        if present.shape != (b, k):
            raise ValueError("B45 present mask shape mismatch")
        if series_meta.shape != (b, k, 3):
            raise ValueError("B45 series metadata shape mismatch")
        if slice_position.shape != (b, k, s):
            raise ValueError("B45 slice-position shape mismatch")

        x = F.layer_norm(spatial.float(), (d,)).to(dtype=spatial.dtype)
        pos = self.position_projection(_position_basis(slice_position).to(x.device)).to(
            dtype=x.dtype
        )
        region = self.region_embedding.to(dtype=x.dtype)[None, None, None, :, :]
        fluid = self.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
        fat = self.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        metadata = (fluid + fat).to(dtype=x.dtype)
        x = x + pos[:, :, :, None, :] + region + metadata[:, :, None, None, :]
        x = self.token_dropout(x)

        tokens = x.reshape(b, k * s * r, d)
        invalid = (
            (present <= 0)[:, :, None, None]
            .expand(b, k, s, r)
            .reshape(b, k * s * r)
        )
        plane = (
            series_meta[:, :, 0]
            .clamp(0, 3)[:, :, None, None]
            .expand(b, k, s, r)
            .reshape(b, k * s * r)
        )
        if invalid.all(dim=1).any():
            raise RuntimeError("B45 received a study with no readable MRI series")
        return tokens, invalid, plane

    def forward_details(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> B45HeadForward:
        tokens, invalid, token_plane = self._tokens_without_plane(
            spatial, present, series_meta, slice_position
        )
        score = torch.einsum(
            "bnd,td->btn",
            tokens,
            self.evidence_weight.to(dtype=tokens.dtype),
        ) + self.evidence_bias.to(dtype=tokens.dtype)[None, :, None]
        score = score.masked_fill(invalid[:, None, :], float("-inf"))

        b = int(score.shape[0])
        plane_logits = score.new_zeros((b, N_TARGETS, B45_PLANE_COUNT), dtype=torch.float32)
        plane_top_values = score.new_full(
            (b, N_TARGETS, B45_PLANE_COUNT, self.top_k),
            float("-inf"),
            dtype=torch.float32,
        )
        plane_top_indices = torch.full(
            (b, N_TARGETS, B45_PLANE_COUNT, self.top_k),
            -1,
            dtype=torch.long,
            device=score.device,
        )
        plane_available = torch.zeros(
            (b, B45_PLANE_COUNT), dtype=torch.bool, device=score.device
        )

        tau = float(self.temperature)
        for plane_slot, plane_id in enumerate(B45_PLANE_IDS):
            member = (token_plane == int(plane_id)) & (~invalid)
            available = member.any(dim=1)
            plane_available[:, plane_slot] = available
            if not available.any():
                continue
            masked = score.masked_fill(~member[:, None, :], float("-inf"))
            # Every present MRI series supplies 32*R tokens, therefore an
            # available plane always has far more than top_k valid instances.
            top_values, top_indices = torch.topk(
                masked,
                k=self.top_k,
                dim=-1,
                largest=True,
                sorted=True,
            )
            plane_top_values[:, :, plane_slot] = top_values.float()
            plane_top_indices[:, :, plane_slot] = top_indices
            pooled = tau * (
                torch.logsumexp(top_values.float() / tau, dim=-1)
                - math.log(float(self.top_k))
            )
            # Missing-plane rows contain -inf top values; overwrite them with 0
            # because the router masks those planes before softmax.
            pooled = torch.where(available[:, None], pooled, torch.zeros_like(pooled))
            plane_logits[:, :, plane_slot] = pooled

        if (~plane_available).all(dim=1).any():
            raise RuntimeError("B45 study has no recognized available plane")

        route = self.plane_router_logits[None, :, :].expand(b, -1, -1)
        route = route / float(self.router_temperature)
        route = route.masked_fill(~plane_available[:, None, :], float("-inf"))
        plane_weights = torch.softmax(route.float(), dim=-1)
        local_logits = (plane_weights * plane_logits.float()).sum(dim=-1)
        return B45HeadForward(
            local_logits=local_logits,
            plane_logits=plane_logits.float(),
            plane_weights=plane_weights.float(),
            plane_top_indices=plane_top_indices,
            plane_top_values=plane_top_values.float(),
            plane_available=plane_available,
        )

    def forward(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        details = self.forward_details(spatial, present, series_meta, slice_position)
        # Compatibility tuple for B42-style callers.  The top-k dimensions are
        # now [B,T,plane,K] rather than one cross-plane top-k list.
        return (
            details.local_logits,
            details.plane_top_indices,
            details.plane_top_values,
        )

    def state(self) -> dict:
        base = super().state()
        base.update(
            {
                "version": B45_VERSION,
                "plane_pooling": B45_PLANE_POOLING,
                "plane_count": B45_PLANE_COUNT,
                "plane_ids": list(B45_PLANE_IDS),
                "plane_router_temperature": self.router_temperature,
                "plane_router_init": B45_PLANE_ROUTER_INIT,
                "plane_embedding_used_in_token_score": False,
                "plane_embedding_trainable": bool(self.plane_embedding.weight.requires_grad),
                "plane_router_logits": self.plane_router_logits.detach().float().cpu().tolist(),
                "plane_router_weights_all_planes": torch.softmax(
                    self.plane_router_logits.detach().float() / self.router_temperature,
                    dim=-1,
                ).cpu().tolist(),
            }
        )
        return base


@dataclass(frozen=True)
class B45Forward:
    logits: torch.Tensor
    base_logits: torch.Tensor
    local_logits: torch.Tensor
    plane_logits: torch.Tensor
    plane_weights: torch.Tensor
    plane_top_indices: torch.Tensor
    plane_top_values: torch.Tensor
    plane_available: torch.Tensor


class B45PlaneCalibratedSparseMILResidual(B42ConstantAreaAspectSparseMILResidual):
    """B42 ragged image representation with B45 plane-calibrated local routing."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        grid_size: int = 6,
        top_k: int = 8,
        temperature: float = 1.0,
        encoder_trainable_stages: int = 1,
        encoder_chunk_size: int = 4,
        router_temperature: float = B45_PLANE_ROUTER_TEMPERATURE,
    ) -> None:
        super().__init__(
            base_model,
            grid_size=int(grid_size),
            top_k=int(top_k),
            temperature=float(temperature),
            encoder_trainable_stages=int(encoder_trainable_stages),
            encoder_chunk_size=int(encoder_chunk_size),
        )
        b42_head = self.head
        self.head = B45PlaneCalibratedSparseMILHead(
            int(self.base.encoder.out_dim),
            grid_size=int(grid_size),
            top_k=int(top_k),
            temperature=float(temperature),
            router_temperature=float(router_temperature),
            initial_b42_head=b42_head,
        )

    def forward(
        self,
        volumes: list[torch.Tensor],
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> B45Forward:
        if present.ndim == 1:
            present = present.unsqueeze(0)
        if series_meta.ndim == 2:
            series_meta = series_meta.unsqueeze(0)
        if slice_position.ndim == 2:
            slice_position = slice_position.unsqueeze(0)
        global_feature, spatial = self._encode_ragged_study(volumes, present)
        base_logits = self._base_logits_from_global(global_feature, present, series_meta)
        details = self.head.forward_details(spatial, present, series_meta, slice_position)
        gate = self.head.effective_gate().to(dtype=details.local_logits.dtype)
        logits = base_logits.float() + gate[None, :] * details.local_logits.float()
        return B45Forward(
            logits=logits,
            base_logits=base_logits,
            local_logits=details.local_logits,
            plane_logits=details.plane_logits,
            plane_weights=details.plane_weights,
            plane_top_indices=details.plane_top_indices,
            plane_top_values=details.plane_top_values,
            plane_available=details.plane_available,
        )

    def state(self) -> dict:
        state = super().state()
        state.update(
            {
                "version": B45_VERSION,
                "experiment": B45_EXPERIMENT,
                "plane_pooling": B45_PLANE_POOLING,
                "plane_router": self.head.state(),
            }
        )
        return state


__all__ = [
    "B45_EFFECTIVE_BATCH",
    "B45_EXPERIMENT",
    "B45_EXPERT58_ROOT",
    "B45_NUMBERED_CONTAINER",
    "B45_PLANE_COUNT",
    "B45_PLANE_IDS",
    "B45_PLANE_POOLING",
    "B45_PLANE_ROUTER_INIT",
    "B45_PLANE_ROUTER_TEMPERATURE",
    "B45_REMOVE_PLANE_EMBEDDING_FROM_TOKEN_SCORE",
    "B45_RUN_ROOT",
    "B45_VERSION",
    "B45Forward",
    "B45HeadForward",
    "B45PlaneCalibratedSparseMILHead",
    "B45PlaneCalibratedSparseMILResidual",
    "require_b45_contract",
]
