"""B48 — pathology-specific global context softly conditions local evidence.

The B42 model already computes two compatible representations from each
ConvNeXt feature map:

* a global B34 hierarchy, whose pathology queries attend across the available
  MRI series; and
* local 6x6 ConvNeXt tokens, whose B36 sparse-MIL evidence scores are currently
  independent of that study-specific global state.

B48 keeps the complete B42 image path, sparse top-k pooling and residual
composition.  It adds one bounded, zero-start compatibility term between a
pathology query and every local token.  The term never removes a token from the
search and never uses a scalar probability as a hard routing decision.

Two matched arms are deliberately supported by one implementation:

``static_prior_control``
    The local head receives the frozen pathology query before it has attended
    to this study's series memory.  It controls for the added low-rank
    compatibility capacity.

``post_cross_attention_candidate``
    The local head receives the pathology query after its frozen B34
    cross-attention over this study's series memory.  This is the actual
    study-dependent global-to-spatial hypothesis.

Both query tensors are detached before the B48 head.  B48 therefore does not
turn the local auxiliary loss into a second training route through the frozen
B34 hierarchy; it only lets the already-computed global state guide local
ranking.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .b35_target_spatial_residual import B35_BASE_SLICES
from .b36_sparse_mil import B36SparseMILHead
from .b37_highres_sparse_mil import (
    B37_GRID_SIZE,
    B37_LOCAL_AUX_WEIGHT,
    B37_TEMPERATURE,
    B37_TOP_K,
)
from .b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectSparseMILResidual,
    require_b42_contract,
)
from .constants import N_TARGETS

B48_VERSION = "b48_global_query_conditioned_cross_series_sparse_mil_v1"
B48_EXPERIMENT = "B48_global_query_conditioned_cross_series_sparse_MIL"
B48_NUMBERED_CONTAINER = "runs/081_Experiment_B48_global_conditioned_spatial_mil"
B48_RUN_ROOT = f"{B48_NUMBERED_CONTAINER}/b48_global_conditioned_spatial_mil"

# 768 / the frozen B34 eight attention heads.  Using one existing head-width
# keeps the new compatibility space small and tied to the parent representation,
# rather than treating its rank as another tuning dimension.
B48_CONTEXT_DIM = 96
B48_CONTEXT_METRIC = "cosine_low_rank_query_token_compatibility"
B48_CONTEXT_EPS = 1e-6
B48_CONTEXT_GATE_INIT = "zero_tanh_targetwise"
B48_CONTEXT_QUERY_GRADIENT = "detached_before_local_head"
B48_FIXED_EPOCHS = 2
B48_SUPERVISION = "report_only_weak_no_official_gold"
B48_VALIDATION_SURFACE = "frozen_scanner_grouped_domain_split_v1"

B48_STATIC_PRIOR_CONTROL = "static_prior_control"
B48_POST_CROSS_ATTENTION_CANDIDATE = "post_cross_attention_candidate"
B48_ARMS = (B48_STATIC_PRIOR_CONTROL, B48_POST_CROSS_ATTENTION_CANDIDATE)
B48_ARM_CONTEXT_SOURCE = {
    B48_STATIC_PRIOR_CONTROL: "pathology_prior_before_series_cross_attention",
    B48_POST_CROSS_ATTENTION_CANDIDATE: "post_series_cross_attention_query",
}


@dataclass(frozen=True)
class B48HeadForward:
    """Sparse-MIL output plus lightweight conditioning audit values."""

    local_logits: torch.Tensor
    top_indices: torch.Tensor
    top_values: torch.Tensor
    # These are only populated when evaluation requests `audit_context=True`.
    base_top_indices: torch.Tensor | None
    base_top_values: torch.Tensor | None
    context_abs_mean: torch.Tensor
    topk_overlap_with_static: torch.Tensor | None


class B48GlobalConditionedSparseMILHead(B36SparseMILHead):
    """B42's sparse-MIL scorer plus a bounded global-query compatibility term.

    For a B42 local token ``x_i`` and a pathology-specific B34 global query
    ``q_t`` the evidence score is:

    ``e_t(x_i) + tanh(a_t) * cosine(Wq LN(q_t), Wk LN(x_i))``.

    ``a_t`` starts at zero, so the inherited B42 evidence surface is exact at
    initialization.  B42's existing direct local BCE gives ``a_t`` gradient
    immediately; after the first non-zero gate update, the two projections also
    receive gradients.  This staged behaviour is intentional and tested.
    """

    def __init__(
        self,
        dim: int = 768,
        *,
        grid_size: int = B37_GRID_SIZE,
        top_k: int = B37_TOP_K,
        temperature: float = B37_TEMPERATURE,
        context_dim: int = B48_CONTEXT_DIM,
        initial_b42_head: B36SparseMILHead | None = None,
    ) -> None:
        super().__init__(
            dim,
            grid_size=int(grid_size),
            top_k=int(top_k),
            temperature=float(temperature),
        )
        self.context_dim = int(context_dim)
        if self.context_dim < 1:
            raise ValueError("B48 context dimension must be positive")
        self.context_query = nn.Linear(self.dim, self.context_dim, bias=False)
        self.context_key = nn.Linear(self.dim, self.context_dim, bias=False)
        self.context_gate = nn.Parameter(torch.zeros(N_TARGETS, dtype=torch.float32))
        nn.init.xavier_uniform_(self.context_query.weight)
        nn.init.xavier_uniform_(self.context_key.weight)

        # Constructing B42 first and copying all of its initialized sparse-head
        # state means the two B48 arms share B42's exact inherited initialization.
        # The only non-inherited state is the new context branch, whose gate is
        # explicitly zero at step zero.
        if initial_b42_head is not None:
            inherited = initial_b42_head.state_dict()
            missing, unexpected = self.load_state_dict(inherited, strict=False)
            expected_missing = {
                "context_gate",
                "context_query.weight",
                "context_key.weight",
            }
            if set(missing) != expected_missing or unexpected:
                raise RuntimeError(
                    "B48 could not inherit exact B42 sparse-head initialization: "
                    f"missing={missing} unexpected={unexpected}"
                )
        with torch.no_grad():
            self.context_gate.zero_()

    def effective_context_gate(self) -> torch.Tensor:
        """Bound each pathology's global-conditioning contribution to [-1, 1]."""
        return torch.tanh(self.context_gate)

    def _base_score(self, tokens: torch.Tensor) -> torch.Tensor:
        """The B36/B42 evidence calculation, kept in its inherited precision."""
        return torch.einsum(
            "bnd,td->btn",
            tokens,
            self.evidence_weight.to(dtype=tokens.dtype),
        ) + self.evidence_bias.to(dtype=tokens.dtype)[None, :, None]

    def _context_residual(
        self,
        tokens: torch.Tensor,
        global_query: torch.Tensor,
        invalid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the detached-query compatibility residual and its valid mean.

        The context branch intentionally works in fp32 before its residual is
        cast back to B42's score dtype.  The score/top-k path itself is not
        upgraded to B47's fp32-ranking mechanism; B48 changes only the evidence
        term, not the inherited ranking precision contract.
        """
        if global_query.ndim != 3:
            raise ValueError("B48 global queries must be [B,T,D]")
        b, t, d = global_query.shape
        if b != int(tokens.shape[0]) or t != N_TARGETS or d != self.dim:
            raise ValueError("B48 global-query shape does not match local head")

        # This is a deliberate stop-gradient boundary.  The frozen B34 query is
        # evidence for the local head, not a second supervision route to B34.
        query = global_query.detach().float()
        token = F.layer_norm(tokens.float(), (self.dim,))
        query = F.layer_norm(query, (self.dim,))
        q = self.context_query(query)
        k = self.context_key(token)
        q = F.normalize(q.float(), p=2.0, dim=-1, eps=B48_CONTEXT_EPS)
        k = F.normalize(k.float(), p=2.0, dim=-1, eps=B48_CONTEXT_EPS)
        cosine = torch.einsum("btr,bnr->btn", q, k)
        residual = self.effective_context_gate().float()[None, :, None] * cosine

        valid = (~invalid).float()
        denominator = valid.sum(dim=-1).clamp_min(1.0)[:, None]
        mean_abs = (residual.abs() * valid[:, None, :]).sum(dim=-1) / denominator
        return residual, mean_abs

    def forward_details(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
        global_query: torch.Tensor,
        *,
        audit_context: bool = False,
    ) -> B48HeadForward:
        tokens, invalid = self._tokens(
            spatial,
            present,
            series_meta,
            slice_position,
        )
        base_score = self._base_score(tokens)
        context_residual, context_abs_mean = self._context_residual(
            tokens,
            global_query,
            invalid,
        )
        # Casting only the added term preserves B42's existing evidence/top-k
        # precision when the zero-start context gate is still closed.
        score = base_score + context_residual.to(dtype=base_score.dtype)
        score = score.masked_fill(invalid[:, None, :], float("-inf"))

        top_values, top_indices = torch.topk(
            score,
            k=self.top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )
        tau = float(self.temperature)
        local_logits = tau * (
            torch.logsumexp(top_values.float() / tau, dim=-1)
            - math.log(float(self.top_k))
        )

        base_top_indices = base_top_values = overlap = None
        if audit_context:
            static = base_score.masked_fill(invalid[:, None, :], float("-inf"))
            base_top_values, base_top_indices = torch.topk(
                static,
                k=self.top_k,
                dim=-1,
                largest=True,
                sorted=True,
            )
            # Set overlap, not ordered equality: a reordered identical top-k set
            # has not changed the locations the local branch considers.
            overlap = (
                (top_indices[..., :, None] == base_top_indices[..., None, :])
                .any(dim=-1)
                .float()
                .mean(dim=-1)
            )

        return B48HeadForward(
            local_logits=local_logits,
            top_indices=top_indices,
            top_values=top_values.float(),
            base_top_indices=base_top_indices,
            base_top_values=None if base_top_values is None else base_top_values.float(),
            context_abs_mean=context_abs_mean.float(),
            topk_overlap_with_static=overlap,
        )

    def forward(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
        global_query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        details = self.forward_details(
            spatial,
            present,
            series_meta,
            slice_position,
            global_query,
        )
        return details.local_logits, details.top_indices, details.top_values

    def state(self) -> dict:
        state = super().state()
        raw = self.context_gate.detach().float().cpu()
        effective = torch.tanh(raw)
        state.update(
            {
                "version": B48_VERSION,
                "context_metric": B48_CONTEXT_METRIC,
                "context_dim": self.context_dim,
                "context_eps": B48_CONTEXT_EPS,
                "context_query_gradient": B48_CONTEXT_QUERY_GRADIENT,
                "context_gate_init": B48_CONTEXT_GATE_INIT,
                "context_gate_raw": [float(x) for x in raw.tolist()],
                "context_gate_effective": [float(x) for x in effective.tolist()],
                "context_gate_effective_abs_mean": float(effective.abs().mean().item()),
                "context_gate_effective_abs_max": float(effective.abs().max().item()),
            }
        )
        return state


@dataclass(frozen=True)
class B48Forward:
    logits: torch.Tensor
    base_logits: torch.Tensor
    local_logits: torch.Tensor
    top_indices: torch.Tensor
    top_values: torch.Tensor
    context_query: torch.Tensor
    context_abs_mean: torch.Tensor
    topk_overlap_with_static: torch.Tensor | None


class B48GlobalConditionedSparseMILResidual(B42ConstantAreaAspectSparseMILResidual):
    """B42 ragged encoder with global-query-conditioned local sparse evidence."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        grid_size: int = B37_GRID_SIZE,
        top_k: int = B37_TOP_K,
        temperature: float = B37_TEMPERATURE,
        encoder_trainable_stages: int = 1,
        encoder_chunk_size: int = 4,
        arm: str = B48_POST_CROSS_ATTENTION_CANDIDATE,
        context_dim: int = B48_CONTEXT_DIM,
    ) -> None:
        if arm not in B48_ARMS:
            raise ValueError(f"B48 arm must be one of {B48_ARMS}; got {arm!r}")
        super().__init__(
            base_model,
            grid_size=int(grid_size),
            top_k=int(top_k),
            temperature=float(temperature),
            encoder_trainable_stages=int(encoder_trainable_stages),
            encoder_chunk_size=int(encoder_chunk_size),
        )
        self.arm = str(arm)
        self.context_source = B48_ARM_CONTEXT_SOURCE[self.arm]
        b42_head = self.head
        # Creating extra B48 parameters must not advance B42's historical
        # constructor RNG stream.  This matters for the paired-arm dropout
        # stream as well as for exact inherited B42 head initialisation.
        with torch.random.fork_rng(devices=[]):
            replacement = B48GlobalConditionedSparseMILHead(
                int(self.base.encoder.out_dim),
                grid_size=int(grid_size),
                top_k=int(top_k),
                temperature=float(temperature),
                context_dim=int(context_dim),
                initial_b42_head=b42_head,
            )
        self.head = replacement

    @torch.no_grad()
    def _global_query_states(
        self,
        global_feature: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reproduce B34 query states without adding a local-loss gradient path.

        Returns the static pre-memory control query, the post-cross-attention
        study query, and logits reconstructed from that post-attention query.
        The ordinary B42 base logit itself is still calculated by the unchanged
        parent method in :meth:`forward`.
        """
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
        prior = base.pathology_context(queries)
        attended, _ = base.cross_attention(
            prior,
            memory,
            memory,
            key_padding_mask=safe_padding,
            need_weights=False,
        )
        # Apply the same normalisation/dropout operator to both arms.  The
        # static arm differs only by the absence of study-memory attention.
        static = base.dropout(base.query_norm(prior))
        post = base.dropout(base.query_norm(prior + attended))
        reconstructed = (
            post * base.target_weight[None, :, :]
        ).sum(dim=-1) + base.target_bias
        reconstructed = torch.where(empty[:, None], base.target_bias[None, :], reconstructed)
        return static.detach(), post.detach(), reconstructed.detach()

    def _select_context_query(
        self,
        global_feature: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> torch.Tensor:
        static, post, _ = self._global_query_states(global_feature, present, series_meta)
        return static if self.arm == B48_STATIC_PRIOR_CONTROL else post

    def forward(
        self,
        volumes: list[torch.Tensor],
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
        *,
        audit_context: bool = False,
    ) -> B48Forward:
        if present.ndim == 1:
            present = present.unsqueeze(0)
        if series_meta.ndim == 2:
            series_meta = series_meta.unsqueeze(0)
        if slice_position.ndim == 2:
            slice_position = slice_position.unsqueeze(0)
        global_feature, spatial = self._encode_ragged_study(volumes, present)
        # Call the unchanged B42/B37 implementation for the global prediction.
        # B48's detached query extraction below is only a conditioning readout.
        base_logits = self._base_logits_from_global(global_feature, present, series_meta)
        context_query = self._select_context_query(global_feature, present, series_meta)
        details = self.head.forward_details(
            spatial,
            present,
            series_meta,
            slice_position,
            context_query,
            audit_context=bool(audit_context),
        )
        gate = self.head.effective_gate().to(dtype=details.local_logits.dtype)
        logits = base_logits.float() + gate[None, :] * details.local_logits.float()
        return B48Forward(
            logits=logits,
            base_logits=base_logits,
            local_logits=details.local_logits,
            top_indices=details.top_indices,
            top_values=details.top_values,
            context_query=context_query,
            context_abs_mean=details.context_abs_mean,
            topk_overlap_with_static=details.topk_overlap_with_static,
        )

    @torch.no_grad()
    def context_reconstruction_error(
        self,
        global_feature: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> float:
        """Assert the extracted post query still reconstructs B42's base logits."""
        expected = self._base_logits_from_global(global_feature, present, series_meta)
        _, _, reconstructed = self._global_query_states(global_feature, present, series_meta)
        return float((expected.float() - reconstructed.float()).abs().max().item())

    def state(self) -> dict:
        state = super().state()
        state.update(
            {
                "version": B48_VERSION,
                "experiment": B48_EXPERIMENT,
                "arm": self.arm,
                "context_source": self.context_source,
                "context_metric": B48_CONTEXT_METRIC,
                "context_dim": int(self.head.context_dim),
                "context_query_gradient": B48_CONTEXT_QUERY_GRADIENT,
            }
        )
        return state


def require_b48_contract(config: dict, *, arm: str) -> dict:
    """Freeze B42 plus B48's sole allowed capability and weak-only protocol."""
    crop_policy = require_b42_contract(config)
    if arm not in B48_ARMS:
        raise ValueError(f"B48 arm must be one of {B48_ARMS}; got {arm!r}")

    expected_int = {
        "b48_context_dim": B48_CONTEXT_DIM,
        "b48_fixed_epochs": B48_FIXED_EPOCHS,
    }
    for key, expected in expected_int.items():
        value = int(config.get(key, expected))
        if value != expected:
            raise ValueError(f"B48 freezes {key}={expected}; got {value}")

    expected_text = {
        "b48_context_metric": B48_CONTEXT_METRIC,
        "b48_context_gate_init": B48_CONTEXT_GATE_INIT,
        "b48_context_query_gradient": B48_CONTEXT_QUERY_GRADIENT,
        "b48_supervision": B48_SUPERVISION,
        "b48_validation_surface": B48_VALIDATION_SURFACE,
        "b48_checkpoint_selection": "none_fixed_epoch_2",
    }
    for key, expected in expected_text.items():
        value = str(config.get(key, expected))
        if value != expected:
            raise ValueError(f"B48 freezes {key}={expected!r}; got {value!r}")

    eps = float(config.get("b48_context_eps", B48_CONTEXT_EPS))
    if not np.isclose(eps, B48_CONTEXT_EPS, atol=1e-12, rtol=0):
        raise ValueError(f"B48 freezes b48_context_eps={B48_CONTEXT_EPS}; got {eps}")
    if float(config.get("b37_local_aux_weight", B37_LOCAL_AUX_WEIGHT)) != B37_LOCAL_AUX_WEIGHT:
        raise ValueError("B48 retains B42's direct local auxiliary loss")
    return {
        "crop_policy": crop_policy,
        "arm": arm,
        "context_source": B48_ARM_CONTEXT_SOURCE[arm],
        "context_dim": B48_CONTEXT_DIM,
        "supervision": B48_SUPERVISION,
    }


def b48_state(arm: str) -> dict:
    """A serialisable arm declaration used by runners and checkpoint audits."""
    if arm not in B48_ARMS:
        raise ValueError(f"B48 arm must be one of {B48_ARMS}; got {arm!r}")
    return {
        "version": B48_VERSION,
        "experiment": B48_EXPERIMENT,
        "arm": arm,
        "context_source": B48_ARM_CONTEXT_SOURCE[arm],
        "context_metric": B48_CONTEXT_METRIC,
        "context_dim": B48_CONTEXT_DIM,
        "context_eps": B48_CONTEXT_EPS,
        "context_gate_init": B48_CONTEXT_GATE_INIT,
        "context_query_gradient": B48_CONTEXT_QUERY_GRADIENT,
        "supervision": B48_SUPERVISION,
        "validation_surface": B48_VALIDATION_SURFACE,
    }


__all__ = [
    "B48_ARMS",
    "B48_ARM_CONTEXT_SOURCE",
    "B48_CONTEXT_DIM",
    "B48_CONTEXT_EPS",
    "B48_CONTEXT_GATE_INIT",
    "B48_CONTEXT_METRIC",
    "B48_CONTEXT_QUERY_GRADIENT",
    "B48_EXPERIMENT",
    "B48_FIXED_EPOCHS",
    "B48GlobalConditionedSparseMILHead",
    "B48GlobalConditionedSparseMILResidual",
    "B48HeadForward",
    "B48Forward",
    "B48_NUMBERED_CONTAINER",
    "B48_POST_CROSS_ATTENTION_CANDIDATE",
    "B48_RUN_ROOT",
    "B48_STATIC_PRIOR_CONTROL",
    "B48_SUPERVISION",
    "B48_VALIDATION_SURFACE",
    "B48_VERSION",
    "b48_state",
    "require_b48_contract",
]
