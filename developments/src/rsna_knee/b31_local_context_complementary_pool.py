"""B31 local-context enhancement of the frozen B29 complementary pool.

B31 keeps B29's successful simple complementary query and changes only the
features used to *score* slices. A zero-initialised depthwise Conv1d(k=3) adds
immediate through-plane neighbour context before the B29 dot-product scorer.
The weighted value sum still uses the original B20 slice tokens.

    H = X + DWConv1d_k3(parameter_free_LN(X))
    w = softmax(<H, q> / sqrt(D))
    C = parameter_free_LN(sum_i w_i X_i)
    T = A + tanh(g) * (C - A)

The complete B29 model is constructed first, so its query/gate and every shared
B20 parameter preserve the historical construction RNG path. The new depthwise
convolution has no bias and is reset to exact zeros after construction.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from .b29_complementary_series_pool import (
    B29_EXPECTED_GATE_PARAMETERS,
    B29_EXPECTED_QUERY_PARAMETERS,
    ComplementarySeriesPoolKneeMILNet,
    b29_model_spec,
)

B31_ARCHITECTURE = "b29_plus_zero_init_depthwise_local_slice_context_v1"
B31_AGGREGATION = "b29_simple_query_scored_on_local_context_values_from_original_tokens_v1"
B31_CONTEXT_VERSION = "parameter_free_ln_plus_zero_init_depthwise_conv1d_k3_v1"
B31_EXPECTED_QUERY_PARAMETERS = B29_EXPECTED_QUERY_PARAMETERS
B31_EXPECTED_GATE_PARAMETERS = B29_EXPECTED_GATE_PARAMETERS
B31_EXPECTED_CONTEXT_PARAMETERS = 768 * 3
B31_EXPECTED_NEW_PARAMETERS = (
    B31_EXPECTED_QUERY_PARAMETERS
    + B31_EXPECTED_GATE_PARAMETERS
    + B31_EXPECTED_CONTEXT_PARAMETERS
)


class LocalContextComplementarySeriesPoolKneeMILNet(ComplementarySeriesPoolKneeMILNet):
    """B29 plus zero-init local through-plane context for attention scoring only."""

    def __init__(self, *args, **kwargs) -> None:
        # B29 (and therefore complete historical B20) is constructed first.
        super().__init__(*args, **kwargs)
        d = int(self.encoder.out_dim)
        self.local_context = nn.Conv1d(
            d,
            d,
            kernel_size=3,
            padding=1,
            groups=d,
            bias=False,
        )
        nn.init.zeros_(self.local_context.weight)
        self._attention_audit_enabled = False
        self._attention_audit_accum = None
        self._require_b31_contract()

    def _require_b31_contract(self) -> None:
        d = int(self.encoder.out_dim)
        if self.local_context.groups != d:
            raise ValueError("B31 local context must remain depthwise")
        if tuple(self.local_context.kernel_size) != (3,):
            raise ValueError("B31 local context kernel must remain 3")
        if self.local_context.bias is not None:
            raise ValueError("B31 local context must remain bias-free")
        if int(self.local_context.weight.numel()) != B31_EXPECTED_CONTEXT_PARAMETERS:
            raise ValueError("B31 local context parameter count changed")

    def _contextualized_slice_features(self, active_slice_features: torch.Tensor) -> torch.Tensor:
        if active_slice_features.ndim != 3:
            raise ValueError("B31 local context expects [N,S,D]")
        d = int(active_slice_features.shape[-1])
        if d != int(self.encoder.out_dim):
            raise ValueError("B31 local context feature dimension mismatch")
        normalized = F.layer_norm(active_slice_features.float(), (d,)).to(
            dtype=active_slice_features.dtype
        )
        delta = self.local_context(normalized.transpose(1, 2)).transpose(1, 2)
        return active_slice_features + delta

    def _contextual_weights(self, active_slice_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        contextual = self._contextualized_slice_features(active_slice_features)
        weights = self._complementary_weights(contextual)
        return weights, contextual

    def _contextual_complementary_summary(
        self, active_slice_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights, contextual = self._contextual_weights(active_slice_features)
        # Values remain the ORIGINAL B20 slice tokens; context only changes scores.
        summary = torch.sum(weights[:, :, None] * active_slice_features, dim=1)
        d = int(summary.shape[-1])
        summary = F.layer_norm(summary.float(), (d,)).to(dtype=active_slice_features.dtype)
        return summary, weights, contextual

    def enable_attention_audit(self, enabled: bool = True, *, reset: bool = False) -> None:
        self._attention_audit_enabled = bool(enabled)
        if reset:
            self._attention_audit_accum = None

    @torch.no_grad()
    def _update_attention_audit(
        self,
        active_slice_features: torch.Tensor,
        context_weights: torch.Tensor,
        contextual_features: torch.Tensor,
        primary_tokens: torch.Tensor,
        complementary_tokens: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        raw_weights = self._complementary_weights(active_slice_features).float().clamp_min(1e-12)
        context_weights = context_weights.float().clamp_min(1e-12)
        s = int(raw_weights.shape[-1])
        log_s = math.log(float(max(s, 2)))

        raw_entropy = -(raw_weights * raw_weights.log()).sum(dim=-1) / log_s
        context_entropy = -(context_weights * context_weights.log()).sum(dim=-1) / log_s
        m = 0.5 * (raw_weights + context_weights)
        js = 0.5 * (
            (raw_weights * (raw_weights / m).log()).sum(dim=-1)
            + (context_weights * (context_weights / m).log()).sum(dim=-1)
        ) / math.log(2.0)

        top1 = (raw_weights.argmax(dim=-1) == context_weights.argmax(dim=-1)).float()
        topk = min(3, s)
        raw_top = torch.topk(raw_weights, k=topk, dim=-1).indices
        ctx_top = torch.topk(context_weights, k=topk, dim=-1).indices
        overlap = (raw_top[:, :, None] == ctx_top[:, None, :]).any(dim=-1).float().sum(dim=-1)
        overlap = overlap / float(topk)

        if s > 1:
            raw_adj = torch.abs(raw_weights[:, 1:] - raw_weights[:, :-1]).mean(dim=-1)
            ctx_adj = torch.abs(context_weights[:, 1:] - context_weights[:, :-1]).mean(dim=-1)
        else:
            raw_adj = torch.zeros_like(raw_entropy)
            ctx_adj = torch.zeros_like(context_entropy)

        context_delta = contextual_features - active_slice_features
        context_delta_ratio = torch.linalg.vector_norm(context_delta.float(), dim=(-2, -1)) / (
            torch.linalg.vector_norm(active_slice_features.float(), dim=(-2, -1)) + 1e-12
        )
        residual = gate[None, :] * (complementary_tokens - primary_tokens)
        residual_ratio = torch.linalg.vector_norm(residual.float(), dim=-1) / (
            torch.linalg.vector_norm(primary_tokens.float(), dim=-1) + 1e-12
        )

        count = torch.tensor(float(raw_weights.shape[0]), device=raw_weights.device)
        sums = torch.stack(
            [
                count,
                raw_entropy.sum(),
                context_entropy.sum(),
                js.sum(),
                top1.sum(),
                overlap.sum(),
                raw_adj.sum(),
                ctx_adj.sum(),
                context_delta_ratio.sum(),
                context_delta_ratio.max(),
                residual_ratio.sum(),
                residual_ratio.max(),
            ]
        ).detach()
        if self._attention_audit_accum is None:
            self._attention_audit_accum = sums.clone()
        else:
            self._attention_audit_accum[:9] = self._attention_audit_accum[:9] + sums[:9]
            self._attention_audit_accum[9] = torch.maximum(self._attention_audit_accum[9], sums[9])
            self._attention_audit_accum[10] = self._attention_audit_accum[10] + sums[10]
            self._attention_audit_accum[11] = torch.maximum(self._attention_audit_accum[11], sums[11])

    def attention_audit_state(self, *, reset: bool = False) -> dict:
        acc = self._attention_audit_accum
        if acc is None:
            state = {
                "series_count": 0,
                "raw_b29_attention_entropy_normalized_mean": None,
                "context_attention_entropy_normalized_mean": None,
                "raw_vs_context_js_divergence_normalized_mean": None,
                "raw_vs_context_top1_agreement": None,
                "raw_vs_context_top3_overlap_fraction_mean": None,
                "raw_attention_adjacent_absdiff_mean": None,
                "context_attention_adjacent_absdiff_mean": None,
                "context_delta_norm_ratio_mean": None,
                "context_delta_norm_ratio_max": None,
                "effective_residual_norm_ratio_mean": None,
                "effective_residual_norm_ratio_max": None,
            }
        else:
            x = acc.detach().float().cpu()
            count = max(float(x[0].item()), 1.0)
            state = {
                "series_count": int(round(float(x[0].item()))),
                "raw_b29_attention_entropy_normalized_mean": float(x[1].item() / count),
                "context_attention_entropy_normalized_mean": float(x[2].item() / count),
                "raw_vs_context_js_divergence_normalized_mean": float(x[3].item() / count),
                "raw_vs_context_top1_agreement": float(x[4].item() / count),
                "raw_vs_context_top3_overlap_fraction_mean": float(x[5].item() / count),
                "raw_attention_adjacent_absdiff_mean": float(x[6].item() / count),
                "context_attention_adjacent_absdiff_mean": float(x[7].item() / count),
                "context_delta_norm_ratio_mean": float(x[8].item() / count),
                "context_delta_norm_ratio_max": float(x[9].item()),
                "effective_residual_norm_ratio_mean": float(x[10].item() / count),
                "effective_residual_norm_ratio_max": float(x[11].item()),
            }
        if reset:
            self._attention_audit_accum = None
        return state

    def local_context_state(self) -> dict:
        w = self.local_context.weight.detach().float().cpu()
        return {
            "version": B31_CONTEXT_VERSION,
            "parameter_count": int(w.numel()),
            "kernel_size": 3,
            "depthwise_groups": int(self.local_context.groups),
            "bias": False,
            "weight_max_abs": float(w.abs().max().item()),
            "weight_mean_abs": float(w.abs().mean().item()),
            "weight_l2": float(torch.linalg.vector_norm(w).item()),
        }

    def _pool_real_series_b31(
        self,
        slice_features: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        if slice_features.ndim != 4:
            raise ValueError("B31 slice features must be [B,K,S,D]")
        b, k, s, d = slice_features.shape
        flat = slice_features.reshape(b * k, s, d)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return slice_features.new_zeros((b, k, d))

        active_slices = flat.index_select(0, active_indices)
        active_primary = self.series_pool(active_slices)
        active_complement, context_weights, contextual = self._contextual_complementary_summary(
            active_slices
        )
        gate = self.effective_complementary_gate().to(dtype=active_primary.dtype)
        active_tokens = active_primary + gate[None, :] * (active_complement - active_primary)

        if self._attention_audit_enabled:
            self._update_attention_audit(
                active_slices,
                context_weights,
                contextual,
                active_primary,
                active_complement,
                gate,
            )

        all_tokens = active_tokens.new_zeros((b * k, d)).index_copy(0, active_indices, active_tokens)
        return all_tokens.reshape(b, k, d)

    def _study_memory(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slice_features = self._encode_slices(volumes, present, series_meta)
        tokens = self._pool_real_series_b31(slice_features, present)
        padding = present <= 0
        empty = padding.all(dim=1)
        safe_padding = padding.clone()
        if empty.any():
            safe_padding[empty, 0] = False
            tokens = tokens.clone()
            tokens[empty, 0] = 0
        contextual = self.context(tokens, src_key_padding_mask=safe_padding)
        contextual = contextual.masked_fill(padding[:, :, None], 0.0)
        return contextual, safe_padding, empty

    def b31_state(self) -> dict:
        return {
            "complementary_pool": self.complementary_state(),
            "local_context": self.local_context_state(),
        }


def b31_model_spec(config: dict, *, normalize_input: bool) -> dict:
    spec = copy.deepcopy(b29_model_spec(config, normalize_input=normalize_input))
    spec["architecture"] = B31_ARCHITECTURE
    spec["aggregation"] = B31_AGGREGATION
    spec["b31_context_version"] = B31_CONTEXT_VERSION
    spec["b31_context_parameter_count"] = B31_EXPECTED_CONTEXT_PARAMETERS
    spec["b31_new_parameter_count"] = B31_EXPECTED_NEW_PARAMETERS
    spec["b31_context"] = "parameter-free per-slice LN + zero-init bias-free depthwise Conv1d(k=3)"
    spec["b31_scoring"] = "B29 query scores contextualized tokens; value sum uses original B20 slice tokens"
    spec["b31_zero_init"] = "context conv exactly zero; outer B29 gate exactly zero"
    return spec


def build_b31_model(
    spec: dict,
    *,
    encoder_state: dict | None = None,
    pretrained_weights: bool = False,
) -> LocalContextComplementarySeriesPoolKneeMILNet:
    if spec.get("architecture") != B31_ARCHITECTURE:
        raise ValueError("not a B31 local-context complementary model spec")
    if spec.get("aggregation") != B31_AGGREGATION:
        raise ValueError("B31 aggregation policy mismatch")
    if spec.get("b31_context_version") != B31_CONTEXT_VERSION:
        raise ValueError("B31 context version mismatch")
    if int(spec.get("b31_context_parameter_count", -1)) != B31_EXPECTED_CONTEXT_PARAMETERS:
        raise ValueError("B31 context parameter count mismatch")
    if int(spec.get("b31_new_parameter_count", -1)) != B31_EXPECTED_NEW_PARAMETERS:
        raise ValueError("B31 total new parameter count mismatch")
    if encoder_state is not None and pretrained_weights:
        raise ValueError("encoder_state and pretrained_weights are mutually exclusive")

    model = LocalContextComplementarySeriesPoolKneeMILNet(
        int(spec["n_slices"]),
        in_channels=int(spec.get("in_channels", 3)),
        pretrained_weights=bool(pretrained_weights),
        normalize_input=bool(spec["normalize_input"]),
        dropout=float(spec["dropout"]),
        encoder_batch_size=int(spec["encoder_batch_size"]),
        gradient_checkpointing=bool(spec["gradient_checkpointing"]),
        transformer_layers=int(spec["transformer_layers"]),
        transformer_heads=int(spec["transformer_heads"]),
        transformer_ff_mult=float(spec["transformer_ff_mult"]),
        pathology_layers=int(spec["pathology_layers"]),
        series_pool_heads=int(spec["series_pool_heads"]),
    )
    if int(model.complementary_query.numel()) != B31_EXPECTED_QUERY_PARAMETERS:
        raise ValueError("B31 query dimension changed")
    if int(model.complementary_gate.numel()) != B31_EXPECTED_GATE_PARAMETERS:
        raise ValueError("B31 gate dimension changed")
    if int(model.local_context.weight.numel()) != B31_EXPECTED_CONTEXT_PARAMETERS:
        raise ValueError("B31 context dimension changed")
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state, strict=True)
    return model
