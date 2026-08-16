"""B32 second-order complementary summary on the frozen B20/B29 family.

B32 is deliberately branched from B29 rather than B31. B31 remains frozen after
its reused-expert inspection. B32 keeps B29's simple learned softmax weights and
mean-like complementary summary, then adds a second zero-gated weighted standard-
deviation summary over the same original B20 slice tokens.

For each real MRI series with slice tokens X and B29 weights w:

    mu_raw = sum_i w_i X_i
    C_mu   = LN0(mu_raw)
    sigma  = sqrt(sum_i w_i (X_i - mu_raw)^2 + eps)
    C_sig  = LN0(sigma)

    T = A
        + tanh(g_mu)  * (C_mu - A)
        + tanh(g_sig) * C_sig

A is the unchanged historical B20 learned-attention series token. g_mu is B29's
existing zero-init feature gate. g_sig is a new zero-init feature gate. The same
B29 query determines both first- and second-order weighted statistics.

The complete B29 model is constructed first and the new dispersion gate is then
added at exact zero, preserving all historical B20/B29 random draws and the B20
training RNG path at initialization.
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

B32_ARCHITECTURE = "b29_plus_zero_gated_weighted_dispersion_summary_v1"
B32_AGGREGATION = "b20_attention_plus_b29_weighted_mean_plus_weighted_std_residual_v1"
B32_DISPERSION_VERSION = "weighted_feature_std_same_b29_weights_parameter_free_ln_v1"
B32_EXPECTED_QUERY_PARAMETERS = B29_EXPECTED_QUERY_PARAMETERS
B32_EXPECTED_MEAN_GATE_PARAMETERS = B29_EXPECTED_GATE_PARAMETERS
B32_EXPECTED_DISPERSION_GATE_PARAMETERS = 768
B32_EXPECTED_NEW_PARAMETERS = (
    B32_EXPECTED_QUERY_PARAMETERS
    + B32_EXPECTED_MEAN_GATE_PARAMETERS
    + B32_EXPECTED_DISPERSION_GATE_PARAMETERS
)
B32_VARIANCE_EPS = 1e-6


class DispersionComplementarySeriesPoolKneeMILNet(ComplementarySeriesPoolKneeMILNet):
    """B29 plus a zero-gated weighted through-series feature-dispersion summary."""

    def __init__(self, *args, **kwargs) -> None:
        # Construct complete B29 (and therefore historical B20) first.
        super().__init__(*args, **kwargs)
        d = int(self.encoder.out_dim)
        self.dispersion_gate = nn.Parameter(torch.zeros(d))
        self._dispersion_audit_enabled = False
        self._dispersion_audit_accum = None
        self._require_b32_contract()

    def _require_b32_contract(self) -> None:
        d = int(self.encoder.out_dim)
        if int(self.complementary_query.numel()) != B32_EXPECTED_QUERY_PARAMETERS:
            raise ValueError("B32 query dimension changed")
        if int(self.complementary_gate.numel()) != B32_EXPECTED_MEAN_GATE_PARAMETERS:
            raise ValueError("B32 mean gate dimension changed")
        if int(self.dispersion_gate.numel()) != B32_EXPECTED_DISPERSION_GATE_PARAMETERS:
            raise ValueError("B32 dispersion gate dimension changed")
        if d != B32_EXPECTED_DISPERSION_GATE_PARAMETERS:
            raise ValueError("B32 frozen B20 feature dimension changed")

    def effective_dispersion_gate(self) -> torch.Tensor:
        return torch.tanh(self.dispersion_gate)

    def _weighted_moments(
        self,
        active_slice_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return B29 weights, normalized mean/std summaries and raw mean/std.

        All moment arithmetic is performed in float32 for numerical stability.
        The weighted variance uses the same learned B29 weights as the mean.
        """
        if active_slice_features.ndim != 3:
            raise ValueError("B32 weighted moments expect [N,S,D]")
        if int(active_slice_features.shape[-1]) != int(self.encoder.out_dim):
            raise ValueError("B32 weighted moment feature dimension mismatch")

        weights = self._complementary_weights(active_slice_features).float()
        x = active_slice_features.float()
        mu_raw = torch.sum(weights[:, :, None] * x, dim=1)
        centered = x - mu_raw[:, None, :]
        var_raw = torch.sum(weights[:, :, None] * centered.square(), dim=1).clamp_min(0.0)
        sigma_raw = torch.sqrt(var_raw + B32_VARIANCE_EPS)
        d = int(x.shape[-1])
        mean_summary = F.layer_norm(mu_raw, (d,)).to(dtype=active_slice_features.dtype)
        dispersion_summary = F.layer_norm(sigma_raw, (d,)).to(dtype=active_slice_features.dtype)
        return (
            weights.to(dtype=active_slice_features.dtype),
            mean_summary,
            dispersion_summary,
            mu_raw,
            sigma_raw,
        )

    def enable_dispersion_audit(self, enabled: bool = True, *, reset: bool = False) -> None:
        self._dispersion_audit_enabled = bool(enabled)
        if reset:
            self._dispersion_audit_accum = None

    @torch.no_grad()
    def _update_dispersion_audit(
        self,
        active_slice_features: torch.Tensor,
        weights: torch.Tensor,
        mu_raw: torch.Tensor,
        sigma_raw: torch.Tensor,
        primary_tokens: torch.Tensor,
        mean_summary: torch.Tensor,
        dispersion_summary: torch.Tensor,
        mean_gate: torch.Tensor,
        dispersion_gate: torch.Tensor,
    ) -> None:
        w = weights.float().clamp_min(1e-12)
        n, s, _ = active_slice_features.shape
        log_s = math.log(float(max(int(s), 2)))
        entropy = -(w * w.log()).sum(dim=-1) / log_s

        x = active_slice_features.float()
        uniform_mu = x.mean(dim=1)
        uniform_centered = x - uniform_mu[:, None, :]
        uniform_sigma = torch.sqrt(uniform_centered.square().mean(dim=1) + B32_VARIANCE_EPS)

        mean_vs_uniform = torch.linalg.vector_norm(mu_raw - uniform_mu, dim=-1) / (
            torch.linalg.vector_norm(uniform_mu, dim=-1) + 1e-12
        )
        dispersion_vs_uniform = torch.linalg.vector_norm(sigma_raw - uniform_sigma, dim=-1) / (
            torch.linalg.vector_norm(uniform_sigma, dim=-1) + 1e-12
        )
        sigma_to_mu = torch.linalg.vector_norm(sigma_raw, dim=-1) / (
            torch.linalg.vector_norm(mu_raw, dim=-1) + 1e-12
        )

        a = primary_tokens.float()
        mean_residual = mean_gate.float()[None, :] * (mean_summary.float() - a)
        dispersion_residual = dispersion_gate.float()[None, :] * dispersion_summary.float()
        combined_residual = mean_residual + dispersion_residual
        a_norm = torch.linalg.vector_norm(a, dim=-1) + 1e-12
        mean_ratio = torch.linalg.vector_norm(mean_residual, dim=-1) / a_norm
        dispersion_ratio = torch.linalg.vector_norm(dispersion_residual, dim=-1) / a_norm
        combined_ratio = torch.linalg.vector_norm(combined_residual, dim=-1) / a_norm
        residual_cosine = F.cosine_similarity(
            mean_residual,
            dispersion_residual,
            dim=-1,
            eps=1e-12,
        )

        count = torch.tensor(float(n), device=w.device)
        sums = torch.stack(
            [
                count,
                entropy.sum(),
                mean_vs_uniform.sum(),
                dispersion_vs_uniform.sum(),
                sigma_to_mu.sum(),
                mean_ratio.sum(),
                dispersion_ratio.sum(),
                combined_ratio.sum(),
                combined_ratio.max(),
                residual_cosine.sum(),
            ]
        ).detach()
        if self._dispersion_audit_accum is None:
            self._dispersion_audit_accum = sums.clone()
        else:
            self._dispersion_audit_accum[:8] = self._dispersion_audit_accum[:8] + sums[:8]
            self._dispersion_audit_accum[8] = torch.maximum(
                self._dispersion_audit_accum[8], sums[8]
            )
            self._dispersion_audit_accum[9] = self._dispersion_audit_accum[9] + sums[9]

    def dispersion_audit_state(self, *, reset: bool = False) -> dict:
        acc = self._dispersion_audit_accum
        if acc is None:
            state = {
                "series_count": 0,
                "attention_entropy_normalized_mean": None,
                "weighted_mean_vs_uniform_mean_norm_ratio_mean": None,
                "weighted_dispersion_vs_uniform_dispersion_norm_ratio_mean": None,
                "raw_dispersion_to_raw_mean_norm_ratio_mean": None,
                "mean_residual_norm_ratio_mean": None,
                "dispersion_residual_norm_ratio_mean": None,
                "combined_residual_norm_ratio_mean": None,
                "combined_residual_norm_ratio_max": None,
                "mean_dispersion_residual_cosine_mean": None,
            }
        else:
            x = acc.detach().float().cpu()
            count = max(float(x[0].item()), 1.0)
            state = {
                "series_count": int(round(float(x[0].item()))),
                "attention_entropy_normalized_mean": float(x[1].item() / count),
                "weighted_mean_vs_uniform_mean_norm_ratio_mean": float(x[2].item() / count),
                "weighted_dispersion_vs_uniform_dispersion_norm_ratio_mean": float(x[3].item() / count),
                "raw_dispersion_to_raw_mean_norm_ratio_mean": float(x[4].item() / count),
                "mean_residual_norm_ratio_mean": float(x[5].item() / count),
                "dispersion_residual_norm_ratio_mean": float(x[6].item() / count),
                "combined_residual_norm_ratio_mean": float(x[7].item() / count),
                "combined_residual_norm_ratio_max": float(x[8].item()),
                "mean_dispersion_residual_cosine_mean": float(x[9].item() / count),
            }
        if reset:
            self._dispersion_audit_accum = None
        return state

    def dispersion_gate_state(self) -> dict:
        raw = self.dispersion_gate.detach().float().cpu()
        effective = torch.tanh(raw)
        return {
            "version": B32_DISPERSION_VERSION,
            "parameter_count": int(raw.numel()),
            "variance_epsilon": B32_VARIANCE_EPS,
            "gate_raw_max_abs": float(raw.abs().max().item()),
            "gate_raw_mean_abs": float(raw.abs().mean().item()),
            "gate_raw_l2": float(torch.linalg.vector_norm(raw).item()),
            "gate_effective_max_abs": float(effective.abs().max().item()),
            "gate_effective_mean_abs": float(effective.abs().mean().item()),
            "gate_effective_l2": float(torch.linalg.vector_norm(effective).item()),
        }

    def _pool_real_series_b32(
        self,
        slice_features: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        if slice_features.ndim != 4:
            raise ValueError("B32 slice features must be [B,K,S,D]")
        b, k, s, d = slice_features.shape
        flat = slice_features.reshape(b * k, s, d)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return slice_features.new_zeros((b, k, d))

        active_slices = flat.index_select(0, active_indices)
        # Historical B20 stochastic pool remains first and unchanged.
        active_primary = self.series_pool(active_slices)
        weights, mean_summary, dispersion_summary, mu_raw, sigma_raw = self._weighted_moments(
            active_slices
        )
        mean_gate = self.effective_complementary_gate().to(dtype=active_primary.dtype)
        dispersion_gate = self.effective_dispersion_gate().to(dtype=active_primary.dtype)
        active_tokens = (
            active_primary
            + mean_gate[None, :] * (mean_summary - active_primary)
            + dispersion_gate[None, :] * dispersion_summary
        )

        if self._dispersion_audit_enabled:
            self._update_dispersion_audit(
                active_slices,
                weights,
                mu_raw,
                sigma_raw,
                active_primary,
                mean_summary,
                dispersion_summary,
                mean_gate,
                dispersion_gate,
            )

        all_tokens = active_tokens.new_zeros((b * k, d)).index_copy(
            0, active_indices, active_tokens
        )
        return all_tokens.reshape(b, k, d)

    def _study_memory(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slice_features = self._encode_slices(volumes, present, series_meta)
        tokens = self._pool_real_series_b32(slice_features, present)
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

    def b32_state(self) -> dict:
        return {
            "mean_complementary_pool": self.complementary_state(),
            "dispersion_gate": self.dispersion_gate_state(),
        }


def b32_model_spec(config: dict, *, normalize_input: bool) -> dict:
    spec = copy.deepcopy(b29_model_spec(config, normalize_input=normalize_input))
    spec["architecture"] = B32_ARCHITECTURE
    spec["aggregation"] = B32_AGGREGATION
    spec["b32_dispersion_version"] = B32_DISPERSION_VERSION
    spec["b32_variance_epsilon"] = B32_VARIANCE_EPS
    spec["b32_dispersion_gate_parameter_count"] = B32_EXPECTED_DISPERSION_GATE_PARAMETERS
    spec["b32_new_parameter_count"] = B32_EXPECTED_NEW_PARAMETERS
    spec["b32_moments"] = "same B29 softmax weights for weighted mean and weighted feature standard deviation"
    spec["b32_dispersion_summary"] = "parameter-free LayerNorm(weighted std); zero-init featurewise tanh gate"
    spec["b32_base"] = "prospective branch from frozen B29; B31 local-context mechanism not included"
    return spec


def build_b32_model(
    spec: dict,
    *,
    encoder_state: dict | None = None,
    pretrained_weights: bool = False,
) -> DispersionComplementarySeriesPoolKneeMILNet:
    if spec.get("architecture") != B32_ARCHITECTURE:
        raise ValueError("not a B32 dispersion-complementary model spec")
    if spec.get("aggregation") != B32_AGGREGATION:
        raise ValueError("B32 aggregation policy mismatch")
    if spec.get("b32_dispersion_version") != B32_DISPERSION_VERSION:
        raise ValueError("B32 dispersion version mismatch")
    if float(spec.get("b32_variance_epsilon", -1.0)) != B32_VARIANCE_EPS:
        raise ValueError("B32 variance epsilon mismatch")
    if int(spec.get("b32_dispersion_gate_parameter_count", -1)) != B32_EXPECTED_DISPERSION_GATE_PARAMETERS:
        raise ValueError("B32 dispersion gate parameter count mismatch")
    if int(spec.get("b32_new_parameter_count", -1)) != B32_EXPECTED_NEW_PARAMETERS:
        raise ValueError("B32 total new parameter count mismatch")
    if encoder_state is not None and pretrained_weights:
        raise ValueError("encoder_state and pretrained_weights are mutually exclusive")

    model = DispersionComplementarySeriesPoolKneeMILNet(
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
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state, strict=True)
    return model
