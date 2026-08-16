"""B33 exact-uniform complementary mean on the frozen B20 family.

B33 tests the simplest interpretation of the B29/B31 development signal: the
useful second series representation may be a broad mean-like statistic rather
than learned slice selection.

For every real series with slice tokens X:

    A       = historical B20 learned attention-pooled series token
    C_mean  = parameter_free_LN(mean_i X_i)
    token   = A + tanh(g) * (C_mean - A)

Only the feature-wise gate g is new (768 parameters), initialized to exact zero.
No complementary query, local-context convolution, dispersion statistic,
projection, trainable normalization, target-specific routing, or extra encoder
pass is present.

The complete historical B20 model is constructed first. The new gate is created
with zeros and consumes no random draw, so shared initialization and the B20
training-mode RNG path are preserved at zero gate.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .b12_1_hierarchical import HierarchicalSeriesKneeMILNet, b12_1_model_spec

B33_ARCHITECTURE = "hierarchical_series_token_plus_zero_gated_uniform_mean_v1"
B33_AGGREGATION = "historical_b20_attention_plus_exact_uniform_slice_mean_v1"
B33_RESIDUAL_VERSION = "zero_init_tanh_feature_gate_uniform_mean_v1"
B33_EXPECTED_GATE_PARAMETERS = 768
B33_EXPECTED_NEW_PARAMETERS = 768


class UniformComplementaryMeanKneeMILNet(HierarchicalSeriesKneeMILNet):
    """B20 hierarchy plus one exact-uniform zero-gated complementary mean."""

    def __init__(self, *args, **kwargs) -> None:
        # Construct the complete historical B20/B12.1 model first. Creating a
        # zeros parameter afterwards consumes no RNG and preserves shared draws.
        super().__init__(*args, **kwargs)
        d = int(self.encoder.out_dim)
        self.uniform_complementary_gate = nn.Parameter(torch.zeros(d))
        self._uniform_audit_enabled = False
        self._uniform_audit_accum = None

    def effective_uniform_gate(self) -> torch.Tensor:
        return torch.tanh(self.uniform_complementary_gate)

    def _uniform_complementary_summary(self, active_slice_features: torch.Tensor) -> torch.Tensor:
        if active_slice_features.ndim != 3:
            raise ValueError("B33 uniform complementary mean expects [N,S,D]")
        if int(active_slice_features.shape[-1]) != int(self.encoder.out_dim):
            raise ValueError("B33 uniform complementary mean feature dimension mismatch")
        summary = active_slice_features.float().mean(dim=1)
        d = int(summary.shape[-1])
        return F.layer_norm(summary, (d,)).to(dtype=active_slice_features.dtype)

    def enable_uniform_audit(self, enabled: bool = True, *, reset: bool = False) -> None:
        self._uniform_audit_enabled = bool(enabled)
        if reset:
            self._uniform_audit_accum = None

    @torch.no_grad()
    def _update_uniform_audit(
        self,
        active_slice_features: torch.Tensor,
        primary_tokens: torch.Tensor,
        uniform_tokens: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        raw_mean = active_slice_features.float().mean(dim=1)
        primary = primary_tokens.float()
        uniform = uniform_tokens.float()
        primary_norm = torch.linalg.vector_norm(primary, dim=-1).clamp_min(1e-12)

        raw_mean_norm_ratio = torch.linalg.vector_norm(raw_mean, dim=-1) / primary_norm
        uniform_cos = F.cosine_similarity(primary, uniform, dim=-1)
        uniform_delta = uniform - primary
        uniform_delta_ratio = torch.linalg.vector_norm(uniform_delta, dim=-1) / primary_norm
        residual = gate.float()[None, :] * uniform_delta
        residual_ratio = torch.linalg.vector_norm(residual, dim=-1) / primary_norm

        count = torch.tensor(float(primary.shape[0]), device=primary.device)
        sums = torch.stack(
            [
                count,
                raw_mean_norm_ratio.sum(),
                uniform_cos.sum(),
                uniform_delta_ratio.sum(),
                uniform_delta_ratio.max(),
                residual_ratio.sum(),
                residual_ratio.max(),
            ]
        ).detach()
        if self._uniform_audit_accum is None:
            self._uniform_audit_accum = sums.clone()
        else:
            self._uniform_audit_accum[:4] = self._uniform_audit_accum[:4] + sums[:4]
            self._uniform_audit_accum[4] = torch.maximum(self._uniform_audit_accum[4], sums[4])
            self._uniform_audit_accum[5] = self._uniform_audit_accum[5] + sums[5]
            self._uniform_audit_accum[6] = torch.maximum(self._uniform_audit_accum[6], sums[6])

    def uniform_audit_state(self, *, reset: bool = False) -> dict:
        acc = self._uniform_audit_accum
        if acc is None:
            state = {
                "series_count": 0,
                "raw_uniform_mean_norm_to_primary_mean": None,
                "uniform_summary_to_primary_cosine_mean": None,
                "uniform_minus_primary_norm_ratio_mean": None,
                "uniform_minus_primary_norm_ratio_max": None,
                "effective_residual_norm_ratio_mean": None,
                "effective_residual_norm_ratio_max": None,
            }
        else:
            x = acc.detach().float().cpu()
            count = max(float(x[0].item()), 1.0)
            state = {
                "series_count": int(round(float(x[0].item()))),
                "raw_uniform_mean_norm_to_primary_mean": float(x[1].item() / count),
                "uniform_summary_to_primary_cosine_mean": float(x[2].item() / count),
                "uniform_minus_primary_norm_ratio_mean": float(x[3].item() / count),
                "uniform_minus_primary_norm_ratio_max": float(x[4].item()),
                "effective_residual_norm_ratio_mean": float(x[5].item() / count),
                "effective_residual_norm_ratio_max": float(x[6].item()),
            }
        if reset:
            self._uniform_audit_accum = None
        return state

    def uniform_gate_state(self) -> dict:
        raw = self.uniform_complementary_gate.detach().float().cpu()
        effective = torch.tanh(raw)
        return {
            "version": B33_RESIDUAL_VERSION,
            "parameter_count": int(raw.numel()),
            "gate_raw_max_abs": float(raw.abs().max().item()),
            "gate_raw_mean_abs": float(raw.abs().mean().item()),
            "gate_raw_l2": float(torch.linalg.vector_norm(raw).item()),
            "gate_effective_max_abs": float(effective.abs().max().item()),
            "gate_effective_mean_abs": float(effective.abs().mean().item()),
            "gate_effective_l2": float(torch.linalg.vector_norm(effective).item()),
        }

    def _pool_real_series_b33(
        self,
        slice_features: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        if slice_features.ndim != 4:
            raise ValueError("B33 slice features must be [B,K,S,D]")
        b, k, s, d = slice_features.shape
        flat = slice_features.reshape(b * k, s, d)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return slice_features.new_zeros((b, k, d))

        active_slices = flat.index_select(0, active_indices)

        # Historical B20 series pool executes first and remains unchanged.
        active_primary = self.series_pool(active_slices)
        active_uniform = self._uniform_complementary_summary(active_slices)
        gate = self.effective_uniform_gate().to(dtype=active_primary.dtype)
        active_tokens = active_primary + gate[None, :] * (active_uniform - active_primary)

        if self._uniform_audit_enabled:
            self._update_uniform_audit(active_slices, active_primary, active_uniform, gate)

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
        tokens = self._pool_real_series_b33(slice_features, present)
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


def b33_model_spec(config: dict, *, normalize_input: bool) -> dict:
    spec = copy.deepcopy(b12_1_model_spec(config, normalize_input=normalize_input))
    spec["architecture"] = B33_ARCHITECTURE
    spec["aggregation"] = B33_AGGREGATION
    spec["b33_residual_version"] = B33_RESIDUAL_VERSION
    spec["b33_new_parameter_count"] = B33_EXPECTED_NEW_PARAMETERS
    spec["b33_complementary_summary"] = "exact arithmetic mean of B20 slice tokens plus parameter-free layer norm"
    spec["b33_query_parameters"] = 0
    spec["b33_gate_constraint"] = "featurewise tanh gate; exactly zero initialisation"
    spec["b33_stochastic_path"] = "uniform branch deterministic; historical B20 series pool executes first"
    return spec


def build_b33_model(
    spec: dict,
    *,
    encoder_state: dict | None = None,
    pretrained_weights: bool = False,
) -> UniformComplementaryMeanKneeMILNet:
    if spec.get("architecture") != B33_ARCHITECTURE:
        raise ValueError("not a B33 uniform-complementary-mean model spec")
    if spec.get("aggregation") != B33_AGGREGATION:
        raise ValueError("B33 aggregation policy mismatch")
    if spec.get("b33_residual_version") != B33_RESIDUAL_VERSION:
        raise ValueError("B33 residual version mismatch")
    if int(spec.get("b33_new_parameter_count", -1)) != B33_EXPECTED_NEW_PARAMETERS:
        raise ValueError("B33 new parameter count mismatch")
    if int(spec.get("b33_query_parameters", -1)) != 0:
        raise ValueError("B33 must not contain a complementary query")
    if encoder_state is not None and pretrained_weights:
        raise ValueError("encoder_state and pretrained_weights are mutually exclusive")

    model = UniformComplementaryMeanKneeMILNet(
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
    if int(model.uniform_complementary_gate.numel()) != B33_EXPECTED_GATE_PARAMETERS:
        raise ValueError("B33 gate dimension changed from the frozen B20 contract")
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state, strict=True)
    return model
