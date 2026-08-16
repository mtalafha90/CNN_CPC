"""B29 zero-gated complementary learned series summary on the frozen B20 family.

B20/B12.1 compresses the 16 encoded slice tokens of each real MRI series into one
learned attention-pooled token. B29 keeps that historical token exactly and adds
a second deterministic learned softmax summary behind a feature-wise tanh gate.

For each real series:

    A = historical B20 learned attention-pooled token
    C = parameter-free-LN(weighted sum of the same B20 slice tokens),
        where weights come from a new learned query
    token = A + tanh(g) * (C - A)

The gate g is initialised to exactly zero, so B29 is functionally identical to
B20 at initialisation. The complementary branch contains no dropout or other
random operation, so evaluating it while the gate is zero does not perturb the
historical B20 stochastic RNG path. The new query and gate are created only
after every B20 parameter, preserving shared initialisation under the historical
construction seed.

No target-specific routing, labels, thresholds, expert outcomes, or extra image
encoder pass enter this module.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from .b12_1_hierarchical import HierarchicalSeriesKneeMILNet, b12_1_model_spec

B29_ARCHITECTURE = "hierarchical_series_token_plus_zero_gated_complementary_softmax_pool_v1"
B29_AGGREGATION = "learned_series_query_attention_plus_complementary_softmax_summary_v1"
B29_RESIDUAL_VERSION = "zero_init_tanh_feature_gate_complementary_query_v1"
B29_EXPECTED_QUERY_PARAMETERS = 768
B29_EXPECTED_GATE_PARAMETERS = 768
B29_EXPECTED_NEW_PARAMETERS = 1536


class ComplementarySeriesPoolKneeMILNet(HierarchicalSeriesKneeMILNet):
    """B20 hierarchy plus one zero-gated complementary learned slice summary."""

    def __init__(self, *args, **kwargs) -> None:
        # Construct the complete historical B20/B12.1 model first so the same
        # construction seed preserves every shared random draw.
        super().__init__(*args, **kwargs)
        d = int(self.encoder.out_dim)
        self.complementary_query = nn.Parameter(torch.randn(d) * 0.02)
        self.complementary_gate = nn.Parameter(torch.zeros(d))

    def effective_complementary_gate(self) -> torch.Tensor:
        return torch.tanh(self.complementary_gate)

    def _complementary_weights(self, active_slice_features: torch.Tensor) -> torch.Tensor:
        """Return deterministic learned softmax weights [N,S] for real series."""
        if active_slice_features.ndim != 3:
            raise ValueError("B29 complementary pool expects [N,S,D]")
        if active_slice_features.shape[-1] != self.encoder.out_dim:
            raise ValueError("B29 complementary pool feature dimension mismatch")
        d = int(active_slice_features.shape[-1])
        query = self.complementary_query.to(dtype=active_slice_features.dtype)
        scores = torch.matmul(active_slice_features, query) / math.sqrt(float(d))
        # Float32 softmax is deterministic and numerically stable under bf16;
        # no dropout is used in this branch, preserving the B20 RNG path.
        return torch.softmax(scores.float(), dim=1).to(dtype=active_slice_features.dtype)

    def _complementary_summary(self, active_slice_features: torch.Tensor) -> torch.Tensor:
        """Build the second learned summary without adding trainable projection layers."""
        weights = self._complementary_weights(active_slice_features)
        summary = torch.sum(weights[:, :, None] * active_slice_features, dim=1)
        d = int(summary.shape[-1])
        # Parameter-free normalisation keeps C on a scale comparable with the
        # LayerNorm-normalised historical B20 token while adding no parameters.
        return F.layer_norm(summary.float(), (d,)).to(dtype=active_slice_features.dtype)

    def _pool_real_series_b29(
        self,
        slice_features: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        if slice_features.ndim != 4:
            raise ValueError("B29 slice features must be [B,K,S,D]")
        b, k, s, d = slice_features.shape
        flat = slice_features.reshape(b * k, s, d)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return slice_features.new_zeros((b, k, d))

        active_slices = flat.index_select(0, active_indices)

        # Historical B20 series token. This call and its stochastic operations
        # are unchanged and occur before the new deterministic branch.
        active_primary = self.series_pool(active_slices)

        active_complement = self._complementary_summary(active_slices)
        gate = self.effective_complementary_gate().to(dtype=active_primary.dtype)
        active_tokens = active_primary + gate[None, :] * (active_complement - active_primary)

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
        tokens = self._pool_real_series_b29(slice_features, present)
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

    def complementary_state(self) -> dict:
        gate_raw = self.complementary_gate.detach().float().cpu()
        gate_effective = torch.tanh(gate_raw)
        query = self.complementary_query.detach().float().cpu()
        primary_query = self.series_pool.query.detach().float().cpu().reshape(-1)
        query_cosine = float(
            F.cosine_similarity(query.reshape(1, -1), primary_query.reshape(1, -1), dim=1).item()
        )
        return {
            "version": B29_RESIDUAL_VERSION,
            "new_parameter_count": int(gate_raw.numel() + query.numel()),
            "query_parameter_count": int(query.numel()),
            "gate_parameter_count": int(gate_raw.numel()),
            "query_max_abs": float(query.abs().max().item()),
            "query_mean_abs": float(query.abs().mean().item()),
            "query_l2": float(torch.linalg.vector_norm(query).item()),
            "query_cosine_to_primary": query_cosine,
            "gate_raw_max_abs": float(gate_raw.abs().max().item()),
            "gate_raw_mean_abs": float(gate_raw.abs().mean().item()),
            "gate_raw_l2": float(torch.linalg.vector_norm(gate_raw).item()),
            "gate_effective_max_abs": float(gate_effective.abs().max().item()),
            "gate_effective_mean_abs": float(gate_effective.abs().mean().item()),
            "gate_effective_l2": float(torch.linalg.vector_norm(gate_effective).item()),
        }


def b29_model_spec(config: dict, *, normalize_input: bool) -> dict:
    spec = copy.deepcopy(b12_1_model_spec(config, normalize_input=normalize_input))
    spec["architecture"] = B29_ARCHITECTURE
    spec["aggregation"] = B29_AGGREGATION
    spec["b29_residual_version"] = B29_RESIDUAL_VERSION
    spec["b29_new_parameter_count"] = B29_EXPECTED_NEW_PARAMETERS
    spec["b29_complementary_scorer"] = "single learned D-vector query; scaled dot-product softmax; no dropout"
    spec["b29_complementary_summary"] = "weighted B20 slice-token sum plus parameter-free layer norm"
    spec["b29_gate_constraint"] = "featurewise tanh gate; exactly zero initialisation"
    spec["b29_stochastic_path"] = "complement branch deterministic; zero gate preserves B20 RNG path"
    return spec


def build_b29_model(
    spec: dict,
    *,
    encoder_state: dict | None = None,
    pretrained_weights: bool = False,
) -> ComplementarySeriesPoolKneeMILNet:
    if spec.get("architecture") != B29_ARCHITECTURE:
        raise ValueError("not a B29 complementary-series-pool model spec")
    if spec.get("aggregation") != B29_AGGREGATION:
        raise ValueError("B29 aggregation policy mismatch")
    if spec.get("b29_residual_version") != B29_RESIDUAL_VERSION:
        raise ValueError("B29 residual version mismatch")
    if int(spec.get("b29_new_parameter_count", -1)) != B29_EXPECTED_NEW_PARAMETERS:
        raise ValueError("B29 new parameter count mismatch")
    if encoder_state is not None and pretrained_weights:
        raise ValueError("encoder_state and pretrained_weights are mutually exclusive")

    model = ComplementarySeriesPoolKneeMILNet(
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
    if int(model.complementary_query.numel()) != B29_EXPECTED_QUERY_PARAMETERS:
        raise ValueError("B29 encoder feature dimension changed from the frozen B20 contract")
    if int(model.complementary_gate.numel()) != B29_EXPECTED_GATE_PARAMETERS:
        raise ValueError("B29 gate dimension changed from the frozen B20 contract")
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state, strict=True)
    return model
