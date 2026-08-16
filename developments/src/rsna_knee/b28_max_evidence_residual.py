"""B28 zero-gated max-evidence residual on the frozen B20 family.

B20/B12.1 compresses 16 encoded slice tokens from each real MRI series into one
learned attention-pooled series token. B28 keeps that complete path and adds one
narrow, outcome-independent intervention: an element-wise max summary of the
*image-content* slice embeddings is injected as a bounded residual.

The residual gate is a D-dimensional parameter initialised to exactly zero. At
initialisation B28 is therefore functionally identical to B20, including the
same attention-pooling computation and stochastic path. Training may then learn
whether sparse/extreme within-series evidence carries information that the
single learned attention summary does not preserve.

No target-specific routing, labels, thresholds, or expert outcomes enter this
module.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .b12_1_hierarchical import (
    HierarchicalSeriesKneeMILNet,
    b12_1_model_spec,
)

B28_ARCHITECTURE = "hierarchical_series_token_plus_zero_gated_max_content_residual_v1"
B28_AGGREGATION = "learned_series_query_attention_plus_max_content_residual_v1"
B28_RESIDUAL_VERSION = "zero_init_tanh_feature_gate_v1"
B28_EXPECTED_GATE_PARAMETERS = 768


class MaxEvidenceResidualKneeMILNet(HierarchicalSeriesKneeMILNet):
    """B20 hierarchy plus a zero-initialised max-evidence residual per series."""

    def __init__(self, *args, **kwargs) -> None:
        # Build every historical B20/B12.1 parameter first. This preserves the
        # exact shared RNG construction order; the only new parameter is created
        # after the complete parent model exists.
        super().__init__(*args, **kwargs)
        d = int(self.encoder.out_dim)
        self.max_residual_gate = nn.Parameter(torch.zeros(d))

    def effective_max_residual_gate(self) -> torch.Tensor:
        """Bound the residual feature gate to [-1, 1] without changing zero init."""
        return torch.tanh(self.max_residual_gate)

    def _content_slice_features(
        self,
        slice_features: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> torch.Tensor:
        """Recover encoder image-content features before B20 metadata/position additions."""
        if slice_features.ndim != 4:
            raise ValueError("B28 slice features must be [B,K,S,D]")
        b, k, s, d = slice_features.shape
        if present.shape != (b, k):
            raise ValueError("B28 present mask shape mismatch")
        if series_meta.shape != (b, k, 3):
            raise ValueError("B28 series_meta must have shape [B,K,3]")
        if s != self.n_slices or d != self.encoder.out_dim:
            raise ValueError("B28 slice feature contract mismatch")

        plane = self.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
        fluid = self.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
        fat = self.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        metadata = plane + fluid + fat
        content = (
            slice_features
            - self.slice_position[None, None, :, :]
            - metadata[:, :, None, :]
        )
        return content * present[:, :, None, None].to(content.dtype)

    def _pool_real_series_b28(
        self,
        slice_features: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> torch.Tensor:
        """Preserve the B20 token and add gated max image evidence."""
        b, k, s, d = slice_features.shape
        flat = slice_features.reshape(b * k, s, d)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return slice_features.new_zeros((b, k, d))

        # Historical B20 attention-pooled series token, unchanged.
        active_primary = self.series_pool(flat.index_select(0, active_indices))

        # MRNet-style element-wise max is applied only to recovered encoder
        # image-content features, not to B20's plane/sequence metadata embeddings
        # or slice-position embeddings. Parameter-free layer normalisation keeps
        # the residual scale comparable without adding another learned module.
        content = self._content_slice_features(slice_features, present, series_meta)
        active_content = content.reshape(b * k, s, d).index_select(0, active_indices)
        active_max = active_content.amax(dim=1)
        active_max = F.layer_norm(active_max, (d,))

        gate = self.effective_max_residual_gate().to(dtype=active_primary.dtype)
        active_tokens = active_primary + gate[None, :] * active_max

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
        tokens = self._pool_real_series_b28(slice_features, present, series_meta)
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

    def residual_state(self) -> dict:
        raw = self.max_residual_gate.detach().float().cpu()
        effective = torch.tanh(raw)
        return {
            "version": B28_RESIDUAL_VERSION,
            "parameter_count": int(raw.numel()),
            "raw_max_abs": float(raw.abs().max().item()) if raw.numel() else 0.0,
            "raw_mean_abs": float(raw.abs().mean().item()) if raw.numel() else 0.0,
            "raw_l2": float(torch.linalg.vector_norm(raw).item()) if raw.numel() else 0.0,
            "effective_max_abs": float(effective.abs().max().item()) if raw.numel() else 0.0,
            "effective_mean_abs": float(effective.abs().mean().item()) if raw.numel() else 0.0,
            "effective_l2": float(torch.linalg.vector_norm(effective).item()) if raw.numel() else 0.0,
        }


def b28_model_spec(config: dict, *, normalize_input: bool) -> dict:
    spec = copy.deepcopy(b12_1_model_spec(config, normalize_input=normalize_input))
    spec["architecture"] = B28_ARCHITECTURE
    spec["aggregation"] = B28_AGGREGATION
    spec["b28_residual_version"] = B28_RESIDUAL_VERSION
    spec["b28_gate_parameter_count"] = B28_EXPECTED_GATE_PARAMETERS
    spec["b28_max_source"] = "encoder image-content only; metadata/position removed"
    spec["b28_gate_constraint"] = "featurewise tanh; zero initialisation"
    return spec


def build_b28_model(
    spec: dict,
    *,
    encoder_state: dict | None = None,
    pretrained_weights: bool = False,
) -> MaxEvidenceResidualKneeMILNet:
    if spec.get("architecture") != B28_ARCHITECTURE:
        raise ValueError("not a B28 max-evidence residual model spec")
    if spec.get("aggregation") != B28_AGGREGATION:
        raise ValueError("B28 aggregation policy mismatch")
    if spec.get("b28_residual_version") != B28_RESIDUAL_VERSION:
        raise ValueError("B28 residual version mismatch")
    if int(spec.get("b28_gate_parameter_count", -1)) != B28_EXPECTED_GATE_PARAMETERS:
        raise ValueError("B28 gate parameter count mismatch")
    if encoder_state is not None and pretrained_weights:
        raise ValueError("encoder_state and pretrained_weights are mutually exclusive")

    model = MaxEvidenceResidualKneeMILNet(
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
    if int(model.max_residual_gate.numel()) != B28_EXPECTED_GATE_PARAMETERS:
        raise ValueError("B28 encoder feature dimension changed from the frozen B20 contract")
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state, strict=True)
    return model
