"""B30 projected complementary attention on the frozen B20 family.

B30 keeps the historical B20 learned attention-pooled series token A exactly.
It replaces B29's raw dot-product complementary summary with a second query that
uses the *current B20 series-pool Q/K/V, output-projection and LayerNorm affine
parameters as detached operators*. The complementary branch therefore sees the
same learned attention coordinate system as B20 but cannot directly update those
shared projection parameters.

For each real series:

    A  = historical B20 learned attention-pooled token
    C2 = deterministic no-dropout projected complementary attention summary
    token = A + tanh(g) * (C2 - A)

Only q2 and g are new trainable parameters (768 + 768 = 1,536). The gate is
initialised to exactly zero. The complete B20 model is constructed before q2/g,
so shared initialisation and the training-mode RNG path remain identical to B20
at zero gate.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from .b12_1_hierarchical import HierarchicalSeriesKneeMILNet, b12_1_model_spec

B30_ARCHITECTURE = "hierarchical_series_token_plus_zero_gated_projected_complementary_pool_v1"
B30_AGGREGATION = "b20_attention_plus_detached_shared_projection_complementary_attention_v1"
B30_RESIDUAL_VERSION = "zero_init_tanh_feature_gate_detached_shared_mha_projection_v1"
B30_EXPECTED_QUERY_PARAMETERS = 768
B30_EXPECTED_GATE_PARAMETERS = 768
B30_EXPECTED_NEW_PARAMETERS = 1536


class ProjectedComplementarySeriesPoolKneeMILNet(HierarchicalSeriesKneeMILNet):
    """B20 hierarchy plus one zero-gated projected complementary slice summary."""

    def __init__(self, *args, **kwargs) -> None:
        # Construct all historical B20/B12.1 parameters first so the historical
        # construction seed preserves every shared random draw.
        super().__init__(*args, **kwargs)
        d = int(self.encoder.out_dim)
        self.complementary_query = nn.Parameter(torch.randn(d) * 0.02)
        self.complementary_gate = nn.Parameter(torch.zeros(d))
        self._attention_audit_enabled = False
        self._attention_audit_accum = None
        self._require_shared_projection_contract()

    def _require_shared_projection_contract(self) -> None:
        mha = self.series_pool.attention
        d = int(self.encoder.out_dim)
        if int(mha.embed_dim) != d or int(mha.kdim or d) != d or int(mha.vdim or d) != d:
            raise ValueError("B30 requires the square historical B20 series-pool attention")
        if mha.in_proj_weight is None or tuple(mha.in_proj_weight.shape) != (3 * d, d):
            raise ValueError("B30 requires B20 packed Q/K/V projection weights")
        if bool(mha.add_zero_attn) or mha.bias_k is not None or mha.bias_v is not None:
            raise ValueError("B30 requires the historical B20 attention without extra K/V tokens")
        if int(mha.num_heads) < 1 or d % int(mha.num_heads) != 0:
            raise ValueError("B30 historical B20 attention head geometry changed")

    def effective_complementary_gate(self) -> torch.Tensor:
        return torch.tanh(self.complementary_gate)

    def _detached_projected_attention(
        self,
        active_slice_features: torch.Tensor,
        query_vector: torch.Tensor,
        *,
        need_summary: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Apply B20's attention projections as detached deterministic operators.

        Returns per-head no-dropout attention weights [N,H,S]. If ``need_summary``
        is true, also returns a B20-scale token built with the detached historical
        output projection and detached historical LayerNorm affine parameters.

        Gradients can flow into ``query_vector`` and ``active_slice_features``;
        they cannot flow into the reused B20 projection/norm parameters through
        this complementary path.
        """
        if active_slice_features.ndim != 3:
            raise ValueError("B30 projected pool expects [N,S,D]")
        n, s, d = active_slice_features.shape
        if d != int(self.encoder.out_dim):
            raise ValueError("B30 projected pool feature dimension mismatch")

        mha = self.series_pool.attention
        heads = int(mha.num_heads)
        head_dim = d // heads
        dtype = active_slice_features.dtype
        device = active_slice_features.device

        packed_w = mha.in_proj_weight.detach().to(device=device, dtype=dtype)
        q_w, k_w, v_w = packed_w.chunk(3, dim=0)
        if mha.in_proj_bias is None:
            q_b = k_b = v_b = None
        else:
            packed_b = mha.in_proj_bias.detach().to(device=device, dtype=dtype)
            q_b, k_b, v_b = packed_b.chunk(3, dim=0)

        query = query_vector.reshape(-1).to(device=device, dtype=dtype)
        if int(query.numel()) != d:
            raise ValueError("B30 complementary query dimension mismatch")
        query_batch = query[None, :].expand(n, -1)

        q = F.linear(query_batch, q_w, q_b).reshape(n, heads, head_dim)
        k = F.linear(active_slice_features, k_w, k_b).reshape(n, s, heads, head_dim)
        k = k.permute(0, 2, 1, 3)
        scores = torch.sum(q[:, :, None, :] * k, dim=-1) / math.sqrt(float(head_dim))
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=dtype)

        if not need_summary:
            return None, weights

        v = F.linear(active_slice_features, v_w, v_b).reshape(n, s, heads, head_dim)
        v = v.permute(0, 2, 1, 3)
        attended = torch.sum(weights[:, :, :, None] * v, dim=2).reshape(n, d)

        out_w = mha.out_proj.weight.detach().to(device=device, dtype=dtype)
        out_b = None if mha.out_proj.bias is None else mha.out_proj.bias.detach().to(device=device, dtype=dtype)
        attended = F.linear(attended, out_w, out_b)

        # Match B20's residual-query + LayerNorm geometry, but use no dropout and
        # detach the historical affine parameters from the complementary path.
        norm = self.series_pool.norm
        norm_input = (query_batch + attended).float()
        norm_weight = None if norm.weight is None else norm.weight.detach().float().to(device)
        norm_bias = None if norm.bias is None else norm.bias.detach().float().to(device)
        summary = F.layer_norm(
            norm_input,
            (d,),
            weight=norm_weight,
            bias=norm_bias,
            eps=float(norm.eps),
        ).to(dtype=dtype)
        return summary, weights

    def _projected_complementary_summary(
        self, active_slice_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        summary, weights = self._detached_projected_attention(
            active_slice_features,
            self.complementary_query,
            need_summary=True,
        )
        if summary is None:
            raise RuntimeError("B30 complementary summary unexpectedly missing")
        return summary, weights

    def enable_attention_audit(self, enabled: bool = True, *, reset: bool = False) -> None:
        self._attention_audit_enabled = bool(enabled)
        if reset:
            self._attention_audit_accum = None

    @torch.no_grad()
    def _update_attention_audit(
        self,
        active_slice_features: torch.Tensor,
        complementary_weights: torch.Tensor,
        primary_tokens: torch.Tensor,
        complementary_tokens: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        _, primary_weights = self._detached_projected_attention(
            active_slice_features,
            self.series_pool.query.reshape(-1),
            need_summary=False,
        )
        p = primary_weights.float().mean(dim=1).clamp_min(1e-12)
        c = complementary_weights.float().mean(dim=1).clamp_min(1e-12)
        s = int(p.shape[-1])
        log_s = math.log(float(max(s, 2)))
        p_entropy = -(p * p.log()).sum(dim=-1) / log_s
        c_entropy = -(c * c.log()).sum(dim=-1) / log_s

        m = 0.5 * (p + c)
        js = 0.5 * ((p * (p / m).log()).sum(dim=-1) + (c * (c / m).log()).sum(dim=-1))
        js = js / math.log(2.0)  # normalized to [0,1]

        top1 = (p.argmax(dim=-1) == c.argmax(dim=-1)).float()
        topk = min(3, s)
        p_top = torch.topk(p, k=topk, dim=-1).indices
        c_top = torch.topk(c, k=topk, dim=-1).indices
        overlap = (p_top[:, :, None] == c_top[:, None, :]).any(dim=-1).float().sum(dim=-1)
        overlap = overlap / float(topk)

        residual = gate[None, :] * (complementary_tokens - primary_tokens)
        residual_ratio = torch.linalg.vector_norm(residual.float(), dim=-1) / (
            torch.linalg.vector_norm(primary_tokens.float(), dim=-1) + 1e-12
        )

        count = torch.tensor(float(p.shape[0]), device=p.device)
        sums = torch.stack(
            [
                count,
                p_entropy.sum(),
                c_entropy.sum(),
                js.sum(),
                top1.sum(),
                overlap.sum(),
                residual_ratio.sum(),
                residual_ratio.max(),
            ]
        ).detach()
        if self._attention_audit_accum is None:
            self._attention_audit_accum = sums.clone()
        else:
            self._attention_audit_accum[:7] = self._attention_audit_accum[:7] + sums[:7]
            self._attention_audit_accum[7] = torch.maximum(
                self._attention_audit_accum[7], sums[7]
            )

    def attention_audit_state(self, *, reset: bool = False) -> dict:
        acc = self._attention_audit_accum
        if acc is None:
            state = {
                "series_count": 0,
                "primary_attention_entropy_normalized_mean": None,
                "complementary_attention_entropy_normalized_mean": None,
                "js_divergence_normalized_mean": None,
                "top1_slice_agreement": None,
                "top3_slice_overlap_fraction_mean": None,
                "effective_residual_norm_ratio_mean": None,
                "effective_residual_norm_ratio_max": None,
            }
        else:
            x = acc.detach().float().cpu()
            count = max(float(x[0].item()), 1.0)
            state = {
                "series_count": int(round(float(x[0].item()))),
                "primary_attention_entropy_normalized_mean": float(x[1].item() / count),
                "complementary_attention_entropy_normalized_mean": float(x[2].item() / count),
                "js_divergence_normalized_mean": float(x[3].item() / count),
                "top1_slice_agreement": float(x[4].item() / count),
                "top3_slice_overlap_fraction_mean": float(x[5].item() / count),
                "effective_residual_norm_ratio_mean": float(x[6].item() / count),
                "effective_residual_norm_ratio_max": float(x[7].item()),
            }
        if reset:
            self._attention_audit_accum = None
        return state

    def _pool_real_series_b30(
        self,
        slice_features: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        if slice_features.ndim != 4:
            raise ValueError("B30 slice features must be [B,K,S,D]")
        b, k, s, d = slice_features.shape
        flat = slice_features.reshape(b * k, s, d)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return slice_features.new_zeros((b, k, d))

        active_slices = flat.index_select(0, active_indices)

        # Historical B20 pooling occurs first and is unchanged, preserving its
        # stochastic path. The complementary branch itself has no dropout.
        active_primary = self.series_pool(active_slices)
        active_complement, complementary_weights = self._projected_complementary_summary(active_slices)
        gate = self.effective_complementary_gate().to(dtype=active_primary.dtype)
        active_tokens = active_primary + gate[None, :] * (active_complement - active_primary)

        if self._attention_audit_enabled:
            self._update_attention_audit(
                active_slices,
                complementary_weights,
                active_primary,
                active_complement,
                gate,
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
        tokens = self._pool_real_series_b30(slice_features, present)
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
        return {
            "version": B30_RESIDUAL_VERSION,
            "new_parameter_count": int(gate_raw.numel() + query.numel()),
            "query_parameter_count": int(query.numel()),
            "gate_parameter_count": int(gate_raw.numel()),
            "query_max_abs": float(query.abs().max().item()),
            "query_mean_abs": float(query.abs().mean().item()),
            "query_l2": float(torch.linalg.vector_norm(query).item()),
            "gate_raw_max_abs": float(gate_raw.abs().max().item()),
            "gate_raw_mean_abs": float(gate_raw.abs().mean().item()),
            "gate_raw_l2": float(torch.linalg.vector_norm(gate_raw).item()),
            "gate_effective_max_abs": float(gate_effective.abs().max().item()),
            "gate_effective_mean_abs": float(gate_effective.abs().mean().item()),
            "gate_effective_l2": float(torch.linalg.vector_norm(gate_effective).item()),
        }


def b30_model_spec(config: dict, *, normalize_input: bool) -> dict:
    spec = copy.deepcopy(b12_1_model_spec(config, normalize_input=normalize_input))
    spec["architecture"] = B30_ARCHITECTURE
    spec["aggregation"] = B30_AGGREGATION
    spec["b30_residual_version"] = B30_RESIDUAL_VERSION
    spec["b30_new_parameter_count"] = B30_EXPECTED_NEW_PARAMETERS
    spec["b30_complementary_attention"] = (
        "new D-vector query through detached current B20 Q/K/V + out projection + LayerNorm affine; no dropout"
    )
    spec["b30_shared_projection_gradient"] = "detached from complementary branch; historical A branch unchanged"
    spec["b30_gate_constraint"] = "featurewise tanh gate; exactly zero initialisation"
    spec["b30_stochastic_path"] = "historical B20 pool executes first; complementary branch deterministic"
    return spec


def build_b30_model(
    spec: dict,
    *,
    encoder_state: dict | None = None,
    pretrained_weights: bool = False,
) -> ProjectedComplementarySeriesPoolKneeMILNet:
    if spec.get("architecture") != B30_ARCHITECTURE:
        raise ValueError("not a B30 projected-complementary model spec")
    if spec.get("aggregation") != B30_AGGREGATION:
        raise ValueError("B30 aggregation policy mismatch")
    if spec.get("b30_residual_version") != B30_RESIDUAL_VERSION:
        raise ValueError("B30 residual version mismatch")
    if int(spec.get("b30_new_parameter_count", -1)) != B30_EXPECTED_NEW_PARAMETERS:
        raise ValueError("B30 new parameter count mismatch")
    if encoder_state is not None and pretrained_weights:
        raise ValueError("encoder_state and pretrained_weights are mutually exclusive")

    model = ProjectedComplementarySeriesPoolKneeMILNet(
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
    if int(model.complementary_query.numel()) != B30_EXPECTED_QUERY_PARAMETERS:
        raise ValueError("B30 encoder feature dimension changed from the frozen B20 contract")
    if int(model.complementary_gate.numel()) != B30_EXPECTED_GATE_PARAMETERS:
        raise ValueError("B30 gate dimension changed from the frozen B20 contract")
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state, strict=True)
    return model
