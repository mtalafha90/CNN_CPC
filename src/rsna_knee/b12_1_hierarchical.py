"""B12.1 hierarchical variable-series MRI representation.

B12 retains every eligible real MRI acquisition but sends all K x S slice tokens
into one study Transformer. B12.1 keeps the exact same B12 series surface and
metadata, then compresses each real series to one learned series token before the
unchanged study-level Transformer and pathology-query heads.

The scientific change is intentionally narrow:

    16 slice tokens / series -> learned attention-pooled series token
    K series tokens          -> study Transformer -> pathology queries

There is no series-rank/position embedding, so acquisition order remains arbitrary.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .constants import N_TARGETS
from .model import ConvNeXtSliceEncoder

B12_1_ARCHITECTURE = "hierarchical_series_token_pathology_queries_v1"
B12_1_AGGREGATION = "learned_series_query_attention_v1"


class LearnedSeriesPool(nn.Module):
    """Compress one real MRI series from S slice tokens to one learned token."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % int(heads) != 0:
            raise ValueError("series pooling heads must divide feature dimension")
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            dim,
            int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, slice_tokens: torch.Tensor) -> torch.Tensor:
        if slice_tokens.ndim != 3:
            raise ValueError("series pool expects [N,S,D]")
        query = self.query.expand(slice_tokens.shape[0], -1, -1)
        attended, _ = self.attention(
            query,
            slice_tokens,
            slice_tokens,
            need_weights=False,
        )
        return self.dropout(self.norm(query + attended)).squeeze(1)


class HierarchicalSeriesKneeMILNet(nn.Module):
    """B12.1: ConvNeXt slices -> learned series tokens -> study Transformer."""

    def __init__(
        self,
        n_slices: int,
        *,
        in_channels: int = 3,
        pretrained_weights: bool = False,
        normalize_input: bool = False,
        dropout: float = 0.25,
        encoder_batch_size: int = 24,
        gradient_checkpointing: bool = True,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_ff_mult: float = 2.0,
        pathology_layers: int = 1,
        series_pool_heads: int | None = None,
    ) -> None:
        super().__init__()
        if n_slices < 1 or encoder_batch_size < 1:
            raise ValueError("n_slices and encoder_batch_size must be positive")
        self.n_slices = int(n_slices)
        self.in_channels = int(in_channels)
        self.encoder_batch_size = int(encoder_batch_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.encoder = ConvNeXtSliceEncoder(
            in_channels,
            pretrained_weights=pretrained_weights,
            normalize_input=normalize_input,
        )
        d = self.encoder.out_dim
        heads = int(transformer_heads)
        if d % heads != 0:
            raise ValueError("transformer_heads must divide encoder feature dimension")
        pool_heads = heads if series_pool_heads is None else int(series_pool_heads)
        if d % pool_heads != 0:
            raise ValueError("series_pool_heads must divide encoder feature dimension")

        self.slice_position = nn.Parameter(torch.randn(self.n_slices, d) * 0.02)
        self.plane_embedding = nn.Embedding(4, d, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, d, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, d, padding_idx=0)
        self.series_pool = LearnedSeriesPool(d, pool_heads, float(dropout))

        study_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=int(d * float(transformer_ff_mult)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            study_layer,
            num_layers=int(transformer_layers),
            norm=nn.LayerNorm(d),
        )

        self.pathology_tokens = nn.Parameter(torch.randn(N_TARGETS, d) * 0.02)
        pathology_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=int(d * float(transformer_ff_mult)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pathology_context = nn.TransformerEncoder(
            pathology_layer,
            num_layers=int(pathology_layers),
            norm=nn.LayerNorm(d),
        )
        self.cross_attention = nn.MultiheadAttention(
            d,
            heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(float(dropout))
        self.target_weight = nn.Parameter(torch.empty(N_TARGETS, d))
        self.target_bias = nn.Parameter(torch.zeros(N_TARGETS))
        nn.init.xavier_uniform_(self.target_weight)

    def _encode_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self.encoder, chunk, use_reentrant=False)
        return self.encoder(chunk)

    def _encode_slices(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> torch.Tensor:
        if volumes.ndim != 6:
            raise ValueError("B12.1 model expects [B,K,S,3,H,W]")
        b, k, s, c, h, w = volumes.shape
        if s != self.n_slices or c != self.in_channels:
            raise ValueError("B12.1 slice/channel contract mismatch")
        if present.shape != (b, k):
            raise ValueError("B12.1 present mask shape mismatch")
        if series_meta.shape != (b, k, 3):
            raise ValueError("B12.1 series_meta must have shape [B,K,3]")

        flat_series = volumes.reshape(b * k, s, c, h, w)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        d = self.encoder.out_dim
        if active_indices.numel() == 0:
            return volumes.new_zeros((b, k, s, d))

        active = flat_series.index_select(0, active_indices)
        flat_slices = active.reshape(-1, c, h, w)
        encoded = torch.cat(
            [self._encode_chunk(chunk) for chunk in flat_slices.split(self.encoder_batch_size, dim=0)],
            dim=0,
        ).reshape(active.shape[0], s, d)
        all_features = encoded.new_zeros((b * k, s, d)).index_copy(
            0, active_indices, encoded
        )
        features = all_features.reshape(b, k, s, d)

        plane = self.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
        fluid = self.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
        fat = self.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        metadata = plane + fluid + fat
        mask = present[:, :, None, None].to(features.dtype)
        return (
            features
            + self.slice_position[None, None, :, :]
            + metadata[:, :, None, :]
        ) * mask

    def _pool_real_series(
        self,
        slice_features: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        if slice_features.ndim != 4:
            raise ValueError("B12.1 slice features must be [B,K,S,D]")
        b, k, s, d = slice_features.shape
        flat = slice_features.reshape(b * k, s, d)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return slice_features.new_zeros((b, k, d))
        active_tokens = self.series_pool(flat.index_select(0, active_indices))
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
        tokens = self._pool_real_series(slice_features, present)
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

    def forward(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> torch.Tensor:
        memory, padding, empty = self._study_memory(volumes, present, series_meta)
        b = memory.shape[0]
        queries = self.pathology_tokens[None, :, :].expand(b, -1, -1)
        queries = self.pathology_context(queries)
        attended, _ = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=padding,
            need_weights=False,
        )
        queries = self.dropout(self.query_norm(queries + attended))
        logits = (queries * self.target_weight[None, :, :]).sum(dim=-1) + self.target_bias
        return torch.where(empty[:, None], self.target_bias[None, :], logits)


def b12_1_model_spec(config: dict, *, normalize_input: bool) -> dict:
    return {
        "architecture": B12_1_ARCHITECTURE,
        "series_policy": "all_repaired_anatomical_series_v1",
        "aggregation": B12_1_AGGREGATION,
        "n_slices": int(config.get("b7_n_slices", 16)),
        "in_channels": 3,
        "image_size": int(config.get("b7_image_size", 224)),
        "triplet_gap": int(config.get("b7_triplet_gap", 1)),
        "dropout": float(config.get("b7_dropout", 0.25)),
        "normalize_input": bool(normalize_input),
        "encoder_batch_size": int(config.get("b7_encoder_batch_size", 24)),
        "gradient_checkpointing": bool(config.get("b7_gradient_checkpointing", True)),
        "transformer_layers": int(config.get("b7_transformer_layers", 2)),
        "transformer_heads": int(config.get("b7_transformer_heads", 8)),
        "transformer_ff_mult": float(config.get("b7_transformer_ff_mult", 2.0)),
        "pathology_layers": int(config.get("b7_pathology_layers", 1)),
        "series_pool_heads": int(config.get("b12_1_series_pool_heads", config.get("b7_transformer_heads", 8))),
        "metadata_embeddings": "plane/fluid/fat categorical; no series-position embedding",
    }


def build_b12_1_model(
    spec: dict,
    *,
    encoder_state: dict | None = None,
) -> HierarchicalSeriesKneeMILNet:
    if spec.get("architecture") != B12_1_ARCHITECTURE:
        raise ValueError("not a B12.1 hierarchical series-token model spec")
    if spec.get("aggregation") != B12_1_AGGREGATION:
        raise ValueError("B12.1 aggregation policy mismatch")
    model = HierarchicalSeriesKneeMILNet(
        int(spec["n_slices"]),
        in_channels=int(spec.get("in_channels", 3)),
        pretrained_weights=False,
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
