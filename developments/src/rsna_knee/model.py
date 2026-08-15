from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from .constants import N_TARGETS


class ConvNeXtSliceEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, *, pretrained_weights: bool = True, normalize_input: bool = True) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained_weights else None
        net = convnext_tiny(weights=weights)
        first = net.features[0][0]
        if in_channels != 3:
            replacement = nn.Conv2d(
                in_channels,
                first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                bias=first.bias is not None,
            )
            if pretrained_weights:
                with torch.no_grad():
                    replacement.weight.copy_(first.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
                    if first.bias is not None:
                        replacement.bias.copy_(first.bias)
            net.features[0][0] = replacement
        self.features = net.features
        self.avgpool = net.avgpool
        self.pre_classifier = nn.Sequential(*list(net.classifier.children())[:-1])
        self.out_dim = int(net.classifier[-1].in_features)
        self.normalize_input = bool(normalize_input)
        rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        mean = rgb_mean if in_channels == 3 else rgb_mean.mean().repeat(in_channels)
        std = rgb_std if in_channels == 3 else rgb_std.mean().repeat(in_channels)
        self.register_buffer("input_mean", mean.view(1, in_channels, 1, 1), persistent=False)
        self.register_buffer("input_std", std.view(1, in_channels, 1, 1), persistent=False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            return (x - self.input_mean.to(dtype=x.dtype)) / self.input_std.to(dtype=x.dtype)
        return x

    def forward(self, x):
        x = self._normalize(x)
        return self.pre_classifier(self.avgpool(self.features(x)))

    def forward_spatial(self, x: torch.Tensor, grid_size: int = 2) -> torch.Tensor:
        """Return coarse ConvNeXt spatial tokens shaped [N, grid_size**2, D].

        B7 collapses every 2.5D slice to one global vector. B8 reuses the same
        ConvNeXt weights but retains a small spatial grid from the final feature
        map. The classifier normalization is reused before the grid is flattened,
        so B7/B5 encoder initialization remains meaningful.
        """
        grid_size = int(grid_size)
        if grid_size < 1:
            raise ValueError("grid_size must be >=1")
        x = self._normalize(x)
        feature_map = self.features(x)
        pooled = F.adaptive_avg_pool2d(feature_map, (grid_size, grid_size))
        # ConvNeXt's first classifier module is the learned channel normalization
        # used by the ordinary globally pooled path and accepts NCHW tensors.
        normalized = self.pre_classifier[0](pooled)
        return normalized.permute(0, 2, 3, 1).reshape(x.shape[0], grid_size * grid_size, self.out_dim)


class KneeMILNet(nn.Module):
    """2.5D ConvNeXt + MRI Transformer + interacting pathology queries."""

    def __init__(
        self,
        n_streams: int,
        n_slices: int,
        *,
        in_channels: int = 3,
        pretrained_weights: bool = True,
        normalize_input: bool = True,
        dropout: float = 0.25,
        encoder_batch_size: int = 24,
        gradient_checkpointing: bool = True,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_ff_mult: float = 2.0,
        pathology_layers: int = 1,
    ) -> None:
        super().__init__()
        if n_streams < 1 or n_slices < 1 or encoder_batch_size < 1:
            raise ValueError("n_streams, n_slices and encoder_batch_size must be positive")
        self.n_streams = int(n_streams)
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
        if d % int(transformer_heads) != 0:
            raise ValueError("transformer_heads must divide encoder feature dimension")

        self.slice_position = nn.Parameter(torch.randn(n_slices, d) * 0.02)
        self.stream_embedding = nn.Parameter(torch.randn(n_streams, d) * 0.02)
        mri_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(transformer_heads),
            dim_feedforward=int(d * float(transformer_ff_mult)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            mri_layer,
            num_layers=int(transformer_layers),
            norm=nn.LayerNorm(d),
        )
        self.pathology_tokens = nn.Parameter(torch.randn(N_TARGETS, d) * 0.02)
        pathology_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(transformer_heads),
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
            int(transformer_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)
        self.target_weight = nn.Parameter(torch.empty(N_TARGETS, d))
        self.target_bias = nn.Parameter(torch.zeros(N_TARGETS))
        nn.init.xavier_uniform_(self.target_weight)

    def _encode_chunk(self, chunk):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self.encoder, chunk, use_reentrant=False)
        return self.encoder(chunk)

    def _reshape(self, volumes):
        if volumes.ndim != 6:
            raise ValueError("production model expects [B,K,S,3,H,W]")
        b, k, s, c, h, w = volumes.shape
        if k != self.n_streams:
            raise ValueError(f"model expects {self.n_streams} streams, received {k}")
        if s != self.n_slices:
            raise ValueError(f"model expects {self.n_slices} sampled slices, received {s}")
        if c != self.in_channels:
            raise ValueError(f"model expects {self.in_channels} channels, received {c}")
        return volumes.reshape(b * k, s, c, h, w), b, k, s

    def _encode_slices(self, volumes, present):
        stream_volumes, b, k, s = self._reshape(volumes)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        d = self.encoder.out_dim
        if active_indices.numel() == 0:
            return volumes.new_zeros((b, k, s, d))
        active = stream_volumes.index_select(0, active_indices)
        flat = active.reshape(-1, *active.shape[2:])
        encoded = torch.cat(
            [self._encode_chunk(chunk) for chunk in flat.split(self.encoder_batch_size, dim=0)],
            dim=0,
        ).reshape(active.shape[0], s, d)
        all_features = encoded.new_zeros((b * k, s, d)).index_copy(0, active_indices, encoded)
        features = all_features.reshape(b, k, s, d)
        active_mask = present[:, :, None, None].to(features.dtype)
        return (
            features
            + self.slice_position[None, None, :, :]
            + self.stream_embedding[None, :, None, :]
        ) * active_mask

    def _mri_tokens(self, volumes, present):
        features = self._encode_slices(volumes, present)
        b, k, s, d = features.shape
        tokens = features.reshape(b, k * s, d)
        padding = (present <= 0)[:, :, None].expand(b, k, s).reshape(b, k * s)
        empty = padding.all(dim=1)
        safe_padding = padding.clone()
        # Attention with every key masked is undefined. Expose one zero token for
        # empty studies, then zero the contextual output again afterward.
        if empty.any():
            safe_padding[empty, 0] = False
            tokens = tokens.clone()
            tokens[empty, 0] = 0
        contextual = self.context(tokens, src_key_padding_mask=safe_padding)
        contextual = contextual.masked_fill(padding[:, :, None], 0.0)
        return contextual, safe_padding, empty

    def forward(self, volumes, present):
        if present.ndim != 2 or present.shape[1] != self.n_streams:
            raise ValueError("present mask does not match stream contract")
        memory, padding, empty = self._mri_tokens(volumes, present)
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


class SpatialAnatomyKneeMILNet(KneeMILNet):
    """B8 model: B7 pathology queries over coarse within-slice spatial tokens.

    The fixed attention bias is deliberately limited to stream and relative-slice
    priors. Region embeddings are learned from weak supervision; no hard-coded
    left/right or anterior/posterior quadrant semantics are assumed because the
    current image pipeline does not guarantee canonical in-plane orientation.
    """

    def __init__(
        self,
        n_streams: int,
        n_slices: int,
        *,
        spatial_grid_size: int = 2,
        anatomy_attention_bias: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(n_streams, n_slices, **kwargs)
        self.spatial_grid_size = int(spatial_grid_size)
        if self.spatial_grid_size < 1:
            raise ValueError("spatial_grid_size must be >=1")
        self.n_regions = self.spatial_grid_size * self.spatial_grid_size
        d = self.encoder.out_dim
        self.region_embedding = nn.Parameter(torch.zeros(self.n_regions, d))
        if anatomy_attention_bias is None:
            anatomy_attention_bias = torch.zeros(
                N_TARGETS,
                self.n_streams * self.n_slices * self.n_regions,
                dtype=torch.float32,
            )
        bias = torch.as_tensor(anatomy_attention_bias, dtype=torch.float32)
        expected = (N_TARGETS, self.n_streams * self.n_slices * self.n_regions)
        if tuple(bias.shape) != expected:
            raise ValueError(f"anatomy_attention_bias shape {tuple(bias.shape)} != {expected}")
        self.register_buffer("anatomy_attention_bias", bias.clone(), persistent=True)

    def _encode_spatial_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                lambda x: self.encoder.forward_spatial(x, self.spatial_grid_size),
                chunk,
                use_reentrant=False,
            )
        return self.encoder.forward_spatial(chunk, self.spatial_grid_size)

    def _encode_spatial_slices(self, volumes: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        stream_volumes, b, k, s = self._reshape(volumes)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        d, r = self.encoder.out_dim, self.n_regions
        if active_indices.numel() == 0:
            return volumes.new_zeros((b, k, s, r, d))
        active = stream_volumes.index_select(0, active_indices)
        flat = active.reshape(-1, *active.shape[2:])
        encoded = torch.cat(
            [self._encode_spatial_chunk(chunk) for chunk in flat.split(self.encoder_batch_size, dim=0)],
            dim=0,
        ).reshape(active.shape[0], s, r, d)
        all_features = encoded.new_zeros((b * k, s, r, d)).index_copy(0, active_indices, encoded)
        features = all_features.reshape(b, k, s, r, d)
        active_mask = present[:, :, None, None, None].to(features.dtype)
        return (
            features
            + self.slice_position[None, None, :, None, :]
            + self.stream_embedding[None, :, None, None, :]
            + self.region_embedding[None, None, None, :, :]
        ) * active_mask

    def _mri_tokens(self, volumes, present):
        features = self._encode_spatial_slices(volumes, present)
        b, k, s, r, d = features.shape
        tokens = features.reshape(b, k * s * r, d)
        padding = (
            (present <= 0)[:, :, None, None]
            .expand(b, k, s, r)
            .reshape(b, k * s * r)
        )
        empty = padding.all(dim=1)
        safe_padding = padding.clone()
        if empty.any():
            safe_padding[empty, 0] = False
            tokens = tokens.clone()
            tokens[empty, 0] = 0
        contextual = self.context(tokens, src_key_padding_mask=safe_padding)
        contextual = contextual.masked_fill(padding[:, :, None], 0.0)
        return contextual, safe_padding, empty

    def forward(self, volumes, present):
        if present.ndim != 2 or present.shape[1] != self.n_streams:
            raise ValueError("present mask does not match stream contract")
        memory, padding, empty = self._mri_tokens(volumes, present)
        b = memory.shape[0]
        queries = self.pathology_tokens[None, :, :].expand(b, -1, -1)
        queries = self.pathology_context(queries)
        attention_bias = self.anatomy_attention_bias.to(device=queries.device, dtype=queries.dtype)
        attended, _ = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=padding,
            attn_mask=attention_bias,
            need_weights=False,
        )
        queries = self.dropout(self.query_norm(queries + attended))
        logits = (queries * self.target_weight[None, :, :]).sum(dim=-1) + self.target_bias
        return torch.where(empty[:, None], self.target_bias[None, :], logits)
