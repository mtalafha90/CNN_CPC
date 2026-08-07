from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from .constants import N_TARGETS


class ConvNeXtSliceEncoder(nn.Module):
    """ConvNeXt-Tiny encoder for grayscale slices or 2.5D triplets.

    ``pretrained_weights`` controls whether ImageNet weights are loaded.
    ``normalize_input`` is independent so inference can rebuild a trained model
    without downloading weights while still applying the exact normalization
    used during training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        *,
        pretrained_weights: bool = True,
        normalize_input: bool = True,
    ) -> None:
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
                    mean_weight = first.weight.mean(dim=1, keepdim=True)
                    replacement.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
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
        if in_channels == 3:
            mean, std = rgb_mean, rgb_std
        else:
            mean = rgb_mean.mean().repeat(in_channels)
            std = rgb_std.mean().repeat(in_channels)
        self.register_buffer("input_mean", mean.view(1, in_channels, 1, 1), persistent=False)
        self.register_buffer("input_std", std.view(1, in_channels, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x = (x - self.input_mean.to(dtype=x.dtype)) / self.input_std.to(dtype=x.dtype)
        return self.pre_classifier(self.avgpool(self.features(x)))


class KneeMILNet(nn.Module):
    """Hierarchical target-specific MIL model for multi-sequence knee MRI.

    Each 2.5D slice triplet is encoded by a shared ConvNeXt-Tiny. Every target
    gets its own attention over slice positions within each MRI stream and its
    own attention over the available streams.

    Encoder work is split into bounded micro-batches. During training, optional
    gradient checkpointing recomputes ConvNeXt activations in backward rather
    than retaining all slice activations simultaneously, substantially reducing
    peak memory before multi-GPU parallelism is introduced.
    """

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
    ) -> None:
        super().__init__()
        if n_streams < 1 or n_slices < 1:
            raise ValueError("n_streams and n_slices must be positive")
        if encoder_batch_size < 1:
            raise ValueError("encoder_batch_size must be positive")

        self.n_streams = int(n_streams)
        self.n_slices = int(n_slices)
        self.encoder_batch_size = int(encoder_batch_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.encoder = ConvNeXtSliceEncoder(
            in_channels,
            pretrained_weights=pretrained_weights,
            normalize_input=normalize_input,
        )
        d = self.encoder.out_dim

        self.slice_position = nn.Parameter(torch.randn(n_slices, d) * 0.02)
        self.stream_embedding = nn.Parameter(torch.randn(n_streams, d) * 0.02)

        self.slice_key = nn.Linear(d, d, bias=False)
        self.slice_query = nn.Parameter(torch.randn(N_TARGETS, d) * 0.02)
        self.stream_key = nn.Linear(d, d, bias=False)
        self.stream_query = nn.Parameter(torch.randn(N_TARGETS, d) * 0.02)

        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)
        self.target_weight = nn.Parameter(torch.empty(N_TARGETS, d))
        self.target_bias = nn.Parameter(torch.zeros(N_TARGETS))
        nn.init.xavier_uniform_(self.target_weight)

    def _encode_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self.encoder, chunk, use_reentrant=False)
        return self.encoder(chunk)

    def _encode_slices(self, volumes: torch.Tensor) -> torch.Tensor:
        if volumes.ndim == 5:
            b, k, s, h, w = volumes.shape
            flat = volumes.reshape(b * k * s, 1, h, w)
        elif volumes.ndim == 6:
            b, k, s, c, h, w = volumes.shape
            flat = volumes.reshape(b * k * s, c, h, w)
        else:
            raise ValueError(f"expected [B,K,S,H,W] or [B,K,S,C,H,W], got {tuple(volumes.shape)}")
        if k != self.n_streams:
            raise ValueError(f"model expects {self.n_streams} streams, received {k}")
        if s != self.n_slices:
            raise ValueError(f"model expects {self.n_slices} sampled slices, received {s}")

        encoded = [self._encode_chunk(chunk) for chunk in flat.split(self.encoder_batch_size, dim=0)]
        features = torch.cat(encoded, dim=0).reshape(b, k, s, -1)
        return features + self.slice_position[None, None, :, :]

    def forward(self, volumes: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        if present.ndim != 2:
            raise ValueError(f"present mask must be [B,K], got {tuple(present.shape)}")
        features = self._encode_slices(volumes)
        d = features.shape[-1]
        scale = math.sqrt(d)

        slice_keys = self.slice_key(features)
        slice_scores = torch.einsum("bksd,td->bkts", slice_keys, self.slice_query) / scale
        slice_weights = torch.softmax(slice_scores, dim=-1)
        series = torch.einsum("bkts,bksd->bktd", slice_weights, features)
        series = series + self.stream_embedding[None, :, None, :]

        stream_keys = self.stream_key(series)
        stream_scores = torch.einsum("bktd,td->btk", stream_keys, self.stream_query) / scale
        stream_scores = stream_scores.masked_fill(present[:, None, :] <= 0, -1e4)
        stream_weights = torch.softmax(stream_scores, dim=-1)
        pooled = torch.einsum("btk,bktd->btd", stream_weights, series)
        pooled[present.sum(dim=1) <= 0] = 0

        pooled = self.dropout(self.norm(pooled))
        return (pooled * self.target_weight[None, :, :]).sum(dim=-1) + self.target_bias
