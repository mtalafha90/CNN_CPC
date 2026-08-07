from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
    convnext_tiny,
    resnet18,
)

from .constants import N_TARGETS


class SliceEncoder(nn.Module):
    """Shared 2D encoder for either grayscale slices or 2.5D triplets."""

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 1,
        backbone: str = "resnet18",
    ):
        super().__init__()
        backbone = str(backbone).lower()
        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet18(weights=weights)
            old = net.conv1
            if in_channels != 3:
                net.conv1 = nn.Conv2d(
                    in_channels,
                    old.out_channels,
                    kernel_size=old.kernel_size,
                    stride=old.stride,
                    padding=old.padding,
                    bias=False,
                )
                if pretrained:
                    with torch.no_grad():
                        if in_channels == 1:
                            net.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                        else:
                            mean = old.weight.mean(dim=1, keepdim=True)
                            net.conv1.weight.copy_(mean.repeat(1, in_channels, 1, 1))
            self.features = nn.Sequential(*list(net.children())[:-1])
            self.out_dim = int(net.fc.in_features)
            self._forward_impl = self._forward_resnet

        elif backbone == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
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
                if pretrained:
                    with torch.no_grad():
                        if in_channels == 1:
                            replacement.weight.copy_(first.weight.mean(dim=1, keepdim=True))
                        else:
                            replacement.weight.copy_(
                                first.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                            )
                        if first.bias is not None:
                            replacement.bias.copy_(first.bias)
                net.features[0][0] = replacement
            self.features = net.features
            self.pool = net.avgpool
            self.norm = net.classifier[0]
            self.out_dim = int(net.classifier[-1].in_features)
            self._forward_impl = self._forward_convnext
        else:
            raise ValueError(f"unsupported backbone: {backbone}")

    def _forward_resnet(self, x):
        return self.features(x).flatten(1)

    def _forward_convnext(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.norm(x)

    def forward(self, x):
        return self._forward_impl(x)


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, max(16, dim // 4)),
            nn.Tanh(),
            nn.Linear(max(16, dim // 4), 1),
        )

    def forward(self, x):
        a = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * a.unsqueeze(-1), dim=1)


class MultiSeriesKneeNet(nn.Module):
    """Multi-series MIL network with optional target-specific stream attention.

    ``target_attention=False`` reproduces the original shared-attention baseline.
    ``target_attention=True`` gives each of the 12 diagnoses an independent query
    over the MRI streams, which is useful because different abnormalities prefer
    different planes/sequences.
    """

    def __init__(
        self,
        n_streams: int,
        pretrained: bool = False,
        dropout: float = 0.25,
        in_channels: int = 1,
        backbone: str = "resnet18",
        target_attention: bool = False,
    ):
        super().__init__()
        self.target_attention = bool(target_attention)
        self.encoder = SliceEncoder(pretrained, in_channels=in_channels, backbone=backbone)
        d = self.encoder.out_dim
        self.slice_pool = AttentionPool(d)
        self.stream_embeddings = nn.Parameter(torch.randn(n_streams, d) * 0.02)

        if self.target_attention:
            self.target_queries = nn.Parameter(torch.randn(N_TARGETS, d) * 0.02)
            self.target_key = nn.Linear(d, d, bias=False)
            self.target_norm = nn.LayerNorm(d)
            self.target_dropout = nn.Dropout(dropout)
            self.target_weight = nn.Parameter(torch.empty(N_TARGETS, d))
            self.target_bias = nn.Parameter(torch.zeros(N_TARGETS))
            nn.init.xavier_uniform_(self.target_weight)
        else:
            self.stream_score = nn.Sequential(
                nn.Linear(d, max(16, d // 4)),
                nn.Tanh(),
                nn.Linear(max(16, d // 4), 1),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, N_TARGETS)
            )

    def _encode_streams(self, volumes):
        if volumes.ndim == 5:
            # [B,K,S,H,W] grayscale
            b, k, s, h, w = volumes.shape
            x = volumes.view(b * k * s, 1, h, w)
        elif volumes.ndim == 6:
            # [B,K,S,C,H,W] 2.5D
            b, k, s, c, h, w = volumes.shape
            x = volumes.view(b * k * s, c, h, w)
        else:
            raise ValueError(f"expected 5D or 6D MRI tensor, got {tuple(volumes.shape)}")
        feat = self.encoder(x).view(b * k, s, -1)
        return self.slice_pool(feat).view(b, k, -1) + self.stream_embeddings[:k].unsqueeze(0)

    def _target_specific_forward(self, series, present):
        d = series.shape[-1]
        keys = self.target_key(series)
        scores = torch.einsum("bkd,td->btk", keys, self.target_queries) / (d ** 0.5)
        scores = scores.masked_fill(present[:, None, :] <= 0, -1e4)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("btk,bkd->btd", weights, series)
        none_present = present.sum(dim=1) <= 0
        pooled[none_present] = 0
        pooled = self.target_dropout(self.target_norm(pooled))
        return (pooled * self.target_weight.unsqueeze(0)).sum(dim=-1) + self.target_bias

    def forward(self, volumes, present):
        series = self._encode_streams(volumes)
        if self.target_attention:
            return self._target_specific_forward(series, present)

        score = self.stream_score(series).squeeze(-1).masked_fill(present <= 0, -1e4)
        pooled = torch.sum(series * torch.softmax(score, dim=1).unsqueeze(-1), dim=1)
        pooled[present.sum(dim=1) <= 0] = 0
        return self.head(pooled)
