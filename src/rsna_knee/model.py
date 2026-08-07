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
from .model3d import SharedVolumeEncoder


class SliceEncoder(nn.Module):
    """Shared 2D encoder for grayscale slices or 2.5D triplets.

    Built-in backbones are ``resnet18`` and ``convnext_tiny``. Any timm model
    that supports ``num_classes=0`` can be requested as ``timm:<model_name>``.
    """

    def __init__(self, pretrained: bool = False, in_channels: int = 1, backbone: str = "resnet18"):
        super().__init__()
        backbone = str(backbone)
        self.backbone_name = backbone

        if backbone.lower() == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet18(weights=weights)
            old = net.conv1
            if in_channels != 3:
                net.conv1 = nn.Conv2d(
                    in_channels, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=False,
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

        elif backbone.lower() == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            net = convnext_tiny(weights=weights)
            first = net.features[0][0]
            if in_channels != 3:
                replacement = nn.Conv2d(
                    in_channels, first.out_channels, kernel_size=first.kernel_size,
                    stride=first.stride, padding=first.padding,
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
            self.pre_classifier = nn.Sequential(*list(net.classifier.children())[:-1])
            self.out_dim = int(net.classifier[-1].in_features)
            self._forward_impl = self._forward_convnext

        elif backbone.lower().startswith("timm:"):
            import timm
            model_name = backbone.split(":", 1)[1]
            self.timm_model = timm.create_model(
                model_name, pretrained=pretrained, in_chans=in_channels,
                num_classes=0, global_pool="avg",
            )
            self.out_dim = int(self.timm_model.num_features)
            self._forward_impl = self._forward_timm
        else:
            raise ValueError(f"unsupported backbone: {backbone}")

    def _forward_resnet(self, x):
        return self.features(x).flatten(1)

    def _forward_convnext(self, x):
        return self.pre_classifier(self.pool(self.features(x)))

    def _forward_timm(self, x):
        return self.timm_model(x)

    def forward(self, x):
        return self._forward_impl(x)


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden = max(16, dim // 4)
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        a = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * a.unsqueeze(-1), dim=1)


class TopKAttentionPool(nn.Module):
    """After slice encoding, retain the highest-scoring feature fraction."""

    def __init__(self, dim: int, fraction: float = 0.25):
        super().__init__()
        if not 0 < fraction <= 1:
            raise ValueError("topk_fraction must be in (0, 1]")
        self.fraction = float(fraction)
        hidden = max(16, dim // 4)
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        scores = self.score(x).squeeze(-1)
        k = max(1, int(round(x.shape[1] * self.fraction)))
        top = torch.topk(scores, k=k, dim=1).indices
        gather = top.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        selected = torch.gather(x, 1, gather)
        selected_scores = torch.gather(scores, 1, top)
        weights = torch.softmax(selected_scores, dim=1)
        return torch.sum(selected * weights.unsqueeze(-1), dim=1)


class MeanPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


class MaxPool(nn.Module):
    def forward(self, x):
        return x.amax(dim=1)


def build_slice_pooling(name: str, dim: int, topk_fraction: float = 0.25) -> nn.Module:
    name = str(name).lower()
    if name == "attention":
        return AttentionPool(dim)
    if name == "topk":
        return TopKAttentionPool(dim, topk_fraction)
    if name == "mean":
        return MeanPool()
    if name == "max":
        return MaxPool()
    raise ValueError(f"unsupported slice_pooling: {name}")


class MultiSeriesKneeNet(nn.Module):
    """Multi-series MRI model for 2D/2.5D MIL or a compact 3D arm.

    Use ``backbone: 3d`` with ``input_mode: 2d`` for the volumetric arm. It
    encodes each complete sampled stream with shared 3D convolutions, then uses
    the same stream-fusion logic as the 2D/2.5D family.
    """

    def __init__(
        self,
        n_streams: int,
        pretrained: bool = False,
        dropout: float = 0.25,
        in_channels: int = 1,
        backbone: str = "resnet18",
        target_attention: bool = False,
        slice_pooling: str = "attention",
        topk_fraction: float = 0.25,
        base_channels_3d: int = 16,
    ):
        super().__init__()
        self.target_attention = bool(target_attention)
        self.is_3d = str(backbone).lower() == "3d"

        if self.is_3d:
            if in_channels != 1:
                raise ValueError("backbone=3d requires grayscale input_mode=2d")
            self.volume_encoder = SharedVolumeEncoder(base_channels_3d)
            d = self.volume_encoder.out_dim
            self.slice_pool = None
        else:
            self.encoder = SliceEncoder(pretrained, in_channels=in_channels, backbone=backbone)
            d = self.encoder.out_dim
            self.slice_pool = build_slice_pooling(slice_pooling, d, topk_fraction)

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
            hidden = max(16, d // 4)
            self.stream_score = nn.Sequential(
                nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1)
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, N_TARGETS)
            )

    def _encode_streams(self, volumes):
        if self.is_3d:
            if volumes.ndim != 5:
                raise ValueError("3D backbone expects [B,K,S,H,W]")
            b, k, s, h, w = volumes.shape
            x = volumes.view(b * k, 1, s, h, w)
            feat = self.volume_encoder(x).view(b, k, -1)
            return feat + self.stream_embeddings[:k].unsqueeze(0)

        if volumes.ndim == 5:
            b, k, s, h, w = volumes.shape
            x = volumes.view(b * k * s, 1, h, w)
        elif volumes.ndim == 6:
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
