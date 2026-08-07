from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .constants import N_TARGETS


class SliceEncoder(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        old = net.conv1
        net.conv1 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                net.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        self.features = nn.Sequential(*list(net.children())[:-1])
        self.out_dim = net.fc.in_features

    def forward(self, x):
        return self.features(x).flatten(1)


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1))

    def forward(self, x):
        a = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * a.unsqueeze(-1), dim=1)


class MultiSeriesKneeNet(nn.Module):
    """MRNet-style shared slice CNN with attention over slices and MRI streams."""
    def __init__(self, n_streams: int, pretrained: bool = False, dropout: float = 0.25):
        super().__init__()
        self.encoder = SliceEncoder(pretrained)
        d = self.encoder.out_dim
        self.slice_pool = AttentionPool(d)
        self.stream_embeddings = nn.Parameter(torch.randn(n_streams, d) * 0.02)
        self.stream_score = nn.Sequential(nn.Linear(d, d // 4), nn.Tanh(), nn.Linear(d // 4, 1))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, N_TARGETS))

    def forward(self, volumes, present):
        b, k, s, h, w = volumes.shape
        feat = self.encoder(volumes.view(b * k * s, 1, h, w)).view(b * k, s, -1)
        series = self.slice_pool(feat).view(b, k, -1) + self.stream_embeddings[:k].unsqueeze(0)
        score = self.stream_score(series).squeeze(-1).masked_fill(present <= 0, -1e4)
        pooled = torch.sum(series * torch.softmax(score, dim=1).unsqueeze(-1), dim=1)
        pooled[present.sum(dim=1) <= 0] = 0
        return self.head(pooled)
