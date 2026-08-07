from __future__ import annotations

import torch
from torch import nn

from .constants import N_TARGETS


class SharedVolumeEncoder(nn.Module):
    """Small shared 3D encoder for one MRI stream.

    The goal of this arm is orthogonal inter-slice signal, not maximum capacity.
    It deliberately stays compact so it can be compared against 2.5D under the
    same runtime constraints.
    """

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = int(base_channels)
        self.net = nn.Sequential(
            nn.Conv3d(1, c, 3, padding=1, bias=False),
            nn.BatchNorm3d(c),
            nn.GELU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(c, c * 2, 3, padding=1, bias=False),
            nn.BatchNorm3d(c * 2),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(c * 2, c * 4, 3, padding=1, bias=False),
            nn.BatchNorm3d(c * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.out_dim = c * 4

    def forward(self, x):
        return self.net(x).flatten(1)


class Small3DKneeNet(nn.Module):
    """Shared 3D stream encoder plus optional target-specific stream attention."""

    def __init__(
        self,
        n_streams: int,
        dropout: float = 0.25,
        base_channels: int = 16,
        target_attention: bool = True,
    ):
        super().__init__()
        self.target_attention = bool(target_attention)
        self.encoder = SharedVolumeEncoder(base_channels)
        d = self.encoder.out_dim
        self.stream_embeddings = nn.Parameter(torch.randn(n_streams, d) * 0.02)

        if self.target_attention:
            self.target_queries = nn.Parameter(torch.randn(N_TARGETS, d) * 0.02)
            self.target_key = nn.Linear(d, d, bias=False)
            self.norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(dropout)
            self.weight = nn.Parameter(torch.empty(N_TARGETS, d))
            self.bias = nn.Parameter(torch.zeros(N_TARGETS))
            nn.init.xavier_uniform_(self.weight)
        else:
            hidden = max(16, d // 2)
            self.stream_score = nn.Sequential(
                nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1)
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, N_TARGETS)
            )

    def _encode(self, volumes):
        if volumes.ndim != 5:
            raise ValueError(
                "Small3DKneeNet expects [B,K,S,H,W]; set input_mode=2d for the 3D arm"
            )
        b, k, s, h, w = volumes.shape
        x = volumes.view(b * k, 1, s, h, w)
        feat = self.encoder(x).view(b, k, -1)
        return feat + self.stream_embeddings[:k].unsqueeze(0)

    def forward(self, volumes, present):
        series = self._encode(volumes)
        none_present = present.sum(dim=1) <= 0

        if self.target_attention:
            d = series.shape[-1]
            keys = self.target_key(series)
            score = torch.einsum("bkd,td->btk", keys, self.target_queries) / (d ** 0.5)
            score = score.masked_fill(present[:, None, :] <= 0, -1e4)
            a = torch.softmax(score, dim=-1)
            pooled = torch.einsum("btk,bkd->btd", a, series)
            pooled[none_present] = 0
            pooled = self.drop(self.norm(pooled))
            return (pooled * self.weight.unsqueeze(0)).sum(dim=-1) + self.bias

        score = self.stream_score(series).squeeze(-1).masked_fill(present <= 0, -1e4)
        pooled = torch.sum(series * torch.softmax(score, dim=1).unsqueeze(-1), dim=1)
        pooled[none_present] = 0
        return self.head(pooled)
