from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .constants import DUAL_STREAMS, N_TARGETS, TARGETS
from .model import ConvNeXtSliceEncoder


# Soft, predeclared anatomical/sequence priors. These are intentionally coarse
# (preferred/secondary/background), not fitted from any OOF outcome. They bias
# attention without excluding any stream, and content evidence can override
# them study by study.
_PRIOR_STRENGTHS = {
    "ACL": [3, 3, 1, 1, 1, 1],
    "MCL": [1, 1, 3, 3, 1, 1],
    "Medial Meniscus": [3, 3, 3, 3, 1, 1],
    "Lateral Meniscus": [3, 3, 3, 3, 1, 1],
    "Medial OA": [1, 2, 2, 4, 1, 2],
    "Lateral OA": [1, 2, 2, 4, 1, 2],
    "PF OA": [1, 1, 1, 1, 4, 4],
    "Effusion": [3, 1, 3, 1, 4, 1],
    "Synovitis": [3, 1, 3, 1, 4, 1],
    "Baker's": [4, 1, 2, 1, 3, 1],
    "Contusion": [3, 1, 4, 1, 4, 1],
    "Fracture": [2, 3, 2, 3, 2, 3],
}


def default_target_stream_priors() -> torch.Tensor:
    """Return a positive row-normalized ``[target, stream]`` prior matrix."""
    if len(DUAL_STREAMS) != 6:
        raise ValueError("pathology-aware priors require the six-stream dual contract")
    rows = []
    for target in TARGETS:
        strengths = torch.tensor(_PRIOR_STRENGTHS[target], dtype=torch.float32)
        rows.append(strengths / strengths.sum())
    priors = torch.stack(rows)
    if priors.shape != (N_TARGETS, len(DUAL_STREAMS)):
        raise RuntimeError("target/stream prior shape mismatch")
    return priors


class PathologyAwareMILNet(nn.Module):
    """Low-capacity pathology-aware MIL over SSL slice features.

    The model removes the global MRI Transformer and pathology-to-pathology
    Transformer used by ``KneeMILNet``. Each target instead has one query that
    performs two transparent pooling operations:

    1. attention over sampled 2.5D positions inside each stream;
    2. attention over the six streams using a soft fixed anatomical prior plus
       a small learned residual and content-dependent evidence.

    No stream is hard-disabled. Missing streams are masked dynamically.
    """

    architecture = "pathology_aware_stream_mil_v1"

    def __init__(
        self,
        n_streams: int,
        n_slices: int,
        *,
        in_channels: int = 3,
        pretrained_weights: bool = False,
        normalize_input: bool = False,
        dropout: float = 0.25,
        encoder_batch_size: int = 24,
        gradient_checkpointing: bool = True,
        prior_strength: float = 1.0,
        prior_residual_scale: float = 0.50,
    ) -> None:
        super().__init__()
        if int(n_streams) != len(DUAL_STREAMS):
            raise ValueError("PathologyAwareMILNet requires the six-stream dual contract")
        if n_slices < 1 or encoder_batch_size < 1:
            raise ValueError("n_slices and encoder_batch_size must be positive")
        if prior_strength < 0 or prior_residual_scale < 0:
            raise ValueError("prior strengths must be non-negative")

        self.n_streams = int(n_streams)
        self.n_slices = int(n_slices)
        self.in_channels = int(in_channels)
        self.encoder_batch_size = int(encoder_batch_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.prior_strength = float(prior_strength)
        self.prior_residual_scale = float(prior_residual_scale)

        self.encoder = ConvNeXtSliceEncoder(
            in_channels,
            pretrained_weights=pretrained_weights,
            normalize_input=normalize_input,
        )
        d = int(self.encoder.out_dim)
        self.feature_dim = d
        self.slice_position = nn.Parameter(torch.randn(n_slices, d) * 0.02)
        self.target_query = nn.Parameter(torch.randn(N_TARGETS, d) * 0.02)
        self.target_weight = nn.Parameter(torch.empty(N_TARGETS, d))
        self.target_bias = nn.Parameter(torch.zeros(N_TARGETS))
        nn.init.xavier_uniform_(self.target_weight)

        priors = default_target_stream_priors()
        self.register_buffer("target_stream_prior", priors, persistent=True)
        self.stream_residual = nn.Parameter(torch.zeros(N_TARGETS, n_streams))
        self.summary_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(float(dropout))
        self.score_scale = 1.0 / math.sqrt(float(d))

    def _encode_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self.encoder, chunk, use_reentrant=False)
        return self.encoder(chunk)

    def _reshape(self, volumes: torch.Tensor):
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

    def _encode_slices(self, volumes: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        stream_volumes, b, k, s = self._reshape(volumes)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        d = self.feature_dim
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
        return (features + self.slice_position[None, None, :, :]) * active_mask

    def _attention(self, features: torch.Tensor, present: torch.Tensor):
        query = nn.functional.normalize(self.target_query, dim=-1)

        # [B,T,K,S]: pathology-specific attention over sampled positions.
        slice_scores = torch.einsum("bksd,td->btks", features, query) * self.score_scale
        slice_attention = torch.softmax(slice_scores, dim=-1)
        stream_features = torch.einsum("btks,bksd->btkd", slice_attention, features)

        # Soft prior + bounded learned residual + content evidence.
        prior_logits = self.prior_strength * torch.log(self.target_stream_prior.clamp_min(1e-6))
        residual = self.prior_residual_scale * torch.tanh(self.stream_residual)
        content = torch.einsum("btkd,td->btk", stream_features, query) * self.score_scale
        stream_scores = content + prior_logits[None, :, :] + residual[None, :, :]

        missing = present[:, None, :] <= 0
        empty = (present <= 0).all(dim=1)
        safe_missing = missing.expand(-1, N_TARGETS, -1).clone()
        if empty.any():
            safe_missing[empty, :, 0] = False
        stream_scores = stream_scores.masked_fill(safe_missing, -1e4)
        stream_attention = torch.softmax(stream_scores, dim=-1)
        summary = torch.einsum("btk,btkd->btd", stream_attention, stream_features)
        return summary, slice_attention, stream_attention, empty

    def forward(self, volumes: torch.Tensor, present: torch.Tensor, *, return_attention: bool = False):
        if present.ndim != 2 or present.shape[1] != self.n_streams:
            raise ValueError("present mask does not match stream contract")
        features = self._encode_slices(volumes, present)
        summary, slice_attention, stream_attention, empty = self._attention(features, present)
        summary = self.dropout(self.summary_norm(summary))
        logits = (summary * self.target_weight[None, :, :]).sum(dim=-1) + self.target_bias
        logits = torch.where(empty[:, None], self.target_bias[None, :], logits)
        if return_attention:
            return logits, {
                "slice_attention": slice_attention,
                "stream_attention": stream_attention,
                "target_stream_prior": self.target_stream_prior,
            }
        return logits
