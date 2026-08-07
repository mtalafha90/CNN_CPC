"""The exam-level model.

Architecture, and why it is shaped this way:

1. **A 2D backbone over slices (a "2.5D" model), not a 3D CNN.** Knee MRI
   series are anisotropic — roughly 0.3 mm in plane against 3-4 mm between
   slices — so a symmetric 3D kernel wastes capacity on an axis with almost no
   resolution. Feeding three neighbouring slices as the three input channels
   gives the network local through-plane context while keeping ImageNet
   pre-trained weights, which matters enormously at ~5,000 exams.

2. **A transformer over the slice axis.** A tear appears on a handful of
   slices. Mean pooling dilutes that evidence; attention pooling learns to find
   it. The transformer also sees slice order, so it can use the fact that a
   finding persists across adjacent slices.

3. **Attention fusion across series.** Each exam holds several series in
   different planes and weightings, and different findings live in different
   ones — a meniscal tear reads best on sagittal proton density, bone marrow
   oedema on a fat-saturated sequence. A learned series-type embedding plus
   attention pooling lets the model weight them per finding rather than
   assuming a fixed protocol, which is essential when sixteen sites use
   sixteen different protocols.

4. **Per-series auxiliary heads.** Supervising each series directly gives a
   much shorter gradient path than back-propagating only through the fused
   representation, and it stabilises early training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


@dataclass
class ModelConfig:
    backbone: str = "convnext_tiny"
    pretrained: bool = True
    num_labels: int = 12
    num_series_types: int = 32
    embed_dim: int = 512
    slice_layers: int = 2
    slice_heads: int = 8
    dropout: float = 0.1
    drop_path: float = 0.1
    max_slices: int = 24
    max_series: int = 6
    grad_checkpoint: bool = False
    in_channels: int = 3
    channels_last: bool = False


class AttentionPool(nn.Module):
    """Gated attention pooling (Ilse et al.), with masking for ragged inputs."""

    def __init__(self, dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or dim // 2
        self.attention_v = nn.Linear(dim, hidden)
        self.attention_u = nn.Linear(dim, hidden)
        self.attention_w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, N, C], mask: [B, N] with 1 for valid entries.
        scores = self.attention_w(torch.tanh(self.attention_v(x)) * torch.sigmoid(self.attention_u(x)))
        scores = scores.squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask < 0.5, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class SliceTransformer(nn.Module):
    """Contextualise slice features along the through-plane axis."""

    def __init__(self, dim: int, layers: int, heads: int, dropout: float, max_slices: int) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, max_slices, dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Nested tensors do not apply here: every series is padded to the same
        # slice count, and the fast path is unavailable with norm_first anyway.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.pool = AttentionPool(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D, C]
        length = x.shape[1]
        if length <= self.position.shape[1]:
            x = x + self.position[:, :length]
        else:
            # Interpolate the positional table when a series is longer than planned.
            position = F.interpolate(
                self.position.transpose(1, 2), size=length, mode="linear", align_corners=False
            ).transpose(1, 2)
            x = x + position
        x = self.encoder(x)
        return self.pool(x)


class KneeExamModel(nn.Module):
    """Multi-series, multi-label knee MRI classifier.

    Input is ``[B, S, D, C, H, W]``: a batch of exams, each holding up to ``S``
    series of ``D`` slices. ``series_type`` carries an integer code per series
    (plane and weighting), and ``series_mask`` marks which series are real, so
    exams with fewer series need no special handling.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone, feature_dim = _build_backbone(config)

        self.project = (
            nn.Identity() if feature_dim == config.embed_dim else nn.Linear(feature_dim, config.embed_dim)
        )
        self.slice_transformer = SliceTransformer(
            config.embed_dim,
            config.slice_layers,
            config.slice_heads,
            config.dropout,
            config.max_slices,
        )
        self.series_embedding = nn.Embedding(config.num_series_types, config.embed_dim)
        nn.init.trunc_normal_(self.series_embedding.weight, std=0.02)

        self.series_norm = nn.LayerNorm(config.embed_dim)
        self.series_pool = AttentionPool(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.embed_dim, config.num_labels)
        # Auxiliary head applied to each series on its own.
        self.series_head = nn.Linear(config.embed_dim, config.num_labels)

    def encode_slices(self, pixels: torch.Tensor) -> torch.Tensor:
        """Run the backbone over every slice of every series."""
        batch, series, depth, channels, height, width = pixels.shape
        flat = pixels.reshape(batch * series * depth, channels, height, width)
        if self.config.channels_last:
            # The memory format only applies once the tensor is four-dimensional,
            # so it is set here rather than on the six-dimensional input.
            flat = flat.contiguous(memory_format=torch.channels_last)
        if self.config.grad_checkpoint and self.training:
            features = torch.utils.checkpoint.checkpoint(
                self.backbone, flat, use_reentrant=False
            )
        else:
            features = self.backbone(flat)
        features = self.project(features)
        return features.reshape(batch * series, depth, -1)

    def forward(
        self,
        pixels: torch.Tensor,
        series_type: torch.Tensor,
        series_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, series = pixels.shape[0], pixels.shape[1]
        if series_mask is None:
            series_mask = torch.ones(batch, series, device=pixels.device)

        slice_features = self.encode_slices(pixels)          # [B*S, D, C]
        series_features = self.slice_transformer(slice_features)  # [B*S, C]
        series_features = series_features.reshape(batch, series, -1)

        series_features = series_features + self.series_embedding(series_type.clamp(min=0))
        series_features = self.series_norm(series_features)

        exam_features = self.series_pool(series_features, series_mask)
        logits = self.head(self.dropout(exam_features))
        series_logits = self.series_head(series_features)

        return {
            "logits": logits,
            "series_logits": series_logits,
            "series_mask": series_mask,
            "features": exam_features,
        }


def _build_backbone(config: ModelConfig) -> tuple[nn.Module, int]:
    """Create a timm backbone that outputs a pooled feature vector per slice."""
    import timm

    backbone = timm.create_model(
        config.backbone,
        pretrained=config.pretrained,
        num_classes=0,
        global_pool="avg",
        in_chans=config.in_channels,
        drop_path_rate=config.drop_path,
    )
    feature_dim = backbone.num_features
    return backbone, feature_dim


class ModelEma:
    """Exponential moving average of the weights.

    On datasets this size the raw weights bounce between epochs; the averaged
    copy is consistently worth a little AUC and costs one extra model in memory.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        import copy

        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_value, model_value in zip(
            self.module.state_dict().values(), model.state_dict().values()
        ):
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value.detach(), alpha=1 - self.decay)
            else:
                ema_value.copy_(model_value)
