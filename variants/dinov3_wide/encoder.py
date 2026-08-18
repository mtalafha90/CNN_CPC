"""DINOv3 ConvNeXt encoder without the 768-d restriction.

The supported encoder in `rsna_knee.dinov3_encoder` refuses anything other than
768-d, because the frozen working model's head is built around that width and a
mismatch there would silently change more than the representation.

This variant lifts that restriction, for the case where the head is rebuilt to
match the encoder rather than the encoder being forced to match the head. It is
otherwise the same encoder.
"""
from __future__ import annotations

import torch
from torch import nn

DINOV3_MODELS = {
    "tiny": ("convnext_tiny.dinov3_lvd1689m", 768),
    "small": ("convnext_small.dinov3_lvd1689m", 768),
    "base": ("convnext_base.dinov3_lvd1689m", 1024),
    "large": ("convnext_large.dinov3_lvd1689m", 1536),
}


class WideDinoV3SliceEncoder(nn.Module):
    """DINOv3 ConvNeXt slice encoder of any published width."""

    def __init__(
        self,
        in_channels: int = 3,
        *,
        variant: str = "base",
        pretrained_weights: bool = True,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "the DINOv3 encoder needs `timm`; install it with `pip install timm`"
            ) from exc

        if variant not in DINOV3_MODELS:
            known = ", ".join(sorted(DINOV3_MODELS))
            raise ValueError(f"unknown DINOv3 variant {variant!r}; known: {known}")

        self.variant = variant
        self.model_name, expected_width = DINOV3_MODELS[variant]
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=bool(pretrained_weights),
            num_classes=0,
            in_chans=int(in_channels),
        )
        self.out_dim = int(self.backbone.num_features)
        if self.out_dim != expected_width:
            raise RuntimeError(
                f"{self.model_name} emitted {self.out_dim}-d features, expected "
                f"{expected_width}; the published checkpoint may have changed"
            )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self._normalize(x))

    def describe(self) -> dict:
        return {
            "model_name": self.model_name,
            "variant": self.variant,
            "out_dim": self.out_dim,
            "normalize_input": self.normalize_input,
            "pretraining": "DINOv3 self-supervised, LVD-1689M",
            "licence": "Meta DINOv3 licence",
        }
