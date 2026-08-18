"""DINOv3 ConvNeXt slice encoder, interface-compatible with the frozen encoder.

The working model's encoder is a ConvNeXt-Tiny adapted to the reports and then
frozen.  This module offers a drop-in alternative whose weights come from
DINOv3's self-supervised pretraining on LVD-1689M instead.

The swap is unusually clean, and that is the point of using this variant rather
than a ViT:

    output width      768, identical to the frozen encoder
    input size        224, identical
    normalisation     ImageNet mean/std, identical
    classifier head   none to strip

Because the width matches, nothing above the encoder changes: the series
pooling, the study Transformer and the pathology queries all see tensors of
exactly the shape they saw before.  The encoder is the single altered variable.

Weights are resolved through `timm`, which publishes the DINOv3 ConvNeXt
checkpoints under standard ConvNeXt architectures, so no key mapping is needed
and there is no hand-written conversion to get wrong.  They are released under
Meta's DINOv3 licence; obtaining and complying with it is the caller's
responsibility.
"""
from __future__ import annotations

import torch
from torch import nn

DINOV3_CONVNEXT_MODELS = {
    "tiny": "convnext_tiny.dinov3_lvd1689m",
    "small": "convnext_small.dinov3_lvd1689m",
    "base": "convnext_base.dinov3_lvd1689m",
    "large": "convnext_large.dinov3_lvd1689m",
}

# The frozen head is built around this width.  A wider encoder would change the
# study representation as well as the features, so it is refused rather than
# silently reshaped.
REQUIRED_OUTPUT_WIDTH = 768

# ConvNeXt tiny and small share the same channel widths and differ only in
# depth, so both emit 768-d features and both drop in unchanged.  Base (1024)
# and large (1536) do not.
DROP_IN_VARIANTS = ("tiny", "small")

DINOV3_ENCODER_VERSION = "dinov3_convnext_slice_encoder_v1"


class DinoV3SliceEncoder(nn.Module):
    """DINOv3 ConvNeXt encoder exposing the frozen encoder's interface.

    Mirrors :class:`rsna_knee.model.ConvNeXtSliceEncoder`: same constructor
    keywords, same `out_dim` attribute, same `forward` contract mapping a batch
    of slice images to one vector per slice.
    """

    def __init__(
        self,
        in_channels: int = 3,
        *,
        variant: str = "tiny",
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

        if variant not in DINOV3_CONVNEXT_MODELS:
            known = ", ".join(sorted(DINOV3_CONVNEXT_MODELS))
            raise ValueError(f"unknown DINOv3 variant {variant!r}; known: {known}")

        self.variant = variant
        self.model_name = DINOV3_CONVNEXT_MODELS[variant]
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=bool(pretrained_weights),
            num_classes=0,
            in_chans=int(in_channels),
        )
        self.out_dim = int(self.backbone.num_features)
        if self.out_dim != REQUIRED_OUTPUT_WIDTH:
            drop_in = ", ".join(sorted(DROP_IN_VARIANTS))
            raise ValueError(
                f"{self.model_name} emits {self.out_dim}-d features but the frozen "
                f"head requires {REQUIRED_OUTPUT_WIDTH}; drop-in variants: {drop_in}"
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
            "version": DINOV3_ENCODER_VERSION,
            "model_name": self.model_name,
            "variant": self.variant,
            "out_dim": self.out_dim,
            "normalize_input": self.normalize_input,
            "pretraining": "DINOv3 self-supervised, LVD-1689M",
            "licence": "Meta DINOv3 licence",
        }


def attach_dinov3_encoder(
    model: nn.Module,
    *,
    variant: str = "tiny",
    pretrained_weights: bool = True,
) -> DinoV3SliceEncoder:
    """Replace a built model's encoder with the DINOv3 one, in place.

    The head is constructed first and its encoder swapped afterwards, which is
    the same shape of operation as loading report-aligned weights into it.  The
    replacement is refused unless the widths match, because a mismatch would
    change the study representation rather than only the features.
    """
    existing = getattr(model, "encoder", None)
    if existing is None:
        raise ValueError("model has no encoder attribute to replace")

    in_channels = int(getattr(existing, "input_mean").shape[1])
    normalize_input = bool(getattr(existing, "normalize_input", True))
    replacement = DinoV3SliceEncoder(
        in_channels,
        variant=variant,
        pretrained_weights=pretrained_weights,
        normalize_input=normalize_input,
    )
    if replacement.out_dim != int(existing.out_dim):
        raise ValueError(
            f"encoder width mismatch: existing {existing.out_dim}, "
            f"replacement {replacement.out_dim}"
        )

    model.encoder = replacement
    return replacement
