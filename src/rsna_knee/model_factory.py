from __future__ import annotations

from .model import MultiSeriesKneeNet
from .model3d import Small3DKneeNet


def build_model(config: dict, n_streams: int, in_channels: int, *, pretrained: bool | None = None):
    """Build the configured study-level model.

    ``model_type=mil2d`` is the standard slice/2.5D MIL path. ``model_type=3d``
    is the compact volumetric complementary arm and requires grayscale volume
    input (``input_mode=2d`` in the dataset configuration).
    """
    model_type = str(config.get("model_type", "mil2d")).lower()
    pretr = bool(config.get("pretrained", False) if pretrained is None else pretrained)

    if model_type in {"mil2d", "2d", "2p5d"}:
        return MultiSeriesKneeNet(
            n_streams,
            pretrained=pretr,
            dropout=float(config.get("dropout", 0.25)),
            in_channels=in_channels,
            backbone=str(config.get("backbone", "resnet18")),
            target_attention=bool(config.get("target_attention", False)),
            slice_pooling=str(config.get("slice_pooling", "attention")),
            topk_fraction=float(config.get("topk_fraction", 0.25)),
        )

    if model_type == "3d":
        if str(config.get("input_mode", "2d")) != "2d" or in_channels != 1:
            raise ValueError("model_type=3d requires input_mode=2d (grayscale MRI volumes)")
        return Small3DKneeNet(
            n_streams=n_streams,
            dropout=float(config.get("dropout", 0.25)),
            base_channels=int(config.get("base_channels_3d", 16)),
            target_attention=bool(config.get("target_attention", True)),
        )

    raise ValueError(f"unsupported model_type: {model_type}")
