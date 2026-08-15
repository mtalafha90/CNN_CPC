from __future__ import annotations

from pathlib import Path

import torch

from .b12_1_hierarchical import build_b12_1_model
from .b22_duration_protocol import (
    B22_CROP_FRACTION,
    B22_TRAIN_CELLS,
    B22_TRAIN_SERIES,
    B22_TRAIN_STUDIES,
    B22_VARIANT,
)


def load_b22_epoch(path: str | Path, expected_epoch: int, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("variant") != B22_VARIANT:
        raise ValueError("not a B22 duration-audit checkpoint")
    if int(payload.get("model_epoch", -1)) != int(expected_epoch):
        raise ValueError("B22 checkpoint epoch mismatch")
    if int(payload.get("training_studies", -1)) != B22_TRAIN_STUDIES:
        raise ValueError("B22 training-study count changed")
    if int(payload.get("training_series", -1)) != B22_TRAIN_SERIES:
        raise ValueError("B22 training-series count changed")
    if int(payload.get("training_supervision_cells", -1)) != B22_TRAIN_CELLS:
        raise ValueError("B22 supervision-cell count changed")
    if float(payload.get("crop_fraction", -1.0)) != B22_CROP_FRACTION:
        raise ValueError("B22 crop fraction changed")
    if payload.get("crop_stage") != "native_array_pre_resize":
        raise ValueError("B22 crop stage changed")
    if payload.get("gold_evaluation_during_training") is not False:
        raise ValueError("B22 checkpoint does not certify zero gold use during training")
    model = build_b12_1_model(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload
