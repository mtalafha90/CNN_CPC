from __future__ import annotations

import torch

from .b12_1_hierarchical import b12_1_model_spec, build_b12_1_model
from .b17_training import encoder_state_sha256, freeze_encoder
from .b22_duration_protocol import B22_SCHEDULER_HORIZON
from .runtime import make_scaler


def build_b22_training_state(config: dict, report_payload: dict, runtime):
    spec = b12_1_model_spec(config, normalize_input=True)
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.encoder.load_state_dict(report_payload["encoder"], strict=True)
    freeze_encoder(model)
    model.gradient_checkpointing = False
    encoder_sha = encoder_state_sha256(model.encoder)
    model = model.to(runtime.device)

    head_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("encoder.") and p.requires_grad
    ]
    if not head_params or any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("B22 frozen/trainable parameter contract failed")

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": float(config.get("b7_head_lr", 1e-4))}],
        weight_decay=float(config.get("b7_weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=B22_SCHEDULER_HORIZON,
        eta_min=float(config.get("b7_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    return model, spec, head_params, optimizer, scheduler, scaler, encoder_sha
