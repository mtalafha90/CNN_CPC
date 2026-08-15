from __future__ import annotations

import time

from torch import nn

from .b7_weak_supervision import target_balanced_weak_bce
from .b17_training import encoder_state_sha256
from .b22_duration_protocol import (
    B22_BATCHES,
    B22_NEGATIVE_CELLS,
    B22_POSITIVE_CELLS,
    B22_TRAIN_CELLS,
    B22_TRAIN_SERIES,
    B22_TRAIN_STUDIES,
)
from .runtime import autocast


def run_b22_epoch(
    *,
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    head_params,
    target_multiplier_t,
    runtime,
    budget,
    encoder_sha_initial: str,
    clip: float,
    epoch_number: int,
) -> dict:
    if not budget.can_start(120.0):
        raise RuntimeError("budget cannot start next complete B22 epoch")
    start = time.monotonic()
    model.train()
    model.encoder.eval()
    loss_sum = 0.0
    steps = studies = cells = pos = neg = series = 0
    max_series = 0

    for batch in loader:
        if not budget.can_start(120.0):
            raise RuntimeError("partial B22 epochs are forbidden")
        volumes = batch["volumes"].to(runtime.device, non_blocking=True)
        present = batch["present"].to(runtime.device, non_blocking=True)
        series_meta = batch["series_meta"].to(runtime.device, non_blocking=True)
        target = batch["target"].to(runtime.device, non_blocking=True)
        weight = batch["weight"].to(runtime.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(runtime):
            logits = model(volumes, present, series_meta)
            loss = target_balanced_weak_bce(logits, target, weight, target_multiplier_t)
        scaler.scale(loss).backward()
        if any(p.grad is not None for p in model.encoder.parameters()):
            raise RuntimeError("B22 frozen encoder received a gradient")
        if clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(head_params, clip)
        scaler.step(optimizer)
        scaler.update()

        active = weight > 0
        loss_sum += float(loss.item())
        steps += 1
        studies += int(volumes.shape[0])
        cells += int(active.sum().item())
        pos += int((active & (target > 0.5)).sum().item())
        neg += int((active & (target < 0.5)).sum().item())
        series += int((present > 0).sum().item())
        max_series = max(max_series, int(volumes.shape[1]))

    scheduler.step()
    sha = encoder_state_sha256(model.encoder)
    if sha != encoder_sha_initial:
        raise RuntimeError("B22 frozen encoder changed")
    if (
        steps != B22_BATCHES
        or studies != B22_TRAIN_STUDIES
        or cells != B22_TRAIN_CELLS
        or pos != B22_POSITIVE_CELLS
        or neg != B22_NEGATIVE_CELLS
        or series != B22_TRAIN_SERIES
    ):
        raise RuntimeError(f"B22 epoch {epoch_number} incomplete")

    return {
        "epoch": int(epoch_number),
        "loss": loss_sum / steps,
        "head_lr": float(optimizer.param_groups[0]["lr"]),
        "batches": steps,
        "studies_seen": studies,
        "active_cells_seen": cells,
        "positive_cells_seen": pos,
        "negative_cells_seen": neg,
        "series_seen": series,
        "max_series_in_batch": max_series,
        "encoder_frozen": True,
        "encoder_sha256": sha,
        "crop_stage": "native_array_pre_resize",
        "training_seconds": float(time.monotonic() - start),
        "gold_evaluation_performed": False,
        "full_coverage": True,
    }
