"""Full-B6 fixed-E2 B21 refit for the one-look gold acceptance step."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .b7_weak_supervision import _read_config, seed_everything, target_balanced_weak_bce
from .b12_1_hierarchical import b12_1_model_spec, build_b12_1_model
from .b13_training import B13_INPUT_NORMALIZATION
from .b16_report_ssl import (
    B16_REPORT_SSL_EXPERIMENT,
    B16_REPORT_SSL_OBJECTIVE,
    B16_REPORT_SSL_VARIANT,
    load_b16_report_encoder,
)
from .b17_training import encoder_state_sha256, freeze_encoder
from .b21_acceptance_protocol import (
    B21_CROP_FRACTION,
    B21_FIXED_EPOCHS,
    B21_FULL_EXPERIMENT,
    B21_FULL_NEGATIVE_CELLS,
    B21_FULL_POSITIVE_CELLS,
    B21_FULL_TRAIN_CELLS,
    B21_FULL_TRAIN_SERIES,
    B21_FULL_VARIANT,
    B21_SCHEDULER_HORIZON,
    require_passed_weak_v2_gate,
)
from .b21_contract import require_b21_contract
from .b21_full_setup import prepare_b21_full_surface
from .budget import RuntimeBudget
from .constants import TARGETS
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime


def train_b21_full(
    config: dict,
    *,
    b6_root: str | Path,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    weak_v2_comparison: str | Path,
    out_root: str | Path = "runs/b21_full_acceptance",
) -> Path:
    validate_competition_config(config, purpose="train")
    crop_fraction = require_b21_contract(config)
    if crop_fraction != B21_CROP_FRACTION:
        raise ValueError("B21 full refit freezes crop_fraction=0.90")
    weak_gate = require_passed_weak_v2_gate(weak_v2_comparison)
    report_payload = load_b16_report_encoder(report_ssl_checkpoint)

    seed = int(config.get("seed", 2026))
    seed_everything(seed + 19_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    print(
        "[B21 full] full B6 surface | fixed E2 | historical B16 frozen | "
        "native 90% crop before resize | no gold selection | scheduler_horizon=5"
    )
    surface = prepare_b21_full_surface(
        config,
        b6_root=b6_root,
        series_policy_path=series_policy_path,
        runtime=runtime,
    )

    spec = b12_1_model_spec(config, normalize_input=True)
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.encoder.load_state_dict(report_payload["encoder"], strict=True)
    freeze_encoder(model)
    model.gradient_checkpointing = False
    encoder_sha_initial = encoder_state_sha256(model.encoder)
    model = model.to(runtime.device)

    head_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("encoder.") and p.requires_grad
    ]
    if not head_params or any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("B21 full frozen/trainable parameter contract failed")
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": float(config.get("b7_head_lr", 1e-4))}],
        weight_decay=float(config.get("b7_weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=B21_SCHEDULER_HORIZON,
        eta_min=float(config.get("b7_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    clip = float(config.get("b7_grad_clip", 1.0))
    target_multiplier_t = torch.from_numpy(surface["target_multiplier"]).to(runtime.device)
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )

    paired_gate = weak_gate["paired_candidate_minus_control"]
    policy = {
        "variant": B21_FULL_VARIANT,
        "experiment": B21_FULL_EXPERIMENT,
        "mode": "full_preresize",
        "status": "frozen full-data acceptance candidate after favorable weak-v2 gate",
        "working_model_before_gold_acceptance": "B20_crop_only_joint_focus",
        "working_model_replaced": False,
        "architecture": "historical B20 hierarchical one-token-per-series model",
        "initialization": B16_REPORT_SSL_VARIANT,
        "initialization_experiment": B16_REPORT_SSL_EXPERIMENT,
        "initialization_objective": B16_REPORT_SSL_OBJECTIVE,
        "input_normalization": B13_INPUT_NORMALIZATION,
        "encoder_frozen": True,
        "fixed_epochs": B21_FIXED_EPOCHS,
        "scheduler_horizon_epochs": B21_SCHEDULER_HORIZON,
        "scheduler_matches_historical_b20_first_two_epochs": True,
        "seed_offset_model": 19_000_000,
        "seed_offset_loader": 19_100_000,
        "epoch_selection": "none; epoch 2 predeclared",
        "gold_checkpoint_selection": False,
        "gold_labels_used_for_development": False,
        "gold_studies_used_in_gradient": 0,
        "training_studies": len(surface["uids"]),
        "training_series": B21_FULL_TRAIN_SERIES,
        "training_supervision_cells": B21_FULL_TRAIN_CELLS,
        "crop_fraction": crop_fraction,
        "crop_stage": "native_array_pre_resize",
        "normalization_support": "cropped native field before percentile normalization",
        "output_resolution": 224,
        "robust_loss": "none",
        "additional_label_smoothing": 0.0,
        "weak_v2_gate_path": str(Path(weak_v2_comparison).resolve()),
        "weak_v2_gate_raw_delta": float(paired_gate["raw_difference_b_minus_a"]),
        "weak_v2_gate_ci_lower": float(paired_gate["ci_lower"]),
        "weak_v2_gate_ci_upper": float(paired_gate["ci_upper"]),
        "weak_v2_gate_passed_before_full_refit": True,
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": surface["b6_audit"].get("b6_version"),
        "b6_policy": surface["b6_policy"],
        "series_policy": surface["series_policy"],
        "supervision": surface["supervision"],
        "metadata_repair": surface["metadata_stats"],
        "report_ssl_checkpoint": str(Path(report_ssl_checkpoint).resolve()),
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "b21_full_model.pt"
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (out / "supervision_plan.json").write_text(
        json.dumps(surface["supervision"], indent=2), encoding="utf-8"
    )

    history = []
    for epoch in range(B21_FIXED_EPOCHS):
        if not budget.can_start(120.0):
            raise RuntimeError("budget cannot start next complete B21 full epoch")
        start = time.monotonic()
        model.train()
        model.encoder.eval()
        loss_sum = 0.0
        steps = studies = cells = pos = neg = series = 0
        max_series = 0
        for batch in surface["loader"]:
            if not budget.can_start(120.0):
                raise RuntimeError("partial B21 full epochs are forbidden")
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
                raise RuntimeError("B21 full encoder received a gradient")
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
            raise RuntimeError("B21 full frozen encoder changed")
        if (
            steps != 1560 or studies != 3120 or cells != B21_FULL_TRAIN_CELLS
            or pos != B21_FULL_POSITIVE_CELLS or neg != B21_FULL_NEGATIVE_CELLS
            or series != B21_FULL_TRAIN_SERIES
        ):
            raise RuntimeError(f"B21 full epoch {epoch + 1} incomplete")
        row = {
            "epoch": epoch + 1,
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
        history.append(row)
        print(row)

    encoder_sha_final = encoder_state_sha256(model.encoder)
    if encoder_sha_final != encoder_sha_initial:
        raise RuntimeError("B21 full encoder changed before save")
    payload = {
        **policy,
        "model_state": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "model_spec": spec,
        "config": config,
        "completed_epochs": len(history),
        "model_epoch": B21_FIXED_EPOCHS,
        "history": history,
        "encoder_sha256_initial": encoder_sha_initial,
        "encoder_sha256_final": encoder_sha_final,
        "budget": budget.to_dict(),
    }
    torch.save(payload, checkpoint_path)
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print({"checkpoint": str(checkpoint_path), "fixed_epoch": 2, "gold_evaluation_performed": False})
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b21-full")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--report-ssl-checkpoint", required=True)
    parser.add_argument("--weak-v2-comparison", required=True)
    parser.add_argument("--out-root", default="runs/b21_full_acceptance")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = train_b21_full(
        config,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        weak_v2_comparison=args.weak_v2_comparison,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
