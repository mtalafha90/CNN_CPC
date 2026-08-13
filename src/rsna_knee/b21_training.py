"""Matched fixed-E2 training for B20-v2 control and B21 pre-resize crop."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .b7_weak_supervision import (
    _read_config,
    load_frozen_b6_export,
    make_b7_dataset_config,
    seed_everything,
    target_balance_multipliers,
    target_balanced_weak_bce,
)
from .b12_training import _load_series_policy
from .b12_variable_series import audit_variable_series_surface, collate_variable_series
from .b12_1_hierarchical import b12_1_model_spec, build_b12_1_model
from .b13_training import B13_INPUT_NORMALIZATION, B13_SERIES_SIGNATURE
from .b15_downstream import _subset_v2_supervision
from .b15_ssl import WEAK_V2_MANIFEST_SHA256, WEAK_V2_SURFACE, load_frozen_v2_manifest
from .b16_report_ssl import (
    B16_REPORT_SSL_EXPERIMENT,
    B16_REPORT_SSL_OBJECTIVE,
    B16_REPORT_SSL_VARIANT,
    load_b16_report_encoder,
)
from .b17_training import encoder_state_sha256, freeze_encoder
from .b21_contract import require_b21_contract
from .b21_dataset import make_matched_crop_dataset
from .b21_protocol import (
    B21_EXPECTED_BATCHES,
    B21_FIXED_EPOCHS,
    B21_SCHEDULER_HORIZON,
    B21_WEAK_HOLDOUT_STUDIES,
    B21_WEAK_TRAIN_STUDIES,
    mode_identity,
)
from .budget import RuntimeBudget
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime


def train_matched_crop(
    config: dict,
    *,
    mode: str,
    b6_root: str | Path,
    series_policy_path: str | Path,
    weak_holdout_root: str | Path,
    report_ssl_checkpoint: str | Path,
    out_root: str | Path,
) -> Path:
    validate_competition_config(config, purpose="train")
    crop_fraction = require_b21_contract(config)
    variant, experiment, crop_stage = mode_identity(mode)
    weak_payload, manifest = load_frozen_v2_manifest(weak_holdout_root)
    report_payload = load_b16_report_encoder(report_ssl_checkpoint)

    seed = int(config.get("seed", 2026))
    # Reuse B20's construction and DataLoader seed offsets. The candidate and
    # matched control therefore share the historical B20 initialization path.
    seed_everything(seed + 19_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    print(
        f"[{experiment}] weak-v2 train only | fixed epoch 2 | encoder frozen | "
        f"crop_stage={crop_stage} | gold development disabled | "
        f"scheduler_horizon={B21_SCHEDULER_HORIZON}"
    )

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    all_uids, train_uids, targets, weights, supervision = _subset_v2_supervision(
        train, b6_frame, manifest
    )
    holdout_uids = set(
        manifest.loc[manifest["split"] == "holdout", "StudyInstanceUID"].astype(str)
    )
    if len(train_uids) != B21_WEAK_TRAIN_STUDIES or len(holdout_uids) != B21_WEAK_HOLDOUT_STUDIES:
        raise ValueError("weak-v2 study counts changed")
    if set(train_uids).intersection(holdout_uids):
        raise RuntimeError("weak-v2 holdout leakage into training")
    gold_uids = set(train.loc[gold_mask(train), "StudyInstanceUID"].astype(str))
    if set(train_uids).intersection(gold_uids):
        raise RuntimeError("gold study leaked into matched weak-v2 training")

    series_policy = _load_series_policy(series_policy_path)
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    full_summary, full_index = audit_variable_series_surface(series, all_uids)
    if full_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("full all-series SHA changed")
    if int(full_summary.get("eligible_recognized_plane_series", -1)) != 17475:
        raise ValueError("expected the frozen 17,475-series all-active surface")
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("series policy is not the frozen B12/B13 policy")
    variable_index = {uid: full_index[uid] for uid in train_uids}
    if any(not variable_index[uid] for uid in train_uids):
        raise ValueError("weak-v2 training study has zero eligible series")
    expected_series = int(sum(len(variable_index[uid]) for uid in train_uids))

    batch_size = int(config.get("b7_batch_size", 2))
    expected_batches = int(math.ceil(len(train_uids) / batch_size))
    if expected_batches != B21_EXPECTED_BATCHES:
        raise ValueError("weak-v2 batch count changed")
    target_multiplier = target_balance_multipliers(weights)
    train_ds = make_matched_crop_dataset(
        mode,
        train_uids,
        variable_index,
        make_b7_dataset_config(config, root, train=True),
        crop_fraction=crop_fraction,
        targets=targets,
        weights=weights,
        train=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=seed + 19_100_000),
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
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("encoder is not frozen")
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": float(config.get("b7_head_lr", 1e-4))}],
        weight_decay=float(config.get("b7_weight_decay", 1e-4)),
    )
    # Stop after E2, but preserve B20's five-epoch cosine horizon. This keeps
    # the first two scheduled learning rates identical to historical B20.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=B21_SCHEDULER_HORIZON,
        eta_min=float(config.get("b7_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    clip = float(config.get("b7_grad_clip", 1.0))
    target_multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )

    expected_cells = int((weights > 0).sum())
    expected_pos = int(((weights > 0) & (targets > 0.5)).sum())
    expected_neg = int(((weights > 0) & (targets < 0.5)).sum())
    supervision = dict(supervision)
    supervision.update(
        {
            "training_studies": len(train_uids),
            "holdout_studies_excluded": len(holdout_uids),
            "training_cells": expected_cells,
            "training_positive_cells": expected_pos,
            "training_negative_cells": expected_neg,
            "training_series": expected_series,
            "batches_per_epoch": expected_batches,
        }
    )
    policy = {
        "variant": variant,
        "experiment": experiment,
        "mode": mode,
        "working_model_remains": "B20_crop_only_joint_focus",
        "working_model_replaced": False,
        "architecture": "B20 hierarchical one-token-per-series model",
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
        "weak_holdout_surface": WEAK_V2_SURFACE,
        "weak_holdout_manifest_sha256": WEAK_V2_MANIFEST_SHA256,
        "weak_train_studies": len(train_uids),
        "weak_holdout_studies": len(holdout_uids),
        "weak_holdout_studies_used_in_gradient": 0,
        "crop_fraction": crop_fraction,
        "crop_stage": crop_stage,
        "output_resolution": 224,
        "additional_label_smoothing": 0.0,
        "robust_loss": "none",
        "single_change_between_matched_arms": (
            "same centered 90% crop applied after the 224 resize in control versus "
            "on the native array before the 224 resize in B21"
        ),
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "b6_policy": b6_policy,
        "series_policy": series_policy,
        "supervision": supervision,
        "metadata_repair": metadata_stats,
        "report_ssl_checkpoint": str(Path(report_ssl_checkpoint).resolve()),
        "weak_holdout_metadata": weak_payload,
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / ("b20_v2_control.pt" if mode == "control" else "b21_model.pt")
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (out / "supervision_plan.json").write_text(json.dumps(supervision, indent=2), encoding="utf-8")

    history = []
    for epoch in range(B21_FIXED_EPOCHS):
        if not budget.can_start(120.0):
            raise RuntimeError("runtime budget cannot start the next complete fixed epoch")
        start = time.monotonic()
        model.train()
        model.encoder.eval()
        loss_sum = 0.0
        steps = studies_seen = cells_seen = pos_seen = neg_seen = series_seen = 0
        max_series = 0
        for batch in train_loader:
            if not budget.can_start(120.0):
                raise RuntimeError("partial fixed epochs are forbidden")
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
                raise RuntimeError("frozen encoder received a gradient")
            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(head_params, clip)
            scaler.step(optimizer)
            scaler.update()

            active = weight > 0
            loss_sum += float(loss.item())
            steps += 1
            studies_seen += int(volumes.shape[0])
            cells_seen += int(active.sum().item())
            pos_seen += int((active & (target > 0.5)).sum().item())
            neg_seen += int((active & (target < 0.5)).sum().item())
            series_seen += int((present > 0).sum().item())
            max_series = max(max_series, int(volumes.shape[1]))

        scheduler.step()
        encoder_sha_epoch = encoder_state_sha256(model.encoder)
        if encoder_sha_epoch != encoder_sha_initial:
            raise RuntimeError("frozen encoder state changed")
        if (
            steps != expected_batches
            or studies_seen != len(train_uids)
            or cells_seen != expected_cells
            or pos_seen != expected_pos
            or neg_seen != expected_neg
            or series_seen != expected_series
        ):
            raise RuntimeError(f"epoch {epoch + 1} did not cover the exact matched surface")
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / steps,
            "head_lr": float(optimizer.param_groups[0]["lr"]),
            "batches": steps,
            "studies_seen": studies_seen,
            "active_cells_seen": cells_seen,
            "positive_cells_seen": pos_seen,
            "negative_cells_seen": neg_seen,
            "series_seen": series_seen,
            "max_series_in_batch": max_series,
            "encoder_frozen": True,
            "encoder_sha256": encoder_sha_epoch,
            "crop_stage": crop_stage,
            "training_seconds": float(time.monotonic() - start),
            "development_evaluation_performed": False,
        }
        history.append(row)
        print(row)

    encoder_sha_final = encoder_state_sha256(model.encoder)
    payload = {
        **policy,
        "model_state": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "model_spec": spec,
        "config": config,
        "completed_epochs": len(history),
        "history": history,
        "encoder_sha256_initial": encoder_sha_initial,
        "encoder_sha256_final": encoder_sha_final,
    }
    if encoder_sha_final != encoder_sha_initial:
        raise RuntimeError("encoder changed before checkpoint save")
    torch.save(payload, checkpoint_path)
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print({"checkpoint": str(checkpoint_path), "mode": mode, "fixed_epoch": 2})
    return checkpoint_path


def _run(mode: str) -> None:
    parser = argparse.ArgumentParser(
        "rsna-knee-b20-v2-control" if mode == "control" else "rsna-knee-b21"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--weak-holdout-root", required=True)
    parser.add_argument("--report-ssl-checkpoint", required=True)
    parser.add_argument(
        "--out-root",
        default="runs/b20_v2_control" if mode == "control" else "runs/b21_preresize_crop",
    )
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = train_matched_crop(
        config,
        mode=mode,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        weak_holdout_root=args.weak_holdout_root,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root,
    )
    print(path)


def main_control() -> None:
    _run("control")


def main() -> None:
    _run("preresize")


if __name__ == "__main__":
    main()
