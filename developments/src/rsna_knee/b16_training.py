"""B16 downstream training: B13 hierarchy initialized from B16 full-report alignment."""
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
    B7_MIN_CONFIDENCE,
    B7_NEGATIVE_TARGET,
    B7_NEGATIVE_WEIGHT,
    B7_POSITIVE_TARGET,
    B7_POSITIVE_WEIGHT,
    _read_config,
    load_frozen_b6_export,
    make_b7_dataset_config,
    prepare_b7_supervision,
    seed_everything,
    target_balance_multipliers,
    target_balanced_weak_bce,
)
from .b12_training import _load_series_policy
from .b12_variable_series import (
    VariableSeriesKneeDataset,
    audit_variable_series_surface,
    collate_variable_series,
)
from .b12_1_hierarchical import b12_1_model_spec, build_b12_1_model
from .b13_training import B13_INPUT_NORMALIZATION, B13_SERIES_SIGNATURE, _require_b13_contract
from .b16_report_ssl import (
    B16_REPORT_SSL_EXPERIMENT,
    B16_REPORT_SSL_OBJECTIVE,
    B16_REPORT_SSL_VARIANT,
    load_b16_report_encoder,
)
from .budget import RuntimeBudget
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime

B16_VARIANT = "b16_b15_report_ssl_b6_hierarchical_full_surface_v1"
B16_EXPERIMENT = "B16_full_report_semantic_downstream"


def train_b16(
    config: dict,
    *,
    b6_root: str | Path,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    out_root: str | Path = "runs/b16_full_report/downstream",
) -> Path:
    validate_competition_config(config, purpose="train")
    _require_b13_contract(config)
    if int(config.get("b7_max_batches_per_epoch", 1560)) != 1560:
        raise ValueError("B16 downstream freezes the full 3,120-study surface: 1,560 batches/epoch")

    report_payload = load_b16_report_encoder(report_ssl_checkpoint)
    seed = int(config.get("seed", 2026))
    seed_everything(seed + 19_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    print(f"[B16] initialization={B16_REPORT_SSL_VARIANT}")

    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    uids, targets, weights, supervision = prepare_b7_supervision(train, b6_frame)
    positive_cells = int(((weights > 0) & (targets > 0.5)).sum())
    negative_cells = int(((weights > 0) & (targets < 0.5)).sum())
    if len(uids) != 3120 or int((weights > 0).sum()) != 14123 or positive_cells != 6871 or negative_cells != 7252:
        raise ValueError("B16 must retain the exact full B13 B6 training surface")

    series_policy = _load_series_policy(series_policy_path)
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    frozen_summary = series_policy.get("series_summary", {})
    if series_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B16 reconstructed series surface does not match frozen B13 SHA-256")
    if frozen_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B16 supplied series policy is not the frozen B12/B13 policy")
    if int(series_summary.get("eligible_recognized_plane_series", -1)) != 17475:
        raise ValueError("B16 requires the frozen 17,475-series full surface")
    if series_summary.get("viability_passed") is not True:
        raise ValueError("B16 reconstructed series surface no longer passes viability")

    target_multiplier = target_balance_multipliers(weights)
    batch_size = int(config.get("b7_batch_size", 2))
    batches_per_epoch = int(math.ceil(len(uids) / batch_size))
    if batches_per_epoch != 1560:
        raise ValueError("B16 full B6 surface must produce 1,560 batches/epoch")
    expected_series = 17475
    supervision.update({
        "training_studies": len(uids),
        "training_cells": int((weights > 0).sum()),
        "training_positive_cells": positive_cells,
        "training_negative_cells": negative_cells,
        "eligible_series_expected_per_full_epoch": expected_series,
        "full_coverage_batches_per_epoch": batches_per_epoch,
        "series_signature_sha256": series_summary["series_signature_sha256"],
    })

    ds = VariableSeriesKneeDataset(
        uids,
        variable_index,
        make_b7_dataset_config(config, root, train=True),
        targets=targets,
        weights=weights,
        train=True,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=seed + 19_100_000),
    )

    spec = b12_1_model_spec(config, normalize_input=True)
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.encoder.load_state_dict(report_payload["encoder"], strict=True)
    model = model.to(runtime.device)

    encoder_params = list(model.encoder.parameters())
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(config.get("b7_encoder_lr", 1e-5))},
            {"params": head_params, "lr": float(config.get("b7_head_lr", 1e-4))},
        ],
        weight_decay=float(config.get("b7_weight_decay", 1e-4)),
    )
    epochs = int(config.get("b7_epochs", 4))
    if epochs != 4:
        raise ValueError("B16 downstream freezes four epochs")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(config.get("b7_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    clip = float(config.get("b7_grad_clip", 1.0))
    target_multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "b16_model.pt"
    policy = {
        "variant": B16_VARIANT,
        "experiment": B16_EXPERIMENT,
        "architecture": "B13 hierarchical learned one-token-per-series aggregation",
        "initialization": B16_REPORT_SSL_VARIANT,
        "initialization_experiment": B16_REPORT_SSL_EXPERIMENT,
        "initialization_objective": B16_REPORT_SSL_OBJECTIVE,
        "initialization_detail": report_payload.get("initialization_detail"),
        "input_normalization": B13_INPUT_NORMALIZATION,
        "external_pretrained": True,
        "full_report_alignment": True,
        "weak_v2_used_for_selection": False,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_for_early_stopping": False,
        "training_studies": len(uids),
        "training_series": expected_series,
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "b6_policy": b6_policy,
        "series_policy": series_policy,
        "supervision": supervision,
        "target_balance_multiplier": {target: float(target_multiplier[j]) for j, target in enumerate(TARGETS)},
        "model_spec": spec,
        "metadata_repair": metadata_stats,
        "report_ssl_checkpoint": str(Path(report_ssl_checkpoint).resolve()),
        "scientific_change_vs_b13": (
            "B13 direct ImageNet initialization is replaced by ImageNet -> completed B15 knee-MRI SSL "
            "-> B16 full-report semantic alignment; the downstream B13 hierarchy, full 3,120-study "
            "B6 surface, optimizer, augmentation, epochs and TTA remain fixed"
        ),
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (out / "supervision_plan.json").write_text(json.dumps(supervision, indent=2), encoding="utf-8")

    history: list[dict] = []
    epoch_times: list[float] = []
    budget_exhausted = False
    for epoch in range(epochs):
        if epoch_times and not budget.can_start(float(np.median(epoch_times)) * 1.20):
            print("[budget] stopping B16 downstream before next epoch")
            break
        start = time.monotonic()
        model.train()
        loss_sum = 0.0
        steps = study_draws = active_cells = pos_seen = neg_seen = series_seen = 0
        max_series = 0
        for batch in loader:
            if not budget.can_start(120.0):
                budget_exhausted = True
                print("[budget] stopping B16 downstream batches before wall-clock reserve")
                break
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
            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()

            active = weight > 0
            loss_sum += float(loss.item())
            steps += 1
            study_draws += int(volumes.shape[0])
            active_cells += int(active.sum().item())
            pos_seen += int((active & (target > 0.5)).sum().item())
            neg_seen += int((active & (target < 0.5)).sum().item())
            series_seen += int((present > 0).sum().item())
            max_series = max(max_series, int(volumes.shape[1]))

        seconds = time.monotonic() - start
        epoch_times.append(seconds)
        if steps == 0:
            raise RuntimeError("B16 downstream completed no training batches")
        scheduler.step()
        full_study = steps == batches_per_epoch and study_draws == len(ds)
        full_series = full_study and series_seen == expected_series
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / steps,
            "encoder_lr": float(optimizer.param_groups[0]["lr"]),
            "head_lr": float(optimizer.param_groups[1]["lr"]),
            "epoch_seconds": float(seconds),
            "batches": int(steps),
            "expected_full_coverage_batches": batches_per_epoch,
            "study_draws": int(study_draws),
            "expected_full_coverage_studies": len(ds),
            "active_supervision_cells_seen": int(active_cells),
            "expected_active_supervision_cells": int((weights > 0).sum()),
            "positive_cells_seen": int(pos_seen),
            "expected_positive_cells": positive_cells,
            "negative_cells_seen": int(neg_seen),
            "expected_negative_cells": negative_cells,
            "series_instances_seen": int(series_seen),
            "expected_series_instances": expected_series,
            "max_series_in_any_batch": int(max_series),
            "full_coverage": bool(full_study),
            "full_series_coverage": bool(full_series),
            "budget_limited": bool(budget_exhausted),
        }
        history.append(row)
        print(row)
        torch.save(
            {
                **policy,
                "model_state": model.state_dict(),
                "encoder": model.encoder.state_dict(),
                "config": config,
                "completed_epochs": len(history),
                "history": history,
                "budget": budget.to_dict(),
            },
            checkpoint_path,
        )
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if budget_exhausted:
            break

    if len(history) != 4 or not all(row["full_coverage"] and row["full_series_coverage"] and not row["budget_limited"] for row in history):
        print("[warning] B16 downstream did not complete four exact full B6 passes; do not run gold evaluation")
    return checkpoint_path


def load_b16_checkpoint(checkpoint: str | Path, *, device: torch.device | str = "cpu"):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("variant") != B16_VARIANT:
        raise ValueError("not a frozen B16 downstream checkpoint")
    if payload.get("initialization") != B16_REPORT_SSL_VARIANT:
        raise ValueError("B16 checkpoint initialization mismatch")
    if payload.get("input_normalization") != B13_INPUT_NORMALIZATION:
        raise ValueError("B16 checkpoint normalization mismatch")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B16 checkpoint does not certify zero gold-gradient use")
    if bool(payload.get("gold_labels_for_early_stopping", True)):
        raise ValueError("B16 checkpoint does not certify zero gold early stopping")
    if int(payload.get("training_studies", -1)) != 3120:
        raise ValueError("B16 checkpoint must use full 3,120-study B6 surface")
    if int(payload.get("training_series", -1)) != 17475:
        raise ValueError("B16 checkpoint must use frozen 17,475-series surface")
    history = payload.get("history", [])
    if int(payload.get("completed_epochs", -1)) != 4 or len(history) != 4:
        raise ValueError("B16 downstream requires four completed epochs")
    if not all(bool(row.get("full_coverage")) and bool(row.get("full_series_coverage")) and not bool(row.get("budget_limited")) for row in history):
        raise ValueError("B16 checkpoint lacks four complete downstream passes")
    spec = payload.get("model_spec")
    state = payload.get("model_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("B16 checkpoint missing model specification/state")
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.load_state_dict(state, strict=True)
    return model.to(device), payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b16")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--report-ssl-checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b16_full_report/downstream")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = train_b16(
        config,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
