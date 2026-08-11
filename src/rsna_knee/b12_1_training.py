"""B12.1 training: B12 series surface with learned per-series token compression."""
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
    _require_frozen_policy,
    load_b5_encoder_payload,
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
from .budget import RuntimeBudget
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime
from .ssl import SSL_SOURCE

B12_1_VARIANT = "b12_1_b5_init_b6_hierarchical_series_token_v1"
B12_1_EXPERIMENT = "B12.1_hierarchical_series_tokens"


def _require_b12_1_contract(config: dict) -> None:
    _require_frozen_policy(config)
    if str(config.get("b12_1_experiment_name", B12_1_EXPERIMENT)) != B12_1_EXPERIMENT:
        raise ValueError(f"B12.1 experiment name must remain {B12_1_EXPERIMENT!r}")
    if int(config.get("b7_epochs", 4)) != 4:
        raise ValueError("B12.1 requires exactly four epochs")
    if int(config.get("b7_batch_size", 2)) != 2:
        raise ValueError("B12.1 requires batch size 2")
    if int(config.get("b12_1_series_pool_heads", 8)) != 8:
        raise ValueError("B12.1-v1 freezes series_pool_heads=8")
    if bool(config.get("b12_use_physical_scale", False)):
        raise ValueError("B12.1 freezes legacy resize; B10 physical normalization is disabled")


def _checkpoint_payload(
    *,
    model,
    config,
    spec,
    history,
    b5_checkpoint,
    b5_payload,
    b6_root,
    b6_audit,
    supervision,
    target_multiplier,
    metadata_stats,
    series_policy,
    budget,
) -> dict:
    return {
        "variant": B12_1_VARIANT,
        "experiment": B12_1_EXPERIMENT,
        "source": SSL_SOURCE,
        "model_state": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "model_spec": spec,
        "config": config,
        "completed_epochs": len(history),
        "gold_studies_used_in_gradient": 0,
        "gold_studies_used_for_early_stopping": 0,
        "b6_gold_audit_informed_global_policy": True,
        # None under ImageNet initialisation, where there is no B5 checkpoint.
        "b5_checkpoint": None if b5_checkpoint is None else str(Path(b5_checkpoint).resolve()),
        "b5_variant": b5_payload.get("variant"),
        "external_pretrained": bool(config.get("pretrained", False)),
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "series_policy": series_policy,
        "single_scientific_change_vs_b12": (
            "compress each real MRI series from 16 slice tokens to one learned attention-pooled "
            "series token before the unchanged study Transformer"
        ),
        "supervision_policy": {
            "b6_min_confidence": B7_MIN_CONFIDENCE,
            "b6_positive_target": B7_POSITIVE_TARGET,
            "b6_negative_target": B7_NEGATIVE_TARGET,
            "b6_positive_weight": B7_POSITIVE_WEIGHT,
            "b6_negative_weight": B7_NEGATIVE_WEIGHT,
            "target_balancing": "unchanged B7.1/B12 B6-derived multipliers",
        },
        "target_balance_multiplier": {
            target: float(target_multiplier[j]) for j, target in enumerate(TARGETS)
        },
        "supervision": supervision,
        "metadata_repair": metadata_stats,
        "history": history,
        "budget": budget.to_dict(),
    }


def train_b12_1(
    config: dict,
    *,
    b5_checkpoint: str | Path | None,
    b6_root: str | Path,
    series_policy_path: str | Path,
    out_root: str | Path = "runs/b12_1_hierarchical",
) -> Path:
    validate_competition_config(config, purpose="train")
    _require_b12_1_contract(config)
    seed = int(config.get("seed", 2026))
    # Intentionally identical to B12 for controlled batch/augmentation ordering.
    seed_everything(seed + 12_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    uids, targets, weights, supervision = prepare_b7_supervision(train, b6_frame)
    if len(uids) != 3120 or int((weights > 0).sum()) != 14123:
        raise ValueError("B12.1 must retain the exact B7.1/B12 B6 training surface")

    series_policy = _load_series_policy(series_policy_path)
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    frozen_summary = series_policy.get("series_summary", {})
    if series_summary.get("series_signature_sha256") != frozen_summary.get("series_signature_sha256"):
        raise ValueError("B12.1 variable-series mapping changed since the frozen B12 audit")
    if int(series_summary.get("eligible_recognized_plane_series", -1)) != int(
        frozen_summary.get("eligible_recognized_plane_series", -2)
    ):
        raise ValueError("B12.1 eligible series count changed since the frozen B12 audit")
    if series_summary.get("viability_passed") is not True:
        raise ValueError("B12.1 reconstructed B12 series surface no longer passes viability")

    target_multiplier = target_balance_multipliers(weights)
    batch_size = int(config.get("b7_batch_size", 2))
    batches_per_epoch = int(math.ceil(len(uids) / batch_size))
    expected_series = int(series_summary["eligible_recognized_plane_series"])
    if expected_series != 17475:
        raise ValueError(f"B12.1-v1 requires the frozen 17,475-series surface; got {expected_series}")
    supervision.update(
        {
            "training_studies": len(uids),
            "training_cells": int((weights > 0).sum()),
            "training_positive_cells": int(((weights > 0) & (targets > 0.5)).sum()),
            "training_negative_cells": int(((weights > 0) & (targets < 0.5)).sum()),
            "eligible_series_expected_per_full_epoch": expected_series,
            "historical_dual_unique_series": int(series_summary["historical_dual_unique_series"]),
            "extra_series_retained": int(series_summary["extra_series_retained"]),
            "full_coverage_batches_per_epoch": batches_per_epoch,
            "series_signature_sha256": series_summary["series_signature_sha256"],
        }
    )

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
        # Intentionally identical to B12 for controlled batch order.
        **runtime.loader_kwargs(seed=seed + 12_100_000),
    )

    # Encoder initialisation is the independent variable of B13. Exactly one of
    # the two sources is used: competition-only SSL (B5) or ImageNet.
    use_imagenet = bool(config.get("pretrained", False))
    if use_imagenet:
        if not bool(config.get("allow_external_pretrained", False)):
            raise ValueError(
                "pretrained=true requires allow_external_pretrained=true; external "
                "pretrained weights must be declared explicitly"
            )
        # ImageNet weights expect ImageNet channel statistics, which the encoder
        # applies itself when normalize_input is set.
        normalize_input = True
        spec = b12_1_model_spec(config, normalize_input=normalize_input)
        model = build_b12_1_model(spec, pretrained_weights=True).to(runtime.device)
        b5_path = None
        b5_payload = {}
        initialization = "imagenet:convnext_tiny:IMAGENET1K_V1"
        initialization_variant = "external_imagenet"
    else:
        if b5_checkpoint is None:
            raise ValueError("competition-only initialisation requires --b5-checkpoint")
        b5_path = Path(b5_checkpoint)
        b5_payload = load_b5_encoder_payload(b5_path)
        normalize_input = bool(b5_payload.get("config", {}).get("normalize_input", False))
        spec = b12_1_model_spec(config, normalize_input=normalize_input)
        model = build_b12_1_model(spec, encoder_state=b5_payload["encoder"]).to(runtime.device)
        initialization = str(b5_path.resolve())
        initialization_variant = b5_payload.get("variant")

    encoder_lr = float(config.get("b7_encoder_lr", 1e-5))
    head_lr = float(config.get("b7_head_lr", 1e-4))
    encoder_params = list(model.encoder.parameters())
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": encoder_lr}, {"params": head_params, "lr": head_lr}],
        weight_decay=float(config.get("b7_weight_decay", 1e-4)),
    )
    epochs = int(config.get("b7_epochs", 4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(config.get("b7_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    clip = float(config.get("b7_grad_clip", 1.0))
    target_multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)

    outdir = Path(out_root)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "b12_1_model.pt"
    (outdir / "supervision_plan.json").write_text(
        json.dumps(supervision, indent=2), encoding="utf-8"
    )
    policy_payload = {
        "experiment": B12_1_EXPERIMENT,
        "variant": B12_1_VARIANT,
        "status": "B12.1-v1 recipe frozen before first gold evaluation",
        "single_scientific_change_vs_b12": (
            "learned per-series attention compression: 16 slice tokens -> 1 series token "
            "before the unchanged study Transformer"
        ),
        "student_initialization": initialization,
        "student_initialization_variant": initialization_variant,
        "external_pretrained": use_imagenet,
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "b6_policy": b6_policy,
        "series_policy": series_policy,
        "preprocessing": "historical B7.1/B12 legacy resize; no B10 physical normalization",
        "gold_labels_in_training_loss": False,
        "gold_labels_for_early_stopping": False,
        "fixed_epochs": epochs,
        "full_coverage_batches_per_epoch": batches_per_epoch,
        "model_spec": spec,
        "supervision": supervision,
    }
    (outdir / "policy.json").write_text(
        json.dumps(policy_payload, indent=2), encoding="utf-8"
    )

    history: list[dict] = []
    epoch_times: list[float] = []
    budget_exhausted = False
    for epoch in range(epochs):
        if epoch_times and not budget.can_start(float(np.median(epoch_times)) * 1.20):
            print("[budget] stopping B12.1 before next epoch")
            break
        start = time.monotonic()
        model.train()
        loss_sum = 0.0
        steps = study_draws = active_cells = positive_cells = negative_cells = 0
        series_present = 0
        max_series_in_batch = 0
        for batch in loader:
            if not budget.can_start(120.0):
                budget_exhausted = True
                print("[budget] stopping B12.1 batches before wall-clock reserve")
                break
            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            series_meta = batch["series_meta"].to(runtime.device, non_blocking=True)
            target = batch["target"].to(runtime.device, non_blocking=True)
            weight = batch["weight"].to(runtime.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                logits = model(volumes, present, series_meta)
                loss = target_balanced_weak_bce(
                    logits, target, weight, target_multiplier_t
                )
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
            positive_cells += int((active & (target > 0.5)).sum().item())
            negative_cells += int((active & (target < 0.5)).sum().item())
            series_present += int((present > 0).sum().item())
            max_series_in_batch = max(max_series_in_batch, int(volumes.shape[1]))

        seconds = time.monotonic() - start
        epoch_times.append(seconds)
        if steps == 0:
            raise RuntimeError("B12.1 completed no training batches")
        scheduler.step()
        full_study_epoch = steps == batches_per_epoch and study_draws == len(ds)
        full_series_epoch = full_study_epoch and series_present == expected_series
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / steps,
            "encoder_lr": float(optimizer.param_groups[0]["lr"]),
            "head_lr": float(optimizer.param_groups[1]["lr"]),
            "epoch_seconds": float(seconds),
            "batches": int(steps),
            "expected_full_coverage_batches": int(batches_per_epoch),
            "study_draws": int(study_draws),
            "expected_full_coverage_studies": int(len(ds)),
            "active_supervision_cells_seen": int(active_cells),
            "positive_cells_seen": int(positive_cells),
            "negative_cells_seen": int(negative_cells),
            "series_instances_seen": int(series_present),
            "expected_series_instances": int(expected_series),
            "max_series_in_any_batch": int(max_series_in_batch),
            "full_coverage": bool(full_study_epoch),
            "full_series_coverage": bool(full_series_epoch),
            "budget_limited": bool(budget_exhausted),
        }
        history.append(row)
        print(row)
        torch.save(
            _checkpoint_payload(
                model=model,
                config=config,
                spec=spec,
                history=history,
                b5_checkpoint=b5_path,
                b5_payload=b5_payload,
                b6_root=Path(b6_root),
                b6_audit=b6_audit,
                supervision=supervision,
                target_multiplier=target_multiplier,
                metadata_stats=metadata_stats,
                series_policy=series_policy,
                budget=budget,
            ),
            checkpoint_path,
        )
        (outdir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if budget_exhausted:
            break

    if len(history) != 4 or not all(
        bool(row.get("full_coverage")) and bool(row.get("full_series_coverage"))
        for row in history
    ):
        print(
            "[warning] B12.1 did not complete the full four-pass study/series contract; "
            "do not run gold evaluation"
        )
    return checkpoint_path


def load_b12_1_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("variant") != B12_1_VARIANT:
        raise ValueError(f"not a {B12_1_VARIANT} checkpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B12.1 checkpoint does not certify zero gold-gradient studies")
    spec = payload.get("model_spec")
    state = payload.get("model_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("B12.1 checkpoint missing model_spec/model_state")
    model = build_b12_1_model(spec)
    model.load_state_dict(state, strict=True)
    return model.to(device), payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b12-1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    # Not required under ImageNet initialisation (config pretrained: true),
    # which replaces the B5 competition-only SSL encoder entirely.
    parser.add_argument("--b5-checkpoint", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--out-root", default="runs/b12_1_hierarchical")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = train_b12_1(
        config,
        b5_checkpoint=args.b5_checkpoint,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
