"""B13 training: B12.1 hierarchical architecture with an ImageNet encoder protocol.

Single controlled intervention versus B12.1:
    B5 competition-only encoder protocol -> torchvision ConvNeXt-Tiny
    ImageNet-1K IMAGENET1K_V1 protocol (weights + standard ImageNet normalization).

The B12 all-series surface, B6 supervision, hierarchical aggregation, optimizer,
augmentation, four-epoch schedule, TTA policy and development evaluation contract
remain frozen. Non-encoder trainable parameters are constructed before ImageNet
weights are loaded so their seeded initialization follows the B12.1 path.
"""
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
from .model import ConvNeXtSliceEncoder
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime

B13_VARIANT = "b13_imagenet_init_b6_hierarchical_series_token_v1"
B13_EXPERIMENT = "B13_imagenet_encoder_init"
B13_INITIALIZATION = "torchvision:convnext_tiny:IMAGENET1K_V1"
B13_INPUT_NORMALIZATION = "imagenet_mean_std"
B13_SERIES_SIGNATURE = "5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376"


def _require_close(config: dict, key: str, expected: float) -> None:
    value = float(config.get(key, expected))
    if not np.isclose(value, expected, atol=1e-12, rtol=0):
        raise ValueError(f"B13-v1 freezes {key}={expected}; got {value}")


def _require_b13_contract(config: dict) -> None:
    """Freeze B13 as the ImageNet encoder-protocol experiment versus B12.1."""
    _require_frozen_policy(config)
    if str(config.get("b13_experiment_name", B13_EXPERIMENT)) != B13_EXPERIMENT:
        raise ValueError(f"B13 experiment name must remain {B13_EXPERIMENT!r}")
    if not bool(config.get("allow_external_pretrained", False)):
        raise ValueError("B13 requires allow_external_pretrained=true")
    if not bool(config.get("pretrained", False)):
        raise ValueError("B13 requires pretrained=true")
    if bool(config.get("b12_use_physical_scale", False)):
        raise ValueError("B13 freezes legacy resize; B10 physical normalization is disabled")

    integer_contract = {
        "seed": 2026,
        "requested_gpus": 1,
        "b7_n_slices": 16,
        "b7_image_size": 224,
        "b7_triplet_gap": 1,
        "b7_batch_size": 2,
        "b7_encoder_batch_size": 24,
        "b7_transformer_layers": 2,
        "b7_transformer_heads": 8,
        "b7_pathology_layers": 1,
        "b7_epochs": 4,
        "b7_max_batches_per_epoch": 1560,
        "b12_1_series_pool_heads": 8,
        "b7_eval_batch_size": 2,
        "b7_n_bootstrap": 5000,
    }
    # An ensemble member deliberately varies the stochastic path: that is the
    # entire point of it, since averaging models that shared a seed gains
    # nothing. The contract still refuses an undeclared seed change, which is
    # what it exists to catch -- a run that drifted off the protocol without
    # anyone noticing. Declaring `ensemble_member` is the deviation being made
    # on purpose and on the record.
    declared_member = int(config.get("ensemble_member", 0) or 0)
    for key, expected in integer_contract.items():
        value = int(config.get(key, expected))
        if value == expected:
            continue
        if key == "seed" and declared_member:
            continue
        raise ValueError(f"B13-v1 freezes {key}={expected}; got {value}")

    bool_contract = {
        "competition_mode": True,
        "b7_gradient_checkpointing": True,
    }
    for key, expected in bool_contract.items():
        value = bool(config.get(key, expected))
        if value is not expected:
            raise ValueError(f"B13-v1 freezes {key}={expected}; got {value}")

    float_contract = {
        "b7_dropout": 0.25,
        "b7_transformer_ff_mult": 2.0,
        "b7_encoder_lr": 1e-5,
        "b7_head_lr": 1e-4,
        "b7_min_lr": 1e-6,
        "b7_weight_decay": 1e-4,
        "b7_grad_clip": 1.0,
        "b7_noise_std": 0.02,
        "b7_slice_dropout": 0.08,
        "b7_center_jitter": 2.0,
        "b7_rotation_deg": 5.0,
        "b7_translate_frac": 0.03,
        "b7_scale_jitter": 0.05,
        "b7_gamma_jitter": 0.12,
        "b7_bias_field_strength": 0.08,
    }
    for key, expected in float_contract.items():
        _require_close(config, key, expected)

    if tuple(int(x) for x in config.get("b7_train_gap_choices", [1, 2])) != (1, 2):
        raise ValueError("B13-v1 freezes b7_train_gap_choices=[1,2]")
    if tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1])) != (-1, 0, 1):
        raise ValueError("B13-v1 freezes b7_eval_tta_offsets=[-1,0,1]")


def _load_imagenet_encoder_state(model) -> None:
    """Load ImageNet weights after shared B12.1 parameters are initialized.

    Constructing the full hierarchical model first keeps the seeded non-encoder
    parameter initialization independent of the external pretrained-weight load.
    """
    pretrained = ConvNeXtSliceEncoder(
        3,
        pretrained_weights=True,
        normalize_input=True,
    )
    model.encoder.load_state_dict(pretrained.state_dict(), strict=True)
    del pretrained


def _checkpoint_payload(
    *,
    model,
    config,
    spec,
    history,
    b6_root,
    b6_audit,
    supervision,
    target_multiplier,
    metadata_stats,
    series_policy,
    budget,
) -> dict:
    return {
        "variant": B13_VARIANT,
        "experiment": B13_EXPERIMENT,
        "source": "external_imagenet",
        "initialization": B13_INITIALIZATION,
        "input_normalization": B13_INPUT_NORMALIZATION,
        "external_pretrained": True,
        "model_state": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "model_spec": spec,
        "config": config,
        "completed_epochs": len(history),
        "gold_studies_used_in_gradient": 0,
        "gold_studies_used_for_early_stopping": 0,
        "b6_gold_audit_informed_global_policy": True,
        "b5_checkpoint": None,
        "b5_variant": None,
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "series_policy": series_policy,
        "single_scientific_change_vs_b12_1": (
            "encoder initialization protocol changed from B5 competition-only SSL to "
            "torchvision ConvNeXt-Tiny ImageNet-1K V1 weights with standard ImageNet normalization"
        ),
        "supervision_policy": {
            "b6_min_confidence": B7_MIN_CONFIDENCE,
            "b6_positive_target": B7_POSITIVE_TARGET,
            "b6_negative_target": B7_NEGATIVE_TARGET,
            "b6_positive_weight": B7_POSITIVE_WEIGHT,
            "b6_negative_weight": B7_NEGATIVE_WEIGHT,
            "target_balancing": "unchanged B7.1/B12/B12.1 B6-derived multipliers",
        },
        "target_balance_multiplier": {
            target: float(target_multiplier[j]) for j, target in enumerate(TARGETS)
        },
        "supervision": supervision,
        "metadata_repair": metadata_stats,
        "history": history,
        "budget": budget.to_dict(),
    }


def train_b13(
    config: dict,
    *,
    b6_root: str | Path,
    series_policy_path: str | Path,
    out_root: str | Path = "runs/b13_imagenet",
) -> Path:
    validate_competition_config(config, purpose="train")
    _require_b13_contract(config)

    seed = int(config.get("seed", 2026))
    seed_everything(seed + 12_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    print(f"[B13] initialization={B13_INITIALIZATION}")
    print(f"[B13] input_normalization={B13_INPUT_NORMALIZATION}")
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
    if (
        len(uids) != 3120
        or int((weights > 0).sum()) != 14123
        or positive_cells != 6871
        or negative_cells != 7252
    ):
        raise ValueError("B13 must retain the exact B7.1/B12/B12.1 B6 training surface")

    series_policy = _load_series_policy(series_policy_path)
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    frozen_summary = series_policy.get("series_summary", {})
    if series_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B13 reconstructed series surface does not match the frozen B12 SHA-256")
    if frozen_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B13 supplied series policy is not the frozen B12 policy")
    if int(series_summary.get("eligible_recognized_plane_series", -1)) != 17475:
        raise ValueError("B13 requires the frozen 17,475-series B12 surface")
    if series_summary.get("viability_passed") is not True:
        raise ValueError("B13 reconstructed B12 series surface no longer passes viability")

    target_multiplier = target_balance_multipliers(weights)
    batch_size = int(config.get("b7_batch_size", 2))
    batches_per_epoch = int(math.ceil(len(uids) / batch_size))
    expected_series = 17475
    supervision.update(
        {
            "training_studies": len(uids),
            "training_cells": int((weights > 0).sum()),
            "training_positive_cells": positive_cells,
            "training_negative_cells": negative_cells,
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
        **runtime.loader_kwargs(seed=seed + 12_100_000),
    )

    # Build the complete B12.1 architecture first, using the same seeded random
    # construction path for all non-encoder parameters. Then replace only the
    # encoder state with ImageNet weights. normalize_input=True is the standard
    # preprocessing expected by the external pretrained encoder.
    spec = b12_1_model_spec(config, normalize_input=True)
    model = build_b12_1_model(spec, pretrained_weights=False)
    _load_imagenet_encoder_state(model)
    model = model.to(runtime.device)

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
    checkpoint_path = outdir / "b13_model.pt"
    (outdir / "supervision_plan.json").write_text(
        json.dumps(supervision, indent=2), encoding="utf-8"
    )
    policy_payload = {
        "experiment": B13_EXPERIMENT,
        "variant": B13_VARIANT,
        "status": "B13-v1 recipe frozen before first gold evaluation",
        "single_scientific_change_vs_b12_1": (
            "B5 competition-only encoder protocol -> torchvision ConvNeXt-Tiny ImageNet-1K V1 "
            "weights with standard ImageNet normalization"
        ),
        "student_initialization": B13_INITIALIZATION,
        "student_initialization_variant": "external_imagenet",
        "input_normalization": B13_INPUT_NORMALIZATION,
        "external_pretrained": True,
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "b6_policy": b6_policy,
        "series_policy": series_policy,
        "preprocessing": (
            "same MRI sampling and legacy 224x224 resize as B12.1; ImageNet mean/std normalization "
            "is applied inside the pretrained ConvNeXt encoder"
        ),
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
            print("[budget] stopping B13 before next epoch")
            break
        start = time.monotonic()
        model.train()
        loss_sum = 0.0
        steps = study_draws = active_cells = pos_seen = neg_seen = 0
        series_present = 0
        max_series_in_batch = 0
        for batch in loader:
            if not budget.can_start(120.0):
                budget_exhausted = True
                print("[budget] stopping B13 batches before wall-clock reserve")
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
            series_present += int((present > 0).sum().item())
            max_series_in_batch = max(max_series_in_batch, int(volumes.shape[1]))

        seconds = time.monotonic() - start
        epoch_times.append(seconds)
        if steps == 0:
            raise RuntimeError("B13 completed no training batches")
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
            "positive_cells_seen": int(pos_seen),
            "negative_cells_seen": int(neg_seen),
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
        (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if budget_exhausted:
            break

    if len(history) != 4 or not all(
        bool(row.get("full_coverage"))
        and bool(row.get("full_series_coverage"))
        and not bool(row.get("budget_limited"))
        for row in history
    ):
        print(
            "[warning] B13 did not complete the frozen four-pass study/series contract; "
            "do not run gold evaluation"
        )
    return checkpoint_path


def load_b13_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("variant") != B13_VARIANT:
        raise ValueError(f"not a {B13_VARIANT} checkpoint")
    if payload.get("initialization") != B13_INITIALIZATION:
        raise ValueError("B13 checkpoint does not certify the frozen ImageNet initialization")
    if payload.get("input_normalization") != B13_INPUT_NORMALIZATION:
        raise ValueError("B13 checkpoint does not certify the frozen ImageNet normalization")
    if payload.get("external_pretrained") is not True:
        raise ValueError("B13 checkpoint does not certify external ImageNet pretraining")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B13 checkpoint does not certify zero gold-gradient studies")
    spec = payload.get("model_spec")
    state = payload.get("model_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("B13 checkpoint missing model_spec/model_state")
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.load_state_dict(state, strict=True)
    return model.to(device), payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b13")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--out-root", default="runs/b13_imagenet")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = train_b13(
        config,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
