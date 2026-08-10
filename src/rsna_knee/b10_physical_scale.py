"""B10: B7.1 recipe with label-free in-plane physical-scale normalization.

The only scientific change versus B7.1 is preprocessing geometry:
- historical B7.1 dual stream routing is retained;
- each selected MRI series is resampled to a plane-specific canonical PixelSpacing;
- the resampled series is centrally cropped/padded to a canonical physical FOV;
- only then does the unchanged pipeline resize triplets to 224x224.

The canonical geometry is derived from the 3,120 active weak-training studies only,
using no gold target labels, and is frozen in physical_scale_policy.json before training.
"""

from __future__ import annotations

import argparse
import json
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
    b7_model_spec,
    build_b7_model,
    load_b5_encoder_payload,
    load_frozen_b6_export,
    make_b7_dataset_config,
    prepare_b7_supervision,
    seed_everything,
    target_balance_multipliers,
    target_balanced_weak_bce,
)
from .budget import RuntimeBudget
from .constants import DUAL_STREAMS, TARGETS
from .data import (
    backfill_series_metadata,
    build_series_index,
    load_series_csv,
    load_train_csv,
)
from .dataset import KneeStudyDataset
from .physical_scale import (
    B10_MIN_GEOMETRY_COVERAGE,
    B10_MISSING_SPACING_ACTION,
    B10_PHYSICAL_POLICY,
    load_physical_scale_policy,
    physical_policy_digest,
    selected_series_signature,
)
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime
from .ssl import SSL_SOURCE

B10_VARIANT = "b10_b5_init_b6_physical_scale_v1"
B10_EXPERIMENT = "B10_physical_scale_normalization"


def _require_b10_contract(config: dict) -> None:
    _require_frozen_policy(config)
    if str(config.get("b10_experiment_name", B10_EXPERIMENT)) != B10_EXPERIMENT:
        raise ValueError(f"B10 experiment name must remain {B10_EXPERIMENT!r}")
    if str(config.get("b10_physical_policy", B10_PHYSICAL_POLICY)) != B10_PHYSICAL_POLICY:
        raise ValueError(f"B10 physical policy must remain {B10_PHYSICAL_POLICY!r}")
    if str(
        config.get("b10_missing_spacing_action", B10_MISSING_SPACING_ACTION)
    ) != B10_MISSING_SPACING_ACTION:
        raise ValueError(
            f"B10 missing spacing action must remain {B10_MISSING_SPACING_ACTION!r}"
        )
    if not np.isclose(
        float(config.get("b10_min_geometry_coverage", B10_MIN_GEOMETRY_COVERAGE)),
        B10_MIN_GEOMETRY_COVERAGE,
    ):
        raise ValueError(
            f"B10 minimum geometry coverage must remain {B10_MIN_GEOMETRY_COVERAGE}"
        )
    if int(config.get("b7_epochs", 4)) != 4:
        raise ValueError("B10-v1 requires exactly 4 epochs")
    if int(config.get("b7_max_batches_per_epoch", 1560)) != 1560:
        raise ValueError("B10-v1 requires 1560 batches per epoch")
    if int(config.get("b7_batch_size", 2)) != 2:
        raise ValueError("B10-v1 requires batch size 2")


def _dataset_config(config: dict, root: Path, *, train: bool, policy: dict, offsets=()):
    ds_config = make_b7_dataset_config(
        config,
        root,
        train=train,
        tta_offsets=tuple(int(x) for x in offsets),
    )
    ds_config.physical_scale_policy = policy
    ds_config.__post_init__()
    return ds_config


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
    physical_policy,
    budget,
) -> dict:
    return {
        "variant": B10_VARIANT,
        "experiment": B10_EXPERIMENT,
        "source": SSL_SOURCE,
        "model_state": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "model_spec": spec,
        "config": config,
        "completed_epochs": len(history),
        "gold_studies_used_in_gradient": 0,
        "gold_studies_used_for_early_stopping": 0,
        "gold_labels_used_to_choose_physical_policy": False,
        "b6_gold_audit_informed_global_policy": True,
        "b5_checkpoint": str(b5_checkpoint.resolve()),
        "b5_variant": b5_payload.get("variant"),
        "b5_completed_epochs": b5_payload.get("completed_epochs"),
        "b6_root": str(b6_root.resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "single_scientific_change_vs_b7_1": (
            "plane-specific in-plane PixelSpacing/FOV normalization before the "
            "unchanged 224x224 resize"
        ),
        "routing_mode": "historical_b7_1_dual",
        "physical_policy_name": B10_PHYSICAL_POLICY,
        "physical_policy_sha256": physical_policy["policy_sha256"],
        "physical_scale_policy": physical_policy,
        "supervision_policy": {
            "min_confidence": B7_MIN_CONFIDENCE,
            "positive_target": B7_POSITIVE_TARGET,
            "negative_target": B7_NEGATIVE_TARGET,
            "positive_weight": B7_POSITIVE_WEIGHT,
            "negative_weight": B7_NEGATIVE_WEIGHT,
            "uncertain_weight": 0.0,
            "unmentioned_weight": 0.0,
            "target_balancing": "inverse total B7 base supervision mass per target",
        },
        "target_balance_multiplier": {
            target: float(target_multiplier[j]) for j, target in enumerate(TARGETS)
        },
        "supervision": supervision,
        "metadata_repair": metadata_stats,
        "history": history,
        "budget": budget.to_dict(),
    }


def train_b10(
    config: dict,
    *,
    b5_checkpoint: str | Path,
    b6_root: str | Path,
    physical_policy_path: str | Path,
    out_root: str | Path = "runs/b10_physical_scale",
) -> Path:
    validate_competition_config(config, purpose="train")
    _require_b10_contract(config)

    seed = int(config.get("seed", 2026))
    seed_everything(seed + 7_000_000)
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

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_series_index(series, uids, mode="dual")
    has_mri = np.asarray(
        [any(index.get(uid, {}).get(stream) for stream in DUAL_STREAMS) for uid in uids],
        dtype=bool,
    )
    if not has_mri.any():
        raise ValueError("B10 found no active weakly labelled studies with MRI series")
    supervision["active_studies_before_mri_filter"] = int(len(uids))
    supervision["studies_without_any_selected_mri_series"] = int((~has_mri).sum())
    uids = [uid for uid, keep in zip(uids, has_mri) if keep]
    targets, weights = targets[has_mri], weights[has_mri]
    supervision["training_studies"] = int(len(uids))
    supervision["training_usable_cells"] = int((weights > 0).sum())

    physical_policy = load_physical_scale_policy(physical_policy_path)
    if int(physical_policy.get("source_study_count", -1)) != len(uids):
        raise ValueError(
            "B10 physical policy source_study_count does not match active training studies"
        )
    current_signature = selected_series_signature(index, uids)
    if physical_policy.get("selected_series_signature") != current_signature:
        raise ValueError(
            "B10 selected-series signature differs from the frozen physical-scale audit"
        )
    if physical_policy.get("policy_sha256") != physical_policy_digest(physical_policy):
        raise ValueError("B10 physical policy digest mismatch")

    target_multiplier = target_balance_multipliers(weights)
    for j, target in enumerate(TARGETS):
        supervision["targets"][target]["training_base_weight_sum"] = float(
            weights[:, j].sum()
        )
        supervision["targets"][target]["target_balance_multiplier"] = float(
            target_multiplier[j]
        )
        supervision["targets"][target]["balanced_weight_sum"] = float(
            (weights[:, j] * target_multiplier[j]).sum()
        )

    ds = KneeStudyDataset(
        uids,
        index,
        _dataset_config(config, root, train=True, policy=physical_policy),
        targets=targets,
        weights=weights,
        train=True,
    )
    batch_size = int(config.get("b7_batch_size", 2))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **runtime.loader_kwargs(seed=seed + 7_100_000),
    )

    b5_path = Path(b5_checkpoint)
    b5_payload = load_b5_encoder_payload(b5_path)
    normalize_input = bool(b5_payload.get("config", {}).get("normalize_input", False))
    spec = b7_model_spec(config, normalize_input=normalize_input)
    model = build_b7_model(spec, encoder_state=b5_payload["encoder"]).to(runtime.device)

    encoder_lr = float(config.get("b7_encoder_lr", 1e-5))
    head_lr = float(config.get("b7_head_lr", 1e-4))
    encoder_params = list(model.encoder.parameters())
    head_params = [
        p for name, p in model.named_parameters() if not name.startswith("encoder.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=float(config.get("b7_weight_decay", 1e-4)),
    )
    epochs = int(config.get("b7_epochs", 4))
    max_batches = int(config.get("b7_max_batches_per_epoch", 1560))
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
    checkpoint_path = outdir / "b10_model.pt"
    history: list[dict] = []
    epoch_times: list[float] = []
    budget_exhausted = False

    (outdir / "supervision_plan.json").write_text(
        json.dumps(supervision, indent=2), encoding="utf-8"
    )
    (outdir / "physical_scale_policy.json").write_text(
        json.dumps(physical_policy, indent=2), encoding="utf-8"
    )
    policy_payload = {
        "experiment": B10_EXPERIMENT,
        "variant": B10_VARIANT,
        "status": "B10-v1 frozen before first gold evaluation",
        "single_scientific_change": (
            "plane-specific in-plane PixelSpacing/FOV normalization before "
            "the unchanged 224x224 resize"
        ),
        "routing_mode": "historical B7.1 dual routing",
        "physical_policy_name": B10_PHYSICAL_POLICY,
        "physical_policy_sha256": physical_policy["policy_sha256"],
        "gold_labels_used_to_choose_physical_policy": False,
        "b5_initialization": str(b5_path.resolve()),
        "b5_variant": b5_payload.get("variant"),
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "b6_policy": b6_policy,
        "gold_labels_in_training_loss": False,
        "gold_labels_for_early_stopping": False,
        "fixed_epochs": epochs,
        "max_batches_per_epoch": max_batches,
        "model_spec": spec,
    }
    (outdir / "policy.json").write_text(
        json.dumps(policy_payload, indent=2), encoding="utf-8"
    )

    for epoch in range(epochs):
        if epoch_times and not budget.can_start(float(np.median(epoch_times)) * 1.20):
            print("[budget] stopping B10 before next epoch")
            break
        epoch_start = time.monotonic()
        model.train()
        loss_sum = 0.0
        steps = study_draws = active_cells = positive_cells = negative_cells = 0

        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            if not budget.can_start(120.0):
                budget_exhausted = True
                print("[budget] stopping B10 batches before wall-clock reserve")
                break

            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            target = batch["target"].to(runtime.device, non_blocking=True)
            weight = batch["weight"].to(runtime.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                logits = model(volumes, present)
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

        epoch_seconds = time.monotonic() - epoch_start
        epoch_times.append(epoch_seconds)
        if steps == 0:
            raise RuntimeError("B10 completed no training batches inside the runtime budget")
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / steps,
            "encoder_lr": float(optimizer.param_groups[0]["lr"]),
            "head_lr": float(optimizer.param_groups[1]["lr"]),
            "epoch_seconds": float(epoch_seconds),
            "batches": int(steps),
            "study_draws": int(study_draws),
            "active_supervision_cells_seen": int(active_cells),
            "positive_cells_seen": int(positive_cells),
            "negative_cells_seen": int(negative_cells),
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
                physical_policy=physical_policy,
                budget=budget,
            ),
            checkpoint_path,
        )
        (outdir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if budget_exhausted:
            break

    if not history:
        raise RuntimeError("B10 did not complete an epoch")
    return checkpoint_path


def load_b10_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("variant") != B10_VARIANT:
        raise ValueError(f"not a {B10_VARIANT} checkpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B10 checkpoint does not certify zero gold-gradient studies")
    if bool(payload.get("gold_labels_used_to_choose_physical_policy", True)):
        raise ValueError("B10 checkpoint does not certify label-free physical policy")
    policy = payload.get("physical_scale_policy")
    if not isinstance(policy, dict):
        raise ValueError("B10 checkpoint is missing physical_scale_policy")
    if payload.get("physical_policy_sha256") != physical_policy_digest(policy):
        raise ValueError("B10 checkpoint physical policy digest mismatch")
    spec = payload.get("model_spec")
    state = payload.get("model_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("B10 checkpoint is missing model_spec/model_state")
    model = build_b7_model(spec)
    model.load_state_dict(state, strict=True)
    return model.to(device), payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b10")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None, help="override data_root from YAML")
    parser.add_argument("--b5-checkpoint", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--physical-policy", required=True)
    parser.add_argument("--out-root", default="runs/b10_physical_scale")
    args = parser.parse_args()

    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    checkpoint = train_b10(
        config,
        b5_checkpoint=args.b5_checkpoint,
        b6_root=args.b6_root,
        physical_policy_path=args.physical_policy,
        out_root=args.out_root,
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
