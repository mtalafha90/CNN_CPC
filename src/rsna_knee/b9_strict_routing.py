"""B9: B7.1 recipe with strict semantic MRI stream routing.

B9 is a controlled routing experiment.  Relative to B7.1 it keeps the B5
encoder initialization, frozen B6 v1.2.1 supervision, model architecture,
optimization, augmentations, four full corpus passes, and gold-evaluation
contract unchanged.  The only scientific change is series routing:

- ``*_fluid`` slots may use only ``Fluid_Sensitive == True``;
- ``*_structural`` slots may use only ``Fluid_Sensitive == False``;
- unavailable contrasts remain missing/masked rather than receiving a series
  from the opposite semantic class.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
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
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .dataset import KneeStudyDataset
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime
from .ssl import SSL_SOURCE
from .strict_routing import STRICT_ROUTING_POLICY, build_strict_series_index, routing_audit

B9_VARIANT = "b9_b5_init_b6_strict_semantic_routing_v1"
B9_EXPERIMENT = "B9_strict_semantic_routing"


def _require_b9_contract(config: dict) -> None:
    _require_frozen_policy(config)
    if str(config.get("b9_routing_policy", STRICT_ROUTING_POLICY)) != STRICT_ROUTING_POLICY:
        raise ValueError(f"B9 routing policy must remain {STRICT_ROUTING_POLICY!r}")
    if str(config.get("b9_experiment_name", B9_EXPERIMENT)) != B9_EXPERIMENT:
        raise ValueError(f"B9 experiment name must remain {B9_EXPERIMENT!r}")
    if int(config.get("b7_epochs", 4)) != 4:
        raise ValueError("B9-v1 requires exactly 4 epochs")
    if int(config.get("b7_max_batches_per_epoch", 1560)) != 1560:
        raise ValueError("B9-v1 requires 1560 batches per epoch")
    if int(config.get("b7_batch_size", 2)) != 2:
        raise ValueError("B9-v1 requires batch size 2")


def _checkpoint_payload(
    *, model, config, spec, history, b5_checkpoint, b5_payload, b6_root,
    b6_audit, supervision, target_multiplier, metadata_stats, route_audit, budget,
) -> dict:
    return {
        "variant": B9_VARIANT,
        "experiment": B9_EXPERIMENT,
        "source": SSL_SOURCE,
        "routing_policy": STRICT_ROUTING_POLICY,
        "model_state": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "model_spec": spec,
        "config": config,
        "completed_epochs": len(history),
        "gold_studies_used_in_gradient": 0,
        "gold_studies_used_for_early_stopping": 0,
        "b6_gold_audit_informed_global_policy": True,
        "b5_checkpoint": str(b5_checkpoint.resolve()),
        "b5_variant": b5_payload.get("variant"),
        "b5_completed_epochs": b5_payload.get("completed_epochs"),
        "b6_root": str(b6_root.resolve()),
        "b6_version": b6_audit.get("b6_version"),
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
        "routing_audit": route_audit,
        "history": history,
        "budget": budget.to_dict(),
    }


def train_b9(
    config: dict,
    *,
    b5_checkpoint: str | Path,
    b6_root: str | Path,
    out_root: str | Path = "runs/b9_strict_routing",
) -> Path:
    validate_competition_config(config, purpose="train")
    _require_b9_contract(config)
    seed = int(config.get("seed", 2026))
    # Preserve B7.1 training randomness so routing is the intended scientific
    # change rather than a newly chosen seed.
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
    route_audit = routing_audit(series, uids)
    index = build_strict_series_index(series, uids)
    has_mri = np.asarray(
        [any(index.get(uid, {}).get(stream) for stream in DUAL_STREAMS) for uid in uids],
        dtype=bool,
    )
    if not has_mri.any():
        raise ValueError("B9 found no active weakly labelled studies with MRI series")
    supervision["active_studies_before_mri_filter"] = int(len(uids))
    supervision["studies_without_any_selected_mri_series"] = int((~has_mri).sum())
    uids = [uid for uid, keep in zip(uids, has_mri) if keep]
    targets, weights = targets[has_mri], weights[has_mri]
    supervision["training_studies"] = int(len(uids))
    supervision["training_usable_cells"] = int((weights > 0).sum())

    target_multiplier = target_balance_multipliers(weights)
    for j, target in enumerate(TARGETS):
        supervision["targets"][target]["training_base_weight_sum"] = float(weights[:, j].sum())
        supervision["targets"][target]["target_balance_multiplier"] = float(target_multiplier[j])
        supervision["targets"][target]["balanced_weight_sum"] = float(
            (weights[:, j] * target_multiplier[j]).sum()
        )

    ds = KneeStudyDataset(
        uids,
        index,
        make_b7_dataset_config(config, root, train=True),
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
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
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
    checkpoint_path = outdir / "b9_model.pt"
    history: list[dict] = []
    epoch_times: list[float] = []
    budget_exhausted = False
    (outdir / "supervision_plan.json").write_text(
        json.dumps(supervision, indent=2), encoding="utf-8"
    )
    (outdir / "routing_audit.json").write_text(
        json.dumps(route_audit, indent=2), encoding="utf-8"
    )
    policy_payload = {
        "experiment": B9_EXPERIMENT,
        "variant": B9_VARIANT,
        "status": "B9-v1 recipe frozen before first gold evaluation",
        "single_scientific_change": "strict Fluid_Sensitive semantic routing; no cross-contrast substitution",
        "routing_policy": STRICT_ROUTING_POLICY,
        "b5_initialization": str(b5_path.resolve()),
        "b5_variant": b5_payload.get("variant"),
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "b6_policy": b6_policy,
        "gold_labels_in_training_loss": False,
        "gold_labels_for_early_stopping": False,
        "b6_gold_audit_informed_global_policy": True,
        "fixed_epochs": epochs,
        "max_batches_per_epoch": max_batches,
        "supervision_policy": {
            "min_confidence": B7_MIN_CONFIDENCE,
            "positive_target": B7_POSITIVE_TARGET,
            "negative_target": B7_NEGATIVE_TARGET,
            "positive_weight": B7_POSITIVE_WEIGHT,
            "negative_weight": B7_NEGATIVE_WEIGHT,
            "uncertain_weight": 0.0,
            "unmentioned_weight": 0.0,
        },
        "target_balancing": {
            target: float(target_multiplier[j]) for j, target in enumerate(TARGETS)
        },
        "model_spec": spec,
        "routing_audit": route_audit,
    }
    (outdir / "policy.json").write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")

    for epoch in range(epochs):
        if epoch_times and not budget.can_start(float(np.median(epoch_times)) * 1.20):
            print("[budget] stopping B9 before next epoch")
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
                print("[budget] stopping B9 batches before wall-clock reserve")
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
            raise RuntimeError("B9 completed no training batches inside the runtime budget")
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
                route_audit=route_audit,
                budget=budget,
            ),
            checkpoint_path,
        )
        (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if budget_exhausted:
            break

    if not history:
        raise RuntimeError("B9 did not complete an epoch")
    return checkpoint_path


def load_b9_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("variant") != B9_VARIANT:
        raise ValueError(f"not a {B9_VARIANT} checkpoint")
    if payload.get("routing_policy") != STRICT_ROUTING_POLICY:
        raise ValueError("B9 checkpoint does not certify strict semantic routing")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B9 checkpoint does not certify zero gold-gradient studies")
    spec = payload.get("model_spec")
    state = payload.get("model_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("B9 checkpoint is missing model_spec/model_state")
    model = build_b7_model(spec)
    model.load_state_dict(state, strict=True)
    return model.to(device), payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b9")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None, help="override data_root from YAML")
    parser.add_argument("--b5-checkpoint", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--out-root", default="runs/b9_strict_routing")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = train_b9(
        config,
        b5_checkpoint=args.b5_checkpoint,
        b6_root=args.b6_root,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
