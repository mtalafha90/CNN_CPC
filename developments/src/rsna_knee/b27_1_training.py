"""B27.1 fixed-E2 training on the exact historical B20 surface.

B27.1 is the pre-outcome correction to B27 after the training metadata audit
showed perfect collinearity between Fluid_Sensitive and Fat_Suppression on all
17,475 eligible series.  It replaces B27's duplicated fluid+fat routing terms
with one paired sequence term, reducing the new routing layer from 84 to 60
parameters while preserving every B20 training control.

No B27/B27.1 expert/gold result is read or used here.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .b7_weak_supervision import (
    _read_config,
    load_frozen_b6_export,
    make_b7_dataset_config,
    prepare_b7_supervision,
    seed_everything,
    target_balance_multipliers,
    target_balanced_weak_bce,
)
from .b12_training import _load_series_policy
from .b12_variable_series import audit_variable_series_surface, collate_variable_series
from .b13_training import B13_INPUT_NORMALIZATION, B13_SERIES_SIGNATURE
from .b16_report_ssl import (
    B16_REPORT_SSL_EXPERIMENT,
    B16_REPORT_SSL_OBJECTIVE,
    B16_REPORT_SSL_VARIANT,
    load_b16_report_encoder,
)
from .b17_training import B17_HEAD_LR, encoder_state_sha256, freeze_encoder
from .b18_fisher_selection import B18_CANDIDATE_EPOCHS
from .b20_crop_focus import CropFocusedVariableSeriesKneeDataset, require_b20_contract
from .b27_1_pathology_routing import (
    B27_1_ROUTE_PARAMETER_COUNT,
    B27_1_ROUTING_VERSION,
    b27_1_model_spec,
    build_b27_1_model,
)
from .budget import RuntimeBudget
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime

B27_1_TRAIN_VERSION = "1.0.0"
B27_1_EXPERIMENT = "B27_1_collinearity_safe_pathology_routing"
B27_1_VARIANT = "b20_plus_zero_init_plane_paired_sequence_attention_bias_fixed_e2_v1"
B27_1_FIXED_EPOCHS = 2
B27_1_EXPECTED_STUDIES = 3120
B27_1_EXPECTED_SERIES = 17475
B27_1_EXPECTED_CELLS = 14123
B27_1_EXPECTED_POS = 6871
B27_1_EXPECTED_NEG = 7252
B27_1_EXPECTED_PAIR_11 = 7459
B27_1_EXPECTED_PAIR_22 = 10016
B27_1_EXPECTED_OTHER_PAIRS = 0
B27_1_MAX_HARD_HOURS = 8.25
B27_1_MIN_RESERVE_MINUTES = 30.0


def paired_sequence_surface_audit(variable_index, study_uids) -> dict:
    counts: dict[str, int] = {}
    total = 0
    discordant = 0
    unknown = 0
    for uid in study_uids:
        for row in variable_index[str(uid)]:
            fluid = int(row["fluid_id"])
            fat = int(row["fat_id"])
            key = f"{fluid},{fat}"
            counts[key] = counts.get(key, 0) + 1
            total += 1
            if fluid == 0 or fat == 0:
                unknown += 1
            elif fluid != fat:
                discordant += 1
    return {
        "series": total,
        "pair_counts": dict(sorted(counts.items())),
        "fluid_equals_fat": total - discordant,
        "fluid_not_equals_fat": discordant,
        "unknown_pair_series": unknown,
        "fraction_identical": float((total - discordant) / max(total, 1)),
    }


def _routing_norms(model) -> dict:
    params = torch.cat(
        [
            model.route_plane_bias.detach().float().reshape(-1).cpu(),
            model.route_sequence_bias.detach().float().reshape(-1).cpu(),
        ]
    )
    return {
        "parameter_count": int(params.numel()),
        "max_abs": float(params.abs().max().item()) if params.numel() else 0.0,
        "mean_abs": float(params.abs().mean().item()) if params.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(params).item()) if params.numel() else 0.0,
    }


def train_b27_1(
    config: dict,
    *,
    b6_root: str | Path,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    out_root: str | Path = "runs/b27_1_pathology_routing",
) -> Path:
    validate_competition_config(config, purpose="train")
    crop_policy = require_b20_contract(config)
    report_payload = load_b16_report_encoder(report_ssl_checkpoint)

    seed = int(config.get("seed", 2026))
    seed_everything(seed + 19_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())

    max_hours = min(
        float(config.get("runtime_budget_hours", 8.5)), B27_1_MAX_HARD_HOURS
    )
    reserve_minutes = max(
        float(config.get("runtime_reserve_minutes", 10.0)),
        B27_1_MIN_RESERVE_MINUTES,
    )
    budget = RuntimeBudget(max_hours=max_hours, reserve_minutes=reserve_minutes)

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv")).copy()
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    study_uids, targets, weights, supervision = prepare_b7_supervision(train, b6_frame)
    study_uids = [str(uid) for uid in study_uids]

    if len(study_uids) != B27_1_EXPECTED_STUDIES:
        raise RuntimeError(f"B27.1 requires {B27_1_EXPECTED_STUDIES} studies")
    active_cells = int((weights > 0).sum())
    positive_cells = int(((weights > 0) & (targets > 0.5)).sum())
    negative_cells = int(((weights > 0) & (targets < 0.5)).sum())
    if (active_cells, positive_cells, negative_cells) != (
        B27_1_EXPECTED_CELLS,
        B27_1_EXPECTED_POS,
        B27_1_EXPECTED_NEG,
    ):
        raise RuntimeError("B27.1 B6 supervision surface drifted from B20")

    gold_by_uid = torch.as_tensor(gold_mask(train).to_numpy(dtype=bool), dtype=torch.bool)
    uid_to_row = {uid: i for i, uid in enumerate(train["StudyInstanceUID"].tolist())}
    if any(bool(gold_by_uid[uid_to_row[uid]]) for uid in study_uids):
        raise RuntimeError("expert-gold study unexpectedly entered B27.1 gradients")

    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B27.1 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, study_uids)
    if series_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B27.1 reconstructed series surface does not match frozen SHA-256")
    if int(series_summary.get("eligible_recognized_plane_series", -1)) != B27_1_EXPECTED_SERIES:
        raise ValueError("B27.1 requires the exact 17,475-series B20 surface")
    if series_summary.get("viability_passed") is not True:
        raise ValueError("B27.1 reconstructed series surface failed viability")

    pair_audit = paired_sequence_surface_audit(variable_index, study_uids)
    pair_11 = int(pair_audit["pair_counts"].get("1,1", 0))
    pair_22 = int(pair_audit["pair_counts"].get("2,2", 0))
    other = int(pair_audit["series"] - pair_11 - pair_22)
    if (
        pair_11 != B27_1_EXPECTED_PAIR_11
        or pair_22 != B27_1_EXPECTED_PAIR_22
        or other != B27_1_EXPECTED_OTHER_PAIRS
        or int(pair_audit["fluid_not_equals_fat"]) != 0
        or int(pair_audit["unknown_pair_series"]) != 0
    ):
        raise RuntimeError(
            "B27.1 paired-metadata surface changed; the pre-outcome collinearity contract no longer holds"
        )

    batch_size = int(config.get("b7_batch_size", 2))
    expected_batches = int(math.ceil(len(study_uids) / batch_size))
    if expected_batches != 1560:
        raise RuntimeError("B27.1 exact B20 surface must yield 1,560 batches per epoch")
    target_multiplier = target_balance_multipliers(weights)

    train_ds = CropFocusedVariableSeriesKneeDataset(
        study_uids,
        variable_index,
        make_b7_dataset_config(config, root, train=True),
        targets=targets,
        weights=weights,
        train=True,
        crop_focus_policy=crop_policy,
    )
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=seed + 19_100_000),
    )

    spec = b27_1_model_spec(config, normalize_input=True)
    model = build_b27_1_model(spec, pretrained_weights=False)
    model.encoder.load_state_dict(report_payload["encoder"], strict=True)
    freeze_encoder(model)
    model.gradient_checkpointing = False
    encoder_sha_initial = encoder_state_sha256(model.encoder)

    route_initial = _routing_norms(model)
    if route_initial["parameter_count"] != B27_1_ROUTE_PARAMETER_COUNT:
        raise RuntimeError("B27.1 routing parameter count changed")
    if route_initial["max_abs"] != 0.0:
        raise RuntimeError("B27.1 routing must be exactly zero-initialised")

    model = model.to(runtime.device)
    head_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("encoder.") and p.requires_grad
    ]
    if not head_params or any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("B27.1 frozen-encoder contract failed")

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": float(config.get("b7_head_lr", B17_HEAD_LR))}],
        weight_decay=float(config.get("b7_weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=B18_CANDIDATE_EPOCHS,
        eta_min=float(config.get("b7_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)
    clip = float(config.get("b7_grad_clip", 1.0))

    print(
        f"[B27.1] studies={len(study_uids)} series={B27_1_EXPECTED_SERIES} "
        f"cells={active_cells} | routing_params={B27_1_ROUTE_PARAMETER_COUNT} "
        f"| fixed E{B27_1_FIXED_EPOCHS}"
    )
    print(
        f"[B27.1] paired sequence: (1,1)={pair_11} (2,2)={pair_22} "
        "discordant=0 unknown=0"
    )
    print(
        f"[B27.1] hard_runtime={max_hours:.2f} h | reserve={reserve_minutes:.0f} min "
        "| expert labels=not read"
    )

    history: list[dict] = []
    for epoch in range(1, B27_1_FIXED_EPOCHS + 1):
        epoch_started = time.monotonic()
        model.train()
        model.encoder.eval()
        loss_sum = 0.0
        steps = seen_studies = seen_series = seen_cells = seen_pos = seen_neg = 0

        for batch in loader:
            if not budget.can_start(120.0):
                raise RuntimeError("B27.1 runtime budget expired before complete fixed-E2")
            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            meta = batch["series_meta"].to(runtime.device, non_blocking=True)
            target = batch["target"].to(runtime.device, non_blocking=True)
            weight = batch["weight"].to(runtime.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                logits = model(volumes, present, meta)
                loss = target_balanced_weak_bce(logits, target, weight, multiplier_t)
            scaler.scale(loss).backward()
            if any(p.grad is not None for p in model.encoder.parameters()):
                raise RuntimeError("B27.1 detected an encoder gradient")
            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(head_params, clip)
            scaler.step(optimizer)
            scaler.update()

            active = weight > 0
            loss_sum += float(loss.item())
            steps += 1
            seen_studies += int(volumes.shape[0])
            seen_series += int(present.sum().item())
            seen_cells += int(active.sum().item())
            seen_pos += int((active & (target > 0.5)).sum().item())
            seen_neg += int((active & (target < 0.5)).sum().item())

        scheduler.step()
        epoch_seconds = time.monotonic() - epoch_started
        if encoder_state_sha256(model.encoder) != encoder_sha_initial:
            raise RuntimeError("B27.1 encoder changed despite freezing")
        full = (
            steps == expected_batches
            and seen_studies == B27_1_EXPECTED_STUDIES
            and seen_series == B27_1_EXPECTED_SERIES
            and seen_cells == B27_1_EXPECTED_CELLS
            and seen_pos == B27_1_EXPECTED_POS
            and seen_neg == B27_1_EXPECTED_NEG
        )
        if not full:
            raise RuntimeError(f"B27.1 epoch {epoch} did not complete the frozen surface")

        row = {
            "epoch": epoch,
            "loss": loss_sum / max(steps, 1),
            "head_lr": float(optimizer.param_groups[0]["lr"]),
            "batches": steps,
            "studies": seen_studies,
            "series_instances": seen_series,
            "supervision_cells": seen_cells,
            "positive_cells": seen_pos,
            "negative_cells": seen_neg,
            "epoch_seconds": round(float(epoch_seconds), 1),
            "full_coverage": True,
            "routing": _routing_norms(model),
        }
        history.append(row)
        print(
            f"[B27.1] E{epoch} loss={row['loss']:.10f} | {epoch_seconds/60:.1f} min "
            f"| route|max|={row['routing']['max_abs']:.6f}"
        )

    encoder_sha_final = encoder_state_sha256(model.encoder)
    if encoder_sha_final != encoder_sha_initial:
        raise RuntimeError("B27.1 encoder fingerprint changed")

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "b27_1_model.pt"
    payload = {
        "experiment": B27_1_EXPERIMENT,
        "variant": B27_1_VARIANT,
        "b27_1_training_version": B27_1_TRAIN_VERSION,
        "b27_1_routing_version": B27_1_ROUTING_VERSION,
        "fixed_endpoint": True,
        "selected_epoch": B27_1_FIXED_EPOCHS,
        "completed_epochs": B27_1_FIXED_EPOCHS,
        "scheduler_horizon": B18_CANDIDATE_EPOCHS,
        "model_spec": spec,
        "model_state": model.state_dict(),
        "config": config,
        "crop_focus_enabled": True,
        "crop_focus_policy": crop_policy,
        "b19_cosine_mask_used": False,
        "initialization": B16_REPORT_SSL_VARIANT,
        "initialization_experiment": B16_REPORT_SSL_EXPERIMENT,
        "initialization_objective": B16_REPORT_SSL_OBJECTIVE,
        "input_normalization": B13_INPUT_NORMALIZATION,
        "encoder_frozen": True,
        "encoder_sha256_initial": encoder_sha_initial,
        "encoder_sha256_final": encoder_sha_final,
        "study_uids": list(study_uids),
        "history": history,
        "routing_initial": route_initial,
        "routing_final": model.routing_tables(),
        "paired_metadata_audit": pair_audit,
        "supervision": supervision,
        "b6_policy": b6_policy,
        "b6_audit": b6_audit,
        "metadata_repair": metadata_stats,
        "series_summary": series_summary,
        "b6_root": str(Path(b6_root).resolve()),
        "series_policy": str(Path(series_policy_path).resolve()),
        "report_ssl_checkpoint": str(Path(report_ssl_checkpoint).resolve()),
        "gold_studies_used_in_gradient": 0,
        "expert_checkpoint_selection": False,
        "evaluation_status": "not yet evaluated; fixed-E2 checkpoint",
        "runtime_budget": budget.to_dict(),
        "pre_outcome_correction": (
            "B27.1 was frozen after observing training-metadata/parameter collinearity and before "
            "any B27 or B27.1 reused-expert performance evaluation."
        ),
    }
    torch.save(payload, checkpoint)

    audit = {
        "experiment": B27_1_EXPERIMENT,
        "variant": B27_1_VARIANT,
        "checkpoint": str(checkpoint),
        "fixed_epoch": B27_1_FIXED_EPOCHS,
        "training_studies": B27_1_EXPECTED_STUDIES,
        "series_instances": B27_1_EXPECTED_SERIES,
        "supervision_cells": B27_1_EXPECTED_CELLS,
        "positive_cells": B27_1_EXPECTED_POS,
        "negative_cells": B27_1_EXPECTED_NEG,
        "routing_parameter_count": B27_1_ROUTE_PARAMETER_COUNT,
        "paired_metadata_audit": pair_audit,
        "routing_initial": route_initial,
        "routing_final": model.routing_tables(),
        "encoder_sha256": encoder_sha_final,
        "history": history,
        "runtime_budget": budget.to_dict(),
        "gold_used": False,
        "pre_outcome_correction": True,
    }
    (out / "training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out / "routing_biases.json").write_text(
        json.dumps(model.routing_tables(), indent=2), encoding="utf-8"
    )
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint)
    return checkpoint


def load_b27_1_checkpoint(path: str | Path, *, device: torch.device | str = "cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != B27_1_EXPERIMENT or payload.get("variant") != B27_1_VARIANT:
        raise ValueError("not a B27.1 fixed-E2 pathology-routing checkpoint")
    if payload.get("fixed_endpoint") is not True or int(payload.get("completed_epochs", -1)) != B27_1_FIXED_EPOCHS:
        raise ValueError("B27.1 checkpoint must be the complete fixed-E2 endpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B27.1 checkpoint does not certify zero gold gradients")
    if payload.get("expert_checkpoint_selection") is not False:
        raise ValueError("B27.1 checkpoint unexpectedly used expert checkpoint selection")
    initial_sha = str(payload.get("encoder_sha256_initial", ""))
    final_sha = str(payload.get("encoder_sha256_final", ""))
    if not initial_sha or initial_sha != final_sha:
        raise ValueError("B27.1 encoder fingerprint changed")
    pair_audit = payload.get("paired_metadata_audit") or {}
    if int(pair_audit.get("fluid_not_equals_fat", -1)) != 0:
        raise ValueError("B27.1 checkpoint paired metadata contract changed")

    model = build_b27_1_model(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    freeze_encoder(model)
    if encoder_state_sha256(model.encoder) != initial_sha:
        raise ValueError("B27.1 reconstructed encoder fingerprint mismatch")
    return model.to(device), payload


def main() -> None:
    parser = argparse.ArgumentParser("B27.1 collinearity-safe pathology routing")
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--report-ssl-checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b27_1_pathology_routing")
    args = parser.parse_args()

    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    train_b27_1(
        config,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
