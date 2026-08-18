"""Phase 9 v2 matched B34 training with the frozen PV2 validation holdout.

Both arms train on the exact same 3,850 report-only studies after removing the
frozen 499-study PV2 validation partition from both arms before any Phase-9 v2
gradient.  All other MRI-side choices are matched.  The only treatment variable
is original B6 versus frozen Phase-8 supervision on the retained training UIDs.
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
    _read_config,
    make_b7_dataset_config,
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
from .b34_training_only_context_scaffold import B34_EXPECTED_NEW_PARAMETERS, b34_model_spec, build_b34_model
from .budget import RuntimeBudget
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .phase9_matched_supervision_training import PHASE9_ARMS
from .phase9_v2_supervision import (
    PHASE9_V2_BATCHES_BATCH2,
    PHASE9_V2_HOLDOUT_STUDIES,
    PHASE9_V2_TRAIN_SERIES,
    PHASE9_V2_TRAIN_STUDIES,
    PHASE9_V2_VERSION,
    prepare_phase9_v2_arm_supervision,
)
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime

PHASE9_V2_EXPERIMENT = "phase9_v2_matched_b34_b6_vs_phase8_pv2_holdout"
PHASE9_V2_FIXED_EPOCHS = 2
PHASE9_V2_CONSTRUCTION_SEED_OFFSET = 40_000_000
PHASE9_V2_LOADER_SEED_OFFSET = 40_100_000
PHASE9_V2_POST_CONSTRUCTION_SEED_OFFSET = 40_200_000
PHASE9_V2_MAX_HARD_HOURS = 10.0
PHASE9_V2_MIN_RESERVE_MINUTES = 30.0


def _candidate_gradient_names() -> tuple[str, ...]:
    return ("complementary_query", "complementary_gate", "local_context.weight")


def train_phase9_v2_arm(
    config: dict,
    *,
    arm: str,
    b6_root: str | Path,
    phase8_root: str | Path,
    parent_pv1_manifest_path: str | Path,
    pv2_manifest_path: str | Path,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    out_root: str | Path = "runs/phase9_matched_supervision_v2",
) -> Path:
    arm = str(arm).lower()
    if arm not in PHASE9_ARMS:
        raise ValueError(f"Phase 9 v2 arm must be one of {PHASE9_ARMS}")
    validate_competition_config(config, purpose="train")
    crop_policy = require_b20_contract(config)
    report_payload = load_b16_report_encoder(report_ssl_checkpoint)

    seed = int(config.get("seed", 2026))
    construction_seed = seed + PHASE9_V2_CONSTRUCTION_SEED_OFFSET
    loader_seed = seed + PHASE9_V2_LOADER_SEED_OFFSET
    post_seed = seed + PHASE9_V2_POST_CONSTRUCTION_SEED_OFFSET
    seed_everything(construction_seed)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    budget = RuntimeBudget(
        max_hours=min(float(config.get("runtime_budget_hours", 10.0)), PHASE9_V2_MAX_HARD_HOURS),
        reserve_minutes=max(float(config.get("runtime_reserve_minutes", 10.0)), PHASE9_V2_MIN_RESERVE_MINUTES),
    )

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    if len(train) != 4407:
        raise ValueError("Phase 9 v2 requires the complete 4,407-study training release")

    uids, targets, weights, supervision, supervision_source, holdout = prepare_phase9_v2_arm_supervision(
        train,
        arm=arm,
        b6_root=b6_root,
        phase8_root=phase8_root,
        parent_pv1_manifest_path=parent_pv1_manifest_path,
        pv2_manifest_path=pv2_manifest_path,
    )
    if len(uids) != PHASE9_V2_TRAIN_STUDIES:
        raise RuntimeError("Phase 9 v2 training population changed")
    if int(supervision["held_out_pv2_studies"]) != PHASE9_V2_HOLDOUT_STUDIES:
        raise RuntimeError("Phase 9 v2 holdout population changed")

    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("Phase 9 v2 requires the frozen B12/B13 all-series policy")
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    expected_series = int(series_summary.get("eligible_recognized_plane_series", -1))
    if expected_series != PHASE9_V2_TRAIN_SERIES:
        raise ValueError(f"Phase 9 v2 requires {PHASE9_V2_TRAIN_SERIES} training MRI series; got {expected_series}")
    if series_summary.get("viability_passed") is not True:
        raise ValueError("Phase 9 v2 training MRI surface no longer passes viability")
    if any(len(variable_index.get(uid, [])) == 0 for uid in uids):
        raise RuntimeError("Phase 9 v2 training population contains a study with zero eligible MRI series")

    batch_size = int(config.get("b7_batch_size", 2))
    if batch_size != 2:
        raise ValueError("Phase 9 v2 freezes b7_batch_size=2")
    expected_batches = int(math.ceil(PHASE9_V2_TRAIN_STUDIES / batch_size))
    if expected_batches != PHASE9_V2_BATCHES_BATCH2:
        raise RuntimeError("Phase 9 v2 expected-batch contract changed")

    target_multiplier = target_balance_multipliers(weights)
    ds = CropFocusedVariableSeriesKneeDataset(
        uids,
        variable_index,
        make_b7_dataset_config(config, root, train=True),
        targets=targets,
        weights=weights,
        train=True,
        crop_focus_policy=crop_policy,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=loader_seed),
    )

    spec = b34_model_spec(config, normalize_input=True)
    model = build_b34_model(spec, pretrained_weights=False)
    model.encoder.load_state_dict(report_payload["encoder"], strict=True)
    freeze_encoder(model)
    model.gradient_checkpointing = False
    encoder_sha_initial = encoder_state_sha256(model.encoder)
    model = model.to(runtime.device)

    # Equalize model/dropout/augmentation RNG after construction in both arms.
    seed_everything(post_seed)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    if not head_params or any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("Phase 9 v2 frozen-encoder contract failed")
    named = dict(model.named_parameters())
    gradient_names = _candidate_gradient_names()
    for name in gradient_names:
        if name not in named:
            raise RuntimeError(f"Phase 9 v2 B34 missing trainable parameter {name}")

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

    active_cells = int((weights > 0).sum())
    positive_cells = int(((weights > 0) & (targets > 0.5)).sum())
    negative_cells = int(((weights > 0) & (targets < 0.5)).sum())
    print(
        f"[Phase9v2/{arm}] train={len(uids)} holdout={PHASE9_V2_HOLDOUT_STUDIES} "
        f"series={expected_series} cells={active_cells} positive={positive_cells} negative={negative_cells} fixed_E2"
    )
    print(
        f"[Phase9v2/{arm}] pv2={holdout['pv2_split_sha256'][:12]} B34_added={B34_EXPECTED_NEW_PARAMETERS} "
        f"construction_seed={construction_seed} loader_seed={loader_seed} post_seed={post_seed}"
    )

    history: list[dict] = []
    for epoch in range(1, PHASE9_V2_FIXED_EPOCHS + 1):
        started = time.monotonic()
        model.train()
        model.encoder.eval()
        loss_sum = 0.0
        steps = seen_studies = seen_series = seen_cells = seen_pos = seen_neg = 0
        gradient_seen = {name: False for name in gradient_names}

        for batch in loader:
            if not budget.can_start(120.0):
                raise RuntimeError("Phase 9 v2 runtime budget expired before fixed E2 completed")
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
                raise RuntimeError("Phase 9 v2 detected an encoder gradient")
            if bool((weight > 0).any().item()):
                for name in gradient_names:
                    grad = named[name].grad
                    if grad is None or not torch.isfinite(grad).all():
                        raise RuntimeError(f"Phase 9 v2 nonfinite/missing B34 gradient for {name}")
                    gradient_seen[name] = gradient_seen[name] or bool(torch.count_nonzero(grad).item() > 0)

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
        if encoder_state_sha256(model.encoder) != encoder_sha_initial:
            raise RuntimeError("Phase 9 v2 encoder fingerprint changed")
        full = (
            steps == expected_batches
            and seen_studies == PHASE9_V2_TRAIN_STUDIES
            and seen_series == expected_series
            and seen_cells == active_cells
            and seen_pos == positive_cells
            and seen_neg == negative_cells
        )
        if not full:
            raise RuntimeError(f"Phase 9 v2 {arm} E{epoch} did not cover its frozen surface")
        missing_gradient = [name for name, seen in gradient_seen.items() if not seen]
        if missing_gradient:
            raise RuntimeError(f"Phase 9 v2 {arm} B34 parameters never coupled: {missing_gradient}")

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
            "candidate_gradient_seen": gradient_seen,
            "candidate_state": model.b34_state(),
            "epoch_seconds": float(time.monotonic() - started),
            "full_coverage": True,
        }
        history.append(row)
        print(f"[Phase9v2/{arm}] E{epoch} loss={row['loss']:.10f} time={row['epoch_seconds']/60:.1f} min")

    encoder_sha_final = encoder_state_sha256(model.encoder)
    if encoder_sha_final != encoder_sha_initial:
        raise RuntimeError("Phase 9 v2 encoder changed after training")

    out = Path(out_root) / arm
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "model.pt"
    payload = {
        "experiment": PHASE9_V2_EXPERIMENT,
        "phase9_version": PHASE9_V2_VERSION,
        "arm": arm,
        "architecture": "B34 training-only context scaffold with exact eval bypass",
        "fixed_endpoint": True,
        "completed_epochs": PHASE9_V2_FIXED_EPOCHS,
        "selected_epoch": PHASE9_V2_FIXED_EPOCHS,
        "validation_used_for_checkpoint_selection": False,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_used": False,
        "training_studies": PHASE9_V2_TRAIN_STUDIES,
        "pv2_holdout_studies": PHASE9_V2_HOLDOUT_STUDIES,
        "pv2_holdout_used_in_gradient": False,
        "training_uid_sha256": supervision["training_uid_sha256"],
        "pv2_split_sha256": holdout["pv2_split_sha256"],
        "pv2_validation_uid_sha256": holdout["pv2_validation_uid_sha256"],
        "parent_pv1_split_sha256": holdout["parent_pv1_split_sha256"],
        "training_series": expected_series,
        "training_supervision_cells": active_cells,
        "training_positive_cells": positive_cells,
        "training_negative_cells": negative_cells,
        "zero_weight_studies_retained": int((weights.sum(axis=1) == 0).sum()),
        "target_balance_rule": "frozen B7 mean target mass / target mass; recomputed mechanically from each arm's retained supervision",
        "target_balance_multipliers": [float(x) for x in target_multiplier],
        "candidate_added_parameters": int(B34_EXPECTED_NEW_PARAMETERS),
        "construction_seed": construction_seed,
        "loader_seed": loader_seed,
        "post_construction_training_seed": post_seed,
        "stochastic_path_matched_after_model_construction": True,
        "model_spec": spec,
        "model_state": model.state_dict(),
        "config": config,
        "crop_focus_policy": crop_policy,
        "initialization": B16_REPORT_SSL_VARIANT,
        "initialization_experiment": B16_REPORT_SSL_EXPERIMENT,
        "initialization_objective": B16_REPORT_SSL_OBJECTIVE,
        "input_normalization": B13_INPUT_NORMALIZATION,
        "encoder_frozen": True,
        "encoder_sha256_initial": encoder_sha_initial,
        "encoder_sha256_final": encoder_sha_final,
        "history": history,
        "b34_state_final": model.b34_state(),
        "b34_inference_contract": "model.eval() bypasses all local-context scaffold parameters exactly",
        "supervision": supervision,
        "supervision_source": supervision_source,
        "series_policy_signature": B13_SERIES_SIGNATURE,
        "training_series_surface": series_summary,
        "metadata_repair": metadata_stats,
        "runtime_budget": budget.to_dict(),
        "pv2_limitation": holdout["limitation"],
        "governance": (
            "Phase-9 v2 supervision experiment only. PV2 is weak-label/historically exposed and is not clinical validation. "
            "Do not use target-wise outcomes to filter rescue cells, retune B34, or promote a model without hidden/external evidence."
        ),
    }
    torch.save(payload, checkpoint)
    audit = {k: v for k, v in payload.items() if k not in {"model_state", "config"}}
    (out / "training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint)
    return checkpoint


def load_phase9_v2_checkpoint(
    path: str | Path,
    *,
    expected_arm: str | None = None,
    expected_pv2_split_sha256: str | None = None,
    device: torch.device | str = "cpu",
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != PHASE9_V2_EXPERIMENT or payload.get("phase9_version") != PHASE9_V2_VERSION:
        raise ValueError("not a Phase-9 v2 PV2-holdout checkpoint")
    arm = str(payload.get("arm", ""))
    if arm not in PHASE9_ARMS:
        raise ValueError("Phase-9 v2 checkpoint has invalid arm")
    if expected_arm is not None and arm != str(expected_arm):
        raise ValueError(f"Phase-9 v2 checkpoint arm mismatch: expected {expected_arm}, got {arm}")
    if payload.get("fixed_endpoint") is not True or int(payload.get("completed_epochs", -1)) != PHASE9_V2_FIXED_EPOCHS:
        raise ValueError("Phase-9 v2 checkpoint must be complete fixed-E2")
    if payload.get("validation_used_for_checkpoint_selection") is not False:
        raise ValueError("Phase-9 v2 holdout unexpectedly used for checkpoint selection")
    if int(payload.get("pv2_holdout_used_in_gradient", -1)) != 0:
        raise ValueError("Phase-9 v2 PV2 holdout unexpectedly entered gradients")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0 or bool(payload.get("gold_labels_used", True)):
        raise ValueError("Phase-9 v2 checkpoint unexpectedly used gold labels")
    if int(payload.get("training_studies", -1)) != PHASE9_V2_TRAIN_STUDIES:
        raise ValueError("Phase-9 v2 training population changed")
    if int(payload.get("training_series", -1)) != PHASE9_V2_TRAIN_SERIES:
        raise ValueError("Phase-9 v2 training series exposure changed")
    if int(payload.get("pv2_holdout_studies", -1)) != PHASE9_V2_HOLDOUT_STUDIES:
        raise ValueError("Phase-9 v2 holdout count changed")
    if expected_pv2_split_sha256 is not None and str(payload.get("pv2_split_sha256", "")) != str(expected_pv2_split_sha256):
        raise ValueError("Phase-9 v2 PV2 split fingerprint mismatch")
    if payload.get("stochastic_path_matched_after_model_construction") is not True:
        raise ValueError("Phase-9 v2 checkpoint lacks matched RNG reset")
    initial_sha = str(payload.get("encoder_sha256_initial", ""))
    final_sha = str(payload.get("encoder_sha256_final", ""))
    if not initial_sha or initial_sha != final_sha:
        raise ValueError("Phase-9 v2 checkpoint encoder fingerprint changed")

    model = build_b34_model(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    freeze_encoder(model)
    if encoder_state_sha256(model.encoder) != initial_sha:
        raise ValueError("Phase-9 v2 reconstructed encoder fingerprint mismatch")
    return model.to(device), payload


def main() -> None:
    ap = argparse.ArgumentParser("Train one Phase-9 v2 matched-supervision B34 arm")
    ap.add_argument("--arm", choices=PHASE9_ARMS, required=True)
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--phase8-root", required=True)
    ap.add_argument("--parent-pv1-manifest", required=True)
    ap.add_argument("--pv2-manifest", required=True)
    ap.add_argument("--series-policy", required=True)
    ap.add_argument("--report-ssl-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/phase9_matched_supervision_v2")
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    train_phase9_v2_arm(
        config,
        arm=args.arm,
        b6_root=args.b6_root,
        phase8_root=args.phase8_root,
        parent_pv1_manifest_path=args.parent_pv1_manifest,
        pv2_manifest_path=args.pv2_manifest,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
