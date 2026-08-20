"""Phase 9 matched MRI training: original B6 versus frozen Phase-8 supervision.

Both arms use the exact same B34 architecture, B16 encoder, B20 crop, all-series
MRI exposure, augmentation, optimizer, scheduler horizon, random seeds, and fixed
E2 endpoint.  All 4,349 report-only studies remain in both dataloaders.  The only
experimental variable is the supervision table and the target-balance multiplier
that follows mechanically from that table under the frozen B7 loss rule.
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
from .b12_variable_series import audit_variable_series_surface, build_variable_series_index, collate_variable_series
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
from .dinov3_encoder import attach_dinov3_encoder
from .encoder_finetune import (
    DEFAULT_ENCODER_LR_SCALE,
    parameter_groups,
    split_parameters,
    unfreeze_encoder_tail,
)
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .label_confidence import rescale_label_confidence
from .phase9_supervision import PHASE9_VERSION, REPORT_ONLY_STUDIES, load_phase9_arm_supervision
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime

PHASE9_EXPERIMENT = "phase9_matched_b34_b6_vs_phase8_supervision"
PHASE9_ARMS = ("control", "candidate", "llm_fill")
PHASE9_ENCODER_SOURCES = ("report-aligned", "dinov3")
PHASE9_FIXED_EPOCHS = 2
PHASE9_EXPECTED_REPORT_ONLY_SERIES = 24035
PHASE9_EXPECTED_BATCHES_BATCH2 = 2175
PHASE9_CONSTRUCTION_SEED_OFFSET = 40_000_000
PHASE9_LOADER_SEED_OFFSET = 40_100_000
PHASE9_POST_CONSTRUCTION_SEED_OFFSET = 40_200_000
PHASE9_MAX_HARD_HOURS = 10.0
PHASE9_MIN_RESERVE_MINUTES = 30.0


def _candidate_gradient_names() -> tuple[str, ...]:
    return ("complementary_query", "complementary_gate", "local_context.weight")


def train_phase9_arm(
    config: dict,
    *,
    arm: str,
    b6_root: str | Path,
    phase8_root: str | Path,
    llm_fill_root: str | Path | None = None,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    out_root: str | Path = "runs/phase9_matched_supervision",
    out_dirname: str | None = None,
    encoder_source: str = "report-aligned",
    dinov3_variant: str = "tiny",
    encoder_trainable_stages: int = 0,
    encoder_lr_scale: float = DEFAULT_ENCODER_LR_SCALE,
) -> Path:
    """Train one matched supervision arm to the fixed endpoint.

    `out_dirname` names the subdirectory of `out_root` the run is written to,
    defaulting to the arm name, which is the historical layout.  It affects
    only where the files land: the recorded arm, the supervision that arm
    selects and every audited quantity are unchanged.
    """
    arm = str(arm).lower()
    if arm not in PHASE9_ARMS:
        raise ValueError(f"Phase 9 arm must be one of {PHASE9_ARMS}")
    directory = str(out_dirname) if out_dirname else arm
    if "/" in directory or "\\" in directory or directory in (".", ".."):
        raise ValueError("out_dirname must be a single directory name")
    if encoder_source not in PHASE9_ENCODER_SOURCES:
        known = ", ".join(sorted(PHASE9_ENCODER_SOURCES))
        raise ValueError(f"encoder_source must be one of: {known}")
    validate_competition_config(config, purpose="train")
    crop_policy = require_b20_contract(config)
    report_payload = (
        load_b16_report_encoder(report_ssl_checkpoint)
        if encoder_source == "report-aligned"
        else None
    )

    seed = int(config.get("seed", 2026))
    construction_seed = seed + PHASE9_CONSTRUCTION_SEED_OFFSET
    loader_seed = seed + PHASE9_LOADER_SEED_OFFSET
    post_seed = seed + PHASE9_POST_CONSTRUCTION_SEED_OFFSET
    seed_everything(construction_seed)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    budget = RuntimeBudget(
        max_hours=min(float(config.get("runtime_budget_hours", 10.0)), PHASE9_MAX_HARD_HOURS),
        reserve_minutes=max(float(config.get("runtime_reserve_minutes", 10.0)), PHASE9_MIN_RESERVE_MINUTES),
    )

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    if len(train) != 4407:
        raise ValueError("Phase 9 requires the complete 4,407-study training release")
    uids, targets, weights, supervision, supervision_source = load_phase9_arm_supervision(
        train,
        arm=arm,
        b6_root=b6_root,
        phase8_root=phase8_root,
        llm_fill_root=llm_fill_root,
    )
    if len(uids) != REPORT_ONLY_STUDIES:
        raise RuntimeError("Phase 9 must retain all 4,349 report-only studies in each arm")

    # The historical B12/B13 policy remains the eligibility/routing contract.
    # Its stored signature covers the original 3,120 B6-active surface; the
    # Phase-9 exposure is deliberately broader and is checked by its exact count.
    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("Phase 9 requires the frozen B12/B13 all-series policy")
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    expected_series = int(series_summary.get("eligible_recognized_plane_series", -1))
    if expected_series != PHASE9_EXPECTED_REPORT_ONLY_SERIES:
        raise ValueError(
            f"Phase 9 requires {PHASE9_EXPECTED_REPORT_ONLY_SERIES} report-only MRI series; got {expected_series}"
        )
    if series_summary.get("viability_passed") is not True:
        raise ValueError("Phase 9 report-only MRI surface no longer passes viability")
    if any(len(variable_index.get(uid, [])) == 0 for uid in uids):
        raise RuntimeError("Phase 9 report-only population contains a study with zero eligible MRI series")

    batch_size = int(config.get("b7_batch_size", 2))
    if batch_size != 2:
        raise ValueError("Phase 9 freezes b7_batch_size=2 for matched stochastic exposure")
    expected_batches = int(math.ceil(REPORT_ONLY_STUDIES / batch_size))
    if expected_batches != PHASE9_EXPECTED_BATCHES_BATCH2:
        raise RuntimeError("Phase 9 expected-batch contract changed")

    # The exported supervision carries the frozen B7 targets, 0.85 and 0.05,
    # baked in when the labels were written. Restating them here is what makes
    # them adjustable at all: nothing downstream reads the config for them, so
    # a config-only setting would be silently ignored.
    targets, confidence = rescale_label_confidence(targets, weights, config)
    if confidence["changed"]:
        print(
            f"[labels] positive cells train towards {confidence['positive_target']:g} "
            f"(was {confidence['exported_positive_target']:g}), negative towards "
            f"{confidence['negative_target']:g} (was {confidence['exported_negative_target']:g})"
        )

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
    if encoder_source == "report-aligned":
        model.encoder.load_state_dict(report_payload["encoder"], strict=True)
        encoder_description = {"source": "report-aligned", "checkpoint": str(report_ssl_checkpoint)}
    else:
        replacement = attach_dinov3_encoder(model, variant=dinov3_variant, pretrained_weights=True)
        encoder_description = {"source": "dinov3", **replacement.describe()}
        spec["dinov3_variant"] = dinov3_variant
    spec["encoder_source"] = encoder_source
    freeze_encoder(model)
    finetune = unfreeze_encoder_tail(model, encoder_trainable_stages)
    spec.update(finetune)
    if encoder_trainable_stages:
        print(
            f"[encoder] freeing {finetune['encoder_trainable_parameters']:,} parameters "
            f"in the last {encoder_trainable_stages} block(s) at "
            f"{encoder_lr_scale:g}x the head learning rate"
        )
    model.gradient_checkpointing = False
    encoder_sha_initial = encoder_state_sha256(model.encoder)
    model = model.to(runtime.device)

    # Ensure both arms see identical dropout/augmentation RNG after construction.
    seed_everything(post_seed)

    head_params, encoder_params = split_parameters(model)
    if not head_params:
        raise RuntimeError("Phase 9 found no trainable head parameters")
    if encoder_trainable_stages == 0 and encoder_params:
        raise RuntimeError("Phase 9 frozen-encoder contract failed")
    if encoder_trainable_stages and not encoder_params:
        raise RuntimeError("Phase 9 was asked to fine-tune the encoder but freed nothing")
    trainable_params = head_params + encoder_params
    named = dict(model.named_parameters())
    gradient_names = _candidate_gradient_names()
    for name in gradient_names:
        if name not in named:
            raise RuntimeError(f"Phase 9 B34 missing trainable parameter {name}")

    optimizer = torch.optim.AdamW(
        parameter_groups(
            model,
            head_lr=float(config.get("b7_head_lr", B17_HEAD_LR)),
            encoder_lr_scale=encoder_lr_scale,
        ),
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
        f"[Phase9/{arm}] studies={len(uids)} series={expected_series} cells={active_cells} "
        f"positive={positive_cells} negative={negative_cells} fixed_E2"
    )
    print(
        f"[Phase9/{arm}] B34 added_parameters={B34_EXPECTED_NEW_PARAMETERS} "
        f"construction_seed={construction_seed} loader_seed={loader_seed} post_seed={post_seed}"
    )

    run_started = time.monotonic()
    total_batches = len(loader) * PHASE9_FIXED_EPOCHS
    print(f"[Phase9/{arm}] starting {PHASE9_FIXED_EPOCHS} epochs, {total_batches} batches total")

    history: list[dict] = []
    for epoch in range(1, PHASE9_FIXED_EPOCHS + 1):
        started = time.monotonic()
        model.train()
        model.encoder.eval()
        loss_sum = 0.0
        steps = seen_studies = seen_series = seen_cells = seen_pos = seen_neg = 0
        gradient_seen = {name: False for name in gradient_names}

        for batch in loader:
            if not budget.can_start(120.0):
                raise RuntimeError("Phase 9 runtime budget expired before fixed E2 completed")
            if steps and steps % 100 == 0:
                done_this_epoch = steps / len(loader)
                per_batch = (time.monotonic() - started) / steps
                left = per_batch * (
                    (len(loader) - steps) + len(loader) * (PHASE9_FIXED_EPOCHS - epoch)
                )
                print(
                    f"[Phase9/{arm}] E{epoch} {steps}/{len(loader)} "
                    f"({done_this_epoch * 100:4.1f}%) loss={loss_sum / steps:.4f} "
                    f"elapsed={(time.monotonic() - run_started) / 60:.1f} min "
                    f"remaining~{left / 60:.1f} min",
                    flush=True,
                )
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
            # A frozen encoder must receive nothing; a deliberately freed tail
            # must receive gradient only where it was freed.
            leaked = any(
                p.grad is not None
                for p in model.encoder.parameters()
                if not p.requires_grad
            )
            if leaked:
                raise RuntimeError("Phase 9 detected a gradient on a frozen encoder parameter")
            # Zero-weight batches are legal in the control arm; require coupling
            # only across the full epoch, not in every individual batch.
            if bool((weight > 0).any().item()):
                for name in gradient_names:
                    grad = named[name].grad
                    if grad is None or not torch.isfinite(grad).all():
                        raise RuntimeError(f"Phase 9 nonfinite/missing B34 gradient for {name}")
                    gradient_seen[name] = gradient_seen[name] or bool(torch.count_nonzero(grad).item() > 0)

            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, clip)
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
        if encoder_trainable_stages == 0 and encoder_state_sha256(model.encoder) != encoder_sha_initial:
            raise RuntimeError("Phase 9 encoder fingerprint changed despite being frozen")
        full = (
            steps == expected_batches
            and seen_studies == REPORT_ONLY_STUDIES
            and seen_series == expected_series
            and seen_cells == active_cells
            and seen_pos == positive_cells
            and seen_neg == negative_cells
        )
        if not full:
            raise RuntimeError(f"Phase 9 {arm} E{epoch} did not cover its frozen surface")
        missing_gradient = [name for name, seen in gradient_seen.items() if not seen]
        if missing_gradient:
            raise RuntimeError(f"Phase 9 {arm} B34 parameters never coupled: {missing_gradient}")

        state = model.b34_state()
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
            "candidate_state": state,
            "epoch_seconds": float(time.monotonic() - started),
            "full_coverage": True,
        }
        history.append(row)
        print(f"[Phase9/{arm}] E{epoch} loss={row['loss']:.10f} time={row['epoch_seconds']/60:.1f} min")

    total_seconds = float(time.monotonic() - run_started)
    print(
        f"[Phase9/{arm}] training finished in {total_seconds / 60:.1f} min "
        f"({total_seconds / 3600:.2f} h)"
    )

    encoder_sha_final = encoder_state_sha256(model.encoder)
    encoder_moved = encoder_sha_final != encoder_sha_initial
    if encoder_trainable_stages == 0 and encoder_moved:
        raise RuntimeError("Phase 9 encoder changed after training despite being frozen")
    if encoder_trainable_stages and not encoder_moved:
        raise RuntimeError(
            "Phase 9 fine-tuned the encoder but its weights are unchanged; "
            "the parameters were not reaching the optimizer"
        )

    out = Path(out_root) / directory
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "model.pt"
    payload = {
        "experiment": PHASE9_EXPERIMENT,
        "phase9_version": PHASE9_VERSION,
        "arm": arm,
        "architecture": "B34 training-only context scaffold with exact eval bypass",
        "fixed_endpoint": True,
        "completed_epochs": PHASE9_FIXED_EPOCHS,
        "selected_epoch": PHASE9_FIXED_EPOCHS,
        "validation_used_for_checkpoint_selection": False,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_used": False,
        "report_only_studies_exposed": REPORT_ONLY_STUDIES,
        "training_series": expected_series,
        "training_supervision_cells": active_cells,
        "training_positive_cells": positive_cells,
        "training_negative_cells": negative_cells,
        "zero_weight_studies_retained": int((weights.sum(axis=1) == 0).sum()),
        "label_confidence": confidence,
        "target_balance_rule": "frozen B7 mean target mass / target mass; recomputed mechanically from each arm's supervision",
        "target_balance_multipliers": [float(x) for x in target_multiplier],
        "candidate_added_parameters": int(B34_EXPECTED_NEW_PARAMETERS),
        # Non-zero means the seed was varied on purpose to build an ensemble
        # member, so this run is not stochastically matched to the others.
        "ensemble_member": int(config.get("ensemble_member", 0) or 0),
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
        "encoder_source": encoder_source,
        "encoder": encoder_description,
        "training_seconds": total_seconds,
        "training_minutes": total_seconds / 60.0,
        "encoder_trainable_stages": int(encoder_trainable_stages),
        "encoder_lr_scale": float(encoder_lr_scale),
        "encoder_finetune": finetune,
        # True for every run before encoder fine-tuning existed, so it used to be
        # written as a constant. It is a claim about this run, not a house rule.
        "encoder_frozen": not encoder_moved,
        "encoder_sha256_initial": encoder_sha_initial,
        "encoder_sha256_final": encoder_sha_final,
        "history": history,
        "b34_state_final": model.b34_state(),
        "b34_inference_contract": "model.eval() bypasses all local-context scaffold parameters exactly",
        "supervision": supervision,
        "supervision_source": supervision_source,
        "series_policy_signature": B13_SERIES_SIGNATURE,
        "report_only_series_surface": series_summary,
        "metadata_repair": metadata_stats,
        "runtime_budget": budget.to_dict(),
        "governance": (
            "Matched Phase-9 supervision experiment only. Do not use target-wise results to filter translated cells, "
            "change B34, or promote a model without independent hidden/external evidence."
        ),
    }
    torch.save(payload, checkpoint)
    audit = {k: v for k, v in payload.items() if k not in {"model_state", "config"}}
    (out / "training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint)
    return checkpoint


def load_phase9_checkpoint(
    path: str | Path,
    *,
    expected_arm: str | None = None,
    device: torch.device | str = "cpu",
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != PHASE9_EXPERIMENT or payload.get("phase9_version") != PHASE9_VERSION:
        raise ValueError("not a Phase-9 matched-supervision checkpoint")
    arm = str(payload.get("arm", ""))
    if arm not in PHASE9_ARMS:
        raise ValueError("Phase-9 checkpoint has invalid arm")
    if expected_arm is not None and arm != str(expected_arm):
        raise ValueError(f"Phase-9 checkpoint arm mismatch: expected {expected_arm}, got {arm}")
    if payload.get("fixed_endpoint") is not True or int(payload.get("completed_epochs", -1)) != PHASE9_FIXED_EPOCHS:
        raise ValueError("Phase-9 checkpoint must be complete fixed-E2")
    if payload.get("validation_used_for_checkpoint_selection") is not False:
        raise ValueError("Phase-9 validation unexpectedly used for checkpoint selection")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0 or bool(payload.get("gold_labels_used", True)):
        raise ValueError("Phase-9 checkpoint unexpectedly used gold labels")
    if int(payload.get("report_only_studies_exposed", -1)) != REPORT_ONLY_STUDIES:
        raise ValueError("Phase-9 MRI exposure population changed")
    if int(payload.get("training_series", -1)) != PHASE9_EXPECTED_REPORT_ONLY_SERIES:
        raise ValueError("Phase-9 MRI series exposure changed")
    if payload.get("stochastic_path_matched_after_model_construction") is not True:
        raise ValueError("Phase-9 checkpoint lacks matched RNG reset")
    initial_sha = str(payload.get("encoder_sha256_initial", ""))
    final_sha = str(payload.get("encoder_sha256_final", ""))
    if not initial_sha or not final_sha:
        raise ValueError("Phase-9 checkpoint is missing an encoder fingerprint")
    if int(payload.get("encoder_trainable_stages", 0)) == 0 and initial_sha != final_sha:
        raise ValueError("Phase-9 checkpoint encoder fingerprint changed")

    model = build_b34_model(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    freeze_encoder(model)
    # The state dict holds the encoder as it ended, which equals the initial
    # fingerprint only when the encoder was frozen throughout.
    if encoder_state_sha256(model.encoder) != final_sha:
        raise ValueError("Phase-9 reconstructed encoder fingerprint mismatch")
    return model.to(device), payload


def main() -> None:
    ap = argparse.ArgumentParser("Train one Phase-9 matched-supervision B34 arm")
    ap.add_argument("--arm", choices=PHASE9_ARMS, required=True)
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--phase8-root", required=True)
    ap.add_argument("--series-policy", required=True)
    ap.add_argument("--report-ssl-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/phase9_matched_supervision")
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    train_phase9_arm(
        config,
        arm=args.arm,
        b6_root=args.b6_root,
        phase8_root=args.phase8_root,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
