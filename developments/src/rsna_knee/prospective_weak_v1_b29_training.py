"""Matched fixed-E2 B29 training for the PV1 mechanistic addendum.

This module deliberately does NOT modify the original predeclared PV1 matched
control set (B20/B31/B33).  The original PV1 result was already observed before
this addendum was defined.  B29 is nevertheless a legitimate mechanistic
follow-up because the B29 architecture was frozen before PV1 existed.

The addendum retrains that unchanged B29 architecture on the exact same 2,496
PV1 training studies, with the same frozen B16 encoder, B20 post-resize crop,
optimizer, scheduler horizon, construction seed, loader seed, post-construction
training seed, augmentation policy, and fixed E2 endpoint used by the original
PV1 controls.  The 624 PV1 validation studies are never loaded during training.

The resulting checkpoint is for mechanism decomposition only.  It must not be
retroactively described as one of the original prospective PV1 controls.
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
    load_frozen_b6_export,
    make_b7_dataset_config,
    prepare_b7_supervision,
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
from .b29_complementary_series_pool import (
    B29_EXPECTED_GATE_PARAMETERS,
    B29_EXPECTED_NEW_PARAMETERS,
    B29_EXPECTED_QUERY_PARAMETERS,
    B29_RESIDUAL_VERSION,
    b29_model_spec,
    build_b29_model,
)
from .budget import RuntimeBudget
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .policy import validate_competition_config
from .prospective_weak_v1 import PV1_TRAIN_STUDIES, PV1_VERSION, validate_prospective_weak_v1_manifest
from .prospective_weak_v1_training import (
    PV1_FIXED_EPOCHS,
    PV1_MAX_HARD_HOURS,
    PV1_MIN_RESERVE_MINUTES,
    PV1_POST_CONSTRUCTION_SEED_OFFSET,
)
from .runtime import autocast, make_scaler, resolve_runtime

PV1_B29_ADDENDUM_TRAIN_VERSION = "1.0.0"
PV1_B29_ADDENDUM_EXPERIMENT = "prospective_weak_v1_b29_frozen_architecture_mechanistic_addendum_fixed_e2"
PV1_B29_ADDENDUM_ROLE = "post_pv1_frozen_architecture_mechanistic_addendum"
PV1_B29_ORIGINAL_PV1_RESULT_ALREADY_OBSERVED = True
PV1_B29_ARCHITECTURE_FROZEN_BEFORE_PV1 = True
PV1_B29_EXPECTED_TRAIN_SERIES = 13931
PV1_B29_EXPECTED_TRAIN_CELLS = 11303
PV1_B29_EXPECTED_TRAIN_POS = 5559
PV1_B29_EXPECTED_TRAIN_NEG = 5744


def _load_split(path: str | Path, active_uids: list[str]) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_prospective_weak_v1_manifest(manifest, active_uids)


def _subset_supervision(
    full_uids: list[str], targets: np.ndarray, weights: np.ndarray, subset_uids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    row = {str(uid): i for i, uid in enumerate(full_uids)}
    try:
        idx = np.asarray([row[str(uid)] for uid in subset_uids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"PV1 split UID missing from B6 supervision: {exc}") from exc
    return targets[idx], weights[idx]


def train_pv1_b29_addendum(
    config: dict,
    *,
    split_manifest_path: str | Path,
    b6_root: str | Path,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    out_root: str | Path = "runs/prospective_weak_v1/b29_addendum/train",
) -> Path:
    validate_competition_config(config, purpose="train")
    crop_policy = require_b20_contract(config)
    report_payload = load_b16_report_encoder(report_ssl_checkpoint)

    seed = int(config.get("seed", 2026))
    construction_seed = seed + 19_000_000
    loader_seed = seed + 19_100_000
    post_construction_seed = seed + PV1_POST_CONSTRUCTION_SEED_OFFSET
    seed_everything(construction_seed)

    runtime = resolve_runtime(config)
    print(runtime.describe())
    budget = RuntimeBudget(
        max_hours=min(float(config.get("runtime_budget_hours", 8.5)), PV1_MAX_HARD_HOURS),
        reserve_minutes=max(float(config.get("runtime_reserve_minutes", 10.0)), PV1_MIN_RESERVE_MINUTES),
    )

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    full_uids, full_targets, full_weights, supervision = prepare_b7_supervision(train, b6_frame)
    full_uids = [str(x) for x in full_uids]

    split = _load_split(split_manifest_path, full_uids)
    train_uids = [str(x) for x in split["training_uids"]]
    if len(train_uids) != PV1_TRAIN_STUDIES:
        raise RuntimeError("PV1 B29 addendum training study count changed")

    targets, weights = _subset_supervision(full_uids, full_targets, full_weights, train_uids)
    split_audit = split["post_assignment_supervision_audit"]["training"]
    active_cells = int((weights > 0).sum())
    positive_cells = int(((weights > 0) & (targets > 0.5)).sum())
    negative_cells = int(((weights > 0) & (targets < 0.5)).sum())
    expected_counts = (
        int(split_audit["usable_cells"]),
        int(split_audit["positive_cells"]),
        int(split_audit["negative_cells"]),
    )
    if (active_cells, positive_cells, negative_cells) != expected_counts:
        raise RuntimeError("PV1 B29 supervision does not match frozen split manifest")
    if (active_cells, positive_cells, negative_cells) != (
        PV1_B29_EXPECTED_TRAIN_CELLS,
        PV1_B29_EXPECTED_TRAIN_POS,
        PV1_B29_EXPECTED_TRAIN_NEG,
    ):
        raise RuntimeError("PV1 B29 expected training-cell counts changed")

    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("PV1 B29 requires the frozen B12/B13 series policy")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    full_series_summary, _ = audit_variable_series_surface(series, full_uids)
    if full_series_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("PV1 B29 full series surface no longer matches frozen B13 signature")
    if int(full_series_summary.get("eligible_recognized_plane_series", -1)) != 17475:
        raise ValueError("PV1 B29 requires the historical 17,475-series source surface")

    train_index = build_variable_series_index(series, train_uids)
    expected_series = int(sum(len(train_index.get(uid, [])) for uid in train_uids))
    if expected_series != PV1_B29_EXPECTED_TRAIN_SERIES:
        raise RuntimeError(
            f"PV1 B29 requires {PV1_B29_EXPECTED_TRAIN_SERIES} training series; got {expected_series}"
        )
    if any(len(train_index.get(uid, [])) == 0 for uid in train_uids):
        raise RuntimeError("PV1 B29 training subset contains a study with no eligible MRI series")

    batch_size = int(config.get("b7_batch_size", 2))
    expected_batches = int(math.ceil(PV1_TRAIN_STUDIES / batch_size))
    target_multiplier = target_balance_multipliers(weights)
    train_ds = CropFocusedVariableSeriesKneeDataset(
        train_uids,
        train_index,
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
        **runtime.loader_kwargs(seed=loader_seed),
    )

    spec = b29_model_spec(config, normalize_input=True)
    model = build_b29_model(spec, pretrained_weights=False)
    model.encoder.load_state_dict(report_payload["encoder"], strict=True)
    freeze_encoder(model)
    model.gradient_checkpointing = False
    encoder_sha_initial = encoder_state_sha256(model.encoder)

    state_initial = model.complementary_state()
    if state_initial["new_parameter_count"] != B29_EXPECTED_NEW_PARAMETERS:
        raise RuntimeError("PV1 B29 new parameter count changed")
    if state_initial["query_parameter_count"] != B29_EXPECTED_QUERY_PARAMETERS:
        raise RuntimeError("PV1 B29 query parameter count changed")
    if state_initial["gate_parameter_count"] != B29_EXPECTED_GATE_PARAMETERS:
        raise RuntimeError("PV1 B29 gate parameter count changed")
    if state_initial["gate_raw_max_abs"] != 0.0 or state_initial["gate_effective_max_abs"] != 0.0:
        raise RuntimeError("PV1 B29 complementary gate must start at exactly zero")

    model = model.to(runtime.device)

    # Match the stochastic training path used by the original PV1 controls.
    # B29 adds construction-time random draws for its query, so reset every
    # global training RNG after construction and device placement.
    seed_everything(post_construction_seed)

    head_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("encoder.") and p.requires_grad
    ]
    if not head_params or any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("PV1 B29 frozen-encoder contract failed")

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
        f"[PV1-B29 addendum] train={len(train_uids)} validation={len(split['validation_uids'])} "
        f"series/train={expected_series} cells/train={active_cells} fixed_E2 "
        f"split={split['split_sha256'][:12]}"
    )
    print(
        f"[PV1-B29 addendum] new_params={B29_EXPECTED_NEW_PARAMETERS}; "
        f"construction_seed={construction_seed}; loader_seed={loader_seed}; "
        f"post_construction_seed={post_construction_seed}"
    )
    print(
        "[PV1-B29 addendum] governance: original B20/B31/B33 PV1 result was already observed; "
        "this frozen B29 run is mechanism decomposition only"
    )

    history: list[dict] = []
    for epoch in range(1, PV1_FIXED_EPOCHS + 1):
        started = time.monotonic()
        model.train()
        model.encoder.eval()
        loss_sum = 0.0
        steps = seen_studies = seen_series = seen_cells = seen_pos = seen_neg = 0
        gate_grad_seen = False
        query_grad_seen = False

        for batch in loader:
            if not budget.can_start(120.0):
                raise RuntimeError("PV1 B29 runtime budget expired before fixed E2 completed")
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
                raise RuntimeError("PV1 B29 detected encoder gradient")
            gate_grad = model.complementary_gate.grad
            query_grad = model.complementary_query.grad
            if gate_grad is None or not torch.isfinite(gate_grad).all():
                raise RuntimeError("PV1 B29 gate gradient missing/nonfinite")
            if query_grad is None or not torch.isfinite(query_grad).all():
                raise RuntimeError("PV1 B29 query gradient missing/nonfinite")
            gate_grad_seen = gate_grad_seen or bool(torch.count_nonzero(gate_grad).item() > 0)
            query_grad_seen = query_grad_seen or bool(torch.count_nonzero(query_grad).item() > 0)

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
            raise RuntimeError("PV1 B29 encoder fingerprint changed")

        full = (
            steps == expected_batches
            and seen_studies == PV1_TRAIN_STUDIES
            and seen_series == expected_series
            and seen_cells == active_cells
            and seen_pos == positive_cells
            and seen_neg == negative_cells
        )
        if not full:
            raise RuntimeError(f"PV1 B29 E{epoch} did not cover the frozen training partition")
        if not gate_grad_seen:
            raise RuntimeError(f"PV1 B29 E{epoch} never produced a nonzero gate gradient")
        if not query_grad_seen:
            raise RuntimeError(f"PV1 B29 E{epoch} never coupled the query to the loss")

        state = model.complementary_state()
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
            "candidate_gradient_seen": {
                "complementary_query": query_grad_seen,
                "complementary_gate": gate_grad_seen,
            },
            "candidate_state": state,
            "epoch_seconds": float(time.monotonic() - started),
            "full_coverage": True,
        }
        history.append(row)
        print(
            f"[PV1-B29 addendum] E{epoch} loss={row['loss']:.10f} "
            f"time={row['epoch_seconds']/60:.1f} min "
            f"gate|max|={state['gate_effective_max_abs']:.6f}"
        )

    encoder_sha_final = encoder_state_sha256(model.encoder)
    if encoder_sha_final != encoder_sha_initial:
        raise RuntimeError("PV1 B29 encoder changed after training")

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "model.pt"
    state_final = model.complementary_state()
    payload = {
        "experiment": PV1_B29_ADDENDUM_EXPERIMENT,
        "training_version": PV1_B29_ADDENDUM_TRAIN_VERSION,
        "validation_framework": PV1_VERSION,
        "analysis_role": PV1_B29_ADDENDUM_ROLE,
        "original_pv1_result_already_observed": PV1_B29_ORIGINAL_PV1_RESULT_ALREADY_OBSERVED,
        "architecture_frozen_before_pv1": PV1_B29_ARCHITECTURE_FROZEN_BEFORE_PV1,
        "model_name": "b29",
        "b29_residual_version": B29_RESIDUAL_VERSION,
        "fixed_endpoint": True,
        "completed_epochs": PV1_FIXED_EPOCHS,
        "selected_epoch": PV1_FIXED_EPOCHS,
        "validation_used_for_checkpoint_selection": False,
        "validation_studies_used_in_gradient": 0,
        "expert_studies_used_in_gradient": 0,
        "expert_labels_used": False,
        "split_sha256": split["split_sha256"],
        "split_manifest_path": str(Path(split_manifest_path).resolve()),
        "training_uid_sha256": split["training_uid_sha256"],
        "validation_uid_sha256": split["validation_uid_sha256"],
        "training_studies": PV1_TRAIN_STUDIES,
        "validation_studies": len(split["validation_uids"]),
        "training_series": expected_series,
        "training_supervision_cells": active_cells,
        "training_positive_cells": positive_cells,
        "training_negative_cells": negative_cells,
        "candidate_added_parameters": B29_EXPECTED_NEW_PARAMETERS,
        "construction_seed": construction_seed,
        "loader_seed": loader_seed,
        "post_construction_training_seed": post_construction_seed,
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
        "candidate_state_initial": state_initial,
        "candidate_state_final": state_final,
        "supervision_source": supervision,
        "b6_policy": b6_policy,
        "b6_audit": b6_audit,
        "metadata_repair": metadata_stats,
        "runtime_budget": budget.to_dict(),
        "governance": (
            "B29 predates PV1 but this matched retraining was requested only after the original B20/B31/B33 "
            "PV1 result was observed. Treat the addendum as post-result global mechanism decomposition, not "
            "as a fourth original prospective control and not as independent clinical validation."
        ),
    }
    torch.save(payload, checkpoint)

    audit = {k: v for k, v in payload.items() if k not in {"model_state", "config", "b6_policy", "b6_audit"}}
    (out / "training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint)
    return checkpoint


def load_pv1_b29_addendum_checkpoint(
    path: str | Path,
    *,
    expected_split_sha256: str | None = None,
    device: torch.device | str = "cpu",
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != PV1_B29_ADDENDUM_EXPERIMENT:
        raise ValueError("not a PV1 B29 mechanistic-addendum checkpoint")
    if payload.get("validation_framework") != PV1_VERSION:
        raise ValueError("PV1 B29 validation framework mismatch")
    if payload.get("analysis_role") != PV1_B29_ADDENDUM_ROLE:
        raise ValueError("PV1 B29 checkpoint analysis-role mismatch")
    if payload.get("original_pv1_result_already_observed") is not True:
        raise ValueError("PV1 B29 checkpoint must record post-result addendum status")
    if payload.get("architecture_frozen_before_pv1") is not True:
        raise ValueError("PV1 B29 checkpoint must certify pre-existing B29 architecture")
    if payload.get("fixed_endpoint") is not True or int(payload.get("completed_epochs", -1)) != PV1_FIXED_EPOCHS:
        raise ValueError("PV1 B29 checkpoint must be complete fixed-E2")
    if payload.get("validation_used_for_checkpoint_selection") is not False:
        raise ValueError("PV1 B29 validation was unexpectedly used for checkpoint selection")
    if int(payload.get("validation_studies_used_in_gradient", -1)) != 0:
        raise ValueError("PV1 B29 validation unexpectedly entered gradients")
    if bool(payload.get("expert_labels_used", True)):
        raise ValueError("PV1 B29 checkpoint unexpectedly used expert labels")
    if payload.get("stochastic_path_matched_after_model_construction") is not True:
        raise ValueError("PV1 B29 checkpoint lacks matched post-construction RNG guardrail")

    split_sha = str(payload.get("split_sha256", ""))
    if expected_split_sha256 is not None and split_sha != str(expected_split_sha256):
        raise ValueError("PV1 B29 checkpoint split fingerprint mismatch")
    initial_sha = str(payload.get("encoder_sha256_initial", ""))
    final_sha = str(payload.get("encoder_sha256_final", ""))
    if not initial_sha or initial_sha != final_sha:
        raise ValueError("PV1 B29 checkpoint encoder fingerprint changed")

    final_state = payload.get("candidate_state_final") or {}
    if int(final_state.get("new_parameter_count", -1)) != B29_EXPECTED_NEW_PARAMETERS:
        raise ValueError("PV1 B29 checkpoint new-parameter contract changed")

    model = build_b29_model(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    freeze_encoder(model)
    if encoder_state_sha256(model.encoder) != initial_sha:
        raise ValueError("PV1 B29 reconstructed encoder fingerprint mismatch")
    return model.to(device), payload


def main() -> None:
    ap = argparse.ArgumentParser("Train frozen B29 on the PV1 mechanistic-addendum training surface")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--series-policy", required=True)
    ap.add_argument("--report-ssl-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/prospective_weak_v1/b29_addendum/train")
    args = ap.parse_args()

    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    train_pv1_b29_addendum(
        config,
        split_manifest_path=args.split_manifest,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
