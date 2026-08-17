"""Matched PV2-B29 control for the predeclared B34 training-scaffold test.

B29 is the essential no-scaffold control because B34 has the same inference-time
complementary-query function as B29 but uses B31's local-context branch only while
training.  A B34-vs-B29 comparison on frozen PV2 therefore tests the optimization
value of the training scaffold directly.
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
    _read_config, load_frozen_b6_export, make_b7_dataset_config,
    prepare_b7_supervision, seed_everything, target_balance_multipliers,
    target_balanced_weak_bce,
)
from .b12_training import _load_series_policy
from .b12_variable_series import audit_variable_series_surface, build_variable_series_index, collate_variable_series
from .b13_training import B13_INPUT_NORMALIZATION, B13_SERIES_SIGNATURE
from .b16_report_ssl import (
    B16_REPORT_SSL_EXPERIMENT, B16_REPORT_SSL_OBJECTIVE, B16_REPORT_SSL_VARIANT,
    load_b16_report_encoder,
)
from .b17_training import B17_HEAD_LR, encoder_state_sha256, freeze_encoder
from .b18_fisher_selection import B18_CANDIDATE_EPOCHS
from .b20_crop_focus import CropFocusedVariableSeriesKneeDataset, require_b20_contract
from .b29_complementary_series_pool import B29_EXPECTED_NEW_PARAMETERS, b29_model_spec, build_b29_model
from .budget import RuntimeBudget
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .policy import validate_competition_config
from .prospective_weak_v1 import validate_prospective_weak_v1_manifest
from .prospective_weak_v2 import (
    PV2_LOCKED_PV1_VALIDATION_STUDIES, PV2_PARENT_PV1_SPLIT_SHA256,
    PV2_TRAIN_STUDIES, PV2_VALIDATION_STUDIES, PV2_VERSION,
    validate_prospective_weak_v2_manifest,
)
from .prospective_weak_v2_training import (
    PV2_CONSTRUCTION_SEED_OFFSET, PV2_FIXED_EPOCHS, PV2_LOADER_SEED_OFFSET,
    PV2_MAX_HARD_HOURS, PV2_MIN_RESERVE_MINUTES, PV2_POST_CONSTRUCTION_SEED_OFFSET,
    PV2_TRAIN_EXPERIMENT,
)
from .runtime import autocast, make_scaler, resolve_runtime

PV2_B29_ROLE = "matched_no_training_scaffold_control_for_b34"


def _subset(full_uids, targets, weights, subset_uids):
    row = {str(uid): i for i, uid in enumerate(full_uids)}
    idx = np.asarray([row[str(uid)] for uid in subset_uids], dtype=np.int64)
    return targets[idx], weights[idx]


def train_pv2_b29(
    config: dict,
    *,
    split_manifest_path: str | Path,
    parent_pv1_manifest_path: str | Path,
    b6_root: str | Path,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    out_root: str | Path = "runs/prospective_weak_v2/b29",
) -> Path:
    validate_competition_config(config, purpose="train")
    crop_policy = require_b20_contract(config)
    report_payload = load_b16_report_encoder(report_ssl_checkpoint)
    seed = int(config.get("seed", 2026))
    construction_seed = seed + PV2_CONSTRUCTION_SEED_OFFSET
    loader_seed = seed + PV2_LOADER_SEED_OFFSET
    post_seed = seed + PV2_POST_CONSTRUCTION_SEED_OFFSET
    seed_everything(construction_seed)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    budget = RuntimeBudget(
        max_hours=min(float(config.get("runtime_budget_hours", 8.5)), PV2_MAX_HARD_HOURS),
        reserve_minutes=max(float(config.get("runtime_reserve_minutes", 10.0)), PV2_MIN_RESERVE_MINUTES),
    )

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    full_uids, full_targets, full_weights, supervision = prepare_b7_supervision(train, b6_frame)
    full_uids = [str(x) for x in full_uids]
    parent = json.loads(Path(parent_pv1_manifest_path).read_text(encoding="utf-8"))
    validate_prospective_weak_v1_manifest(parent, full_uids)
    if str(parent.get("split_sha256", "")) != PV2_PARENT_PV1_SPLIT_SHA256:
        raise ValueError("PV2-B29 requires the exact frozen parent PV1 split")
    split = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    validate_prospective_weak_v2_manifest(split, parent, full_uids)

    train_uids = [str(x) for x in split["training_uids"]]
    if len(train_uids) != PV2_TRAIN_STUDIES:
        raise RuntimeError("PV2-B29 training count changed")
    if len(split["validation_uids"]) != PV2_VALIDATION_STUDIES:
        raise RuntimeError("PV2-B29 validation count changed")
    if len(split["locked_parent_pv1_validation_uids"]) != PV2_LOCKED_PV1_VALIDATION_STUDIES:
        raise RuntimeError("PV2-B29 locked PV1 validation count changed")

    targets, weights = _subset(full_uids, full_targets, full_weights, train_uids)
    audit = split["post_assignment_supervision_audit"]["training"]
    active_cells = int((weights > 0).sum())
    positive_cells = int(((weights > 0) & (targets > 0.5)).sum())
    negative_cells = int(((weights > 0) & (targets < 0.5)).sum())
    if (active_cells, positive_cells, negative_cells) != (
        int(audit["usable_cells"]), int(audit["positive_cells"]), int(audit["negative_cells"])
    ):
        raise RuntimeError("PV2-B29 supervision does not match frozen manifest")

    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("PV2-B29 requires frozen B12/B13 series policy")
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    full_summary, _ = audit_variable_series_surface(series, full_uids)
    if full_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("PV2-B29 full series surface changed")
    train_index = build_variable_series_index(series, train_uids)
    expected_series = int(sum(len(train_index.get(uid, [])) for uid in train_uids))
    if expected_series <= 0 or any(len(train_index.get(uid, [])) == 0 for uid in train_uids):
        raise RuntimeError("PV2-B29 contains study with no eligible MRI series")

    batch_size = int(config.get("b7_batch_size", 2))
    expected_batches = int(math.ceil(PV2_TRAIN_STUDIES / batch_size))
    multiplier = target_balance_multipliers(weights)
    ds = CropFocusedVariableSeriesKneeDataset(
        train_uids, train_index, make_b7_dataset_config(config, root, train=True),
        targets=targets, weights=weights, train=True, crop_focus_policy=crop_policy,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=False,
        collate_fn=collate_variable_series, **runtime.loader_kwargs(seed=loader_seed),
    )

    spec = b29_model_spec(config, normalize_input=True)
    model = build_b29_model(spec, pretrained_weights=False)
    model.encoder.load_state_dict(report_payload["encoder"], strict=True)
    freeze_encoder(model)
    model.gradient_checkpointing = False
    encoder_sha_initial = encoder_state_sha256(model.encoder)
    model = model.to(runtime.device)
    seed_everything(post_seed)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": float(config.get("b7_head_lr", B17_HEAD_LR))}],
        weight_decay=float(config.get("b7_weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=B18_CANDIDATE_EPOCHS, eta_min=float(config.get("b7_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    multiplier_t = torch.from_numpy(multiplier).to(runtime.device)
    clip = float(config.get("b7_grad_clip", 1.0))

    history = []
    for epoch in range(1, PV2_FIXED_EPOCHS + 1):
        started = time.monotonic()
        model.train(); model.encoder.eval()
        loss_sum = 0.0
        steps = seen_studies = seen_series = seen_cells = seen_pos = seen_neg = 0
        query_seen = gate_seen = False
        for batch in loader:
            if not budget.can_start(120.0):
                raise RuntimeError("PV2-B29 runtime budget expired")
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
                raise RuntimeError("PV2-B29 detected encoder gradient")
            qg = model.complementary_query.grad
            gg = model.complementary_gate.grad
            if qg is None or gg is None or not torch.isfinite(qg).all() or not torch.isfinite(gg).all():
                raise RuntimeError("PV2-B29 candidate gradient missing/nonfinite")
            query_seen = query_seen or bool(torch.count_nonzero(qg).item() > 0)
            gate_seen = gate_seen or bool(torch.count_nonzero(gg).item() > 0)
            if clip > 0:
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(head_params, clip)
            scaler.step(optimizer); scaler.update()
            active = weight > 0
            loss_sum += float(loss.item()); steps += 1
            seen_studies += int(volumes.shape[0]); seen_series += int(present.sum().item())
            seen_cells += int(active.sum().item())
            seen_pos += int((active & (target > 0.5)).sum().item())
            seen_neg += int((active & (target < 0.5)).sum().item())
        scheduler.step()
        full = (
            steps == expected_batches and seen_studies == PV2_TRAIN_STUDIES and
            seen_series == expected_series and seen_cells == active_cells and
            seen_pos == positive_cells and seen_neg == negative_cells
        )
        if not full or not query_seen or not gate_seen:
            raise RuntimeError("PV2-B29 fixed training contract failed")
        if encoder_state_sha256(model.encoder) != encoder_sha_initial:
            raise RuntimeError("PV2-B29 encoder changed")
        history.append({
            "epoch": epoch, "loss": loss_sum / steps, "batches": steps,
            "studies": seen_studies, "series_instances": seen_series,
            "supervision_cells": seen_cells, "positive_cells": seen_pos,
            "negative_cells": seen_neg, "query_gradient_seen": query_seen,
            "gate_gradient_seen": gate_seen, "candidate_state": model.complementary_state(),
            "epoch_seconds": float(time.monotonic() - started), "full_coverage": True,
        })
        print(f"[PV2/b29] E{epoch} loss={history[-1]['loss']:.10f} time={history[-1]['epoch_seconds']/60:.1f} min")

    encoder_sha_final = encoder_state_sha256(model.encoder)
    out = Path(out_root); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "model.pt"
    payload = {
        "experiment": PV2_TRAIN_EXPERIMENT,
        "validation_framework": PV2_VERSION,
        "model_name": "b29",
        "analysis_role": PV2_B29_ROLE,
        "fixed_endpoint": True,
        "completed_epochs": PV2_FIXED_EPOCHS,
        "selected_epoch": PV2_FIXED_EPOCHS,
        "validation_used_for_checkpoint_selection": False,
        "validation_studies_used_in_gradient": 0,
        "locked_parent_pv1_validation_studies_used_in_gradient": 0,
        "expert_labels_used": False,
        "split_sha256": split["split_sha256"],
        "parent_pv1_split_sha256": split["parent_pv1_split_sha256"],
        "training_uid_sha256": split["training_uid_sha256"],
        "validation_uid_sha256": split["validation_uid_sha256"],
        "training_studies": PV2_TRAIN_STUDIES,
        "validation_studies": PV2_VALIDATION_STUDIES,
        "locked_parent_pv1_validation_studies": PV2_LOCKED_PV1_VALIDATION_STUDIES,
        "training_series": expected_series,
        "training_supervision_cells": active_cells,
        "training_positive_cells": positive_cells,
        "training_negative_cells": negative_cells,
        "candidate_added_parameters": B29_EXPECTED_NEW_PARAMETERS,
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
        "candidate_state_final": model.complementary_state(),
        "supervision_source": supervision,
        "b6_policy": b6_policy,
        "b6_audit": b6_audit,
        "metadata_repair": metadata_stats,
        "runtime_budget": budget.to_dict(),
        "pv2_limitation": split["exposure_note"],
    }
    torch.save(payload, checkpoint)
    audit_payload = {k: v for k, v in payload.items() if k not in {"model_state", "config", "b6_policy", "b6_audit"}}
    (out / "training_audit.json").write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint)
    return checkpoint


def load_pv2_b29_checkpoint(path: str | Path, *, expected_split_sha256: str | None = None, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != PV2_TRAIN_EXPERIMENT or payload.get("validation_framework") != PV2_VERSION:
        raise ValueError("not a PV2-B29 checkpoint")
    if payload.get("model_name") != "b29" or payload.get("analysis_role") != PV2_B29_ROLE:
        raise ValueError("PV2-B29 role/model mismatch")
    if payload.get("fixed_endpoint") is not True or int(payload.get("completed_epochs", -1)) != PV2_FIXED_EPOCHS:
        raise ValueError("PV2-B29 must be complete fixed-E2")
    if expected_split_sha256 is not None and str(payload.get("split_sha256", "")) != str(expected_split_sha256):
        raise ValueError("PV2-B29 split mismatch")
    if int(payload.get("validation_studies_used_in_gradient", -1)) != 0:
        raise ValueError("PV2-B29 validation entered gradients")
    if int(payload.get("locked_parent_pv1_validation_studies_used_in_gradient", -1)) != 0:
        raise ValueError("PV2-B29 reused locked PV1 validation")
    initial = str(payload.get("encoder_sha256_initial", "")); final = str(payload.get("encoder_sha256_final", ""))
    if not initial or initial != final:
        raise ValueError("PV2-B29 encoder fingerprint changed")
    model = build_b29_model(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    freeze_encoder(model)
    if encoder_state_sha256(model.encoder) != initial:
        raise ValueError("PV2-B29 reconstructed encoder mismatch")
    return model.to(device), payload


def main() -> None:
    ap = argparse.ArgumentParser("Train matched PV2-B29 no-scaffold control")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--parent-pv1-manifest", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--series-policy", required=True)
    ap.add_argument("--report-ssl-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/prospective_weak_v2/b29")
    args = ap.parse_args()
    config = dict(_read_config(args.config)); config["data_root"] = str(Path(args.data_root).resolve())
    train_pv2_b29(
        config, split_manifest_path=args.split_manifest,
        parent_pv1_manifest_path=args.parent_pv1_manifest, b6_root=args.b6_root,
        series_policy_path=args.series_policy, report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
