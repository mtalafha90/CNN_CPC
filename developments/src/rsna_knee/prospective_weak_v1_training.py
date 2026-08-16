"""Matched fixed-E2 training on the prospective weak-validation v1 split.

B20, B31 and B33 are retrained from the same historical B16/B20 initialization
on the frozen 80% study subset. The 20% validation subset is never loaded by
this trainer and is never used for checkpoint selection.

After model construction, the global training RNG is reset to one fixed seed for
all controls. This prevents B31's additional randomly initialized query/Conv1d
parameters from shifting the dropout/augmentation stochastic trajectory relative
to B20/B33 merely by consuming extra construction-time random draws.
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
from .b12_1_hierarchical import b12_1_model_spec, build_b12_1_model
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
from .b31_local_context_complementary_pool import (
    B31_EXPECTED_NEW_PARAMETERS,
    b31_model_spec,
    build_b31_model,
)
from .b33_uniform_complementary_mean import (
    B33_EXPECTED_NEW_PARAMETERS,
    b33_model_spec,
    build_b33_model,
)
from .budget import RuntimeBudget
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .policy import validate_competition_config
from .prospective_weak_v1 import (
    PV1_TRAIN_STUDIES,
    PV1_VERSION,
    validate_prospective_weak_v1_manifest,
)
from .runtime import autocast, make_scaler, resolve_runtime

PV1_TRAIN_EXPERIMENT = "prospective_weak_v1_matched_control_fixed_e2"
PV1_FIXED_EPOCHS = 2
PV1_MAX_HARD_HOURS = 8.25
PV1_MIN_RESERVE_MINUTES = 30.0
PV1_CONTROL_NAMES = ("b20", "b31", "b33")
PV1_POST_CONSTRUCTION_SEED_OFFSET = 19_200_000


def _build_control_model(name: str, config: dict):
    key = str(name).lower()
    if key == "b20":
        spec = b12_1_model_spec(config, normalize_input=True)
        return build_b12_1_model(spec, pretrained_weights=False), spec, 0
    if key == "b31":
        spec = b31_model_spec(config, normalize_input=True)
        return build_b31_model(spec, pretrained_weights=False), spec, B31_EXPECTED_NEW_PARAMETERS
    if key == "b33":
        spec = b33_model_spec(config, normalize_input=True)
        return build_b33_model(spec, pretrained_weights=False), spec, B33_EXPECTED_NEW_PARAMETERS
    raise ValueError(f"PV1 control must be one of {PV1_CONTROL_NAMES}; got {name!r}")


def _candidate_gradient_names(name: str) -> tuple[str, ...]:
    if name == "b31":
        return ("complementary_query", "complementary_gate", "local_context.weight")
    if name == "b33":
        return ("uniform_complementary_gate",)
    return ()


def _candidate_state(model_name: str, model) -> dict:
    if model_name == "b31":
        return model.b31_state()
    if model_name == "b33":
        return {"uniform_gate": model.uniform_gate_state()}
    return {}


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


def train_prospective_weak_v1_control(
    config: dict,
    *,
    model_name: str,
    split_manifest_path: str | Path,
    b6_root: str | Path,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    out_root: str | Path = "runs/prospective_weak_v1",
) -> Path:
    model_name = str(model_name).lower()
    if model_name not in PV1_CONTROL_NAMES:
        raise ValueError(f"unsupported PV1 control {model_name!r}")
    validate_competition_config(config, purpose="train")
    crop_policy = require_b20_contract(config)
    report_payload = load_b16_report_encoder(report_ssl_checkpoint)

    seed = int(config.get("seed", 2026))
    seed_everything(seed + 19_000_000)
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
        raise RuntimeError("PV1 training study count changed")
    targets, weights = _subset_supervision(full_uids, full_targets, full_weights, train_uids)

    split_audit = split["post_assignment_supervision_audit"]["training"]
    active_cells = int((weights > 0).sum())
    positive_cells = int(((weights > 0) & (targets > 0.5)).sum())
    negative_cells = int(((weights > 0) & (targets < 0.5)).sum())
    if (active_cells, positive_cells, negative_cells) != (
        int(split_audit["usable_cells"]),
        int(split_audit["positive_cells"]),
        int(split_audit["negative_cells"]),
    ):
        raise RuntimeError("PV1 training supervision does not match frozen split manifest")

    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("PV1 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    full_series_summary, _ = audit_variable_series_surface(series, full_uids)
    if full_series_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("PV1 full series surface no longer matches frozen B13 signature")
    if int(full_series_summary.get("eligible_recognized_plane_series", -1)) != 17475:
        raise ValueError("PV1 requires the historical 17,475-series source surface")

    train_index = build_variable_series_index(series, train_uids)
    expected_series = int(sum(len(train_index.get(uid, [])) for uid in train_uids))
    if expected_series <= 0 or any(len(train_index.get(uid, [])) == 0 for uid in train_uids):
        raise RuntimeError("PV1 training subset contains a study with no eligible MRI series")

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
        **runtime.loader_kwargs(seed=seed + 19_100_000),
    )

    model, spec, added_parameters = _build_control_model(model_name, config)
    model.encoder.load_state_dict(report_payload["encoder"], strict=True)
    freeze_encoder(model)
    model.gradient_checkpointing = False
    encoder_sha_initial = encoder_state_sha256(model.encoder)
    model = model.to(runtime.device)

    # Critical matched-control guardrail: construction of B31 consumes extra RNG
    # draws for its query/Conv1d. Reset ALL training RNGs here so the subsequent
    # dropout/main-process stochastic path starts identically for B20/B31/B33.
    post_construction_seed = seed + PV1_POST_CONSTRUCTION_SEED_OFFSET
    seed_everything(post_construction_seed)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    if not head_params or any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("PV1 frozen-encoder contract failed")
    named = dict(model.named_parameters())
    candidate_names = _candidate_gradient_names(model_name)
    for candidate_name in candidate_names:
        if candidate_name not in named:
            raise RuntimeError(f"PV1 {model_name} missing candidate parameter {candidate_name}")

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
        f"[PV1/{model_name}] train={len(train_uids)} validation={len(split['validation_uids'])} "
        f"series/train={expected_series} cells/train={active_cells} fixed_E2 split={split['split_sha256'][:12]}"
    )
    print(
        f"[PV1/{model_name}] candidate_added_parameters={added_parameters}; "
        f"post_construction_training_seed={post_construction_seed}; validation subset is not loaded during training"
    )

    history: list[dict] = []
    for epoch in range(1, PV1_FIXED_EPOCHS + 1):
        started = time.monotonic()
        model.train()
        model.encoder.eval()
        loss_sum = 0.0
        steps = seen_studies = seen_series = seen_cells = seen_pos = seen_neg = 0
        grad_seen = {name: False for name in candidate_names}

        for batch in loader:
            if not budget.can_start(120.0):
                raise RuntimeError("PV1 runtime budget expired before fixed E2 completed")
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
                raise RuntimeError("PV1 detected encoder gradient")
            for parameter_name in candidate_names:
                grad = named[parameter_name].grad
                if grad is None or not torch.isfinite(grad).all():
                    raise RuntimeError(f"PV1 {model_name} gradient missing/nonfinite for {parameter_name}")
                grad_seen[parameter_name] = grad_seen[parameter_name] or bool(torch.count_nonzero(grad).item() > 0)

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
            raise RuntimeError("PV1 encoder fingerprint changed")
        full = (
            steps == expected_batches
            and seen_studies == PV1_TRAIN_STUDIES
            and seen_series == expected_series
            and seen_cells == active_cells
            and seen_pos == positive_cells
            and seen_neg == negative_cells
        )
        if not full:
            raise RuntimeError(f"PV1 {model_name} E{epoch} did not cover the frozen training partition")
        missing_grad = [name for name, seen in grad_seen.items() if not seen]
        if missing_grad:
            raise RuntimeError(f"PV1 {model_name} E{epoch} candidate parameters never coupled: {missing_grad}")

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
            "candidate_gradient_seen": grad_seen,
            "candidate_state": _candidate_state(model_name, model),
            "epoch_seconds": float(time.monotonic() - started),
            "full_coverage": True,
        }
        history.append(row)
        print(f"[PV1/{model_name}] E{epoch} loss={row['loss']:.10f} time={row['epoch_seconds']/60:.1f} min")

    encoder_sha_final = encoder_state_sha256(model.encoder)
    if encoder_sha_final != encoder_sha_initial:
        raise RuntimeError("PV1 encoder changed after training")

    out = Path(out_root) / model_name
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "model.pt"
    payload = {
        "experiment": PV1_TRAIN_EXPERIMENT,
        "validation_framework": PV1_VERSION,
        "model_name": model_name,
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
        "candidate_added_parameters": int(added_parameters),
        "construction_seed": seed + 19_000_000,
        "loader_seed": seed + 19_100_000,
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
        "candidate_state_final": _candidate_state(model_name, model),
        "supervision_source": supervision,
        "b6_policy": b6_policy,
        "b6_audit": b6_audit,
        "metadata_repair": metadata_stats,
        "runtime_budget": budget.to_dict(),
    }
    torch.save(payload, checkpoint)
    audit = {k: v for k, v in payload.items() if k not in {"model_state", "config", "b6_policy", "b6_audit"}}
    (out / "training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint)
    return checkpoint


def load_prospective_weak_v1_checkpoint(
    path: str | Path, *, expected_split_sha256: str | None = None, device: torch.device | str = "cpu"
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != PV1_TRAIN_EXPERIMENT or payload.get("validation_framework") != PV1_VERSION:
        raise ValueError("not a prospective weak-v1 matched-control checkpoint")
    if payload.get("fixed_endpoint") is not True or int(payload.get("completed_epochs", -1)) != PV1_FIXED_EPOCHS:
        raise ValueError("PV1 checkpoint must be complete fixed-E2")
    if payload.get("validation_used_for_checkpoint_selection") is not False:
        raise ValueError("PV1 validation was unexpectedly used for checkpoint selection")
    if int(payload.get("validation_studies_used_in_gradient", -1)) != 0:
        raise ValueError("PV1 validation studies unexpectedly entered gradients")
    if bool(payload.get("expert_labels_used", True)):
        raise ValueError("PV1 checkpoint unexpectedly used expert labels")
    if payload.get("stochastic_path_matched_after_model_construction") is not True:
        raise ValueError("PV1 checkpoint lacks the matched post-construction RNG guardrail")
    split_sha = str(payload.get("split_sha256", ""))
    if expected_split_sha256 is not None and split_sha != str(expected_split_sha256):
        raise ValueError("PV1 checkpoint split fingerprint mismatch")
    initial_sha = str(payload.get("encoder_sha256_initial", ""))
    final_sha = str(payload.get("encoder_sha256_final", ""))
    if not initial_sha or initial_sha != final_sha:
        raise ValueError("PV1 checkpoint encoder fingerprint changed")

    model_name = str(payload.get("model_name", ""))
    spec = payload.get("model_spec")
    if model_name == "b20":
        model = build_b12_1_model(spec, pretrained_weights=False)
    elif model_name == "b31":
        model = build_b31_model(spec, pretrained_weights=False)
    elif model_name == "b33":
        model = build_b33_model(spec, pretrained_weights=False)
    else:
        raise ValueError(f"unknown PV1 model_name {model_name!r}")
    model.load_state_dict(payload["model_state"], strict=True)
    freeze_encoder(model)
    if encoder_state_sha256(model.encoder) != initial_sha:
        raise ValueError("PV1 reconstructed encoder fingerprint mismatch")
    return model.to(device), payload


def main() -> None:
    ap = argparse.ArgumentParser("Train one matched control on prospective weak-validation v1")
    ap.add_argument("--model", choices=PV1_CONTROL_NAMES, required=True)
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--series-policy", required=True)
    ap.add_argument("--report-ssl-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/prospective_weak_v1")
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    train_prospective_weak_v1_control(
        config,
        model_name=args.model,
        split_manifest_path=args.split_manifest,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
