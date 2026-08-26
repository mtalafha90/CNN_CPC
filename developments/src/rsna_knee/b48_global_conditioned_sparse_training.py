"""Train one matched B48 global-query-conditioning arm on frozen scanner splits.

B48 is intentionally not a continuation of B46.  It starts from the same
Phase-9/B34 base checkpoint as B42 and uses report-only weak supervision only.
The scanner-grouped split provides a new, no-gold primary comparison surface:
both B48 arms train on ``train`` rows, while seen- and unseen-scanner rows stay
out of every gradient.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .b7_weak_supervision import (
    _read_config,
    make_b7_dataset_config,
    seed_everything,
    target_balance_multipliers,
)
from .b12_training import _load_series_policy
from .b12_variable_series import audit_variable_series_surface
from .b13_training import B13_SERIES_SIGNATURE
from .b17_training import encoder_state_sha256
from .b35_training import B35_EXPECTED_CELLS, _require_base_checkpoint, sha256_file
from .b37_highres_sparse_training import (
    B37_CONSTRUCTION_SEED_OFFSET,
    B37_GRAD_CLIP,
    B37_HEAD_LR,
    B37_LOADER_SEED_OFFSET,
    B37_WEIGHT_DECAY,
    _format_memory_state,
    _memory_state,
    _save_recovery,
    _trim_host_memory,
)
from .b42_constant_area_aspect_sparse_mil import (
    B42_EFFECTIVE_BATCH,
    B42ConstantAreaAspectDataset,
    b42_preprocessing_state,
    collate_b42,
)
from .b42_constant_area_aspect_sparse_training import (
    _batch_scales,
    _losses,
    _move_study,
    _preflight,
)
from .b48_global_conditioned_sparse_mil import (
    B48_ARMS,
    B48_CONTEXT_DIM,
    B48_EXPERIMENT,
    B48_FIXED_EPOCHS,
    B48_RUN_ROOT,
    B48_SUPERVISION,
    B48_VERSION,
    B48GlobalConditionedSparseMILResidual,
    b48_state,
    require_b48_contract,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .domain_shift_split import DOMAIN_SPLIT_VERSION, verify_domain_split
from .label_confidence import rescale_label_confidence
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .phase9_supervision import (
    REPORT_ONLY_STUDIES,
    load_fill_merged_export,
    prepare_all_report_only_supervision,
)
from .runtime import make_scaler, resolve_runtime

B48_CHECKPOINT_TEMPLATE = "b48_{arm}_model.pt"
B48_CONSTRUCTION_SEED_OFFSET = B37_CONSTRUCTION_SEED_OFFSET
B48_LOADER_SEED_OFFSET = B37_LOADER_SEED_OFFSET
# These are predeclared paired runs.  The first is the compute gate; if it
# supports the mechanism, the other two are required replications, never
# best-seed selection.
B48_REPLICATION_SEEDS = (2026, 2037, 2048)
B48_FILL_ARTIFACT_FILES = ("training_targets.csv", "policy.json", "audit.json")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _config_sha256(config: dict) -> str:
    return _sha256_text(json.dumps(config, sort_keys=True, separators=(",", ":"), default=str))


def _uid_sha256(uids: list[str]) -> str:
    return _sha256_text("\n".join(str(uid) for uid in uids) + "\n")


def b48_fill_artifacts(labels_root: str | Path) -> dict[str, str]:
    """Fingerprint each fill-only label artifact consumed by B48.

    ``load_fill_merged_export`` enforces the no-overwrite rules but does not
    itself pin artifact digests.  Recording these inputs prevents a paired
    evaluation from silently using a later-edited report-only label export.
    """
    root = Path(labels_root).resolve()
    result: dict[str, str] = {}
    for name in B48_FILL_ARTIFACT_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"B48 fill-only artifact is missing: {path}")
        result[name] = sha256_file(path)
    return result


def _read_domain_split(path: str | Path) -> tuple[dict, pd.DataFrame, dict]:
    """Load a frozen scanner split and verify all of its companion artifacts."""
    payload_path = Path(path).resolve()
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    root = payload_path.parent
    rows_path = root / "domain_split_by_study.csv"
    sha_path = root / "domain_split.sha256"
    if not rows_path.is_file() or not sha_path.is_file():
        raise FileNotFoundError(
            "B48 requires domain_split.json, domain_split_by_study.csv, and domain_split.sha256"
        )
    digest = sha256_file(payload_path)
    recorded = sha_path.read_text(encoding="utf-8").strip()
    if recorded != digest:
        raise ValueError("B48 domain-split SHA does not match domain_split.json")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if payload.get("version") != DOMAIN_SPLIT_VERSION:
        raise ValueError("B48 requires the frozen official scanner-domain split version")
    if not str(payload.get("status", "")).startswith("frozen_before"):
        raise ValueError("B48 domain split was not declared frozen before scoring")

    rows = pd.read_csv(rows_path)
    required = {"StudyInstanceUID", "scanner_profile", "holdout", "split"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"B48 domain split CSV missing columns: {missing}")
    rows = rows.copy()
    rows["StudyInstanceUID"] = rows["StudyInstanceUID"].astype(str)
    if rows["StudyInstanceUID"].duplicated().any():
        raise ValueError("B48 domain split contains duplicate study UIDs")
    # CSV parsing can represent this boolean as strings in hand-edited files.
    if rows["holdout"].dtype == object:
        values = rows["holdout"].astype(str).str.lower()
        if not values.isin({"true", "false"}).all():
            raise ValueError("B48 domain split has invalid holdout booleans")
        rows["holdout"] = values.eq("true")
    else:
        rows["holdout"] = rows["holdout"].astype(bool)
    verify_domain_split(rows)
    splits = set(rows["split"].astype(str))
    expected_splits = {"train", "validation_seen_scanners", "holdout_unseen_scanners"}
    if splits != expected_splits:
        raise ValueError(f"B48 domain split must contain exactly {expected_splits}; got {splits}")
    metadata = {
        "path": str(payload_path),
        "sha256": digest,
        "rows_path": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "summary": payload.get("summary", {}),
    }
    return payload, rows, metadata


def load_b48_domain_split(path: str | Path) -> tuple[dict, pd.DataFrame, dict]:
    """Public validator shared by B48 training and evaluation."""
    return _read_domain_split(path)


def _report_only_surface(
    *,
    data_root: Path,
    labels_root: str | Path,
    config: dict,
    domain_rows: pd.DataFrame,
    base_payload: dict,
) -> tuple:
    """Return the split-aligned B48 weak-label surface without any gold rows."""
    train = load_train_csv(data_root / config.get("train_csv", "train.csv"))
    if len(train) != 4407:
        raise ValueError("B48 requires the complete 4,407-study training release")
    gold_uids = set(train.loc[gold_mask(train), "StudyInstanceUID"].astype(str))

    frame, fill_policy, fill_audit = load_fill_merged_export(labels_root)
    all_uids, all_targets, all_weights, supervision = prepare_all_report_only_supervision(train, frame)
    all_uids = [str(uid) for uid in all_uids]
    if len(all_uids) != REPORT_ONLY_STUDIES:
        raise ValueError("B48 requires all 4,349 report-only studies before the split")
    if int((all_weights > 0).sum()) != B35_EXPECTED_CELLS:
        raise ValueError("B48 weak supervision surface changed")
    if set(all_uids).intersection(gold_uids):
        raise RuntimeError("B48 report-only supervision includes an official gold study")
    if int(fill_audit.get("base_cells_overridden", -1)) != 0:
        raise ValueError("B48 requires zero B6 overrides")
    if list(fill_audit.get("excluded_targets", [])):
        raise ValueError("B48 requires all 12 targets")

    split_uids = set(domain_rows["StudyInstanceUID"])
    if split_uids != set(all_uids):
        missing = sorted(set(all_uids).difference(split_uids))
        extra = sorted(split_uids.difference(set(all_uids)))
        raise RuntimeError(
            "B48 domain split/report-only population mismatch "
            f"missing={missing[:3]} extra={extra[:3]}"
        )

    all_targets, confidence = rescale_label_confidence(all_targets, all_weights, config)
    base_confidence = base_payload.get("label_confidence", {})
    for key in ("positive_target", "negative_target"):
        if key in base_confidence and not np.isclose(
            float(confidence[key]), float(base_confidence[key]), atol=1e-12, rtol=0
        ):
            raise ValueError(f"B48 label confidence mismatch for {key}")
    lookup = {uid: index for index, uid in enumerate(all_uids)}
    return (
        train,
        all_uids,
        all_targets.astype(np.float32),
        all_weights.astype(np.float32),
        lookup,
        confidence,
        fill_policy,
        fill_audit,
        supervision,
    )


def _indices_for_split(all_uids: list[str], rows: pd.DataFrame, split: str) -> np.ndarray:
    selected = set(rows.loc[rows["split"].eq(str(split)), "StudyInstanceUID"].astype(str))
    result = np.asarray([index for index, uid in enumerate(all_uids) if uid in selected], dtype=np.int64)
    if len(result) != len(selected) or not len(result):
        raise RuntimeError(f"B48 split {split!r} has missing or zero report-only UIDs")
    return result


def _b48_context_preflight(
    model,
    dataset,
    *,
    runtime,
    multiplier_t,
    scaler,
    aux_weight: float,
) -> None:
    """Check B48's new gradient boundary without taking an optimizer step."""
    index = next(
        (
            i
            for i, weight in enumerate(dataset.weights)
            if np.asarray(weight, dtype=np.float32).max() > 0
        ),
        None,
    )
    if index is None:
        raise RuntimeError("B48 preflight cannot find a supervised train study")
    saved_gate = model.head.context_gate.detach().clone()
    item = dataset[int(index)]
    tensors = _move_study(item, runtime.device)
    # Verify on a real B42 rectangular study that the read-only post-attention
    # query still reconstructs the unchanged B34 global logit path exactly.
    was_training = model.training
    model.eval()
    with torch.no_grad():
        global_feature, spatial_probe = model._encode_ragged_study(tensors[0], tensors[2])
        reconstruction_error = model.context_reconstruction_error(
            global_feature,
            tensors[2],
            tensors[3],
        )
    del global_feature, spatial_probe
    if was_training:
        model.train(True)
    if reconstruction_error > 1e-6:
        raise RuntimeError(
            "B48 post-attention query no longer reconstructs the B42 global logits: "
            f"max_abs_error={reconstruction_error:.3e}"
        )
    print(
        f"[B48 preflight] post-attention global-logit reconstruction={reconstruction_error:.3e} PASS",
        flush=True,
    )

    model.train()
    model.zero_grad(set_to_none=True)
    out, total, _combined, _local = _losses(model, runtime, tensors, multiplier_t, aux_weight)
    if out.context_query.requires_grad:
        raise RuntimeError("B48 global query was not detached before local conditioning")
    scaler.scale(total).backward()
    gate_grad = model.head.context_gate.grad
    if gate_grad is None or torch.count_nonzero(gate_grad).item() == 0:
        raise RuntimeError("B48 local auxiliary loss did not reach the context gate")
    # At exact zero the two projections must wait for the context gate to open;
    # this guards an accidental non-zero start as well as gradient starvation.
    for parameter in (model.head.context_query.weight, model.head.context_key.weight):
        if parameter.grad is None or torch.count_nonzero(parameter.grad).item() != 0:
            raise RuntimeError("B48 zero-start projection gradient contract changed")
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        model.head.context_gate.fill_(0.05)
    out, total, _combined, _local = _losses(model, runtime, tensors, multiplier_t, aux_weight)
    scaler.scale(total).backward()
    for parameter in (model.head.context_query.weight, model.head.context_key.weight):
        if parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0:
            raise RuntimeError("B48 opened context gate did not reach low-rank projection")
    leaked = any(
        parameter.grad is not None
        for name, parameter in model.base.named_parameters()
        if not name.startswith("encoder.") and not parameter.requires_grad
    )
    if leaked:
        raise RuntimeError("B48 conditioning reached a frozen B34 non-encoder parameter")
    with torch.no_grad():
        model.head.context_gate.copy_(saved_gate)
    model.zero_grad(set_to_none=True)
    del item, tensors, out, total
    _trim_host_memory()
    print("[B48 preflight] detached global-query conditioning PASS", flush=True)


def train_b48_domain_arm(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    domain_split: str | Path,
    arm: str,
    seed: int = B48_REPLICATION_SEEDS[0],
    out_root: str | Path = B48_RUN_ROOT,
    preflight_only: bool = False,
) -> Path | None:
    """Train a fixed-E2 B48 arm with scanner-held-out rows excluded."""
    arm = str(arm)
    if arm not in B48_ARMS:
        raise ValueError(f"B48 arm must be one of {B48_ARMS}; got {arm!r}")
    seed = int(seed)
    if seed not in B48_REPLICATION_SEEDS:
        raise ValueError(f"B48 seed must be one of {B48_REPLICATION_SEEDS}; got {seed}")
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    contract = require_b48_contract(settings, arm=arm)
    declared_seeds = tuple(int(value) for value in settings.get("b48_replication_seeds", B48_REPLICATION_SEEDS))
    if declared_seeds != B48_REPLICATION_SEEDS:
        raise ValueError(f"B48 freezes b48_replication_seeds={list(B48_REPLICATION_SEEDS)}")

    domain_payload, domain_rows, domain_meta = load_b48_domain_split(domain_split)
    settings["seed"] = seed
    seed_everything(seed + B48_CONSTRUCTION_SEED_OFFSET)
    runtime = resolve_runtime(settings)
    print(runtime.describe(), flush=True)
    print(
        f"[B48 {arm} seed={seed}] domain_split_sha={domain_meta['sha256']} ",
        flush=True,
    )

    base_path = Path(base_checkpoint).resolve()
    base_model, base_payload = load_phase9_checkpoint(
        base_path, expected_arm="llm_fill", device="cpu"
    )
    _require_base_checkpoint(base_payload)
    encoder_initial_sha = encoder_state_sha256(base_model.encoder)

    root = Path(settings["data_root"])
    expected_train_sha = str(domain_payload.get("source_train_csv_sha256", ""))
    if not expected_train_sha:
        raise ValueError("B48 domain split does not pin source train.csv")
    if sha256_file(root / settings.get("train_csv", "train.csv")) != expected_train_sha:
        raise ValueError("B48 domain split source train.csv fingerprint mismatch")
    fill_artifacts = b48_fill_artifacts(labels_root)
    (
        _train,
        all_uids,
        all_targets,
        all_weights,
        _lookup,
        confidence,
        fill_policy,
        fill_audit,
        supervision,
    ) = _report_only_surface(
        data_root=root,
        labels_root=labels_root,
        config=settings,
        domain_rows=domain_rows,
        base_payload=base_payload,
    )
    train_indices = _indices_for_split(all_uids, domain_rows, "train")
    uids = [all_uids[index] for index in train_indices]
    targets = all_targets[train_indices]
    weights = all_weights[train_indices]
    target_multiplier = target_balance_multipliers(weights)

    # The supplied B12/B13 series policy remains frozen.  The smaller training
    # subset has a deliberately new audited series count, recorded rather than
    # pretending it is B42's full-population count.
    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B48 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    expected_series = int(series_summary.get("eligible_recognized_plane_series", -1))
    if expected_series <= 0 or series_summary.get("viability_passed") is not True:
        raise ValueError("B48 scanner-split MRI training surface failed viability")
    expected_cells = int((weights > 0).sum())
    if expected_cells <= 0:
        raise ValueError("B48 scanner-split training surface has no usable weak labels")

    dataset_config = make_b7_dataset_config(settings, root, train=False)
    dataset_config.tta_center_offsets = ()
    dataset = B42ConstantAreaAspectDataset(
        uids,
        variable_index,
        dataset_config,
        crop_focus_policy=contract["crop_policy"],
        center_offsets=(0,),
        targets=targets,
        weights=weights,
    )
    loader = DataLoader(
        dataset,
        batch_size=B42_EFFECTIVE_BATCH,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_b42,
        **runtime.loader_kwargs(seed=seed + B48_LOADER_SEED_OFFSET),
    )

    model = B48GlobalConditionedSparseMILResidual(
        base_model,
        grid_size=int(settings["b37_grid_size"]),
        top_k=int(settings["b37_top_k"]),
        temperature=float(settings["b37_temperature"]),
        encoder_trainable_stages=int(settings["b37_encoder_trainable_stages"]),
        encoder_chunk_size=int(settings["b37_encoder_chunk_size"]),
        arm=arm,
        context_dim=int(settings["b48_context_dim"]),
    ).to(runtime.device)
    model.train()
    head_params = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    encoder_params = [
        parameter for parameter in model.base.encoder.parameters() if parameter.requires_grad
    ]
    if not head_params or not encoder_params:
        raise RuntimeError("B48 requires sparse-head/context and encoder-tail parameters")
    if any(
        parameter.requires_grad
        for name, parameter in model.base.named_parameters()
        if not name.startswith("encoder.")
    ):
        raise RuntimeError("B48 non-encoder B34 parameters must remain frozen")

    head_lr = float(settings.get("b37_head_lr", B37_HEAD_LR))
    encoder_scale = float(settings["b37_encoder_lr_scale"])
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": head_lr, "name": "sparse_context_head"},
            {"params": encoder_params, "lr": head_lr * encoder_scale, "name": "encoder_tail"},
        ],
        weight_decay=float(settings.get("b37_weight_decay", B37_WEIGHT_DECAY)),
    )
    scaler = make_scaler(runtime)
    multiplier_cpu = torch.from_numpy(target_multiplier)
    multiplier_t = multiplier_cpu.to(runtime.device)
    aux_weight = float(settings["b37_local_aux_weight"])
    clip = float(settings.get("b37_grad_clip", B37_GRAD_CLIP))

    arm_root = Path(out_root)
    if preflight_only:
        print(f"[B48 {arm}] inherited B42 ragged geometry preflight", flush=True)
        _preflight(model, loader, runtime, multiplier_t, multiplier_cpu, scaler, aux_weight)
        _b48_context_preflight(
            model,
            dataset,
            runtime=runtime,
            multiplier_t=multiplier_t,
            scaler=scaler,
            aux_weight=aux_weight,
        )
        print(f"[B48 {arm} preflight] PASS", flush=True)
        return None

    arm_root.mkdir(parents=True, exist_ok=True)
    checkpoint = arm_root / B48_CHECKPOINT_TEMPLATE.format(arm=arm)
    if checkpoint.exists():
        raise FileExistsError(
            f"B48 will not overwrite an existing checkpoint: {checkpoint}"
        )
    history: list[dict] = []
    for epoch in range(1, B48_FIXED_EPOCHS + 1):
        started = time.monotonic()
        if runtime.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(runtime.device)
        model.train()
        total_sum = combined_sum = local_sum = 0.0
        batches = studies_seen = series_seen = cells_seen = 0
        gate_gradient_seen = evidence_gradient_seen = encoder_gradient_seen = False
        context_gate_gradient_seen = context_projection_gradient_seen = False

        for step, items in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            scales = _batch_scales(items, multiplier_cpu)
            batch_total = batch_combined = batch_local = 0.0
            batch_series = batch_cells = 0
            for item, scale in zip(items, scales):
                batch_series += int(item["present"].sum().item())
                batch_cells += int((item["weight"] > 0).sum().item())
                tensors = _move_study(item, runtime.device)
                out, total, combined, local = _losses(
                    model, runtime, tensors, multiplier_t, aux_weight
                )
                scaler.scale(total * float(scale)).backward()
                batch_total += float(total.detach().item()) * float(scale)
                batch_combined += float(combined.detach().item()) * float(scale)
                batch_local += float(local.detach().item()) * float(scale)
                del tensors, out, total, combined, local

            leaked = any(
                parameter.grad is not None
                for name, parameter in model.base.named_parameters()
                if not name.startswith("encoder.") and not parameter.requires_grad
            )
            if leaked:
                raise RuntimeError("B48 detected a gradient on the frozen B34 hierarchy")
            gate_gradient_seen = gate_gradient_seen or bool(
                model.head.gate.grad is not None
                and torch.count_nonzero(model.head.gate.grad).item() > 0
            )
            evidence_gradient_seen = evidence_gradient_seen or bool(
                model.head.evidence_weight.grad is not None
                and torch.count_nonzero(model.head.evidence_weight.grad).item() > 0
            )
            context_gate_gradient_seen = context_gate_gradient_seen or bool(
                model.head.context_gate.grad is not None
                and torch.count_nonzero(model.head.context_gate.grad).item() > 0
            )
            context_projection_gradient_seen = context_projection_gradient_seen or all(
                parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
                for parameter in (model.head.context_query.weight, model.head.context_key.weight)
            )
            encoder_gradient_seen = encoder_gradient_seen or any(
                parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
                for parameter in encoder_params
            )
            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(head_params + encoder_params, clip)
            scaler.step(optimizer)
            scaler.update()

            total_sum += batch_total
            combined_sum += batch_combined
            local_sum += batch_local
            batches += 1
            studies_seen += len(items)
            series_seen += batch_series
            cells_seen += batch_cells
            report_progress = step % 50 == 0
            if report_progress:
                elapsed = (time.monotonic() - started) / 60.0
                rate = elapsed / step
                remaining = rate * (len(loader) - step)
                sparse_gate = model.head.effective_gate().detach().abs().mean().item()
                context_gate = model.head.effective_context_gate().detach().abs().mean().item()
            del items
            if report_progress:
                _trim_host_memory()
                print(
                    f"[B48 {arm} S{seed}] E{epoch} {step}/{len(loader)} "
                    f"total={total_sum/batches:.4f} combined={combined_sum/batches:.4f} "
                    f"local={local_sum/batches:.4f} sparse_gate={sparse_gate:.4f} "
                    f"context_gate={context_gate:.4f} elapsed={elapsed:.1f} min "
                    f"remaining~{remaining:.1f} min {_format_memory_state(_memory_state(runtime))}",
                    flush=True,
                )

        if studies_seen != len(uids) or series_seen != expected_series or cells_seen != expected_cells:
            raise RuntimeError(
                "B48 epoch surface changed: "
                f"studies={studies_seen}/{len(uids)} series={series_seen}/{expected_series} "
                f"cells={cells_seen}/{expected_cells}"
            )
        if not (
            gate_gradient_seen
            and evidence_gradient_seen
            and encoder_gradient_seen
            and context_gate_gradient_seen
            and context_projection_gradient_seen
        ):
            raise RuntimeError("B48 required sparse, context, and encoder gradient paths were not active")
        row = {
            "epoch": epoch,
            "arm": arm,
            "seed": seed,
            "loss_total": total_sum / batches,
            "loss_combined": combined_sum / batches,
            "loss_local_aux": local_sum / batches,
            "batches": batches,
            "studies": studies_seen,
            "series": series_seen,
            "supervision_cells": cells_seen,
            "sparse_mil": model.head.state(),
            "encoder_sha256": encoder_state_sha256(model.base.encoder),
            "epoch_seconds": float(time.monotonic() - started),
            "memory": _memory_state(runtime),
        }
        history.append(row)
        print(
            f"[B48 {arm} S{seed}] E{epoch} total={row['loss_total']:.10f} "
            f"combined={row['loss_combined']:.10f} local={row['loss_local_aux']:.10f} "
            f"time={row['epoch_seconds']/60:.1f} min",
            flush=True,
        )
        _save_recovery(arm_root, epoch=epoch, model=model, history=history, version=B48_VERSION)

    encoder_final_sha = encoder_state_sha256(model.base.encoder)
    if encoder_final_sha == encoder_initial_sha:
        raise RuntimeError("B48 encoder fingerprint did not move")
    target_balance = {
        target: float(target_multiplier[index]) for index, target in enumerate(TARGETS)
    }
    source_sha = {
        "model": sha256_file(Path(__file__).with_name("b48_global_conditioned_sparse_mil.py")),
        "training": sha256_file(Path(__file__)),
    }
    matched_pair_identity = {
        # These fields must be byte-for-byte equal for the two arm checkpoints.
        # Arm and query source intentionally stay outside this object.
        "seed": seed,
        "config_sha256": _config_sha256(settings),
        "base_checkpoint_sha256": sha256_file(base_path),
        "training_uids_sha256": _uid_sha256(uids),
        "target_balance_multiplier": target_balance,
        "domain_split_sha256": domain_meta["sha256"],
        "domain_rows_sha256": domain_meta["rows_sha256"],
        "fill_artifacts": fill_artifacts,
        "series_policy_signature": B13_SERIES_SIGNATURE,
        "source_sha256": source_sha,
    }
    payload = {
        "experiment": B48_EXPERIMENT,
        "version": B48_VERSION,
        "fixed_endpoint": True,
        "completed_epochs": B48_FIXED_EPOCHS,
        "checkpoint_selection": "none; fixed epoch 2",
        "arm": arm,
        "seed": seed,
        "hypothesis": (
            "a detached post-cross-attention global pathology query can softly re-rank "
            "B42 local sparse evidence across the current study's series beyond an "
            "otherwise matched static pathology-query control"
        ),
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": sha256_file(base_path),
        "base_payload_experiment": base_payload.get("experiment"),
        "base_state": model.base.state_dict(),
        "head_state": model.head.state_dict(),
        "model_state": model.state(),
        "encoder_sha256_initial": encoder_initial_sha,
        "encoder_sha256_final": encoder_final_sha,
        "head_lr": head_lr,
        "encoder_lr": head_lr * encoder_scale,
        "local_aux_weight": aux_weight,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_used": False,
        "supervision_source": B48_SUPERVISION,
        "training_studies": len(uids),
        "training_series": expected_series,
        "training_supervision_cells": expected_cells,
        "training_uids_sha256": _uid_sha256(uids),
        "target_balance_source": "scanner_split_train_only_weak_labels",
        "target_balance_multiplier": target_balance,
        "domain_split": domain_meta,
        "domain_split_summary": domain_payload.get("summary", {}),
        "preprocessing": b42_preprocessing_state(),
        "crop_policy": contract["crop_policy"],
        "label_confidence": confidence,
        "fill_policy": fill_policy,
        "fill_audit": fill_audit,
        "fill_artifacts": fill_artifacts,
        "supervision": supervision,
        "series_policy_signature": B13_SERIES_SIGNATURE,
        "series_surface": series_summary,
        "metadata_repair": metadata_stats,
        "b48": b48_state(arm),
        "config_sha256": matched_pair_identity["config_sha256"],
        "source_sha256": source_sha,
        "matched_pair_identity": matched_pair_identity,
        "history": history,
        "governance": (
            "B48 is a prospective matched scanner-split weak-label comparison. Do not use "
            "official gold labels, B46 outputs, B47 grid changes, or any checkpoint/parameter "
            "selection. Do not alter rank, query source, seeds, training rows, optimizer, "
            "top-k, grid, or endpoint after a B48 result is inspected."
        ),
    }
    torch.save(payload, checkpoint)
    audit = {key: value for key, value in payload.items() if key not in {"base_state", "head_state"}}
    (arm_root / "training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (arm_root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint, flush=True)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser("Train one fixed B48 global-conditioned scanner-split arm")
    parser.add_argument("--config", default="config/b48_global_conditioned_sparse.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--domain-split", required=True)
    parser.add_argument("--arm", choices=B48_ARMS, required=True)
    parser.add_argument("--seed", type=int, default=B48_REPLICATION_SEEDS[0])
    parser.add_argument("--out-root", default=B48_RUN_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    train_b48_domain_arm(
        dict(_read_config(args.config)),
        data_root=args.data_root,
        labels_root=args.labels_root,
        series_policy_path=args.series_policy,
        base_checkpoint=args.base_checkpoint,
        domain_split=args.domain_split,
        arm=args.arm,
        seed=args.seed,
        out_root=args.out_root,
        preflight_only=bool(args.preflight_only),
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B48_CHECKPOINT_TEMPLATE",
    "B48_FILL_ARTIFACT_FILES",
    "B48_REPLICATION_SEEDS",
    "b48_fill_artifacts",
    "load_b48_domain_split",
    "train_b48_domain_arm",
]
