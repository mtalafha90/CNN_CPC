"""B40: one additional optimizer-reset epoch from the completed B37 endpoint.

The completed B37 checkpoint stores the complete model parameters after its
predeclared two epochs, but it deliberately did not store AdamW moment buffers.
B40 is therefore a new, explicit experiment: load the verified B37 E2 weights,
create a fresh AdamW optimizer with the original B37 parameter groups, and train
exactly one more full-coverage epoch.  It never changes B37 or B39.
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
    _read_config,
    make_b7_dataset_config,
    seed_everything,
    target_balance_multipliers,
    target_balanced_weak_bce,
)
from .b12_training import _load_series_policy
from .b12_variable_series import audit_variable_series_surface
from .b13_training import B13_SERIES_SIGNATURE
from .b17_training import encoder_state_sha256
from .b35_training import B35_EXPECTED_CELLS, B35_EXPECTED_SERIES, sha256_file
from .b37_highres_sparse_eval import load_b37_checkpoint
from .b37_highres_sparse_mil import (
    B37_ENCODER_LR_SCALE,
    B37_GRID_SIZE,
    B37_LOCAL_AUX_WEIGHT,
    B37_TOP_K,
    B37_VERSION,
    B37HighResSparseDataset,
    B37HighResSparseMILResidual,
    collate_b35,
    require_b37_sparse_contract,
)
from .b37_highres_sparse_training import (
    B37_EQUIVALENCE_TOLERANCE,
    B37_EPOCHS,
    B37_EXPERIMENT,
    B37_GRAD_CLIP,
    B37_HEAD_LR,
    B37_MICRO_BATCH,
    B37_WEIGHT_DECAY,
    _format_memory_state,
    _largest_series_indices,
    _memory_state,
    _move_batch,
    _trim_host_memory,
)
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .label_confidence import rescale_label_confidence
from .phase9_supervision import (
    REPORT_ONLY_STUDIES,
    load_fill_merged_export,
    prepare_all_report_only_supervision,
)
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import autocast, make_scaler, resolve_runtime

B40_VERSION = "b40_b37_e2_optimizer_reset_continuation_v1"
B40_EXPERIMENT = "B40_b37_e2_optimizer_reset_one_epoch_continuation"
B40_NUMBERED_CONTAINER = "runs/075_Experiment_B40_b37_e2_optimizer_reset_continuation"
B40_RUN_ROOT = f"{B40_NUMBERED_CONTAINER}/b40_b37_e2_continuation"
B40_PARENT_EPOCHS = 2
B40_ADDITIONAL_EPOCHS = 1
B40_COMPLETED_EPOCHS = B40_PARENT_EPOCHS + B40_ADDITIONAL_EPOCHS
B40_CONSTRUCTION_SEED_OFFSET = 47_600_000
B40_LOADER_SEED_OFFSET = 47_600_002


def _require_close(name: str, value: float, expected: float) -> None:
    if not np.isclose(float(value), float(expected), atol=1e-12, rtol=0.0):
        raise ValueError(f"B40 freezes {name}={expected}; got {value}")


def require_b40_continuation_contract(config: dict) -> dict:
    """Require B40 to change only the declared post-B37 optimization duration."""
    crop_policy = require_b37_sparse_contract(config)
    if int(config.get("b40_parent_completed_epochs", B40_PARENT_EPOCHS)) != B40_PARENT_EPOCHS:
        raise ValueError(f"B40 requires parent completed_epochs={B40_PARENT_EPOCHS}")
    if int(config.get("b40_additional_epochs", B40_ADDITIONAL_EPOCHS)) != B40_ADDITIONAL_EPOCHS:
        raise ValueError(f"B40 freezes b40_additional_epochs={B40_ADDITIONAL_EPOCHS}")
    if bool(config.get("b40_optimizer_reset", True)) is not True:
        raise ValueError("B40 requires the declared fresh-AdamW optimizer reset")
    _require_close(
        "b40_head_lr",
        float(config.get("b40_head_lr", B37_HEAD_LR)),
        B37_HEAD_LR,
    )
    _require_close(
        "b40_encoder_lr_scale",
        float(config.get("b40_encoder_lr_scale", B37_ENCODER_LR_SCALE)),
        B37_ENCODER_LR_SCALE,
    )
    _require_close(
        "b40_weight_decay",
        float(config.get("b40_weight_decay", B37_WEIGHT_DECAY)),
        B37_WEIGHT_DECAY,
    )
    _require_close(
        "b40_grad_clip",
        float(config.get("b40_grad_clip", B37_GRAD_CLIP)),
        B37_GRAD_CLIP,
    )
    if int(config.get("b37_micro_batch", B37_MICRO_BATCH)) != B37_MICRO_BATCH:
        raise ValueError(f"B40 freezes micro-batch={B37_MICRO_BATCH}")
    return crop_policy


def _require_parent_b37(payload: dict, path: Path) -> None:
    """Reject anything other than the completed immutable B37 E2 model."""
    if payload.get("experiment") != B37_EXPERIMENT:
        raise ValueError("B40 parent is not the completed B37 experiment")
    if payload.get("version") != B37_VERSION:
        raise ValueError("B40 parent is not a B37 high-resolution sparse-MIL checkpoint")
    if payload.get("fixed_endpoint") is not True or int(payload.get("completed_epochs", -1)) != B40_PARENT_EPOCHS:
        raise ValueError("B40 requires B37's completed fixed-E2 checkpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0 or bool(payload.get("gold_labels_used", True)):
        raise ValueError("B40 parent unexpectedly used expert labels")
    if int(payload.get("training_studies", -1)) != REPORT_ONLY_STUDIES:
        raise ValueError("B40 parent has the wrong report-only training population")
    if int(payload.get("training_series", -1)) != B35_EXPECTED_SERIES:
        raise ValueError("B40 parent has the wrong MRI series surface")
    if int(payload.get("training_supervision_cells", -1)) != B35_EXPECTED_CELLS:
        raise ValueError("B40 parent has the wrong supervision surface")
    sparse = payload.get("sparse_mil", {})
    if int(sparse.get("grid_size", -1)) != B37_GRID_SIZE:
        raise ValueError("B40 parent does not have B37's 6x6 sparse grid")
    if int(sparse.get("top_k", -1)) != B37_TOP_K:
        raise ValueError("B40 parent does not have B37's top-k=8 sparse MIL")
    if not path.is_file() or not sha256_file(path):
        raise FileNotFoundError(f"B40 parent checkpoint is missing {path}")


def _losses(model, runtime, batch_tensors, multiplier_t: torch.Tensor, aux_weight: float):
    volumes, _, present, meta, target, weight = batch_tensors
    with autocast(runtime):
        output = model(volumes, present, meta, batch_tensors[1])
        combined = target_balanced_weak_bce(
            output.logits,
            target,
            weight,
            multiplier_t,
        )
        local = target_balanced_weak_bce(
            output.local_logits,
            target,
            weight,
            multiplier_t,
        )
        total = combined + float(aux_weight) * local
    return output, total, combined, local


def _preflight_b40(model, loader, runtime, multiplier_t, scaler, aux_weight: float) -> None:
    """Perform one largest-shape forward/backward pass without an optimizer step."""
    print("[B40 preflight] forward/backward only; no optimizer step", flush=True)
    if runtime.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(runtime.device)
    model.train()
    model.zero_grad(set_to_none=True)
    indices = _largest_series_indices(loader.dataset, B37_MICRO_BATCH)
    items = [loader.dataset[index] for index in indices]
    counts = [int(item["present"].shape[0]) for item in items]
    batch = collate_b35(items)
    del items
    print(
        f"[B40 preflight] worst-case series/study={counts} "
        f"padded_series={int(batch['present'].shape[1])}",
        flush=True,
    )
    tensors = _move_batch(batch, runtime.device)
    volumes, _, present, meta, _, _ = tensors
    equivalence = model.base_equivalence_error_448(volumes, present, meta)
    print(
        f"[B40 preflight] reconstructed B37 base max|delta|={equivalence:.8g}",
        flush=True,
    )
    if equivalence > B37_EQUIVALENCE_TOLERANCE:
        raise RuntimeError(f"B40 B37 reconstruction guard failed: {equivalence}")
    _, total, combined, local = _losses(
        model,
        runtime,
        tensors,
        multiplier_t,
        aux_weight,
    )
    scaler.scale(total).backward()
    encoder_grad = any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.base.encoder.parameters()
        if parameter.requires_grad
    )
    evidence_grad = bool(
        model.head.evidence_weight.grad is not None
        and torch.count_nonzero(model.head.evidence_weight.grad).item() > 0
    )
    if not encoder_grad or not evidence_grad:
        raise RuntimeError("B40 preflight did not reach encoder tail and sparse evidence head")
    print(
        f"[B40 preflight] total={total.detach().item():.6f} "
        f"combined={combined.detach().item():.6f} "
        f"local={local.detach().item():.6f}",
        flush=True,
    )
    print(f"[B40 preflight] {_format_memory_state(_memory_state(runtime))}", flush=True)
    model.zero_grad(set_to_none=True)
    del batch, tensors, volumes, present, meta, total, combined, local
    _trim_host_memory()
    print("[B40 preflight] PASS", flush=True)


def _save_recovery(
    out: Path,
    *,
    parent_checkpoint: Path,
    parent_sha256: str,
    absolute_epoch: int,
    model,
    optimizer,
    scaler,
    history: list[dict],
) -> None:
    """Save enough state for a future exact B40 recovery, unlike historical B37."""
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": B40_VERSION,
            "experiment": B40_EXPERIMENT,
            "absolute_epoch": int(absolute_epoch),
            "fixed_endpoint": False,
            "parent_b37_checkpoint": str(parent_checkpoint),
            "parent_b37_checkpoint_sha256": parent_sha256,
            "base_state": model.base.state_dict(),
            "head_state": model.head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if hasattr(scaler, "state_dict") else {},
            "history": history,
        },
        out / "recovery_latest.pt",
    )


def _build_training_surface(config: dict, *, data_root: Path, labels_root: Path, series_policy_path: Path, parent_payload: dict):
    """Rebuild B37's exact all-report-only training surface and validate it."""
    train = load_train_csv(data_root / config.get("train_csv", "train.csv"))
    if len(train) != 4407:
        raise ValueError("B40 requires the complete 4,407-study training release")
    frame, fill_policy, fill_audit = load_fill_merged_export(labels_root)
    uids, targets, weights, supervision = prepare_all_report_only_supervision(train, frame)
    if len(uids) != REPORT_ONLY_STUDIES:
        raise ValueError("B40 requires all 4,349 report-only studies")
    if int((weights > 0).sum()) != B35_EXPECTED_CELLS:
        raise ValueError("B40 supervision surface changed")
    if int(fill_audit.get("base_cells_overridden", -1)) != 0:
        raise ValueError("B40 requires B37's fill-only surface with zero B6 overrides")
    if list(fill_audit.get("excluded_targets", [])):
        raise ValueError("B40 requires all 12 targets")

    targets, confidence = rescale_label_confidence(targets, weights, config)
    for key in ("positive_target", "negative_target"):
        if not np.isclose(
            float(confidence[key]),
            float(parent_payload.get("label_confidence", {}).get(key, np.nan)),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"B40 label confidence mismatch for {key}")

    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B40 requires the frozen B12/B13 series policy")
    series = load_series_csv(data_root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, data_root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    if int(series_summary.get("eligible_recognized_plane_series", -1)) != B35_EXPECTED_SERIES:
        raise ValueError("B40 all-series MRI surface changed")
    if series_summary.get("viability_passed") is not True:
        raise ValueError("B40 all-series MRI surface failed viability")
    return (
        uids,
        targets,
        weights,
        supervision,
        fill_policy,
        fill_audit,
        variable_index,
        metadata_stats,
        series_summary,
        confidence,
    )


def train_b40(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    parent_checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path = B40_RUN_ROOT,
    preflight_only: bool = False,
) -> Path | None:
    """Train B40 from immutable B37 E2 weights for exactly one new epoch."""
    config = dict(config)
    root = Path(data_root).resolve()
    labels_path = Path(labels_root).resolve()
    policy_path = Path(series_policy_path).resolve()
    parent_path = Path(parent_checkpoint).resolve()
    base_path = Path(base_checkpoint).resolve()
    config["data_root"] = str(root)
    crop_policy = require_b40_continuation_contract(config)
    if not labels_path.is_dir():
        raise FileNotFoundError(f"B40 labels root is missing {labels_path}")
    if not policy_path.is_file():
        raise FileNotFoundError(f"B40 series policy is missing {policy_path}")
    if not base_path.is_file():
        raise FileNotFoundError(f"B40 base checkpoint is missing {base_path}")

    seed = int(config.get("seed", 2026))
    seed_everything(seed + B40_CONSTRUCTION_SEED_OFFSET)
    runtime = resolve_runtime(config)
    print(runtime.describe(), flush=True)

    model, parent_payload = load_b37_checkpoint(
        parent_path,
        base_checkpoint=base_path,
        device=runtime.device,
    )
    _require_parent_b37(parent_payload, parent_path)
    parent_sha256 = sha256_file(parent_path)
    parent_encoder_sha256 = str(parent_payload.get("encoder_sha256_final", ""))
    if not parent_encoder_sha256:
        raise ValueError("B40 parent lacks its final encoder fingerprint")
    parent_history = list(parent_payload.get("history", []))
    if len(parent_history) != B40_PARENT_EPOCHS:
        raise ValueError("B40 requires B37's two completed history rows")

    (
        uids,
        targets,
        weights,
        supervision,
        fill_policy,
        fill_audit,
        variable_index,
        metadata_stats,
        series_summary,
        confidence,
    ) = _build_training_surface(
        config,
        data_root=root,
        labels_root=labels_path,
        series_policy_path=policy_path,
        parent_payload=parent_payload,
    )

    dataset_config = make_b7_dataset_config(config, root, train=False)
    dataset_config.tta_center_offsets = ()
    dataset = B37HighResSparseDataset(
        uids,
        variable_index,
        dataset_config,
        crop_focus_policy=crop_policy,
        center_offsets=(0,),
        targets=targets,
        weights=weights,
    )
    loader = DataLoader(
        dataset,
        batch_size=B37_MICRO_BATCH,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_b35,
        **runtime.loader_kwargs(seed=seed + B40_LOADER_SEED_OFFSET),
    )

    model.train()
    head_params = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    encoder_params = [parameter for parameter in model.base.encoder.parameters() if parameter.requires_grad]
    if not head_params or not encoder_params:
        raise RuntimeError("B40 requires both sparse-head and encoder-tail parameters")
    if any(
        parameter.requires_grad
        for name, parameter in model.base.named_parameters()
        if not name.startswith("encoder.")
    ):
        raise RuntimeError("B40 non-encoder B34 parameters must remain frozen")

    head_lr = float(config["b40_head_lr"])
    encoder_scale = float(config["b40_encoder_lr_scale"])
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": head_lr, "name": "sparse_head"},
            {"params": encoder_params, "lr": head_lr * encoder_scale, "name": "encoder_tail"},
        ],
        weight_decay=float(config["b40_weight_decay"]),
    )
    scaler = make_scaler(runtime)
    multiplier_t = torch.from_numpy(target_balance_multipliers(weights)).to(runtime.device)
    aux_weight = float(config.get("b37_local_aux_weight", B37_LOCAL_AUX_WEIGHT))
    clip = float(config["b40_grad_clip"])

    if preflight_only:
        _preflight_b40(model, loader, runtime, multiplier_t, scaler, aux_weight)
        return None

    output_root = Path(out_root)
    output_root.mkdir(parents=True, exist_ok=True)
    history = parent_history.copy()
    equivalence_error = None
    absolute_epoch = B40_COMPLETED_EPOCHS
    started = time.monotonic()
    if runtime.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(runtime.device)
    model.train()
    total_sum = combined_sum = local_sum = 0.0
    batches = studies_seen = series_seen = cells_seen = 0
    gate_gradient_seen = evidence_gradient_seen = encoder_gradient_seen = False

    for step, batch in enumerate(loader, start=1):
        tensors = _move_batch(batch, runtime.device)
        volumes, _, present, meta, target, weight = tensors
        if equivalence_error is None:
            equivalence_error = model.base_equivalence_error_448(volumes, present, meta)
            print(
                f"[B40] reconstructed B37 base max|delta|={equivalence_error:.8g}",
                flush=True,
            )
            if equivalence_error > B37_EQUIVALENCE_TOLERANCE:
                raise RuntimeError("B40 reconstructed B37 guard failed")

        optimizer.zero_grad(set_to_none=True)
        output, total, combined, local = _losses(
            model,
            runtime,
            tensors,
            multiplier_t,
            aux_weight,
        )
        scaler.scale(total).backward()
        leaked = any(
            parameter.grad is not None
            for name, parameter in model.base.named_parameters()
            if not name.startswith("encoder.") and not parameter.requires_grad
        )
        if leaked:
            raise RuntimeError("B40 detected a gradient on frozen B34 hierarchy")
        gate_gradient_seen = gate_gradient_seen or bool(
            model.head.gate.grad is not None
            and torch.count_nonzero(model.head.gate.grad).item() > 0
        )
        evidence_gradient_seen = evidence_gradient_seen or bool(
            model.head.evidence_weight.grad is not None
            and torch.count_nonzero(model.head.evidence_weight.grad).item() > 0
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

        active = weight > 0
        total_sum += float(total.item())
        combined_sum += float(combined.item())
        local_sum += float(local.item())
        batches += 1
        studies_seen += int(volumes.shape[0])
        series_seen += int(present.sum().item())
        cells_seen += int(active.sum().item())
        report_progress = step % 50 == 0
        if report_progress:
            elapsed = (time.monotonic() - started) / 60.0
            remaining = (elapsed / step) * (len(loader) - step)
            gate_abs_mean = model.head.effective_gate().detach().abs().mean().item()

        del (
            batch,
            tensors,
            volumes,
            present,
            meta,
            target,
            weight,
            output,
            total,
            combined,
            local,
            active,
        )
        if report_progress:
            _trim_host_memory()
            print(
                f"[B40] E{absolute_epoch} {step}/{len(loader)} "
                f"total={total_sum/batches:.4f} combined={combined_sum/batches:.4f} "
                f"local={local_sum/batches:.4f} gate_abs_mean={gate_abs_mean:.4f} "
                f"elapsed={elapsed:.1f} min remaining~{remaining:.1f} min "
                f"{_format_memory_state(_memory_state(runtime))}",
                flush=True,
            )

    if studies_seen != REPORT_ONLY_STUDIES:
        raise RuntimeError("B40 epoch did not cover all report-only studies")
    if series_seen != B35_EXPECTED_SERIES:
        raise RuntimeError("B40 epoch did not cover all expected MRI series")
    if cells_seen != B35_EXPECTED_CELLS:
        raise RuntimeError("B40 epoch did not cover all supervision cells")
    if not (gate_gradient_seen and evidence_gradient_seen and encoder_gradient_seen):
        raise RuntimeError("B40 required gradient path was not active")

    encoder_final_sha = encoder_state_sha256(model.base.encoder)
    if encoder_final_sha == parent_encoder_sha256:
        raise RuntimeError("B40 encoder tail did not move from its B37 E2 parent")
    row = {
        "epoch": absolute_epoch,
        "continuation_epoch": 1,
        "loss_total": total_sum / batches,
        "loss_combined": combined_sum / batches,
        "loss_local_aux": local_sum / batches,
        "batches": batches,
        "studies": studies_seen,
        "series": series_seen,
        "supervision_cells": cells_seen,
        "gate": model.head.state(),
        "encoder_sha256": encoder_final_sha,
        "epoch_seconds": float(time.monotonic() - started),
        "memory": _memory_state(runtime),
        "optimizer": "fresh_adamw_from_b37_e2_weights",
    }
    history.append(row)
    print(
        f"[B40] E{absolute_epoch} total={row['loss_total']:.10f} "
        f"combined={row['loss_combined']:.10f} local={row['loss_local_aux']:.10f} "
        f"time={row['epoch_seconds']/60:.1f} min",
        flush=True,
    )
    _save_recovery(
        output_root,
        parent_checkpoint=parent_path,
        parent_sha256=parent_sha256,
        absolute_epoch=absolute_epoch,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        history=history,
    )

    checkpoint = output_root / "b40_model.pt"
    payload = {
        "experiment": B40_EXPERIMENT,
        "version": B40_VERSION,
        "fixed_endpoint": True,
        "completed_epochs": B40_COMPLETED_EPOCHS,
        "parent_b37_checkpoint": str(parent_path),
        "parent_b37_checkpoint_sha256": parent_sha256,
        "parent_b37_completed_epochs": B40_PARENT_EPOCHS,
        "continuation": {
            "additional_epochs": B40_ADDITIONAL_EPOCHS,
            "optimizer": "AdamW",
            "optimizer_state_restored": False,
            "optimizer_reset_reason": "B37 E2 checkpoint did not serialize AdamW moment buffers",
            "head_lr": head_lr,
            "encoder_lr": head_lr * encoder_scale,
            "weight_decay": float(config["b40_weight_decay"]),
            "grad_clip": clip,
            "loader_seed": seed + B40_LOADER_SEED_OFFSET,
        },
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": str(parent_payload.get("base_checkpoint_sha256", "")),
        "base_payload_experiment": parent_payload.get("base_payload_experiment"),
        "base_state": model.base.state_dict(),
        "head_state": model.head.state_dict(),
        "model_state": model.state(),
        "encoder_sha256_initial": parent_encoder_sha256,
        "encoder_sha256_final": encoder_final_sha,
        "local_aux_weight": aux_weight,
        "base_reconstruction_448_max_abs_error": float(equivalence_error or 0.0),
        "training_studies": REPORT_ONLY_STUDIES,
        "training_series": B35_EXPECTED_SERIES,
        "training_supervision_cells": B35_EXPECTED_CELLS,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_used": False,
        "checkpoint_selection": "none; fixed one optimizer-reset continuation epoch",
        "preprocessing": parent_payload.get("preprocessing"),
        "sparse_mil": parent_payload.get("sparse_mil"),
        "encoder_finetune": model.finetune,
        "crop_policy": crop_policy,
        "label_confidence": confidence,
        "fill_policy": fill_policy,
        "fill_audit": fill_audit,
        "supervision": supervision,
        "series_policy_signature": B13_SERIES_SIGNATURE,
        "series_surface": series_summary,
        "metadata_repair": metadata_stats,
        "history": history,
        "governance": (
            "Prospective B40 continuation endpoint. It starts from the immutable "
            "completed B37 E2 model and declares the unavoidable fresh-AdamW reset. "
            "Expert58 is reused diagnostic only; do not select epochs, tune B40, or "
            "alter B37/B39 from that surface. Hidden competition evidence is required "
            "for promotion."
        ),
    }
    torch.save(payload, checkpoint)
    audit = {key: value for key, value in payload.items() if key not in {"base_state", "head_state"}}
    (output_root / "training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output_root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint, flush=True)
    return checkpoint


def load_b40_checkpoint(
    path: str | Path,
    *,
    base_checkpoint: str | Path,
    device,
):
    """Reconstruct the B40 endpoint from its exact B34 base checkpoint."""
    checkpoint = Path(path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("experiment") != B40_EXPERIMENT or payload.get("version") != B40_VERSION:
        raise ValueError("not a B40 optimizer-reset continuation checkpoint")
    if payload.get("fixed_endpoint") is not True or int(payload.get("completed_epochs", -1)) != B40_COMPLETED_EPOCHS:
        raise ValueError("B40 evaluation requires the completed fixed-E3 endpoint")
    if int(payload.get("parent_b37_completed_epochs", -1)) != B40_PARENT_EPOCHS:
        raise ValueError("B40 checkpoint does not certify its B37 E2 parent")
    if not str(payload.get("parent_b37_checkpoint_sha256", "")):
        raise ValueError("B40 checkpoint lacks its B37 parent fingerprint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0 or bool(payload.get("gold_labels_used", True)):
        raise ValueError("B40 checkpoint unexpectedly used expert labels")
    if int(payload.get("training_studies", -1)) != REPORT_ONLY_STUDIES:
        raise ValueError("B40 checkpoint has the wrong report-only training population")
    if int(payload.get("training_series", -1)) != B35_EXPECTED_SERIES:
        raise ValueError("B40 checkpoint has the wrong MRI series training surface")
    if int(payload.get("training_supervision_cells", -1)) != B35_EXPECTED_CELLS:
        raise ValueError("B40 checkpoint has the wrong supervision surface")

    base_path = Path(base_checkpoint).resolve()
    if sha256_file(base_path) != str(payload.get("base_checkpoint_sha256", "")):
        raise ValueError("B40 base checkpoint fingerprint mismatch")
    base, _ = load_phase9_checkpoint(base_path, expected_arm="llm_fill", device="cpu")
    sparse = payload.get("sparse_mil", {})
    finetune = payload.get("encoder_finetune", {})
    model_state = payload.get("model_state", {})
    model = B37HighResSparseMILResidual(
        base,
        grid_size=int(sparse.get("grid_size", B37_GRID_SIZE)),
        top_k=int(sparse.get("top_k", B37_TOP_K)),
        temperature=float(sparse.get("temperature", 1.0)),
        encoder_trainable_stages=int(finetune.get("encoder_trainable_stages", 1)),
        encoder_chunk_size=int(model_state.get("encoder_chunk_size", 4)),
    )
    model.base.load_state_dict(payload["base_state"], strict=True)
    model.head.load_state_dict(payload["head_state"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser("Train B40 B37-E2 optimizer-reset continuation")
    parser.add_argument("--config", default="config/b40_b37_e2_continuation.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--out-root", default=B40_RUN_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = dict(_read_config(args.config))
    train_b40(
        config,
        data_root=args.data_root,
        labels_root=args.labels_root,
        series_policy_path=args.series_policy,
        parent_checkpoint=args.parent_checkpoint,
        base_checkpoint=args.base_checkpoint,
        out_root=args.out_root,
        preflight_only=bool(args.preflight_only),
    )


if __name__ == "__main__":
    main()
