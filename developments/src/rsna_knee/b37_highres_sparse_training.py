"""Train B37: 448x448 native-crop B36 sparse MIL with one encoder stage free.

B37 is a fixed, prospective *joint* mechanism test.  It does not claim to isolate
resolution from localization: the primary question is whether the B36 sparse
pathology-specific localizer becomes useful once the representation retains much
more native in-plane information and the final ConvNeXt stage can adapt.

No expert labels enter gradients, checkpoint selection, stopping, or tuning.
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
from .b35_training import (
    B35_EXPECTED_CELLS,
    B35_EXPECTED_SERIES,
    _require_base_checkpoint,
    sha256_file,
)
from .b37_highres_sparse_mil import (
    B37_ENCODER_LR_SCALE,
    B37_LOCAL_AUX_WEIGHT,
    B37_RUN_ROOT,
    B37_VERSION,
    B37HighResSparseDataset,
    B37HighResSparseMILResidual,
    collate_b35,
    require_b37_sparse_contract,
)
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .label_confidence import rescale_label_confidence
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .phase9_supervision import (
    REPORT_ONLY_STUDIES,
    load_fill_merged_export,
    prepare_all_report_only_supervision,
)
from .runtime import autocast, make_scaler, resolve_runtime

B37_EXPERIMENT = "B37_highres448_pathology_sparse_topk_MIL_encoder_tail"
B37_EPOCHS = 2
B37_MICRO_BATCH = 2
B37_HEAD_LR = 1e-4
B37_WEIGHT_DECAY = 1e-4
B37_GRAD_CLIP = 1.0
B37_EQUIVALENCE_TOLERANCE = 2e-3
B37_CONSTRUCTION_SEED_OFFSET = 47_000_000
B37_LOADER_SEED_OFFSET = 47_100_000


def _save_recovery(
    out: Path,
    *,
    epoch: int,
    model: B37HighResSparseMILResidual,
    history: list[dict],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": B37_VERSION,
            "epoch": int(epoch),
            "fixed_endpoint": False,
            "model_selection_allowed": False,
            "base_state": model.base.state_dict(),
            "head_state": model.head.state_dict(),
            "history": history,
        },
        out / "recovery_latest.pt",
    )


def _move_batch(batch: dict, device) -> tuple:
    return (
        batch["volumes"].to(device, non_blocking=True),
        batch["slice_position"].to(device, non_blocking=True),
        batch["present"].to(device, non_blocking=True),
        batch["series_meta"].to(device, non_blocking=True),
        batch["target"].to(device, non_blocking=True),
        batch["weight"].to(device, non_blocking=True),
    )


def _losses(model, runtime, batch_tensors, multiplier_t, aux_weight: float):
    volumes, position, present, meta, target, weight = batch_tensors
    with autocast(runtime):
        out = model(volumes, present, meta, position)
        combined = target_balanced_weak_bce(
            out.logits,
            target,
            weight,
            multiplier_t,
        )
        local = target_balanced_weak_bce(
            out.local_logits,
            target,
            weight,
            multiplier_t,
        )
        total = combined + float(aux_weight) * local
    return out, total, combined, local


def _preflight(
    model,
    loader,
    runtime,
    multiplier_t,
    scaler,
    aux_weight: float,
) -> None:
    """One no-step forward/backward memory probe on the exact final batch shape."""
    print("[B37 preflight] forward/backward only; no optimizer step", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(runtime.device)
    model.train()
    model.zero_grad(set_to_none=True)
    batch = next(iter(loader))
    tensors = _move_batch(batch, runtime.device)
    volumes, _, present, meta, _, _ = tensors
    equivalence = model.base_equivalence_error_448(volumes, present, meta)
    print(
        f"[B37 preflight] reconstructed 448 B34 max|delta|={equivalence:.8g}",
        flush=True,
    )
    if equivalence > B37_EQUIVALENCE_TOLERANCE:
        raise RuntimeError(
            f"B37 448 reconstruction guard failed: {equivalence}"
        )
    _, total, combined, local = _losses(
        model,
        runtime,
        tensors,
        multiplier_t,
        aux_weight,
    )
    scaler.scale(total).backward()
    encoder_grad = any(
        p.grad is not None and torch.count_nonzero(p.grad).item() > 0
        for p in model.base.encoder.parameters()
        if p.requires_grad
    )
    evidence_grad = bool(
        model.head.evidence_weight.grad is not None
        and torch.count_nonzero(model.head.evidence_weight.grad).item() > 0
    )
    if not encoder_grad or not evidence_grad:
        raise RuntimeError("B37 preflight did not reach encoder tail and sparse evidence head")
    print(
        f"[B37 preflight] total={float(total):.6f} "
        f"combined={float(combined):.6f} local={float(local):.6f}",
        flush=True,
    )
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated(runtime.device) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(runtime.device) / (1024**3)
        print(
            f"[B37 preflight] CUDA peak allocated={peak_alloc:.2f} GiB "
            f"reserved={peak_reserved:.2f} GiB",
            flush=True,
        )
    model.zero_grad(set_to_none=True)
    print("[B37 preflight] PASS", flush=True)


def train_b37(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path = B37_RUN_ROOT,
    preflight_only: bool = False,
) -> Path | None:
    config = dict(config)
    config["data_root"] = str(Path(data_root).resolve())
    crop_policy = require_b37_sparse_contract(config)
    if int(config.get("b37_micro_batch", B37_MICRO_BATCH)) != B37_MICRO_BATCH:
        raise ValueError(f"B37 freezes micro-batch={B37_MICRO_BATCH}")

    seed = int(config.get("seed", 2026))
    seed_everything(seed + B37_CONSTRUCTION_SEED_OFFSET)
    runtime = resolve_runtime(config)
    print(runtime.describe(), flush=True)

    base_path = Path(base_checkpoint).resolve()
    base_model, base_payload = load_phase9_checkpoint(
        base_path,
        expected_arm="llm_fill",
        device="cpu",
    )
    _require_base_checkpoint(base_payload)
    encoder_initial_sha = encoder_state_sha256(base_model.encoder)

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    if len(train) != 4407:
        raise ValueError("B37 requires the complete 4,407-study training release")
    frame, fill_policy, fill_audit = load_fill_merged_export(labels_root)
    uids, targets, weights, supervision = prepare_all_report_only_supervision(
        train,
        frame,
    )
    if len(uids) != REPORT_ONLY_STUDIES:
        raise ValueError("B37 requires all 4,349 report-only studies")
    if int((weights > 0).sum()) != B35_EXPECTED_CELLS:
        raise ValueError("B37 supervision surface changed")
    if int(fill_audit.get("base_cells_overridden", -1)) != 0:
        raise ValueError("B37 requires the fill-only surface with zero B6 overrides")
    if list(fill_audit.get("excluded_targets", [])):
        raise ValueError("B37 requires all 12 targets")

    targets, confidence = rescale_label_confidence(targets, weights, config)
    base_confidence = base_payload.get("label_confidence", {})
    for key in ("positive_target", "negative_target"):
        if key in base_confidence and not np.isclose(
            float(confidence[key]),
            float(base_confidence[key]),
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError(f"B37 label confidence mismatch for {key}")

    series_policy = _load_series_policy(series_policy_path)
    if (
        series_policy.get("series_summary", {}).get("series_signature_sha256")
        != B13_SERIES_SIGNATURE
    ):
        raise ValueError("B37 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    expected_series = int(series_summary.get("eligible_recognized_plane_series", -1))
    if expected_series != B35_EXPECTED_SERIES:
        raise ValueError(
            f"B37 requires {B35_EXPECTED_SERIES} report-only MRI series; got {expected_series}"
        )
    if series_summary.get("viability_passed") is not True:
        raise ValueError("B37 all-series MRI surface failed viability")

    dataset_config = make_b7_dataset_config(config, root, train=False)
    dataset_config.tta_center_offsets = ()
    ds = B37HighResSparseDataset(
        uids,
        variable_index,
        dataset_config,
        crop_focus_policy=crop_policy,
        center_offsets=(0,),
        targets=targets,
        weights=weights,
    )
    loader = DataLoader(
        ds,
        batch_size=B37_MICRO_BATCH,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_b35,
        **runtime.loader_kwargs(seed=seed + B37_LOADER_SEED_OFFSET),
    )

    model = B37HighResSparseMILResidual(
        base_model,
        grid_size=int(config["b37_grid_size"]),
        top_k=int(config["b37_top_k"]),
        temperature=float(config["b37_temperature"]),
        encoder_trainable_stages=int(config["b37_encoder_trainable_stages"]),
        encoder_chunk_size=int(config["b37_encoder_chunk_size"]),
    ).to(runtime.device)
    model.train()

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    encoder_params = [
        p for p in model.base.encoder.parameters() if p.requires_grad
    ]
    if not head_params or not encoder_params:
        raise RuntimeError("B37 requires both sparse-head and encoder-tail parameters")
    if any(
        p.requires_grad
        for name, p in model.base.named_parameters()
        if not name.startswith("encoder.")
    ):
        raise RuntimeError("B37 non-encoder B34 parameters must remain frozen")

    head_lr = float(config.get("b37_head_lr", B37_HEAD_LR))
    encoder_scale = float(config["b37_encoder_lr_scale"])
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": head_lr, "name": "sparse_head"},
            {
                "params": encoder_params,
                "lr": head_lr * encoder_scale,
                "name": "encoder_tail",
            },
        ],
        weight_decay=float(config.get("b37_weight_decay", B37_WEIGHT_DECAY)),
    )
    scaler = make_scaler(runtime)
    target_multiplier = target_balance_multipliers(weights)
    multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)
    aux_weight = float(config.get("b37_local_aux_weight", B37_LOCAL_AUX_WEIGHT))
    clip = float(config.get("b37_grad_clip", B37_GRAD_CLIP))

    if preflight_only:
        _preflight(model, loader, runtime, multiplier_t, scaler, aux_weight)
        return None

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    equivalence_error = None

    for epoch in range(1, B37_EPOCHS + 1):
        started = time.monotonic()
        model.train()
        total_sum = combined_sum = local_sum = 0.0
        batches = studies_seen = series_seen = cells_seen = 0
        gate_gradient_seen = evidence_gradient_seen = encoder_gradient_seen = False

        for step, batch in enumerate(loader, start=1):
            tensors = _move_batch(batch, runtime.device)
            volumes, _, present, meta, target, weight = tensors
            if equivalence_error is None:
                equivalence_error = model.base_equivalence_error_448(
                    volumes,
                    present,
                    meta,
                )
                print(
                    f"[B37] reconstructed 448 B34 max|delta|={equivalence_error:.8g}",
                    flush=True,
                )
                if equivalence_error > B37_EQUIVALENCE_TOLERANCE:
                    raise RuntimeError("B37 reconstructed B34 guard failed")

            optimizer.zero_grad(set_to_none=True)
            out, total, combined, local = _losses(
                model,
                runtime,
                tensors,
                multiplier_t,
                aux_weight,
            )
            scaler.scale(total).backward()

            leaked = any(
                p.grad is not None
                for name, p in model.base.named_parameters()
                if not name.startswith("encoder.") and not p.requires_grad
            )
            if leaked:
                raise RuntimeError("B37 detected a gradient on frozen B34 hierarchy")
            gate_gradient_seen = gate_gradient_seen or bool(
                model.head.gate.grad is not None
                and torch.count_nonzero(model.head.gate.grad).item() > 0
            )
            evidence_gradient_seen = evidence_gradient_seen or bool(
                model.head.evidence_weight.grad is not None
                and torch.count_nonzero(model.head.evidence_weight.grad).item() > 0
            )
            encoder_gradient_seen = encoder_gradient_seen or any(
                p.grad is not None and torch.count_nonzero(p.grad).item() > 0
                for p in encoder_params
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

            if step % 50 == 0:
                elapsed = (time.monotonic() - started) / 60.0
                rate = elapsed / step
                remaining = rate * (len(loader) - step)
                print(
                    f"[B37] E{epoch} {step}/{len(loader)} "
                    f"total={total_sum/batches:.4f} "
                    f"combined={combined_sum/batches:.4f} "
                    f"local={local_sum/batches:.4f} "
                    f"gate_abs_mean={model.head.effective_gate().detach().abs().mean().item():.4f} "
                    f"elapsed={elapsed:.1f} min remaining~{remaining:.1f} min",
                    flush=True,
                )

        if studies_seen != REPORT_ONLY_STUDIES:
            raise RuntimeError("B37 epoch did not cover all report-only studies")
        if series_seen != B35_EXPECTED_SERIES:
            raise RuntimeError("B37 epoch did not cover all expected MRI series")
        if cells_seen != B35_EXPECTED_CELLS:
            raise RuntimeError("B37 epoch did not cover all supervision cells")
        if not (gate_gradient_seen and evidence_gradient_seen and encoder_gradient_seen):
            raise RuntimeError("B37 required gradient path was not active")

        row = {
            "epoch": epoch,
            "loss_total": total_sum / batches,
            "loss_combined": combined_sum / batches,
            "loss_local_aux": local_sum / batches,
            "batches": batches,
            "studies": studies_seen,
            "series": series_seen,
            "supervision_cells": cells_seen,
            "gate": model.head.state(),
            "encoder_sha256": encoder_state_sha256(model.base.encoder),
            "epoch_seconds": float(time.monotonic() - started),
        }
        history.append(row)
        print(
            f"[B37] E{epoch} total={row['loss_total']:.10f} "
            f"combined={row['loss_combined']:.10f} "
            f"local={row['loss_local_aux']:.10f} "
            f"time={row['epoch_seconds']/60:.1f} min",
            flush=True,
        )
        _save_recovery(out_root, epoch=epoch, model=model, history=history)

    encoder_final_sha = encoder_state_sha256(model.base.encoder)
    if encoder_final_sha == encoder_initial_sha:
        raise RuntimeError("B37 encoder tail was trainable but encoder fingerprint did not move")

    checkpoint = out_root / "b37_model.pt"
    payload = {
        "experiment": B37_EXPERIMENT,
        "version": B37_VERSION,
        "fixed_endpoint": True,
        "completed_epochs": B37_EPOCHS,
        "joint_hypothesis": (
            "higher in-plane information plus B36 sparse pathology-specific MIL, "
            "with limited final-stage ConvNeXt adaptation"
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
        "encoder_lr": head_lr * B37_ENCODER_LR_SCALE,
        "local_aux_weight": aux_weight,
        "base_reconstruction_448_max_abs_error": float(equivalence_error or 0.0),
        "training_studies": REPORT_ONLY_STUDIES,
        "training_series": B35_EXPECTED_SERIES,
        "training_supervision_cells": B35_EXPECTED_CELLS,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_used": False,
        "checkpoint_selection": "none; fixed epoch 2",
        "preprocessing": {
            "normalization": "full native volume before crop",
            "crop_fraction": 0.90,
            "crop_stage": "native resolution before deterministic resize",
            "image_size": 448,
            "deterministic_resize_count": 1,
            "resize": "bilinear antialias=True align_corners=False",
        },
        "sparse_mil": {
            "dense_slices": 32,
            "grid_size": int(model.head.grid_size),
            "regions_per_slice": int(model.head.n_regions),
            "top_k": int(model.head.top_k),
            "temperature": float(model.head.temperature),
            "local_aux_weight": aux_weight,
        },
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
            "Prospective joint B37 endpoint. Expert58 is reused diagnostic only. "
            "Do not tune resolution, grid size, top-k, crop fraction, target subset, "
            "or epoch after observing expert58. Hidden competition evidence is required "
            "for promotion."
        ),
    }
    torch.save(payload, checkpoint)
    audit = {k: v for k, v in payload.items() if k not in {"base_state", "head_state"}}
    (out_root / "training_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (out_root / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(checkpoint, flush=True)
    return checkpoint


def main() -> None:
    ap = argparse.ArgumentParser("Train B37 high-resolution sparse MIL")
    ap.add_argument("--config", default="config/b37_highres_sparse_448.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--series-policy", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--out-root", default=B37_RUN_ROOT)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    train_b37(
        config,
        data_root=args.data_root,
        labels_root=args.labels_root,
        series_policy_path=args.series_policy,
        base_checkpoint=args.base_checkpoint,
        out_root=args.out_root,
        preflight_only=bool(args.preflight_only),
    )


if __name__ == "__main__":
    main()
