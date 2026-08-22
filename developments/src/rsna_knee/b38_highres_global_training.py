"""Train B38: 448 native-crop global B34 ablation with one encoder tail stage free.

B38 reuses the frozen all-target B6-preserved LLM-fill supervision and B34
checkpoint surface from B37, but it has no sparse-MIL head or local auxiliary
loss.  The only trainable parameters are the final ConvNeXt stage and output
normalization.  Its 16 centres exactly reproduce the historical B34 centre
positions at each evaluation offset.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
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
from .b38_highres_global import (
    B38_CROP_FRACTION,
    B38_RUN_ROOT,
    B38_TAIL_REFERENCE_LR,
    B38_VERSION,
    B38HighResGlobalDataset,
    B38HighResGlobalTail,
    collate_b35,
    require_b38_global_contract,
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

B38_EXPERIMENT = "B38_highres448_global16_encoder_tail_ablation"
B38_EPOCHS = 2
B38_MICRO_BATCH = 2
B38_WEIGHT_DECAY = 1e-4
B38_GRAD_CLIP = 1.0
B38_EQUIVALENCE_TOLERANCE = 2e-3
B38_CONSTRUCTION_SEED_OFFSET = 48_000_000
B38_LOADER_SEED_OFFSET = 48_100_000


def _largest_series_indices(dataset, batch_size: int) -> tuple[int, ...]:
    """Return a deterministic preflight batch with the largest study series counts."""
    if int(batch_size) < 1:
        raise ValueError("B38 preflight batch size must be positive")
    ranked = sorted(
        range(len(dataset)),
        key=lambda idx: (
            -len(dataset.series_records[dataset.study_uids[idx]]),
            str(dataset.study_uids[idx]),
        ),
    )
    if len(ranked) < int(batch_size):
        raise ValueError("B38 dataset is smaller than the preflight batch")
    return tuple(ranked[: int(batch_size)])


def _memory_state(runtime) -> dict[str, float]:
    """Return process, host, and CUDA memory telemetry in GiB."""
    status: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {"VmRSS", "VmHWM"}:
                status[key] = int(value.strip().split()[0])
    except (FileNotFoundError, OSError, ValueError):
        pass

    available_kib = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_kib = int(line.split()[1])
                break
    except (FileNotFoundError, OSError, ValueError):
        pass

    gib = float(1024**2)
    result = {
        "rss_gib": float(status.get("VmRSS", 0)) / gib,
        "rss_peak_gib": float(status.get("VmHWM", 0)) / gib,
        "system_available_gib": float(available_kib) / gib,
        "cuda_allocated_gib": 0.0,
        "cuda_reserved_gib": 0.0,
        "cuda_peak_allocated_gib": 0.0,
        "cuda_peak_reserved_gib": 0.0,
    }
    if runtime.device.type == "cuda" and torch.cuda.is_available():
        divisor = float(1024**3)
        result.update(
            {
                "cuda_allocated_gib": torch.cuda.memory_allocated(runtime.device)
                / divisor,
                "cuda_reserved_gib": torch.cuda.memory_reserved(runtime.device)
                / divisor,
                "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(
                    runtime.device
                )
                / divisor,
                "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(
                    runtime.device
                )
                / divisor,
            }
        )
    return result


def _format_memory_state(state: dict[str, float]) -> str:
    return (
        f"rss={state['rss_gib']:.2f}GiB "
        f"rss_peak={state['rss_peak_gib']:.2f}GiB "
        f"host_available={state['system_available_gib']:.2f}GiB "
        f"cuda={state['cuda_allocated_gib']:.2f}/"
        f"{state['cuda_reserved_gib']:.2f}GiB "
        f"cuda_peak={state['cuda_peak_allocated_gib']:.2f}/"
        f"{state['cuda_peak_reserved_gib']:.2f}GiB"
    )


def _trim_host_memory() -> None:
    """Release unreachable Python objects and return free glibc arenas."""
    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _save_recovery(
    out: Path,
    *,
    epoch: int,
    model: B38HighResGlobalTail,
    history: list[dict],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": B38_VERSION,
            "epoch": int(epoch),
            "fixed_endpoint": False,
            "model_selection_allowed": False,
            "base_state": model.base.state_dict(),
            "history": history,
        },
        out / "recovery_latest.pt",
    )


def _move_batch(batch: dict, device) -> tuple:
    """Move only B38's required tensors; slice positions are deliberately unused."""
    return (
        batch["volumes"].to(device, non_blocking=True),
        batch["present"].to(device, non_blocking=True),
        batch["series_meta"].to(device, non_blocking=True),
        batch["target"].to(device, non_blocking=True),
        batch["weight"].to(device, non_blocking=True),
    )


def _loss(model, runtime, batch_tensors, multiplier_t):
    volumes, present, meta, target, weight = batch_tensors
    with autocast(runtime):
        out = model(volumes, present, meta)
        loss = target_balanced_weak_bce(
            out.logits,
            target,
            weight,
            multiplier_t,
        )
    return out, loss


def _preflight(model, loader, runtime, multiplier_t, scaler) -> None:
    """One no-step forward/backward probe on the largest-series batch shape."""
    print("[B38 preflight] forward/backward only; no optimizer step", flush=True)
    if runtime.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(runtime.device)
    model.train()
    model.zero_grad(set_to_none=True)
    indices = _largest_series_indices(loader.dataset, B38_MICRO_BATCH)
    items = [loader.dataset[idx] for idx in indices]
    counts = [int(item["present"].shape[0]) for item in items]
    batch = collate_b35(items)
    del items
    print(
        f"[B38 preflight] worst-case series/study={counts} "
        f"padded_series={int(batch['present'].shape[1])}",
        flush=True,
    )
    tensors = _move_batch(batch, runtime.device)
    volumes, present, meta, _, _ = tensors
    equivalence = model.base_equivalence_error_448(volumes, present, meta)
    print(
        f"[B38 preflight] reconstructed 448 B34 max|delta|={equivalence:.8g}",
        flush=True,
    )
    if equivalence > B38_EQUIVALENCE_TOLERANCE:
        raise RuntimeError(
            f"B38 448 reconstruction guard failed: {equivalence}"
        )
    _, loss = _loss(model, runtime, tensors, multiplier_t)
    scaler.scale(loss).backward()
    encoder_grad = any(
        p.grad is not None and torch.count_nonzero(p.grad).item() > 0
        for p in model.base.encoder.parameters()
        if p.requires_grad
    )
    if not encoder_grad:
        raise RuntimeError("B38 preflight did not reach the encoder tail")
    print(
        f"[B38 preflight] loss={loss.detach().item():.6f}",
        flush=True,
    )
    print(
        f"[B38 preflight] {_format_memory_state(_memory_state(runtime))}",
        flush=True,
    )
    model.zero_grad(set_to_none=True)
    del batch, tensors, volumes, present, meta, loss
    _trim_host_memory()
    print("[B38 preflight] PASS", flush=True)


def train_b38(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path = B38_RUN_ROOT,
    preflight_only: bool = False,
) -> Path | None:
    """Train exactly two B38 epochs, or perform only its no-step preflight."""
    config = dict(config)
    config["data_root"] = str(Path(data_root).resolve())
    crop_policy = require_b38_global_contract(config)
    if int(config.get("b38_micro_batch", B38_MICRO_BATCH)) != B38_MICRO_BATCH:
        raise ValueError(f"B38 freezes micro-batch={B38_MICRO_BATCH}")

    seed = int(config.get("seed", 2026))
    seed_everything(seed + B38_CONSTRUCTION_SEED_OFFSET)
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
        raise ValueError("B38 requires the complete 4,407-study training release")
    frame, fill_policy, fill_audit = load_fill_merged_export(labels_root)
    uids, targets, weights, supervision = prepare_all_report_only_supervision(
        train,
        frame,
    )
    if len(uids) != REPORT_ONLY_STUDIES:
        raise ValueError("B38 requires all 4,349 report-only studies")
    if int((weights > 0).sum()) != B35_EXPECTED_CELLS:
        raise ValueError("B38 supervision surface changed")
    if int(fill_audit.get("base_cells_overridden", -1)) != 0:
        raise ValueError("B38 requires the fill-only surface with zero B6 overrides")
    if list(fill_audit.get("excluded_targets", [])):
        raise ValueError("B38 requires all 12 targets")

    targets, confidence = rescale_label_confidence(targets, weights, config)
    base_confidence = base_payload.get("label_confidence", {})
    for key in ("positive_target", "negative_target"):
        if key in base_confidence and not np.isclose(
            float(confidence[key]),
            float(base_confidence[key]),
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError(f"B38 label confidence mismatch for {key}")

    series_policy = _load_series_policy(series_policy_path)
    if (
        series_policy.get("series_summary", {}).get("series_signature_sha256")
        != B13_SERIES_SIGNATURE
    ):
        raise ValueError("B38 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    expected_series = int(series_summary.get("eligible_recognized_plane_series", -1))
    if expected_series != B35_EXPECTED_SERIES:
        raise ValueError(
            f"B38 requires {B35_EXPECTED_SERIES} report-only MRI series; "
            f"got {expected_series}"
        )
    if series_summary.get("viability_passed") is not True:
        raise ValueError("B38 all-series MRI surface failed viability")

    dataset_config = make_b7_dataset_config(config, root, train=False)
    dataset_config.tta_center_offsets = ()
    ds = B38HighResGlobalDataset(
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
        batch_size=B38_MICRO_BATCH,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_b35,
        **runtime.loader_kwargs(seed=seed + B38_LOADER_SEED_OFFSET),
    )

    model = B38HighResGlobalTail(
        base_model,
        encoder_trainable_stages=int(config["b38_encoder_trainable_stages"]),
        encoder_chunk_size=int(config["b38_encoder_chunk_size"]),
    ).to(runtime.device)
    model.train()

    encoder_params = [
        p for p in model.base.encoder.parameters() if p.requires_grad
    ]
    if not encoder_params:
        raise RuntimeError("B38 requires trainable encoder-tail parameters")
    if any(
        p.requires_grad
        for name, p in model.base.named_parameters()
        if not name.startswith("encoder.")
    ):
        raise RuntimeError("B38 non-encoder B34 parameters must remain frozen")

    reference_lr = float(
        config.get("b38_tail_reference_lr", B38_TAIL_REFERENCE_LR)
    )
    encoder_scale = float(config["b38_encoder_lr_scale"])
    tail_lr = reference_lr * encoder_scale
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": tail_lr, "name": "encoder_tail"}],
        weight_decay=float(config.get("b38_weight_decay", B38_WEIGHT_DECAY)),
    )
    scaler = make_scaler(runtime)
    target_multiplier = target_balance_multipliers(weights)
    multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)
    clip = float(config.get("b38_grad_clip", B38_GRAD_CLIP))

    if preflight_only:
        _preflight(model, loader, runtime, multiplier_t, scaler)
        return None

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    equivalence_error = None

    for epoch in range(1, B38_EPOCHS + 1):
        started = time.monotonic()
        if runtime.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(runtime.device)
        model.train()
        loss_sum = 0.0
        batches = studies_seen = series_seen = cells_seen = 0
        encoder_gradient_seen = False

        for step, batch in enumerate(loader, start=1):
            tensors = _move_batch(batch, runtime.device)
            volumes, present, meta, target, weight = tensors
            if equivalence_error is None:
                equivalence_error = model.base_equivalence_error_448(
                    volumes,
                    present,
                    meta,
                )
                print(
                    f"[B38] reconstructed 448 B34 max|delta|="
                    f"{equivalence_error:.8g}",
                    flush=True,
                )
                if equivalence_error > B38_EQUIVALENCE_TOLERANCE:
                    raise RuntimeError("B38 reconstructed B34 guard failed")

            optimizer.zero_grad(set_to_none=True)
            out, loss = _loss(model, runtime, tensors, multiplier_t)
            scaler.scale(loss).backward()

            leaked = any(
                p.grad is not None
                for name, p in model.base.named_parameters()
                if not name.startswith("encoder.") and not p.requires_grad
            )
            if leaked:
                raise RuntimeError("B38 detected a gradient on frozen B34 hierarchy")
            encoder_gradient_seen = encoder_gradient_seen or any(
                p.grad is not None and torch.count_nonzero(p.grad).item() > 0
                for p in encoder_params
            )

            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(encoder_params, clip)
            scaler.step(optimizer)
            scaler.update()

            active = weight > 0
            loss_sum += float(loss.detach().item())
            batches += 1
            studies_seen += int(volumes.shape[0])
            series_seen += int(present.sum().item())
            cells_seen += int(active.sum().item())

            report_progress = step % 50 == 0
            if report_progress:
                elapsed = (time.monotonic() - started) / 60.0
                rate = elapsed / step
                remaining = rate * (len(loader) - step)

            # Drop the completed padded 448 batch before DataLoader constructs
            # another variable-size batch.  This avoids the worker-scope host RAM
            # multiplication that caused B37's first launch to be OOM-killed.
            del (
                batch,
                tensors,
                volumes,
                present,
                meta,
                target,
                weight,
                out,
                loss,
                active,
            )

            if report_progress:
                _trim_host_memory()
                memory = _memory_state(runtime)
                print(
                    f"[B38] E{epoch} {step}/{len(loader)} "
                    f"loss={loss_sum/batches:.4f} "
                    f"elapsed={elapsed:.1f} min remaining~{remaining:.1f} min "
                    f"{_format_memory_state(memory)}",
                    flush=True,
                )

        if studies_seen != REPORT_ONLY_STUDIES:
            raise RuntimeError("B38 epoch did not cover all report-only studies")
        if series_seen != B35_EXPECTED_SERIES:
            raise RuntimeError("B38 epoch did not cover all expected MRI series")
        if cells_seen != B35_EXPECTED_CELLS:
            raise RuntimeError("B38 epoch did not cover all supervision cells")
        if not encoder_gradient_seen:
            raise RuntimeError("B38 required encoder-tail gradient path was not active")

        row = {
            "epoch": epoch,
            "loss_global": loss_sum / batches,
            "batches": batches,
            "studies": studies_seen,
            "series": series_seen,
            "supervision_cells": cells_seen,
            "encoder_sha256": encoder_state_sha256(model.base.encoder),
            "epoch_seconds": float(time.monotonic() - started),
            "memory": _memory_state(runtime),
        }
        history.append(row)
        print(
            f"[B38] E{epoch} global={row['loss_global']:.10f} "
            f"time={row['epoch_seconds']/60:.1f} min",
            flush=True,
        )
        _save_recovery(out_root, epoch=epoch, model=model, history=history)

    encoder_final_sha = encoder_state_sha256(model.base.encoder)
    if encoder_final_sha == encoder_initial_sha:
        raise RuntimeError(
            "B38 encoder tail was trainable but encoder fingerprint did not move"
        )

    checkpoint = out_root / "b38_model.pt"
    payload = {
        "experiment": B38_EXPERIMENT,
        "version": B38_VERSION,
        "fixed_endpoint": True,
        "completed_epochs": B38_EPOCHS,
        "hypothesis": (
            "higher in-plane information plus limited final-stage ConvNeXt "
            "adaptation can improve the frozen B34 global hierarchy without "
            "sparse-MIL, local residuals, or extra slice centres"
        ),
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": sha256_file(base_path),
        "base_payload_experiment": base_payload.get("experiment"),
        "base_state": model.base.state_dict(),
        "model_state": model.state(),
        "encoder_sha256_initial": encoder_initial_sha,
        "encoder_sha256_final": encoder_final_sha,
        "tail_reference_lr": reference_lr,
        "encoder_lr_scale": encoder_scale,
        "encoder_lr": tail_lr,
        "base_reconstruction_448_max_abs_error": float(equivalence_error or 0.0),
        "training_studies": REPORT_ONLY_STUDIES,
        "training_series": B35_EXPECTED_SERIES,
        "training_supervision_cells": B35_EXPECTED_CELLS,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_used": False,
        "checkpoint_selection": "none; fixed epoch 2",
        "preprocessing": {
            "normalization": "full native volume before crop",
            "crop_fraction": B38_CROP_FRACTION,
            "crop_stage": "native resolution before deterministic resize",
            "image_size": 448,
            "deterministic_resize_count": 1,
            "resize": "bilinear antialias=True align_corners=False",
            "centres": "historical B34 sixteen deterministic centres",
        },
        "global_model": {
            "n_slices": 16,
            "aggregation": "frozen B34 global hierarchy only",
            "sparse_mil": False,
            "local_auxiliary_loss": False,
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
            "B38 is a fixed post-B37 global-only ablation. Expert58 is reused "
            "diagnostic only: it must not change B38's resolution, crop, centres, "
            "tail depth, learning rate, target subset, or fixed epoch count. "
            "Hidden competition evidence is required for promotion."
        ),
    }
    torch.save(payload, checkpoint)
    audit = {key: value for key, value in payload.items() if key != "base_state"}
    (out_root / "training_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    (out_root / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    print(checkpoint, flush=True)
    return checkpoint


def main() -> None:
    ap = argparse.ArgumentParser("Train B38 high-resolution global-tail ablation")
    ap.add_argument("--config", default="config/b38_highres_global_448.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--series-policy", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--out-root", default=B38_RUN_ROOT)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    train_b38(
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
