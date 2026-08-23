"""Train B42 constant-area native-aspect rectangular sparse MIL.

The scientific endpoint remains B37's fixed two-epoch sparse-MIL training
contract.  B42 changes only in-plane geometry and the ragged plumbing needed to
avoid reintroducing a large common square.  Two studies are still one optimizer
batch: each is encoded sequentially and its backward contribution is weighted
by its exact target-balanced BCE denominator share, reproducing the historical
batch-2 objective without retaining both image graphs simultaneously.
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
from .b37_highres_sparse_training import (
    B37_CONSTRUCTION_SEED_OFFSET,
    B37_EPOCHS,
    B37_GRAD_CLIP,
    B37_HEAD_LR,
    B37_LOADER_SEED_OFFSET,
    B37_WEIGHT_DECAY,
    _format_memory_state,
    _largest_series_indices,
    _memory_state,
    _save_recovery,
    _trim_host_memory,
)
from .b42_constant_area_aspect_sparse_mil import (
    B42_EFFECTIVE_BATCH,
    B42_EXPERIMENT,
    B42_RUN_ROOT,
    B42_VERSION,
    B42ConstantAreaAspectDataset,
    B42ConstantAreaAspectSparseMILResidual,
    b42_preprocessing_state,
    collate_b42,
    require_b42_contract,
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

B42_EPOCHS = B37_EPOCHS
# B42 intentionally reuses B37's streams so head initialization and study order
# do not become extra variables in the B37-vs-B42 geometry comparison.
B42_CONSTRUCTION_SEED_OFFSET = B37_CONSTRUCTION_SEED_OFFSET
B42_LOADER_SEED_OFFSET = B37_LOADER_SEED_OFFSET
B42_CHECKPOINT = "b42_model.pt"


def _move_study(item: dict, device) -> tuple:
    volumes = [volume.to(device, non_blocking=True) for volume in item["volumes"]]
    return (
        volumes,
        item["slice_position"].to(device, non_blocking=True),
        item["present"].to(device, non_blocking=True),
        item["series_meta"].to(device, non_blocking=True),
        item["target"].to(device, non_blocking=True).unsqueeze(0),
        item["weight"].to(device, non_blocking=True).unsqueeze(0),
    )


def _study_mass(weight: torch.Tensor, target_multiplier: torch.Tensor) -> float:
    w = weight.reshape(-1, weight.shape[-1]).to(dtype=torch.float32, device="cpu")
    m = target_multiplier.to(dtype=torch.float32, device="cpu")
    return float((w * m[None, :]).sum().item())


def _losses(model, runtime, tensors, multiplier_t, aux_weight: float):
    volumes, position, present, meta, target, weight = tensors
    with autocast(runtime):
        out = model(volumes, present, meta, position)
        combined = target_balanced_weak_bce(
            out.logits, target, weight, multiplier_t
        )
        local = target_balanced_weak_bce(
            out.local_logits, target, weight, multiplier_t
        )
        total = combined + float(aux_weight) * local
    return out, total, combined, local


def _batch_scales(items: list[dict], multiplier_cpu: torch.Tensor) -> list[float]:
    masses = [_study_mass(item["weight"], multiplier_cpu) for item in items]
    total = float(sum(masses))
    if total > 0:
        return [float(mass / total) for mass in masses]
    # Historical target_balanced_weak_bce returns a graph-connected zero when an
    # entire batch has no usable cells. Equal zero-loss contributions preserve
    # that behavior and still expose every study's MRI to the forward path.
    return [1.0 / len(items)] * len(items)


def _synthetic_geometry_preflight(model, runtime) -> None:
    shapes = ((448, 448), (320, 640), (640, 320), (256, 800))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for height, width in shapes:
            volume = torch.ones(
                32, 3, height, width, dtype=torch.float32, device=runtime.device
            )
            present = torch.ones(1, device=runtime.device)
            meta = torch.zeros((1, 3), dtype=torch.long, device=runtime.device)
            position = torch.linspace(-1.0, 1.0, 32, device=runtime.device).unsqueeze(0)
            with autocast(runtime):
                out = model([volume], present, meta, position)
            if not torch.isfinite(out.base_logits).all() or not torch.isfinite(out.logits).all():
                raise RuntimeError(f"B42 synthetic {height}x{width} produced non-finite logits")
            print(
                f"[B42 preflight] synthetic={height}x{width} "
                f"feature~{height//32}x{width//32} PASS",
                flush=True,
            )
            del volume, present, meta, position, out
    if was_training:
        model.train(True)


def _preflight(
    model,
    loader,
    runtime,
    multiplier_t,
    multiplier_cpu,
    scaler,
    aux_weight: float,
) -> None:
    print("[B42 preflight] forward/backward only; no optimizer step", flush=True)
    if runtime.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(runtime.device)

    _synthetic_geometry_preflight(model, runtime)
    model.train()
    model.zero_grad(set_to_none=True)
    indices = _largest_series_indices(loader.dataset, B42_EFFECTIVE_BATCH)
    items = [loader.dataset[index] for index in indices]
    counts = [int(item["present"].shape[0]) for item in items]
    shapes = [
        [(int(v.shape[-2]), int(v.shape[-1])) for v in item["volumes"] if v.ndim == 4]
        for item in items
    ]
    print(
        f"[B42 preflight] worst-case series/study={counts} ragged_shapes={shapes}",
        flush=True,
    )

    scales = _batch_scales(items, multiplier_cpu)
    total_value = combined_value = local_value = 0.0
    for item, scale in zip(items, scales):
        tensors = _move_study(item, runtime.device)
        out, total, combined, local = _losses(
            model, runtime, tensors, multiplier_t, aux_weight
        )
        scaler.scale(total * float(scale)).backward()
        total_value += float(total.detach().item()) * float(scale)
        combined_value += float(combined.detach().item()) * float(scale)
        local_value += float(local.detach().item()) * float(scale)
        del tensors, out, total, combined, local

    encoder_grad = any(
        p.grad is not None and torch.count_nonzero(p.grad).item() > 0
        for p in model.base.encoder.parameters()
        if p.requires_grad
    )
    evidence_grad = bool(
        model.head.evidence_weight.grad is not None
        and torch.count_nonzero(model.head.evidence_weight.grad).item() > 0
    )
    leaked = any(
        p.grad is not None
        for name, p in model.base.named_parameters()
        if not name.startswith("encoder.") and not p.requires_grad
    )
    if leaked:
        raise RuntimeError("B42 preflight found a gradient on frozen B34 hierarchy")
    if not encoder_grad or not evidence_grad:
        raise RuntimeError("B42 preflight did not reach encoder tail and sparse evidence head")
    print(
        f"[B42 preflight] total={total_value:.6f} combined={combined_value:.6f} "
        f"local={local_value:.6f}",
        flush=True,
    )
    print(
        f"[B42 preflight] {_format_memory_state(_memory_state(runtime))}",
        flush=True,
    )
    model.zero_grad(set_to_none=True)
    del items
    _trim_host_memory()
    print("[B42 preflight] PASS", flush=True)


def train_b42(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path = B42_RUN_ROOT,
    preflight_only: bool = False,
) -> Path | None:
    config = dict(config)
    config["data_root"] = str(Path(data_root).resolve())
    crop_policy = require_b42_contract(config)
    if int(config.get("b37_micro_batch", B42_EFFECTIVE_BATCH)) != B42_EFFECTIVE_BATCH:
        raise ValueError("B42 retains B37 effective batch size 2")

    seed = int(config.get("seed", 2026))
    seed_everything(seed + B42_CONSTRUCTION_SEED_OFFSET)
    runtime = resolve_runtime(config)
    print(runtime.describe(), flush=True)

    base_path = Path(base_checkpoint).resolve()
    base_model, base_payload = load_phase9_checkpoint(
        base_path, expected_arm="llm_fill", device="cpu"
    )
    _require_base_checkpoint(base_payload)
    encoder_initial_sha = encoder_state_sha256(base_model.encoder)

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    if len(train) != 4407:
        raise ValueError("B42 requires the complete 4,407-study training release")
    frame, fill_policy, fill_audit = load_fill_merged_export(labels_root)
    uids, targets, weights, supervision = prepare_all_report_only_supervision(train, frame)
    if len(uids) != REPORT_ONLY_STUDIES:
        raise ValueError("B42 requires all 4,349 report-only studies")
    if int((weights > 0).sum()) != B35_EXPECTED_CELLS:
        raise ValueError("B42 supervision surface changed")
    if int(fill_audit.get("base_cells_overridden", -1)) != 0:
        raise ValueError("B42 requires zero B6 overrides")
    if list(fill_audit.get("excluded_targets", [])):
        raise ValueError("B42 requires all 12 targets")

    targets, confidence = rescale_label_confidence(targets, weights, config)
    base_confidence = base_payload.get("label_confidence", {})
    for key in ("positive_target", "negative_target"):
        if key in base_confidence and not np.isclose(
            float(confidence[key]), float(base_confidence[key]), atol=1e-12, rtol=0
        ):
            raise ValueError(f"B42 label confidence mismatch for {key}")

    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B42 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    expected_series = int(series_summary.get("eligible_recognized_plane_series", -1))
    if expected_series != B35_EXPECTED_SERIES:
        raise ValueError(
            f"B42 requires {B35_EXPECTED_SERIES} MRI series; got {expected_series}"
        )
    if series_summary.get("viability_passed") is not True:
        raise ValueError("B42 all-series MRI surface failed viability")

    dataset_config = make_b7_dataset_config(config, root, train=False)
    dataset_config.tta_center_offsets = ()
    dataset = B42ConstantAreaAspectDataset(
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
        batch_size=B42_EFFECTIVE_BATCH,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_b42,
        **runtime.loader_kwargs(seed=seed + B42_LOADER_SEED_OFFSET),
    )

    model = B42ConstantAreaAspectSparseMILResidual(
        base_model,
        grid_size=int(config["b37_grid_size"]),
        top_k=int(config["b37_top_k"]),
        temperature=float(config["b37_temperature"]),
        encoder_trainable_stages=int(config["b37_encoder_trainable_stages"]),
        encoder_chunk_size=int(config["b37_encoder_chunk_size"]),
    ).to(runtime.device)
    model.train()

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    encoder_params = [p for p in model.base.encoder.parameters() if p.requires_grad]
    if not head_params or not encoder_params:
        raise RuntimeError("B42 requires sparse-head and encoder-tail parameters")
    if any(
        p.requires_grad
        for name, p in model.base.named_parameters()
        if not name.startswith("encoder.")
    ):
        raise RuntimeError("B42 non-encoder B34 parameters must remain frozen")

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
    multiplier_cpu = torch.from_numpy(target_multiplier)
    multiplier_t = multiplier_cpu.to(runtime.device)
    aux_weight = float(config["b37_local_aux_weight"])
    clip = float(config.get("b37_grad_clip", B37_GRAD_CLIP))

    if preflight_only:
        _preflight(
            model,
            loader,
            runtime,
            multiplier_t,
            multiplier_cpu,
            scaler,
            aux_weight,
        )
        return None

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    for epoch in range(1, B42_EPOCHS + 1):
        started = time.monotonic()
        if runtime.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(runtime.device)
        model.train()
        total_sum = combined_sum = local_sum = 0.0
        batches = studies_seen = series_seen = cells_seen = 0
        gate_gradient_seen = evidence_gradient_seen = encoder_gradient_seen = False

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
                p.grad is not None
                for name, p in model.base.named_parameters()
                if not name.startswith("encoder.") and not p.requires_grad
            )
            if leaked:
                raise RuntimeError("B42 detected a gradient on frozen B34 hierarchy")
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
                gate_abs_mean = model.head.effective_gate().detach().abs().mean().item()

            del items
            if report_progress:
                _trim_host_memory()
                print(
                    f"[B42] E{epoch} {step}/{len(loader)} "
                    f"total={total_sum/batches:.4f} combined={combined_sum/batches:.4f} "
                    f"local={local_sum/batches:.4f} gate_abs_mean={gate_abs_mean:.4f} "
                    f"elapsed={elapsed:.1f} min remaining~{remaining:.1f} min "
                    f"{_format_memory_state(_memory_state(runtime))}",
                    flush=True,
                )

        if studies_seen != REPORT_ONLY_STUDIES:
            raise RuntimeError("B42 epoch did not cover all report-only studies")
        if series_seen != B35_EXPECTED_SERIES:
            raise RuntimeError("B42 epoch did not cover all expected MRI series")
        if cells_seen != B35_EXPECTED_CELLS:
            raise RuntimeError("B42 epoch did not cover all supervision cells")
        if not (gate_gradient_seen and evidence_gradient_seen and encoder_gradient_seen):
            raise RuntimeError("B42 required gradient path was not active")

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
            "memory": _memory_state(runtime),
        }
        history.append(row)
        print(
            f"[B42] E{epoch} total={row['loss_total']:.10f} "
            f"combined={row['loss_combined']:.10f} local={row['loss_local_aux']:.10f} "
            f"time={row['epoch_seconds']/60:.1f} min",
            flush=True,
        )
        _save_recovery(
            out_root, epoch=epoch, model=model, history=history, version=B42_VERSION
        )

    encoder_final_sha = encoder_state_sha256(model.base.encoder)
    if encoder_final_sha == encoder_initial_sha:
        raise RuntimeError("B42 encoder fingerprint did not move")

    checkpoint = out_root / B42_CHECKPOINT
    preprocessing = b42_preprocessing_state()
    payload = {
        "experiment": B42_EXPERIMENT,
        "version": B42_VERSION,
        "fixed_endpoint": True,
        "completed_epochs": B42_EPOCHS,
        "joint_hypothesis": (
            "preserving native in-plane aspect ratio while restoring approximately "
            "the B37 448^2 anatomical pixel/feature budget recovers representation "
            "quality lost by B41's large square padding"
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
        "base_reconstruction_448_max_abs_error": None,
        "base_hierarchy_guard": (
            "B34 non-encoder hierarchy frozen exactly; direct square-tensor equivalence "
            "is not applicable to per-series ragged rectangular inputs"
        ),
        "training_studies": REPORT_ONLY_STUDIES,
        "training_series": B35_EXPECTED_SERIES,
        "training_supervision_cells": B35_EXPECTED_CELLS,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_used": False,
        "checkpoint_selection": "none; fixed epoch 2",
        "preprocessing": preprocessing,
        "effective_batching": {
            "studies_per_optimizer_step": B42_EFFECTIVE_BATCH,
            "ragged_studies_processed_sequentially": True,
            "loss_accumulation": (
                "per-study target-balanced numerator contribution weighted by its "
                "known effective-denominator share; exactly reproduces batch-2 loss"
            ),
            "construction_seed_offset": B42_CONSTRUCTION_SEED_OFFSET,
            "loader_seed_offset": B42_LOADER_SEED_OFFSET,
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
            "Prospective B42 fixed endpoint. Do not tune reference area, aspect policy, "
            "padding mode, local grid, top-k, target subset, learning rates, or epoch "
            "count after Expert-58. Hidden competition evidence is required for promotion."
        ),
    }
    torch.save(payload, checkpoint)
    audit = {k: v for k, v in payload.items() if k not in {"base_state", "head_state"}}
    (out_root / "training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out_root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint, flush=True)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser("Train B42 constant-area native-aspect sparse MIL")
    parser.add_argument("--config", default="config/b42_constant_area_aspect_sparse.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--out-root", default=B42_RUN_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = dict(_read_config(args.config))
    train_b42(
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
