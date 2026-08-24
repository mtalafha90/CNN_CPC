"""Train the prospective B45 plane-calibrated target-conditioned sparse MIL.

B45 inherits the complete B42 supervision, constant-area ragged geometry,
encoder-tail adaptation, effective batch size, optimizer, losses, learning rates,
and fixed two-epoch endpoint.  The only scientific change is the local sparse-MIL
routing implemented in :mod:`b45_plane_calibrated_sparse_mil`.
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
    _memory_state,
    _save_recovery,
    _trim_host_memory,
)
from .b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectDataset,
    b42_preprocessing_state,
    collate_b42,
)
from .b42_constant_area_aspect_sparse_training import (
    _batch_scales,
    _largest_series_indices,
    _losses,
    _move_study,
)
from .b45_plane_calibrated_sparse_mil import (
    B45_EFFECTIVE_BATCH,
    B45_EXPERIMENT,
    B45_PLANE_COUNT,
    B45_PLANE_POOLING,
    B45_PLANE_ROUTER_INIT,
    B45_PLANE_ROUTER_TEMPERATURE,
    B45_REMOVE_PLANE_EMBEDDING_FROM_TOKEN_SCORE,
    B45_RUN_ROOT,
    B45_VERSION,
    B45PlaneCalibratedSparseMILResidual,
    require_b45_contract,
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

B45_EPOCHS = B37_EPOCHS
B45_CONSTRUCTION_SEED_OFFSET = B37_CONSTRUCTION_SEED_OFFSET
B45_LOADER_SEED_OFFSET = B37_LOADER_SEED_OFFSET
B45_CHECKPOINT = "b45_model.pt"


def _router_entropy(weights: torch.Tensor) -> float:
    p = weights.detach().float().clamp_min(1e-12)
    return float((-(p * p.log()).sum(dim=-1)).mean().cpu().item())


def _synthetic_geometry_preflight(model, runtime) -> None:
    """Exercise B42 rectangular geometry with a valid explicit sagittal plane."""
    shapes = ((448, 448), (320, 640), (640, 320), (256, 800))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for height, width in shapes:
            volume = torch.ones(
                32, 3, height, width, dtype=torch.float32, device=runtime.device
            )
            present = torch.ones(1, device=runtime.device)
            # B45 only accepts recognized anatomical plane pools.  Use sagittal
            # with unknown fluid/fat metadata for this geometry-only probe.
            meta = torch.tensor([[1, 0, 0]], dtype=torch.long, device=runtime.device)
            position = torch.linspace(0.0, 1.0, 32, device=runtime.device).unsqueeze(0)
            with autocast(runtime):
                out = model([volume], present, meta, position)
            if not (
                torch.isfinite(out.base_logits).all()
                and torch.isfinite(out.local_logits).all()
                and torch.isfinite(out.logits).all()
                and torch.isfinite(out.plane_weights).all()
            ):
                raise RuntimeError(
                    f"B45 synthetic {height}x{width} produced non-finite values"
                )
            expected = torch.tensor([[[1.0, 0.0, 0.0]]], device=runtime.device)
            expected = expected.expand_as(out.plane_weights)
            if not torch.allclose(out.plane_weights, expected, atol=1e-7, rtol=0):
                raise RuntimeError("B45 single-plane synthetic routing is not deterministic")
            print(
                f"[B45 preflight] synthetic={height}x{width} "
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
    """Forward/backward-only preflight including the new plane-router gradient."""
    print("[B45 preflight] forward/backward only; no optimizer step", flush=True)
    if runtime.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(runtime.device)

    _synthetic_geometry_preflight(model, runtime)

    model.train()
    model.zero_grad(set_to_none=True)
    indices = _largest_series_indices(loader.dataset, B45_EFFECTIVE_BATCH)
    items = [loader.dataset[index] for index in indices]
    counts = [int(item["present"].shape[0]) for item in items]
    plane_sets = [
        sorted(
            {
                int(meta[0].item())
                for meta, flag in zip(item["series_meta"], item["present"])
                if float(flag.item()) > 0 and int(meta[0].item()) in (1, 2, 3)
            }
        )
        for item in items
    ]
    print(
        f"[B45 preflight] worst-case series/study={counts} available_planes={plane_sets}",
        flush=True,
    )

    scales = _batch_scales(items, multiplier_cpu)
    total_value = combined_value = local_value = 0.0
    multi_plane_seen = False
    for item, scale in zip(items, scales):
        tensors = _move_study(item, runtime.device)
        out, total, combined, local = _losses(
            model, runtime, tensors, multiplier_t, aux_weight
        )
        if out.plane_weights.shape[-1] != B45_PLANE_COUNT:
            raise RuntimeError("B45 preflight plane-weight shape changed")
        if not torch.isfinite(out.plane_weights).all():
            raise RuntimeError("B45 preflight produced non-finite plane weights")
        if not torch.allclose(
            out.plane_weights.sum(dim=-1),
            torch.ones_like(out.plane_weights.sum(dim=-1)),
            atol=1e-6,
            rtol=0,
        ):
            raise RuntimeError("B45 preflight plane weights do not sum to one")
        available = out.plane_available[0]
        n_available = int(available.sum().item())
        if n_available > 1:
            multi_plane_seen = True
            expected = 1.0 / float(n_available)
            actual = out.plane_weights[0, :, available]
            if not torch.allclose(
                actual, torch.full_like(actual, expected), atol=1e-6, rtol=0
            ):
                raise RuntimeError("B45 zero-logit router is not uniform at initialization")
        scaler.scale(total * float(scale)).backward()
        total_value += float(total.detach().item()) * float(scale)
        combined_value += float(combined.detach().item()) * float(scale)
        local_value += float(local.detach().item()) * float(scale)
        del tensors, out, total, combined, local

    if not multi_plane_seen:
        raise RuntimeError("B45 preflight did not exercise a multi-plane study")
    encoder_grad = any(
        p.grad is not None and torch.count_nonzero(p.grad).item() > 0
        for p in model.base.encoder.parameters()
        if p.requires_grad
    )
    evidence_grad = bool(
        model.head.evidence_weight.grad is not None
        and torch.count_nonzero(model.head.evidence_weight.grad).item() > 0
    )
    router_grad = bool(
        model.head.plane_router_logits.grad is not None
        and torch.count_nonzero(model.head.plane_router_logits.grad).item() > 0
    )
    if model.head.plane_embedding.weight.grad is not None:
        raise RuntimeError("B45 plane embedding unexpectedly received a gradient")
    leaked = any(
        p.grad is not None
        for name, p in model.base.named_parameters()
        if not name.startswith("encoder.") and not p.requires_grad
    )
    if leaked:
        raise RuntimeError("B45 preflight found a gradient on frozen B34 hierarchy")
    if not (encoder_grad and evidence_grad and router_grad):
        raise RuntimeError(
            "B45 preflight did not reach encoder tail, evidence scorer, and plane router"
        )
    print(
        f"[B45 preflight] total={total_value:.6f} combined={combined_value:.6f} "
        f"local={local_value:.6f}",
        flush=True,
    )
    print(
        f"[B45 preflight] {_format_memory_state(_memory_state(runtime))}", flush=True
    )
    model.zero_grad(set_to_none=True)
    del items
    _trim_host_memory()
    print("[B45 preflight] PASS", flush=True)


def train_b45(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path = B45_RUN_ROOT,
    preflight_only: bool = False,
) -> Path | None:
    config = dict(config)
    config["data_root"] = str(Path(data_root).resolve())
    crop_policy = require_b45_contract(config)
    if int(config.get("b37_micro_batch", B45_EFFECTIVE_BATCH)) != B45_EFFECTIVE_BATCH:
        raise ValueError("B45 retains B42/B37 effective batch size 2")

    seed = int(config.get("seed", 2026))
    seed_everything(seed + B45_CONSTRUCTION_SEED_OFFSET)
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
        raise ValueError("B45 requires the complete 4,407-study training release")
    frame, fill_policy, fill_audit = load_fill_merged_export(labels_root)
    uids, targets, weights, supervision = prepare_all_report_only_supervision(train, frame)
    if len(uids) != REPORT_ONLY_STUDIES:
        raise ValueError("B45 requires all 4,349 report-only studies")
    if int((weights > 0).sum()) != B35_EXPECTED_CELLS:
        raise ValueError("B45 supervision surface changed")
    if int(fill_audit.get("base_cells_overridden", -1)) != 0:
        raise ValueError("B45 requires zero B6 overrides")
    if list(fill_audit.get("excluded_targets", [])):
        raise ValueError("B45 requires all 12 targets")

    targets, confidence = rescale_label_confidence(targets, weights, config)
    base_confidence = base_payload.get("label_confidence", {})
    for key in ("positive_target", "negative_target"):
        if key in base_confidence and not np.isclose(
            float(confidence[key]), float(base_confidence[key]), atol=1e-12, rtol=0
        ):
            raise ValueError(f"B45 label confidence mismatch for {key}")

    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B45 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    expected_series = int(series_summary.get("eligible_recognized_plane_series", -1))
    if expected_series != B35_EXPECTED_SERIES:
        raise ValueError(
            f"B45 requires {B35_EXPECTED_SERIES} MRI series; got {expected_series}"
        )
    if series_summary.get("viability_passed") is not True:
        raise ValueError("B45 all-series MRI surface failed viability")

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
        batch_size=B45_EFFECTIVE_BATCH,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_b42,
        **runtime.loader_kwargs(seed=seed + B45_LOADER_SEED_OFFSET),
    )

    model = B45PlaneCalibratedSparseMILResidual(
        base_model,
        grid_size=int(config["b37_grid_size"]),
        top_k=int(config["b37_top_k"]),
        temperature=float(config["b37_temperature"]),
        encoder_trainable_stages=int(config["b37_encoder_trainable_stages"]),
        encoder_chunk_size=int(config["b37_encoder_chunk_size"]),
        router_temperature=float(config["b45_plane_router_temperature"]),
    ).to(runtime.device)
    model.train()

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    encoder_params = [p for p in model.base.encoder.parameters() if p.requires_grad]
    if not head_params or not encoder_params:
        raise RuntimeError("B45 requires sparse-head/router and encoder-tail parameters")
    if model.head.plane_embedding.weight.requires_grad:
        raise RuntimeError("B45 token plane embedding must remain frozen")
    if any(
        p.requires_grad
        for name, p in model.base.named_parameters()
        if not name.startswith("encoder.")
    ):
        raise RuntimeError("B45 non-encoder B34 parameters must remain frozen")

    head_lr = float(config.get("b37_head_lr", B37_HEAD_LR))
    encoder_scale = float(config["b37_encoder_lr_scale"])
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": head_lr, "name": "plane_calibrated_sparse_head"},
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

    for epoch in range(1, B45_EPOCHS + 1):
        started = time.monotonic()
        if runtime.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(runtime.device)
        model.train()
        total_sum = combined_sum = local_sum = 0.0
        batches = studies_seen = series_seen = cells_seen = 0
        gate_gradient_seen = evidence_gradient_seen = False
        router_gradient_seen = encoder_gradient_seen = False

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
                raise RuntimeError("B45 detected a gradient on frozen B34 hierarchy")
            if model.head.plane_embedding.weight.grad is not None:
                raise RuntimeError("B45 detected gradient through forbidden token plane embedding")
            gate_gradient_seen = gate_gradient_seen or bool(
                model.head.gate.grad is not None
                and torch.count_nonzero(model.head.gate.grad).item() > 0
            )
            evidence_gradient_seen = evidence_gradient_seen or bool(
                model.head.evidence_weight.grad is not None
                and torch.count_nonzero(model.head.evidence_weight.grad).item() > 0
            )
            router_gradient_seen = router_gradient_seen or bool(
                model.head.plane_router_logits.grad is not None
                and torch.count_nonzero(model.head.plane_router_logits.grad).item() > 0
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
                router_all = torch.softmax(
                    model.head.plane_router_logits.detach().float()
                    / model.head.router_temperature,
                    dim=-1,
                )
                router_entropy = _router_entropy(router_all)

            del items
            if report_progress:
                _trim_host_memory()
                print(
                    f"[B45] E{epoch} {step}/{len(loader)} "
                    f"total={total_sum/batches:.4f} combined={combined_sum/batches:.4f} "
                    f"local={local_sum/batches:.4f} gate_abs_mean={gate_abs_mean:.4f} "
                    f"router_entropy={router_entropy:.4f} elapsed={elapsed:.1f} min "
                    f"remaining~{remaining:.1f} min "
                    f"{_format_memory_state(_memory_state(runtime))}",
                    flush=True,
                )

        if studies_seen != REPORT_ONLY_STUDIES:
            raise RuntimeError("B45 epoch did not cover all report-only studies")
        if series_seen != B35_EXPECTED_SERIES:
            raise RuntimeError("B45 epoch did not cover all expected MRI series")
        if cells_seen != B35_EXPECTED_CELLS:
            raise RuntimeError("B45 epoch did not cover all supervision cells")
        if not (
            gate_gradient_seen
            and evidence_gradient_seen
            and router_gradient_seen
            and encoder_gradient_seen
        ):
            raise RuntimeError("B45 required gradient path was not active")

        router_weights = torch.softmax(
            model.head.plane_router_logits.detach().float()
            / model.head.router_temperature,
            dim=-1,
        ).cpu()
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
            "plane_router_weights_all_planes": router_weights.tolist(),
            "plane_router_entropy_mean": _router_entropy(router_weights),
            "encoder_sha256": encoder_state_sha256(model.base.encoder),
            "epoch_seconds": float(time.monotonic() - started),
            "memory": _memory_state(runtime),
        }
        history.append(row)
        print(
            f"[B45] E{epoch} total={row['loss_total']:.10f} "
            f"combined={row['loss_combined']:.10f} local={row['loss_local_aux']:.10f} "
            f"router_entropy={row['plane_router_entropy_mean']:.6f} "
            f"time={row['epoch_seconds']/60:.1f} min",
            flush=True,
        )
        _save_recovery(
            out_root, epoch=epoch, model=model, history=history, version=B45_VERSION
        )

    encoder_final_sha = encoder_state_sha256(model.base.encoder)
    if encoder_final_sha == encoder_initial_sha:
        raise RuntimeError("B45 encoder fingerprint did not move")

    checkpoint = out_root / B45_CHECKPOINT
    preprocessing = b42_preprocessing_state()
    payload = {
        "experiment": B45_EXPERIMENT,
        "version": B45_VERSION,
        "fixed_endpoint": True,
        "completed_epochs": B45_EPOCHS,
        "prospective_hypothesis": (
            "B42 local evidence is impaired by cross-plane score calibration: plane "
            "identity can shift token scores before one global top-k. Factor plane "
            "identity out of token scoring, pool top-k evidence independently inside "
            "each available plane, and learn target-specific plane fusion using only "
            "report-supervised training labels."
        ),
        "diagnostic_basis": {
            "B43": (
                "post-B42 Expert-58 mechanistic audit found strong axial-selection "
                "enrichment and target-dependent plane-specific signal"
            ),
            "B44": (
                "nested 32->64-centre frozen audit changed macro AUC by approximately "
                "zero, rejecting slice-count coverage as the main weak-target mechanism"
            ),
            "guardrail": (
                "Expert-58 diagnostics motivate mechanism only; no Expert-58 label, "
                "target-specific prior, threshold, hyperparameter, or checkpoint "
                "selection enters B45 training"
            ),
        },
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
        "training_studies": REPORT_ONLY_STUDIES,
        "training_series": B35_EXPECTED_SERIES,
        "training_supervision_cells": B35_EXPECTED_CELLS,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_used": False,
        "checkpoint_selection": "none; fixed epoch 2",
        "preprocessing": preprocessing,
        "effective_batching": {
            "studies_per_optimizer_step": B45_EFFECTIVE_BATCH,
            "ragged_studies_processed_sequentially": True,
            "construction_seed_offset": B45_CONSTRUCTION_SEED_OFFSET,
            "loader_seed_offset": B45_LOADER_SEED_OFFSET,
        },
        "sparse_mil": {
            "dense_slices": 32,
            "grid_size": int(model.head.grid_size),
            "regions_per_slice": int(model.head.n_regions),
            "top_k_per_available_plane": int(model.head.top_k),
            "temperature": float(model.head.temperature),
            "local_aux_weight": aux_weight,
        },
        "plane_routing": {
            "pooling": B45_PLANE_POOLING,
            "plane_count": B45_PLANE_COUNT,
            "router_init": B45_PLANE_ROUTER_INIT,
            "router_temperature": B45_PLANE_ROUTER_TEMPERATURE,
            "plane_embedding_used_in_token_score": (
                not B45_REMOVE_PLANE_EMBEDDING_FROM_TOKEN_SCORE
            ),
            "hard_coded_target_plane_priors": False,
            "router_logits": model.head.plane_router_logits.detach().float().cpu().tolist(),
            "router_weights_all_planes": torch.softmax(
                model.head.plane_router_logits.detach().float()
                / model.head.router_temperature,
                dim=-1,
            ).cpu().tolist(),
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
            "Prospective B45 fixed endpoint. Do not tune plane pooling, target-plane "
            "weights, router temperature, token metadata, top-k, grid, crop, geometry, "
            "learning rates, target subset, or epoch count after Expert-58. No model "
            "promotion may be based on the reused Expert-58 surface alone."
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
    parser = argparse.ArgumentParser(
        "Train B45 plane-calibrated target-conditioned sparse MIL"
    )
    parser.add_argument("--config", default="config/b45_plane_calibrated_sparse.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--out-root", default=B45_RUN_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = dict(_read_config(args.config))
    train_b45(
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


__all__ = [
    "B45_CHECKPOINT",
    "B45_CONSTRUCTION_SEED_OFFSET",
    "B45_EPOCHS",
    "B45_LOADER_SEED_OFFSET",
    "train_b45",
]
