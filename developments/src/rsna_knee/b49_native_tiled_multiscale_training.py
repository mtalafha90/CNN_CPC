"""Train one fixed, matched B49 native-tiled scanner-domain arm.

The B49 runner intentionally reuses B48's frozen report-only domain split and
label artefacts.  It never loads B46/B48 weights or official gold labels.  The
only new input capability is B49's full-FOV native local tiling, with a separate
downsampled global-context view used solely by the frozen B34 hierarchy.
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
from .b35_training import _require_base_checkpoint, sha256_file
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
from .b42_constant_area_aspect_sparse_mil import B42_EFFECTIVE_BATCH
from .b42_constant_area_aspect_sparse_training import _batch_scales
from .b48_global_conditioned_sparse_training import (
    _config_sha256,
    _indices_for_split,
    _report_only_surface,
    _uid_sha256,
    b48_fill_artifacts,
    load_b48_domain_split,
)
from .b49_native_tiled_multiscale_mil import (
    B49_ARMS,
    B49_CONTEXT_DIM,
    B49_EXPERIMENT,
    B49_FIXED_EPOCHS,
    B49_RUN_ROOT,
    B49_SUPERVISION,
    B49_VERSION,
    B49NativeTiledFullFOVDataset,
    B49NativeTiledMultiscaleMILResidual,
    b49_preprocessing_state,
    b49_state,
    collate_b49,
    native_tile_layout,
    require_b49_contract,
    verify_native_tile_coverage,
)
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv
from .dicom import DICOM_SUFFIXES, find_series_dir
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import autocast, make_scaler, resolve_runtime


B49_CHECKPOINT_TEMPLATE = "b49_{arm}_model.pt"
B49_CONSTRUCTION_SEED_OFFSET = B37_CONSTRUCTION_SEED_OFFSET
B49_LOADER_SEED_OFFSET = B37_LOADER_SEED_OFFSET
B49_REPLICATION_SEEDS = (2026, 2037, 2048)


def _move_b49_study(item: dict, device) -> tuple:
    """Move only context tensors/labels; native source descriptors stay on CPU."""
    if len(item["views"]) != 1:
        raise ValueError("B49 training/preflight requires exactly one non-TTA view")
    view = item["views"][0]
    return (
        [volume.to(device, non_blocking=True) for volume in view["context_volumes"]],
        view["local_sources"],
        item["present"].to(device, non_blocking=True),
        item["series_meta"].to(device, non_blocking=True),
        view["slice_position"].to(device, non_blocking=True),
        item["target"].to(device, non_blocking=True).unsqueeze(0),
        item["weight"].to(device, non_blocking=True).unsqueeze(0),
    )


def _b49_losses(model, runtime, tensors, multiplier_t, aux_weight: float):
    context, sources, present, meta, position, target, weight = tensors
    with autocast(runtime):
        out = model(context, sources, present, meta, position)
        combined = target_balanced_weak_bce(out.logits, target, weight, multiplier_t)
        local = target_balanced_weak_bce(out.local_logits, target, weight, multiplier_t)
        total = combined + float(aux_weight) * local
    return out, total, combined, local


def _find_tile_heavy_index(dataset) -> tuple[int, dict]:
    """Find a real source that exercises the most demanding common tile grid."""
    import pydicom

    best: tuple[int, int, dict] | None = None
    for index, uid in enumerate(dataset.study_uids):
        if dataset.weights is not None and not np.any(np.asarray(dataset.weights[index]) > 0):
            continue
        for record in dataset.series_records[uid]:
            directory = find_series_dir(
                dataset.config.data_root,
                dataset.config.split,
                str(uid),
                str(record["series_uid"]),
            )
            if directory is None:
                continue
            files = sorted(
                path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in DICOM_SUFFIXES
            )
            for path in files:
                try:
                    header = pydicom.dcmread(
                        str(path), force=True, stop_before_pixels=True, specific_tags=["Rows", "Columns"]
                    )
                    height = int(getattr(header, "Rows", 0) or 0)
                    width = int(getattr(header, "Columns", 0) or 0)
                except Exception:
                    continue
                if height < 2 or width < 2:
                    continue
                tiles = len(native_tile_layout(height, width))
                candidate = {
                    "study_uid": str(uid),
                    "series_uid": str(record["series_uid"]),
                    "native_height": height,
                    "native_width": width,
                    "tile_count": tiles,
                }
                if best is None or tiles > best[1]:
                    best = (index, tiles, candidate)
                # Four 640 tiles is the expected 1024x1024 stress case.  There
                # is no need to scan the entire training disk after finding it.
                if tiles >= 4:
                    return index, candidate
                break
    if best is None:
        raise RuntimeError("B49 could not find a readable native MRI header for preflight")
    return best[0], best[2]


def _geometry_preflight(dataset) -> tuple[int, dict]:
    """Verify exact full-FOV ownership on representative and real geometries."""
    for height, width in ((320, 300), (512, 512), (640, 640), (1024, 1024), (640, 1280)):
        layout = native_tile_layout(height, width)
        verify_native_tile_coverage(height, width, layout)
        print(
            f"[B49 preflight] synthetic native={height}x{width} tiles={len(layout)} "
            "full-FOV ownership=PASS",
            flush=True,
        )
    index, source = _find_tile_heavy_index(dataset)
    layout = native_tile_layout(source["native_height"], source["native_width"])
    verify_native_tile_coverage(source["native_height"], source["native_width"], layout)
    print(
        "[B49 preflight] real native tile stress "
        f"study={source['study_uid']} series={source['series_uid']} "
        f"matrix={source['native_height']}x{source['native_width']} tiles={source['tile_count']} PASS",
        flush=True,
    )
    return index, source


def _b49_context_preflight(
    model,
    dataset,
    *,
    runtime,
    multiplier_t,
    scaler,
    aux_weight: float,
) -> None:
    """Exercise the real native-tiled path and all new zero-start gradients."""
    if runtime.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(runtime.device)
    index, source = _geometry_preflight(dataset)
    item = dataset[int(index)]
    tensors = _move_b49_study(item, runtime.device)
    saved_gate = model.head.context_gate.detach().clone()

    model.eval()
    with torch.no_grad():
        global_feature = model._encode_context_study(tensors[0], tensors[2])
        reconstruction_error = model.context_reconstruction_error(
            global_feature, tensors[2], tensors[3]
        )
    del global_feature
    if reconstruction_error > 1e-6:
        raise RuntimeError(
            "B49 post-attention query no longer reconstructs its full-FOV B34 global logit: "
            f"max_abs_error={reconstruction_error:.3e}"
        )
    print(
        f"[B49 preflight] post-attention global-logit reconstruction={reconstruction_error:.3e} PASS",
        flush=True,
    )

    model.train()
    model.zero_grad(set_to_none=True)
    out, total, combined, local = _b49_losses(model, runtime, tensors, multiplier_t, aux_weight)
    if out.context_query.requires_grad:
        raise RuntimeError("B49 global query was not detached before local conditioning")
    if not (
        torch.isfinite(out.base_logits).all()
        and torch.isfinite(out.local_logits).all()
        and torch.isfinite(out.logits).all()
        and torch.isfinite(total)
    ):
        raise RuntimeError("B49 real native tile preflight produced non-finite values")
    scaler.scale(total).backward()
    required = {
        "encoder_tail": any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in model.base.encoder.parameters()
            if parameter.requires_grad
        ),
        "evidence": bool(
            model.head.evidence_weight.grad is not None
            and torch.count_nonzero(model.head.evidence_weight.grad).item() > 0
        ),
        "sparse_gate": bool(
            model.head.gate.grad is not None and torch.count_nonzero(model.head.gate.grad).item() > 0
        ),
        "coordinate_projection": bool(
            model.head.region_projection.weight.grad is not None
            and torch.count_nonzero(model.head.region_projection.weight.grad).item() > 0
        ),
        "context_gate": bool(
            model.head.context_gate.grad is not None
            and torch.count_nonzero(model.head.context_gate.grad).item() > 0
        ),
    }
    if not all(required.values()):
        raise RuntimeError(f"B49 native preflight gradient path failure: {required}")
    zero_projection_grads = [
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for parameter in (model.head.context_query.weight, model.head.context_key.weight)
    ]
    if not all(zero_projection_grads):
        raise RuntimeError("B49 context projection received gradient before its zero-start gate opened")
    leaked = any(
        parameter.grad is not None
        for name, parameter in model.base.named_parameters()
        if not name.startswith("encoder.") and not parameter.requires_grad
    )
    if leaked:
        raise RuntimeError("B49 native local loss reached the frozen B34 hierarchy")
    print(
        f"[B49 preflight] real total={total.detach().item():.6f} "
        f"combined={combined.detach().item():.6f} local={local.detach().item():.6f} "
        f"tiles={out.native_tile_count} valid_tokens={out.native_valid_token_count} "
        f"{_format_memory_state(_memory_state(runtime))} PASS",
        flush=True,
    )
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        model.head.context_gate.fill_(0.05)
    _out, opened_total, _combined, _local = _b49_losses(
        model, runtime, tensors, multiplier_t, aux_weight
    )
    scaler.scale(opened_total).backward()
    for parameter in (model.head.context_query.weight, model.head.context_key.weight):
        if parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0:
            raise RuntimeError("B49 opened context gate did not reach its low-rank projection")
    with torch.no_grad():
        model.head.context_gate.copy_(saved_gate)
    model.zero_grad(set_to_none=True)
    del item, tensors, out, total, combined, local, _out, opened_total, _combined, _local
    _trim_host_memory()
    print("[B49 preflight] full-FOV native tiled/context gradient contract PASS", flush=True)


def train_b49_domain_arm(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    domain_split: str | Path,
    arm: str,
    seed: int = B49_REPLICATION_SEEDS[0],
    out_root: str | Path = B49_RUN_ROOT,
    preflight_only: bool = False,
) -> Path | None:
    """Train exactly two epochs of one B49 matched scanner-domain arm."""
    arm, seed = str(arm), int(seed)
    if arm not in B49_ARMS:
        raise ValueError(f"B49 arm must be one of {B49_ARMS}; got {arm!r}")
    if seed not in B49_REPLICATION_SEEDS:
        raise ValueError(f"B49 seed must be one of {B49_REPLICATION_SEEDS}; got {seed}")
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    contract = require_b49_contract(settings, arm=arm)
    declared_seeds = tuple(int(value) for value in settings.get("b49_replication_seeds", B49_REPLICATION_SEEDS))
    if declared_seeds != B49_REPLICATION_SEEDS:
        raise ValueError(f"B49 freezes b49_replication_seeds={list(B49_REPLICATION_SEEDS)}")

    domain_payload, domain_rows, domain_meta = load_b48_domain_split(domain_split)
    settings["seed"] = seed
    seed_everything(seed + B49_CONSTRUCTION_SEED_OFFSET)
    runtime = resolve_runtime(settings)
    print(runtime.describe(), flush=True)
    print(f"[B49 {arm} seed={seed}] domain_split_sha={domain_meta['sha256']}", flush=True)

    base_path = Path(base_checkpoint).resolve()
    base_model, base_payload = load_phase9_checkpoint(base_path, expected_arm="llm_fill", device="cpu")
    _require_base_checkpoint(base_payload)
    encoder_initial_sha = encoder_state_sha256(base_model.encoder)

    root = Path(settings["data_root"])
    expected_train_sha = str(domain_payload.get("source_train_csv_sha256", ""))
    if not expected_train_sha or sha256_file(root / settings.get("train_csv", "train.csv")) != expected_train_sha:
        raise ValueError("B49 domain split source train.csv fingerprint mismatch")
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
    targets, weights = all_targets[train_indices], all_weights[train_indices]
    target_multiplier = target_balance_multipliers(weights)

    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B49 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    expected_series = int(series_summary.get("eligible_recognized_plane_series", -1))
    expected_cells = int((weights > 0).sum())
    if expected_series <= 0 or expected_cells <= 0 or series_summary.get("viability_passed") is not True:
        raise RuntimeError("B49 scanner-split MRI/weak-label surface failed viability")

    dataset_config = make_b7_dataset_config(settings, root, train=False)
    dataset_config.tta_center_offsets = ()
    dataset = B49NativeTiledFullFOVDataset(
        uids,
        variable_index,
        dataset_config,
        center_offsets=(0,),
        targets=targets,
        weights=weights,
    )
    loader = DataLoader(
        dataset,
        batch_size=B42_EFFECTIVE_BATCH,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_b49,
        **runtime.loader_kwargs(seed=seed + B49_LOADER_SEED_OFFSET),
    )
    model = B49NativeTiledMultiscaleMILResidual(
        base_model,
        encoder_trainable_stages=int(settings["b37_encoder_trainable_stages"]),
        encoder_chunk_size=int(settings["b37_encoder_chunk_size"]),
        tile_encoder_chunk_size=int(settings["b49_tile_encoder_chunk_size"]),
        arm=arm,
        context_dim=int(settings["b49_context_dim"]),
    ).to(runtime.device)
    model.train()
    head_params = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    encoder_params = [parameter for parameter in model.base.encoder.parameters() if parameter.requires_grad]
    if not head_params or not encoder_params:
        raise RuntimeError("B49 requires native head/context and encoder-tail parameters")
    if any(
        parameter.requires_grad
        for name, parameter in model.base.named_parameters()
        if not name.startswith("encoder.")
    ):
        raise RuntimeError("B49 B34 non-encoder parameters must remain frozen")

    head_lr = float(settings.get("b37_head_lr", B37_HEAD_LR))
    encoder_scale = float(settings["b37_encoder_lr_scale"])
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": head_lr, "name": "native_tiled_sparse_context_head"},
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
        _b49_context_preflight(
            model,
            dataset,
            runtime=runtime,
            multiplier_t=multiplier_t,
            scaler=scaler,
            aux_weight=aux_weight,
        )
        print(f"[B49 {arm} preflight] PASS", flush=True)
        return None

    arm_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arm_root / B49_CHECKPOINT_TEMPLATE.format(arm=arm)
    if checkpoint_path.exists():
        raise FileExistsError(f"B49 will not overwrite an existing checkpoint: {checkpoint_path}")
    history: list[dict] = []
    for epoch in range(1, B49_FIXED_EPOCHS + 1):
        started = time.monotonic()
        if runtime.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(runtime.device)
        model.train()
        total_sum = combined_sum = local_sum = 0.0
        batches = studies_seen = series_seen = cells_seen = 0
        tiles_seen = tokens_seen = 0
        gradient_seen = {
            "sparse_gate": False,
            "evidence": False,
            "coordinate": False,
            "context_gate": False,
            "context_projection": False,
            "encoder": False,
        }
        for step, items in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            scales = _batch_scales(items, multiplier_cpu)
            batch_total = batch_combined = batch_local = 0.0
            for item, scale in zip(items, scales):
                series_seen += int(item["present"].sum().item())
                cells_seen += int((item["weight"] > 0).sum().item())
                tensors = _move_b49_study(item, runtime.device)
                out, total, combined, local = _b49_losses(model, runtime, tensors, multiplier_t, aux_weight)
                scaler.scale(total * float(scale)).backward()
                batch_total += float(total.detach().item()) * float(scale)
                batch_combined += float(combined.detach().item()) * float(scale)
                batch_local += float(local.detach().item()) * float(scale)
                tiles_seen += int(out.native_tile_count)
                tokens_seen += int(out.native_valid_token_count)
                del tensors, out, total, combined, local
            leaked = any(
                parameter.grad is not None
                for name, parameter in model.base.named_parameters()
                if not name.startswith("encoder.") and not parameter.requires_grad
            )
            if leaked:
                raise RuntimeError("B49 detected a gradient on frozen B34 hierarchy")
            gradient_seen["sparse_gate"] |= bool(model.head.gate.grad is not None and torch.count_nonzero(model.head.gate.grad).item() > 0)
            gradient_seen["evidence"] |= bool(model.head.evidence_weight.grad is not None and torch.count_nonzero(model.head.evidence_weight.grad).item() > 0)
            gradient_seen["coordinate"] |= bool(model.head.region_projection.weight.grad is not None and torch.count_nonzero(model.head.region_projection.weight.grad).item() > 0)
            gradient_seen["context_gate"] |= bool(model.head.context_gate.grad is not None and torch.count_nonzero(model.head.context_gate.grad).item() > 0)
            gradient_seen["context_projection"] |= all(
                parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
                for parameter in (model.head.context_query.weight, model.head.context_key.weight)
            )
            gradient_seen["encoder"] |= any(
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
            if step % 20 == 0 or step == len(loader):
                elapsed = (time.monotonic() - started) / 60.0
                remaining = elapsed / step * (len(loader) - step)
                print(
                    f"[B49 {arm} S{seed}] E{epoch} {step}/{len(loader)} "
                    f"total={total_sum/batches:.4f} combined={combined_sum/batches:.4f} "
                    f"local={local_sum/batches:.4f} tiles/study={tiles_seen/max(studies_seen,1):.1f} "
                    f"tokens/study={tokens_seen/max(studies_seen,1):.0f} "
                    f"elapsed={elapsed:.1f} min remaining~{remaining:.1f} min "
                    f"{_format_memory_state(_memory_state(runtime))}",
                    flush=True,
                )
                _trim_host_memory()
            del items
        if studies_seen != len(uids) or series_seen != expected_series or cells_seen != expected_cells:
            raise RuntimeError(
                "B49 epoch surface changed: "
                f"studies={studies_seen}/{len(uids)} series={series_seen}/{expected_series} "
                f"cells={cells_seen}/{expected_cells}"
            )
        if not all(gradient_seen.values()):
            raise RuntimeError(f"B49 required gradient paths were not active: {gradient_seen}")
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
            "native_tiles": tiles_seen,
            "native_valid_tokens": tokens_seen,
            "native_tiles_per_study": tiles_seen / max(studies_seen, 1),
            "native_tokens_per_study": tokens_seen / max(studies_seen, 1),
            "sparse_mil": model.head.state(),
            "encoder_sha256": encoder_state_sha256(model.base.encoder),
            "epoch_seconds": float(time.monotonic() - started),
            "memory": _memory_state(runtime),
        }
        history.append(row)
        print(
            f"[B49 {arm} S{seed}] E{epoch} total={row['loss_total']:.10f} "
            f"combined={row['loss_combined']:.10f} local={row['loss_local_aux']:.10f} "
            f"tiles/study={row['native_tiles_per_study']:.1f} time={row['epoch_seconds']/60:.1f} min",
            flush=True,
        )
        _save_recovery(arm_root, epoch=epoch, model=model, history=history, version=B49_VERSION)

    encoder_final_sha = encoder_state_sha256(model.base.encoder)
    if encoder_final_sha == encoder_initial_sha:
        raise RuntimeError("B49 encoder fingerprint did not move")
    target_balance = {target: float(target_multiplier[index]) for index, target in enumerate(TARGETS)}
    source_sha = {
        "model": sha256_file(Path(__file__).with_name("b49_native_tiled_multiscale_mil.py")),
        "training": sha256_file(Path(__file__)),
        "b48_domain_protocol": sha256_file(
            Path(__file__).with_name("b48_global_conditioned_sparse_training.py")
        ),
    }
    matched_pair_identity = {
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
        "b49_representation": b49_preprocessing_state(),
    }
    payload = {
        "experiment": B49_EXPERIMENT,
        "version": B49_VERSION,
        "fixed_endpoint": True,
        "completed_epochs": B49_FIXED_EPOCHS,
        "checkpoint_selection": "none; fixed epoch 2",
        "arm": arm,
        "seed": seed,
        "hypothesis": (
            "full-FOV native tile evidence, softly conditioned by a detached B34 global pathology "
            "query, improves scanner-held-out weak-label ranking beyond an otherwise matched static query"
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
        "supervision_source": B49_SUPERVISION,
        "training_studies": len(uids),
        "training_series": expected_series,
        "training_supervision_cells": expected_cells,
        "training_uids_sha256": _uid_sha256(uids),
        "target_balance_source": "scanner_split_train_only_weak_labels",
        "target_balance_multiplier": target_balance,
        "domain_split": domain_meta,
        "domain_split_summary": domain_payload.get("summary", {}),
        "preprocessing": b49_preprocessing_state(),
        "label_confidence": confidence,
        "fill_policy": fill_policy,
        "fill_audit": fill_audit,
        "fill_artifacts": fill_artifacts,
        "supervision": supervision,
        "series_policy_signature": B13_SERIES_SIGNATURE,
        "series_surface": series_summary,
        "metadata_repair": metadata_stats,
        "b49": b49_state(arm),
        "config_sha256": matched_pair_identity["config_sha256"],
        "source_sha256": source_sha,
        "matched_pair_identity": matched_pair_identity,
        "history": history,
        "governance": (
            "B49 is a prospective matched scanner-split weak-label representation/mechanism comparison. "
            "Do not use official gold labels, B46/B48 checkpoints, B47 outputs, checkpoint selection, "
            "or parameter/geometry/seed/endpoint tuning after a B49 result is inspected."
        ),
    }
    torch.save(payload, checkpoint_path)
    audit = {key: value for key, value in payload.items() if key not in {"base_state", "head_state"}}
    (arm_root / "training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (arm_root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint_path, flush=True)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser("Train one fixed B49 native-tiled scanner-split arm")
    parser.add_argument("--config", default="config/b49_native_tiled_multiscale.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--domain-split", required=True)
    parser.add_argument("--arm", choices=B49_ARMS, required=True)
    parser.add_argument("--seed", type=int, default=B49_REPLICATION_SEEDS[0])
    parser.add_argument("--out-root", default=B49_RUN_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    train_b49_domain_arm(
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
    "B49_CHECKPOINT_TEMPLATE",
    "B49_REPLICATION_SEEDS",
    "train_b49_domain_arm",
]
