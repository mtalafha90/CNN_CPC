"""Train one frozen B46 cross-fit fold with the unchanged B42 model contract."""
from __future__ import annotations

import argparse
import hashlib
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
    _require_base_checkpoint,
    sha256_file,
)
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
    B42ConstantAreaAspectSparseMILResidual,
    b42_preprocessing_state,
    collate_b42,
)
from .b42_constant_area_aspect_sparse_training import (
    _batch_scales,
    _losses,
    _move_study,
    _preflight,
)
from .b46_gold_crossfit import (
    B46_EXPERIMENT,
    B46_FIXED_EPOCHS,
    B46_GOLD_CELL_WEIGHT,
    B46_N_FOLDS,
    B46_RUN_ROOT,
    B46_VERSION,
    heldout_uids,
    load_gold_fold_manifest,
    require_b46_contract,
    training_gold_uids,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .label_confidence import rescale_label_confidence
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .phase9_supervision import (
    REPORT_ONLY_STUDIES,
    load_fill_merged_export,
    prepare_all_report_only_supervision,
)
from .runtime import make_scaler, resolve_runtime

B46_CHECKPOINT_TEMPLATE = "b46_fold{fold}_model.pt"
B46_CONSTRUCTION_SEED_OFFSET = B37_CONSTRUCTION_SEED_OFFSET
B46_LOADER_SEED_OFFSET = B37_LOADER_SEED_OFFSET


def _manifest_gold_targets(train, manifest: dict, uids: list[str]) -> np.ndarray:
    official = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    official["StudyInstanceUID"] = official["StudyInstanceUID"].astype(str)
    official = official.set_index("StudyInstanceUID")
    by_uid = {str(row["StudyInstanceUID"]): row for row in manifest["rows"]}
    values = []
    for uid in uids:
        if uid not in official.index or uid not in by_uid:
            raise ValueError(f"B46 gold UID missing from official/manifest surface: {uid}")
        raw = official.loc[uid, TARGETS].to_numpy(np.float32)
        recorded = np.asarray([by_uid[uid][target] for target in TARGETS], dtype=np.float32)
        if not np.array_equal(raw, recorded):
            raise RuntimeError(f"B46 manifest labels differ from official train.csv for {uid}")
        values.append(raw)
    return np.stack(values, axis=0).astype(np.float32)


def _prepare_fold_supervision(
    config: dict,
    *,
    root: Path,
    labels_root: str | Path,
    manifest: dict,
    fold: int,
    base_payload: dict,
):
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    if len(train) != 4407:
        raise ValueError("B46 requires the complete 4,407-study training release")

    frame, fill_policy, fill_audit = load_fill_merged_export(labels_root)
    weak_uids, weak_targets, weak_weights, weak_supervision = prepare_all_report_only_supervision(
        train, frame
    )
    if len(weak_uids) != REPORT_ONLY_STUDIES:
        raise ValueError("B46 requires all 4,349 report-only weak studies")
    if int((weak_weights > 0).sum()) != B35_EXPECTED_CELLS:
        raise ValueError("B46 weak supervision surface changed")
    if int(fill_audit.get("base_cells_overridden", -1)) != 0:
        raise ValueError("B46 requires zero B6 overrides")
    if list(fill_audit.get("excluded_targets", [])):
        raise ValueError("B46 requires all 12 targets")

    weak_targets, confidence = rescale_label_confidence(weak_targets, weak_weights, config)
    base_confidence = base_payload.get("label_confidence", {})
    for key in ("positive_target", "negative_target"):
        if key in base_confidence and not np.isclose(
            float(confidence[key]), float(base_confidence[key]), atol=1e-12, rtol=0
        ):
            raise ValueError(f"B46 label confidence mismatch for {key}")

    heldout = heldout_uids(manifest, fold)
    gold_train = training_gold_uids(manifest, fold)
    if set(heldout).intersection(gold_train):
        raise RuntimeError("B46 held-out gold leaked into gold training fold")
    if set(weak_uids).intersection(set(manifest_uid["StudyInstanceUID"] for manifest_uid in manifest["rows"])):
        raise RuntimeError("B46 official gold studies overlap weak/report-only supervision")

    gold_targets = _manifest_gold_targets(train, manifest, gold_train)
    gold_weights = np.full_like(gold_targets, B46_GOLD_CELL_WEIGHT, dtype=np.float32)

    uids = [*weak_uids, *gold_train]
    targets = np.concatenate((weak_targets.astype(np.float32), gold_targets), axis=0)
    weights = np.concatenate((weak_weights.astype(np.float32), gold_weights), axis=0)

    # Critical B46 isolation rule: target balancing is frozen from the historical
    # weak/report supervision. Gold additions do not change per-target multipliers.
    weak_target_multiplier = target_balance_multipliers(weak_weights)

    supervision = {
        "weak": weak_supervision,
        "weak_studies": len(weak_uids),
        "weak_cells": int((weak_weights > 0).sum()),
        "gold_training_studies": len(gold_train),
        "gold_training_cells": int(gold_targets.size),
        "gold_cell_weight": B46_GOLD_CELL_WEIGHT,
        "heldout_gold_studies": len(heldout),
        "combined_training_studies": len(uids),
        "combined_training_cells": int((weights > 0).sum()),
        "target_balance_source": "weak_only_frozen",
        "weak_target_balance_multiplier": {
            target: float(weak_target_multiplier[j]) for j, target in enumerate(TARGETS)
        },
    }
    return (
        train,
        uids,
        targets,
        weights,
        weak_target_multiplier,
        heldout,
        gold_train,
        confidence,
        fill_policy,
        fill_audit,
        supervision,
    )


def _gold_anchor_preflight(
    model,
    dataset,
    *,
    gold_index: int,
    runtime,
    multiplier_t,
    scaler,
    aux_weight: float,
) -> None:
    """Verify one held-in official gold study reaches the declared gradient paths."""
    item = dataset[int(gold_index)]
    target = item["target"].numpy()
    weight = item["weight"].numpy()
    if not np.isin(target, [0.0, 1.0]).all():
        raise RuntimeError("B46 gold preflight target is not exact binary 0/1")
    if not np.allclose(weight, B46_GOLD_CELL_WEIGHT, atol=0, rtol=0):
        raise RuntimeError("B46 gold preflight weight changed")

    model.train()
    model.zero_grad(set_to_none=True)
    tensors = _move_study(item, runtime.device)
    out, total, combined, local = _losses(model, runtime, tensors, multiplier_t, aux_weight)
    scaler.scale(total).backward()
    encoder_grad = any(
        p.grad is not None and torch.count_nonzero(p.grad).item() > 0
        for p in model.base.encoder.parameters()
        if p.requires_grad
    )
    head_grad = bool(
        model.head.evidence_weight.grad is not None
        and torch.count_nonzero(model.head.evidence_weight.grad).item() > 0
    )
    if not encoder_grad or not head_grad:
        raise RuntimeError("B46 gold preflight did not reach encoder tail and sparse head")
    print(
        f"[B46 gold preflight] uid={item['study_uid']} weight={B46_GOLD_CELL_WEIGHT:.1f} "
        f"total={total.detach().item():.6f} combined={combined.detach().item():.6f} "
        f"local={local.detach().item():.6f} PASS",
        flush=True,
    )
    model.zero_grad(set_to_none=True)
    del item, tensors, out, total, combined, local
    _trim_host_memory()


def train_b46_fold(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    fold_manifest: str | Path,
    fold: int,
    out_root: str | Path = B46_RUN_ROOT,
    preflight_only: bool = False,
) -> Path | None:
    config = dict(config)
    config["data_root"] = str(Path(data_root).resolve())
    crop_policy = require_b46_contract(config)
    fold = int(fold)
    if fold < 0 or fold >= B46_N_FOLDS:
        raise ValueError("B46 fold must be in [0,4]")
    if int(config.get("b37_micro_batch", B42_EFFECTIVE_BATCH)) != B42_EFFECTIVE_BATCH:
        raise ValueError("B46 retains B42 effective batch size 2")

    manifest_path = Path(fold_manifest).resolve()
    manifest = load_gold_fold_manifest(manifest_path)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    seed = int(config.get("seed", 2026))
    # Deliberately identical model-construction seed in all folds and in B42.
    seed_everything(seed + B46_CONSTRUCTION_SEED_OFFSET)
    runtime = resolve_runtime(config)
    print(runtime.describe(), flush=True)

    base_path = Path(base_checkpoint).resolve()
    base_model, base_payload = load_phase9_checkpoint(
        base_path, expected_arm="llm_fill", device="cpu"
    )
    _require_base_checkpoint(base_payload)
    encoder_initial_sha = encoder_state_sha256(base_model.encoder)

    root = Path(config["data_root"])
    (
        train,
        uids,
        targets,
        weights,
        target_multiplier,
        heldout,
        gold_train,
        confidence,
        fill_policy,
        fill_audit,
        supervision,
    ) = _prepare_fold_supervision(
        config,
        root=root,
        labels_root=labels_root,
        manifest=manifest,
        fold=fold,
        base_payload=base_payload,
    )

    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B46 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    expected_series = int(series_summary.get("eligible_recognized_plane_series", -1))
    if expected_series <= 0 or series_summary.get("viability_passed") is not True:
        raise ValueError("B46 combined weak+gold MRI surface failed viability")

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
        **runtime.loader_kwargs(seed=seed + B46_LOADER_SEED_OFFSET),
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
        raise RuntimeError("B46 requires sparse-head and encoder-tail parameters")
    if any(
        p.requires_grad
        for name, p in model.base.named_parameters()
        if not name.startswith("encoder.")
    ):
        raise RuntimeError("B46 non-encoder B34 parameters must remain frozen")

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
    multiplier_cpu = torch.from_numpy(target_multiplier)
    multiplier_t = multiplier_cpu.to(runtime.device)
    aux_weight = float(config["b37_local_aux_weight"])
    clip = float(config.get("b37_grad_clip", B37_GRAD_CLIP))

    fold_root = Path(out_root) / f"fold_{fold}"
    fold_root.mkdir(parents=True, exist_ok=True)

    if preflight_only:
        print(f"[B46 fold {fold}] inherited B42 geometry/gradient preflight", flush=True)
        _preflight(
            model,
            loader,
            runtime,
            multiplier_t,
            multiplier_cpu,
            scaler,
            aux_weight,
        )
        _gold_anchor_preflight(
            model,
            dataset,
            gold_index=REPORT_ONLY_STUDIES,
            runtime=runtime,
            multiplier_t=multiplier_t,
            scaler=scaler,
            aux_weight=aux_weight,
        )
        print(f"[B46 fold {fold} preflight] PASS", flush=True)
        return None

    history: list[dict] = []
    expected_studies = len(uids)
    expected_cells = int((weights > 0).sum())

    for epoch in range(1, B46_FIXED_EPOCHS + 1):
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
                raise RuntimeError("B46 detected a gradient on frozen B34 hierarchy")
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
                    f"[B46 F{fold}] E{epoch} {step}/{len(loader)} "
                    f"total={total_sum/batches:.4f} combined={combined_sum/batches:.4f} "
                    f"local={local_sum/batches:.4f} gate_abs_mean={gate_abs_mean:.4f} "
                    f"elapsed={elapsed:.1f} min remaining~{remaining:.1f} min "
                    f"{_format_memory_state(_memory_state(runtime))}",
                    flush=True,
                )

        if studies_seen != expected_studies:
            raise RuntimeError(
                f"B46 fold {fold} epoch covered {studies_seen} studies; expected {expected_studies}"
            )
        if series_seen != expected_series:
            raise RuntimeError(
                f"B46 fold {fold} epoch covered {series_seen} series; expected {expected_series}"
            )
        if cells_seen != expected_cells:
            raise RuntimeError(
                f"B46 fold {fold} epoch covered {cells_seen} cells; expected {expected_cells}"
            )
        if not (gate_gradient_seen and evidence_gradient_seen and encoder_gradient_seen):
            raise RuntimeError("B46 required gradient path was not active")

        row = {
            "epoch": epoch,
            "fold": fold,
            "loss_total": total_sum / batches,
            "loss_combined": combined_sum / batches,
            "loss_local_aux": local_sum / batches,
            "batches": batches,
            "studies": studies_seen,
            "series": series_seen,
            "supervision_cells": cells_seen,
            "gold_training_studies": len(gold_train),
            "gold_training_cells": len(gold_train) * len(TARGETS),
            "gate": model.head.state(),
            "encoder_sha256": encoder_state_sha256(model.base.encoder),
            "epoch_seconds": float(time.monotonic() - started),
            "memory": _memory_state(runtime),
        }
        history.append(row)
        print(
            f"[B46 F{fold}] E{epoch} total={row['loss_total']:.10f} "
            f"combined={row['loss_combined']:.10f} local={row['loss_local_aux']:.10f} "
            f"time={row['epoch_seconds']/60:.1f} min",
            flush=True,
        )
        _save_recovery(
            fold_root, epoch=epoch, model=model, history=history, version=B46_VERSION
        )

    encoder_final_sha = encoder_state_sha256(model.base.encoder)
    if encoder_final_sha == encoder_initial_sha:
        raise RuntimeError("B46 encoder fingerprint did not move")

    checkpoint = fold_root / B46_CHECKPOINT_TEMPLATE.format(fold=fold)
    payload = {
        "experiment": B46_EXPERIMENT,
        "version": B46_VERSION,
        "fixed_endpoint": True,
        "fold": fold,
        "completed_epochs": B46_FIXED_EPOCHS,
        "checkpoint_selection": "none; fixed epoch 2",
        "hypothesis": (
            "adding prospectively cross-fitted official expert supervision to the unchanged "
            "B42 training contract tests whether report-to-expert target mismatch is a primary ceiling"
        ),
        "fold_manifest": str(manifest_path),
        "fold_manifest_sha256": manifest_sha,
        "heldout_gold_uids": heldout,
        "training_gold_uids": gold_train,
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
        "training_studies": expected_studies,
        "training_series": expected_series,
        "training_supervision_cells": expected_cells,
        "weak_training_studies": REPORT_ONLY_STUDIES,
        "weak_training_cells": B35_EXPECTED_CELLS,
        "gold_studies_used_in_gradient": len(gold_train),
        "gold_labels_used": True,
        "heldout_gold_studies_used_in_gradient": 0,
        "gold_cell_weight": B46_GOLD_CELL_WEIGHT,
        "gold_targets": "hard_binary_0_1",
        "target_balance_source": "weak_only_frozen",
        "target_balance_multiplier": {
            target: float(target_multiplier[j]) for j, target in enumerate(TARGETS)
        },
        "preprocessing": b42_preprocessing_state(),
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
            "Prospective B46 cross-fit fold. Held-out gold UIDs never enter gradients. "
            "Do not change gold weight, folds, architecture, learning rates, target balance, "
            "or epoch count after any B46 OOF prediction is inspected."
        ),
    }
    torch.save(payload, checkpoint)
    audit = {k: v for k, v in payload.items() if k not in {"base_state", "head_state"}}
    (fold_root / "training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (fold_root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint, flush=True)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser("Train one fixed B46 gold-anchored cross-fit fold")
    parser.add_argument("--config", default="config/b46_gold_anchored_crossfit.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out-root", default=B46_RUN_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = dict(_read_config(args.config))
    train_b46_fold(
        config,
        data_root=args.data_root,
        labels_root=args.labels_root,
        series_policy_path=args.series_policy,
        base_checkpoint=args.base_checkpoint,
        fold_manifest=args.fold_manifest,
        fold=args.fold,
        out_root=args.out_root,
        preflight_only=bool(args.preflight_only),
    )


if __name__ == "__main__":
    main()
