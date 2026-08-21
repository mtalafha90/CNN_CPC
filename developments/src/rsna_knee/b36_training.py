"""Train B36: pathology-specific sparse top-k spatial MIL residual.

B36 is a prospective mechanism test motivated by the completed B35 diagnostic:
B35's dense softmax attention stayed essentially uniform and its zero gate
starved the local query/scoring parameters of early gradient.  B36 therefore:

* preserves the frozen B34 predictor and exact 16-centre reconstruction;
* keeps the validated 32-centre, 3x3 local ConvNeXt feature surface;
* scores every local token independently for each pathology;
* pools only the strongest top-k evidence locations; and
* directly supervises the local MIL logit in addition to the zero-gated combined
  prediction, so the local head receives full gradient from step one.

No expert labels enter gradients or checkpoint selection.
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
from .b20_crop_focus import b20_crop_focus_policy
from .b35_target_spatial_residual import B35SpatialDataset, collate_b35
from .b35_training import (
    B35_EXPECTED_CELLS,
    B35_EXPECTED_SERIES,
    _require_base_checkpoint,
    sha256_file,
)
from .b36_sparse_mil import (
    B36_LOCAL_AUX_WEIGHT,
    B36_TEMPERATURE,
    B36_TOP_K,
    B36_VERSION,
    B36SparseMILResidual,
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

B36_EXPERIMENT = "B36_pathology_sparse_topk_spatial_MIL_phaseA"
B36_EPOCHS = 2
B36_MICRO_BATCH = 2
B36_ACCUMULATION = 1
B36_HEAD_LR = 1e-4
B36_WEIGHT_DECAY = 1e-4
B36_GRAD_CLIP = 1.0
B36_EQUIVALENCE_TOLERANCE = 2e-3
B36_CONSTRUCTION_SEED_OFFSET = 46_000_000
B36_LOADER_SEED_OFFSET = 46_100_000


def _save_epoch_head(
    out_root: Path,
    *,
    epoch: int,
    model: B36SparseMILResidual,
    history: list[dict],
) -> None:
    """Infrastructure-only recovery snapshot; never a model-selection endpoint."""
    out_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": B36_VERSION,
            "epoch": int(epoch),
            "fixed_endpoint": False,
            "model_selection_allowed": False,
            "head_state": model.head.state_dict(),
            "history": history,
        },
        out_root / "recovery_latest.pt",
    )


def train_b36(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path = "runs/b36_sparse_mil_v1",
) -> Path:
    config = dict(config)
    config["data_root"] = str(Path(data_root).resolve())
    seed = int(config.get("seed", 2026))
    seed_everything(seed + B36_CONSTRUCTION_SEED_OFFSET)
    runtime = resolve_runtime(config)
    print(runtime.describe(), flush=True)

    base_path = Path(base_checkpoint).resolve()
    base_model, base_payload = load_phase9_checkpoint(
        base_path,
        expected_arm="llm_fill",
        device="cpu",
    )
    _require_base_checkpoint(base_payload)

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    if len(train) != 4407:
        raise ValueError("B36 requires the complete 4,407-study training release")

    frame, fill_policy, fill_audit = load_fill_merged_export(labels_root)
    uids, targets, weights, supervision = prepare_all_report_only_supervision(
        train,
        frame,
    )
    if len(uids) != REPORT_ONLY_STUDIES:
        raise ValueError("B36 requires all 4,349 report-only studies")
    if int((weights > 0).sum()) != B35_EXPECTED_CELLS:
        raise ValueError(
            f"B36 requires exactly {B35_EXPECTED_CELLS} usable cells; "
            f"got {int((weights > 0).sum())}"
        )
    if int(fill_audit.get("base_cells_overridden", -1)) != 0:
        raise ValueError("B36 requires the fill-only surface with zero B6 overrides")
    if list(fill_audit.get("excluded_targets", [])):
        raise ValueError("B36 requires the all-target fill surface")

    targets, confidence = rescale_label_confidence(targets, weights, config)
    base_confidence = base_payload.get("label_confidence", {})
    for key in ("positive_target", "negative_target"):
        if key in base_confidence and not np.isclose(
            float(confidence[key]),
            float(base_confidence[key]),
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError(
                f"B36 label confidence {key}={confidence[key]} does not match "
                f"the frozen base checkpoint {base_confidence[key]}"
            )

    series_policy = _load_series_policy(series_policy_path)
    if (
        series_policy.get("series_summary", {}).get("series_signature_sha256")
        != B13_SERIES_SIGNATURE
    ):
        raise ValueError("B36 requires the frozen all-series B12/B13 policy")
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    expected_series = int(series_summary.get("eligible_recognized_plane_series", -1))
    if expected_series != B35_EXPECTED_SERIES:
        raise ValueError(
            f"B36 requires {B35_EXPECTED_SERIES} report-only MRI series; "
            f"got {expected_series}"
        )
    if series_summary.get("viability_passed") is not True:
        raise ValueError("B36 all-series MRI surface failed viability")

    crop_policy = b20_crop_focus_policy(config)
    dataset_config = make_b7_dataset_config(config, root, train=False)
    dataset_config.tta_center_offsets = ()
    ds = B35SpatialDataset(
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
        batch_size=B36_MICRO_BATCH,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_b35,
        **runtime.loader_kwargs(seed=seed + B36_LOADER_SEED_OFFSET),
    )

    top_k = int(config.get("b36_top_k", B36_TOP_K))
    temperature = float(config.get("b36_temperature", B36_TEMPERATURE))
    aux_weight = float(config.get("b36_local_aux_weight", B36_LOCAL_AUX_WEIGHT))
    if aux_weight <= 0:
        raise ValueError("b36_local_aux_weight must be positive")

    base_model = base_model.to(runtime.device)
    model = B36SparseMILResidual(
        base_model,
        top_k=top_k,
        temperature=temperature,
    ).to(runtime.device)
    model.train()
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    if not head_params:
        raise RuntimeError("B36 found no trainable sparse-MIL parameters")
    if any(p.requires_grad for p in model.base.parameters()):
        raise RuntimeError("B36 base model is not fully frozen")

    optimizer = torch.optim.AdamW(
        head_params,
        lr=float(config.get("b36_head_lr", B36_HEAD_LR)),
        weight_decay=float(config.get("b36_weight_decay", B36_WEIGHT_DECAY)),
    )
    scaler = make_scaler(runtime)
    target_multiplier = target_balance_multipliers(weights)
    multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)
    clip = float(config.get("b36_grad_clip", B36_GRAD_CLIP))

    equivalence_error = None
    history: list[dict] = []
    out_root = Path(out_root)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, B36_EPOCHS + 1):
        started = time.monotonic()
        model.train()
        total_sum = combined_sum = local_sum = 0.0
        batches = studies_seen = series_seen = cells_seen = 0
        optimizer_steps = 0
        gate_gradient_seen = False
        evidence_gradient_seen = False

        for step, batch in enumerate(loader, start=1):
            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            position = batch["slice_position"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            meta = batch["series_meta"].to(runtime.device, non_blocking=True)
            target = batch["target"].to(runtime.device, non_blocking=True)
            weight = batch["weight"].to(runtime.device, non_blocking=True)

            if equivalence_error is None:
                with autocast(runtime):
                    equivalence_error = model.base_equivalence_error(
                        volumes,
                        present,
                        meta,
                    )
                print(
                    f"[B36] exact-base reconstruction max|delta|={equivalence_error:.8g}",
                    flush=True,
                )
                if equivalence_error > B36_EQUIVALENCE_TOLERANCE:
                    raise RuntimeError(
                        "B36 cannot reproduce the frozen B34 base closely enough; "
                        f"max error {equivalence_error}"
                    )

            with autocast(runtime):
                out = model(volumes, present, meta, position)
                combined_loss = target_balanced_weak_bce(
                    out.logits,
                    target,
                    weight,
                    multiplier_t,
                )
                # Critical B36 change: full-gradient direct supervision of the
                # sparse local MIL head even while the residual gate is zero.
                local_loss = target_balanced_weak_bce(
                    out.local_logits,
                    target,
                    weight,
                    multiplier_t,
                )
                loss = combined_loss + aux_weight * local_loss
                scaled_loss = loss / float(B36_ACCUMULATION)
            scaler.scale(scaled_loss).backward()

            if any(p.grad is not None for p in model.base.parameters()):
                raise RuntimeError("B36 detected a gradient on the frozen B34 base")
            if model.head.gate.grad is not None:
                gate_gradient_seen = gate_gradient_seen or bool(
                    torch.count_nonzero(model.head.gate.grad).item() > 0
                )
            if model.head.evidence_weight.grad is not None:
                evidence_gradient_seen = evidence_gradient_seen or bool(
                    torch.count_nonzero(model.head.evidence_weight.grad).item() > 0
                )

            should_step = (
                step % B36_ACCUMULATION == 0 or step == len(loader)
            )
            if should_step:
                if clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(head_params, clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            active = weight > 0
            total_sum += float(loss.item())
            combined_sum += float(combined_loss.item())
            local_sum += float(local_loss.item())
            batches += 1
            studies_seen += int(volumes.shape[0])
            series_seen += int(present.sum().item())
            cells_seen += int(active.sum().item())

            if step % 100 == 0:
                elapsed = (time.monotonic() - started) / 60.0
                print(
                    f"[B36] E{epoch} {step}/{len(loader)} "
                    f"total={total_sum/batches:.4f} "
                    f"combined={combined_sum/batches:.4f} "
                    f"local={local_sum/batches:.4f} "
                    f"elapsed={elapsed:.1f} min "
                    f"gate_abs_mean={model.head.effective_gate().detach().abs().mean().item():.4f}",
                    flush=True,
                )

        if studies_seen != REPORT_ONLY_STUDIES:
            raise RuntimeError("B36 epoch did not cover all report-only studies")
        if series_seen != B35_EXPECTED_SERIES:
            raise RuntimeError("B36 epoch did not cover all expected MRI series")
        if cells_seen != B35_EXPECTED_CELLS:
            raise RuntimeError("B36 epoch did not cover all supervision cells")
        if not gate_gradient_seen:
            raise RuntimeError("B36 residual gate never received gradient")
        if not evidence_gradient_seen:
            raise RuntimeError(
                "B36 sparse evidence scorer never received direct auxiliary gradient"
            )

        row = {
            "epoch": epoch,
            "loss_total": total_sum / max(batches, 1),
            "loss_combined": combined_sum / max(batches, 1),
            "loss_local_aux": local_sum / max(batches, 1),
            "micro_batches": batches,
            "optimizer_steps": optimizer_steps,
            "studies": studies_seen,
            "series": series_seen,
            "supervision_cells": cells_seen,
            "gate": model.head.state(),
            "epoch_seconds": float(time.monotonic() - started),
        }
        history.append(row)
        print(
            f"[B36] E{epoch} total={row['loss_total']:.10f} "
            f"combined={row['loss_combined']:.10f} "
            f"local={row['loss_local_aux']:.10f} "
            f"time={row['epoch_seconds']/60:.1f} min",
            flush=True,
        )
        _save_epoch_head(out_root, epoch=epoch, model=model, history=history)

    out_root.mkdir(parents=True, exist_ok=True)
    checkpoint = out_root / "b36_model.pt"
    payload = {
        "experiment": B36_EXPERIMENT,
        "version": B36_VERSION,
        "phase": "A_frozen_encoder_sparse_topk_MIL_with_direct_local_aux",
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": sha256_file(base_path),
        "base_architecture": base_payload.get("architecture"),
        "base_completed_epochs": int(base_payload.get("completed_epochs", -1)),
        "base_training_supervision_cells": int(
            base_payload.get("training_supervision_cells", -1)
        ),
        "base_encoder_trainable_stages": int(
            base_payload.get("encoder_trainable_stages", -1)
        ),
        "base_exact_reconstruction_max_abs_error": float(equivalence_error or 0.0),
        "head_state": model.head.state_dict(),
        "head_spec": {
            "dim": int(model.head.dim),
            "grid_size": int(model.head.grid_size),
            "dense_slices": 32,
            "base_slices": 16,
            "top_k": int(model.head.top_k),
            "temperature": float(model.head.temperature),
            "token_dropout": float(model.head.token_dropout.p),
            "local_aux_weight": aux_weight,
            "pooling": "topk_logmeanexp",
        },
        "head_audit": model.head.state(),
        "completed_epochs": B36_EPOCHS,
        "fixed_endpoint": True,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_used": False,
        "training_studies": REPORT_ONLY_STUDIES,
        "training_series": B35_EXPECTED_SERIES,
        "training_supervision_cells": B35_EXPECTED_CELLS,
        "micro_batch_size": B36_MICRO_BATCH,
        "gradient_accumulation": B36_ACCUMULATION,
        "effective_batch_size": B36_MICRO_BATCH * B36_ACCUMULATION,
        "head_lr": float(config.get("b36_head_lr", B36_HEAD_LR)),
        "weight_decay": float(config.get("b36_weight_decay", B36_WEIGHT_DECAY)),
        "scheduler": "none_constant_lr",
        "label_confidence": confidence,
        "supervision": supervision,
        "supervision_source": {
            "labels_root": str(Path(labels_root).resolve()),
            "merge_version": fill_policy.get("version"),
            "base_cells_overridden": int(fill_audit.get("base_cells_overridden", -1)),
            "excluded_targets": list(fill_audit.get("excluded_targets", [])),
        },
        "series_policy_signature": B13_SERIES_SIGNATURE,
        "series_summary": series_summary,
        "metadata_repair": metadata_stats,
        "crop_policy": crop_policy,
        "history": history,
        "config": config,
        "governance": (
            "Prospective B36 mechanism test after B35. Frozen B34/encoder; direct "
            "local auxiliary loss is trained only on the same report-derived weak "
            "surface. No expert labels enter gradients or checkpoint selection. "
            "Recovery checkpoints are infrastructure-only and not selectable."
        ),
    }
    torch.save(payload, checkpoint)
    audit = {k: v for k, v in payload.items() if k not in {"head_state", "config"}}
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
    ap = argparse.ArgumentParser(
        "Train B36 pathology-specific sparse top-k spatial MIL"
    )
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--labels", required=True, help="runs/b6_plus_llm_fill_all")
    ap.add_argument("--series-policy", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/b36_sparse_mil_v1")
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    train_b36(
        config,
        data_root=args.data_root,
        labels_root=args.labels,
        series_policy_path=args.series_policy,
        base_checkpoint=args.base_checkpoint,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
