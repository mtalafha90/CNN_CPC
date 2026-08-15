"""B26.2 fixed-E2 training: B20 with 171 quality-approved Synovitis fills.

This experiment is intentionally narrow.  It keeps the historical B20 model,
B16 report-aligned frozen encoder, post-resize 90% crop, optimizer, augmentation,
loader seed and five-epoch cosine scheduler horizon.  Training stops at the
canonical B20 endpoint E2 and never reads expert labels during training.

The only supervision change is additive: 76 positive and 95 negated Synovitis
cells that B6 left silent and that passed the B26 -> B26.1 -> B26.2 quality
pipeline.  Existing B6 targets and weights are immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
from .b12_variable_series import (
    audit_variable_series_surface,
    collate_variable_series,
)
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
from .b20_crop_focus import (
    CropFocusedVariableSeriesKneeDataset,
    require_b20_contract,
)
from .budget import RuntimeBudget
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime

B26_2_TRAIN_VERSION = "1.0.0"
B26_2_TRAIN_EXPERIMENT = "B26_2_quality_filtered_synovitis_fill"
B26_2_TRAIN_VARIANT = "b20_historical_recipe_b26_2_fill_fixed_e2_v1"
B26_2_FIXED_EPOCHS = 2
B26_2_TARGET = "Synovitis"
EXPECTED_ACCEPTED_POS = 76
EXPECTED_ACCEPTED_NEG = 95
EXPECTED_ACCEPTED_TOTAL = 171
EXPECTED_FINAL_CELLS = 14294
EXPECTED_FINAL_POS = 6947
EXPECTED_FINAL_NEG = 7347
EXPECTED_FINAL_SYN_POS = 475
EXPECTED_FINAL_SYN_NEG = 112


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_b26_2_fill_to_arrays(
    study_uids: list[str],
    base_targets: np.ndarray,
    base_weights: np.ndarray,
    filtered: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Add accepted B26.2 cells while proving that B6 is unchanged."""
    y0 = np.asarray(base_targets)
    w0 = np.asarray(base_weights)
    if y0.shape != w0.shape or y0.ndim != 2 or y0.shape[1] != len(TARGETS):
        raise ValueError("base supervision must be aligned [n, 12] arrays")
    if len(study_uids) != y0.shape[0]:
        raise ValueError("study_uids must align with base supervision")

    required = {"StudyInstanceUID", "target", "b26_2_accept", "b26_2_state"}
    missing = sorted(required - set(filtered.columns))
    if missing:
        raise ValueError(f"B26.2 filtered candidates missing columns: {missing}")

    frame = filtered.copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("B26.2 filtered candidates contain duplicate StudyInstanceUID rows")
    if set(frame["target"].astype(str)) != {B26_2_TARGET}:
        raise ValueError("B26.2 training accepts only the audited Synovitis target")

    uid_to_i = {str(uid): i for i, uid in enumerate(study_uids)}
    y = np.array(y0, dtype=np.float32, copy=True)
    w = np.array(w0, dtype=np.float32, copy=True)
    j = TARGETS.index(B26_2_TARGET)

    accepted = frame.loc[frame["b26_2_accept"].astype(bool)].copy()
    added_pos = added_neg = 0
    for row in accepted.itertuples(index=False):
        uid = str(row.StudyInstanceUID)
        if uid not in uid_to_i:
            raise ValueError(f"B26.2 accepted UID is outside the exact B20 surface: {uid}")
        i = uid_to_i[uid]
        if w0[i, j] > 0:
            raise RuntimeError("B26.2 attempted to overwrite an occupied B6 Synovitis cell")
        state = str(row.b26_2_state).strip().lower()
        if state == "positive":
            y[i, j] = 0.85
            w[i, j] = 0.50
            added_pos += 1
        elif state == "negated":
            y[i, j] = 0.05
            w[i, j] = 1.00
            added_neg += 1
        else:
            raise ValueError(f"accepted B26.2 row has non-definite state {state!r}")

    base_mask = w0 > 0
    if not np.array_equal(w[base_mask], w0[base_mask]):
        raise RuntimeError("B26.2 changed an existing B6 supervision weight")
    if not np.array_equal(y[base_mask], y0[base_mask]):
        raise RuntimeError("B26.2 changed an existing B6 supervision target")

    changed = (w != w0) | (y != y0)
    if np.any(changed[:, [k for k in range(len(TARGETS)) if k != j]]):
        raise RuntimeError("B26.2 changed a target other than Synovitis")

    diagnostics = {
        "accepted_positive": int(added_pos),
        "accepted_negated": int(added_neg),
        "accepted_total": int(added_pos + added_neg),
        "base_cells_dropped": 0,
        "base_cells_overridden": 0,
        "base_usable_cells": int((w0 > 0).sum()),
        "final_usable_cells": int((w > 0).sum()),
        "final_positive_cells": int(((w > 0) & (y > 0.5)).sum()),
        "final_negative_cells": int(((w > 0) & (y < 0.5)).sum()),
        "final_synovitis_positive": int(((w[:, j] > 0) & (y[:, j] > 0.5)).sum()),
        "final_synovitis_negative": int(((w[:, j] > 0) & (y[:, j] < 0.5)).sum()),
    }
    return y, w, diagnostics


def build_exact_b26_2_supervision(
    train: pd.DataFrame,
    *,
    b6_root: str | Path,
    filtered_candidates: str | Path,
) -> tuple[list[str], np.ndarray, np.ndarray, dict, dict, dict]:
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    uids, base_y, base_w, base_summary = prepare_b7_supervision(train, b6_frame)
    uids = [str(uid) for uid in uids]

    if len(uids) != 3120 or int((base_w > 0).sum()) != 14123:
        raise RuntimeError("B26.2 requires the exact historical B20 B6 gradient surface")
    gold_by_uid = pd.Series(
        gold_mask(train).to_numpy(dtype=bool),
        index=train["StudyInstanceUID"].astype(str),
    )
    if any(bool(gold_by_uid.loc[uid]) for uid in uids):
        raise RuntimeError("expert-gold study unexpectedly entered B26.2 gradients")

    filtered = pd.read_csv(filtered_candidates, dtype={"StudyInstanceUID": str})
    if len(filtered) != 631:
        raise RuntimeError(f"expected 631 B26.2 candidate rows, got {len(filtered)}")

    y, w, diagnostics = apply_b26_2_fill_to_arrays(uids, base_y, base_w, filtered)
    expected = {
        "accepted_positive": EXPECTED_ACCEPTED_POS,
        "accepted_negated": EXPECTED_ACCEPTED_NEG,
        "accepted_total": EXPECTED_ACCEPTED_TOTAL,
        "final_usable_cells": EXPECTED_FINAL_CELLS,
        "final_positive_cells": EXPECTED_FINAL_POS,
        "final_negative_cells": EXPECTED_FINAL_NEG,
        "final_synovitis_positive": EXPECTED_FINAL_SYN_POS,
        "final_synovitis_negative": EXPECTED_FINAL_SYN_NEG,
    }
    for key, value in expected.items():
        if int(diagnostics[key]) != int(value):
            raise RuntimeError(
                f"B26.2 supervision contract changed: {key}={diagnostics[key]} expected {value}"
            )
    return uids, y, w, diagnostics, b6_policy, {
        "b6_audit": b6_audit,
        "base_summary": base_summary,
    }


def train_b26_2(
    config: dict,
    *,
    b6_root: str | Path,
    filtered_candidates: str | Path,
    b26_2_audit: str | Path,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    out_root: str | Path = "runs/b26_2_training",
) -> Path:
    validate_competition_config(config, purpose="train")
    crop_policy = require_b20_contract(config)

    audit = json.loads(Path(b26_2_audit).read_text(encoding="utf-8"))
    if int(audit.get("accepted_positive", -1)) != EXPECTED_ACCEPTED_POS:
        raise RuntimeError("B26.2 audit positive count does not match the quality-approved surface")
    if int(audit.get("accepted_negated", -1)) != EXPECTED_ACCEPTED_NEG:
        raise RuntimeError("B26.2 audit negative count does not match the quality-approved surface")
    if int(audit.get("base_cells_dropped", -1)) != 0 or int(audit.get("base_cells_overridden", -1)) != 0:
        raise RuntimeError("B26.2 audit violates the fill-only contract")

    report_payload = load_b16_report_encoder(report_ssl_checkpoint)
    seed = int(config.get("seed", 2026))
    seed_everything(seed + 19_000_000)  # exact B20 construction seed
    runtime = resolve_runtime(config)
    print(runtime.describe())

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv")).copy()
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    study_uids, targets, weights, supervision_diag, b6_policy, base_meta = (
        build_exact_b26_2_supervision(
            train,
            b6_root=b6_root,
            filtered_candidates=filtered_candidates,
        )
    )

    print(
        f"[B26.2 train] studies={len(study_uids)} cells={int((weights > 0).sum())} | "
        f"Synovitis={supervision_diag['final_synovitis_positive']} pos / "
        f"{supervision_diag['final_synovitis_negative']} neg | fixed E{B26_2_FIXED_EPOCHS}"
    )

    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B26.2 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, study_uids)
    if series_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B26.2 reconstructed series surface does not match frozen B13 SHA-256")
    if int(series_summary.get("eligible_recognized_plane_series", -1)) != 17475:
        raise ValueError("B26.2 requires the exact 17,475-series B20 gradient surface")
    if series_summary.get("viability_passed") is not True:
        raise ValueError("B26.2 reconstructed series surface failed viability")

    batch_size = int(config.get("b7_batch_size", 2))
    expected_batches = int(math.ceil(len(study_uids) / batch_size))
    if expected_batches != 1560:
        raise RuntimeError("B26.2 exact B20 surface must yield 1,560 batches per epoch")
    expected_series = 17475
    target_multiplier = target_balance_multipliers(weights)

    train_ds = CropFocusedVariableSeriesKneeDataset(
        study_uids,
        variable_index,
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
        **runtime.loader_kwargs(seed=seed + 19_100_000),  # exact B20 loader seed
    )

    spec = b12_1_model_spec(config, normalize_input=True)
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.encoder.load_state_dict(report_payload["encoder"], strict=True)
    freeze_encoder(model)
    model.gradient_checkpointing = False
    encoder_sha_initial = encoder_state_sha256(model.encoder)
    model = model.to(runtime.device)

    head_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("encoder.") and p.requires_grad
    ]
    if not head_params or any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("B26.2 frozen-encoder contract failed")

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": float(config.get("b7_head_lr", B17_HEAD_LR))}],
        weight_decay=float(config.get("b7_weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=B18_CANDIDATE_EPOCHS,  # preserve B20's five-epoch LR trajectory
        eta_min=float(config.get("b7_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)
    clip = float(config.get("b7_grad_clip", 1.0))
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )

    history: list[dict] = []
    for epoch in range(1, B26_2_FIXED_EPOCHS + 1):
        epoch_started = time.monotonic()
        model.train()
        model.encoder.eval()
        loss_sum = 0.0
        steps = seen_studies = seen_series = seen_cells = seen_pos = seen_neg = 0
        for batch in loader:
            if not budget.can_start(120.0):
                raise RuntimeError("B26.2 runtime budget expired before a complete fixed-E2 pass")
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
                raise RuntimeError("B26.2 detected an encoder gradient")
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
        epoch_seconds = time.monotonic() - epoch_started
        if encoder_state_sha256(model.encoder) != encoder_sha_initial:
            raise RuntimeError("B26.2 encoder changed despite freezing")
        full = (
            steps == expected_batches
            and seen_studies == len(study_uids)
            and seen_series == expected_series
            and seen_cells == EXPECTED_FINAL_CELLS
            and seen_pos == EXPECTED_FINAL_POS
            and seen_neg == EXPECTED_FINAL_NEG
        )
        if not full:
            raise RuntimeError(f"B26.2 epoch {epoch} did not complete the exact frozen surface")

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
            "epoch_seconds": round(float(epoch_seconds), 1),
            "full_coverage": True,
        }
        history.append(row)
        print(f"[B26.2 train] E{epoch} loss={row['loss']:.10f} | {epoch_seconds/60:.1f} min")

    encoder_sha_final = encoder_state_sha256(model.encoder)
    if encoder_sha_final != encoder_sha_initial:
        raise RuntimeError("B26.2 encoder fingerprint changed")

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "b26_2_model.pt"
    payload = {
        "experiment": B26_2_TRAIN_EXPERIMENT,
        "variant": B26_2_TRAIN_VARIANT,
        "b26_2_training_version": B26_2_TRAIN_VERSION,
        "fixed_endpoint": True,
        "selected_epoch": B26_2_FIXED_EPOCHS,
        "completed_epochs": B26_2_FIXED_EPOCHS,
        "scheduler_horizon": B18_CANDIDATE_EPOCHS,
        "model_spec": spec,
        "model_state": model.state_dict(),
        "config": config,
        "crop_focus_enabled": True,
        "crop_focus_policy": crop_policy,
        "b19_cosine_mask_used": False,
        "initialization": B16_REPORT_SSL_VARIANT,
        "initialization_experiment": B16_REPORT_SSL_EXPERIMENT,
        "initialization_objective": B16_REPORT_SSL_OBJECTIVE,
        "input_normalization": B13_INPUT_NORMALIZATION,
        "encoder_frozen": True,
        "encoder_sha256_initial": encoder_sha_initial,
        "encoder_sha256_final": encoder_sha_final,
        "study_uids": list(study_uids),
        "history": history,
        "supervision": supervision_diag,
        "b6_policy": b6_policy,
        "base_metadata": base_meta,
        "metadata_repair": metadata_stats,
        "b6_root": str(Path(b6_root).resolve()),
        "filtered_candidates": str(Path(filtered_candidates).resolve()),
        "filtered_candidates_sha256": _sha256(filtered_candidates),
        "b26_2_audit": audit,
        "b26_2_audit_sha256": _sha256(b26_2_audit),
        "series_policy": str(Path(series_policy_path).resolve()),
        "report_ssl_checkpoint": str(Path(report_ssl_checkpoint).resolve()),
        "gold_studies_used_in_gradient": 0,
        "expert_checkpoint_selection": False,
        "evaluation_status": "not yet evaluated; fixed-E2 checkpoint",
    }
    torch.save(payload, checkpoint)
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "training_audit.json").write_text(
        json.dumps(
            {
                "experiment": B26_2_TRAIN_EXPERIMENT,
                "variant": B26_2_TRAIN_VARIANT,
                "checkpoint": str(checkpoint),
                "fixed_epoch": B26_2_FIXED_EPOCHS,
                "supervision": supervision_diag,
                "filtered_candidates_sha256": _sha256(filtered_candidates),
                "b26_2_audit_sha256": _sha256(b26_2_audit),
                "encoder_sha256": encoder_sha_final,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser("B26.2 fixed-E2 B20-family training")
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--filtered-candidates", required=True)
    parser.add_argument("--b26-2-audit", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--report-ssl-checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b26_2_training")
    args = parser.parse_args()

    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    train_b26_2(
        config,
        b6_root=args.b6_root,
        filtered_candidates=args.filtered_candidates,
        b26_2_audit=args.b26_2_audit,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
