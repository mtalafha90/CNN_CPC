"""B11 student: B7.1 recipe plus frozen conservative teacher pseudo-labels.

The student starts from the same B5 encoder initialization as B7.1.  It does
not initialize from the B7.1 teacher.  Historical routing, legacy resize,
architecture, optimizer hyperparameters, augmentation, and four-epoch policy
are retained.  The scientific change is additional low-weight supervision on
B6-unsupervised cells, including newly activated report-only studies.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .b11_pseudo_labels import (
    B11_MAX_TTA_RANGE,
    B11_NEGATIVE_THRESHOLD,
    B11_POSITIVE_THRESHOLD,
    B11_PSEUDO_BASE_WEIGHT,
    B11_PSEUDO_MASS_CAP_FRACTION,
    B11_PSEUDO_POLICY,
    _all_non_gold_b6,
    _require_policy,
    _series_signature,
    _sha256_file,
    combine_b6_and_teacher,
)
from .b7_weak_supervision import (
    B7_MIN_CONFIDENCE,
    B7_NEGATIVE_TARGET,
    B7_NEGATIVE_WEIGHT,
    B7_POSITIVE_TARGET,
    B7_POSITIVE_WEIGHT,
    _read_config,
    _require_frozen_policy,
    b7_model_spec,
    build_b7_model,
    load_b5_encoder_payload,
    load_frozen_b6_export,
    make_b7_dataset_config,
    seed_everything,
    target_balance_multipliers,
    target_balanced_weak_bce,
)
from .budget import RuntimeBudget
from .constants import DUAL_STREAMS, TARGETS
from .data import backfill_series_metadata, build_series_index, load_series_csv, load_train_csv
from .dataset import KneeStudyDataset
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime
from .ssl import SSL_SOURCE

B11_VARIANT = "b11_b5_init_b6_b71_teacher_completion_v1"
B11_EXPERIMENT = "B11_teacher_student_completion"


def _require_b11_contract(config: dict) -> None:
    _require_frozen_policy(config)
    _require_policy(config)
    if str(config.get("b11_experiment_name", B11_EXPERIMENT)) != B11_EXPERIMENT:
        raise ValueError(f"B11 experiment name must remain {B11_EXPERIMENT!r}")
    if int(config.get("b7_epochs", 4)) != 4:
        raise ValueError("B11-v1 requires exactly four epochs")
    if int(config.get("b7_batch_size", 2)) != 2:
        raise ValueError("B11-v1 requires batch size 2")


def _load_pseudo_artifacts(pseudo_root: str | Path) -> tuple[pd.DataFrame, dict]:
    root = Path(pseudo_root)
    csv_path = root / "pseudo_labels.csv"
    policy_path = root / "pseudo_policy.json"
    if not csv_path.is_file() or not policy_path.is_file():
        raise FileNotFoundError("B11 requires pseudo_labels.csv and pseudo_policy.json")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("policy") != B11_PSEUDO_POLICY:
        raise ValueError("B11 pseudo policy name mismatch")
    if policy.get("uses_gold_labels_to_choose_pseudo_cells") is not False:
        raise ValueError("B11 pseudo policy does not certify label-free pseudo selection")
    if _sha256_file(csv_path) != str(policy.get("pseudo_labels_sha256")):
        raise ValueError("B11 pseudo_labels.csv SHA-256 does not match frozen policy")
    frame = pd.read_csv(csv_path)
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("B11 pseudo_labels.csv contains duplicate StudyInstanceUIDs")
    return frame, policy


def _reconstruct_combined_supervision(
    train: pd.DataFrame,
    b6_frame: pd.DataFrame,
    pseudo_frame: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    all_uids, b6_y, b6_w, b6_summary = _all_non_gold_b6(train, b6_frame)
    pseudo = pseudo_frame.set_index("StudyInstanceUID").reindex(all_uids)
    if pseudo.index.isna().any() or pseudo.shape[0] != len(all_uids):
        raise ValueError("B11 pseudo UID surface does not match non-gold training studies")
    teacher_mean = np.zeros_like(b6_y)
    teacher_range = np.zeros_like(b6_y)
    for j, target in enumerate(TARGETS):
        mean_col = f"{target}__teacher_mean"
        range_col = f"{target}__tta_range"
        b6_col = f"{target}__b6_weight"
        for col in (mean_col, range_col, b6_col):
            if col not in pseudo.columns:
                raise ValueError(f"B11 pseudo file missing {col}")
        teacher_mean[:, j] = pd.to_numeric(pseudo[mean_col], errors="raise").to_numpy(np.float32)
        teacher_range[:, j] = pd.to_numeric(pseudo[range_col], errors="raise").to_numpy(np.float32)
        stored_b6 = pd.to_numeric(pseudo[b6_col], errors="raise").to_numpy(np.float32)
        if not np.allclose(stored_b6, b6_w[:, j], atol=1e-7, rtol=0):
            raise ValueError(f"B11 stored B6 weights drifted for {target}")

    combined_y, combined_w, pseudo_w, summary = combine_b6_and_teacher(
        b6_y, b6_w, teacher_mean, teacher_range
    )
    # Verify stored combined artifacts too; this catches policy/code drift.
    for j, target in enumerate(TARGETS):
        cy = pd.to_numeric(pseudo[f"{target}__combined_target"], errors="raise").to_numpy(np.float32)
        cw = pd.to_numeric(pseudo[f"{target}__combined_weight"], errors="raise").to_numpy(np.float32)
        pw = pd.to_numeric(pseudo[f"{target}__pseudo_weight"], errors="raise").to_numpy(np.float32)
        if not np.allclose(cy, combined_y[:, j], atol=1e-6, rtol=0):
            raise ValueError(f"B11 combined target drift for {target}")
        if not np.allclose(cw, combined_w[:, j], atol=1e-6, rtol=0):
            raise ValueError(f"B11 combined weight drift for {target}")
        if not np.allclose(pw, pseudo_w[:, j], atol=1e-6, rtol=0):
            raise ValueError(f"B11 pseudo weight drift for {target}")
    summary["b6_summary"] = b6_summary
    return all_uids, combined_y, combined_w, pseudo_w, b6_w, summary


def _checkpoint_payload(
    *, model, config, spec, history, b5_checkpoint, b5_payload, b6_root,
    b6_audit, supervision, target_multiplier, metadata_stats, pseudo_policy,
    pseudo_root, budget,
) -> dict:
    return {
        "variant": B11_VARIANT,
        "experiment": B11_EXPERIMENT,
        "source": SSL_SOURCE,
        "model_state": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "model_spec": spec,
        "config": config,
        "completed_epochs": len(history),
        "gold_studies_used_in_gradient": 0,
        "gold_studies_used_for_early_stopping": 0,
        "gold_labels_used_to_choose_pseudo_cells": False,
        "b6_gold_audit_informed_global_policy": True,
        "b5_checkpoint": str(b5_checkpoint.resolve()),
        "b5_variant": b5_payload.get("variant"),
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "pseudo_root": str(Path(pseudo_root).resolve()),
        "pseudo_policy": pseudo_policy,
        "supervision_policy": {
            "b6_min_confidence": B7_MIN_CONFIDENCE,
            "b6_positive_target": B7_POSITIVE_TARGET,
            "b6_negative_target": B7_NEGATIVE_TARGET,
            "b6_positive_weight": B7_POSITIVE_WEIGHT,
            "b6_negative_weight": B7_NEGATIVE_WEIGHT,
            "teacher_positive_threshold": B11_POSITIVE_THRESHOLD,
            "teacher_negative_threshold": B11_NEGATIVE_THRESHOLD,
            "teacher_max_tta_range": B11_MAX_TTA_RANGE,
            "teacher_base_weight": B11_PSEUDO_BASE_WEIGHT,
            "teacher_mass_cap_fraction": B11_PSEUDO_MASS_CAP_FRACTION,
            "target_balancing": "frozen B7.1 multipliers derived from B6 only",
        },
        "target_balance_multiplier": {target: float(target_multiplier[j]) for j, target in enumerate(TARGETS)},
        "supervision": supervision,
        "metadata_repair": metadata_stats,
        "history": history,
        "budget": budget.to_dict(),
    }


def train_b11(
    config: dict,
    *,
    b5_checkpoint: str | Path,
    b6_root: str | Path,
    pseudo_root: str | Path,
    out_root: str | Path = "runs/b11_teacher_student",
) -> Path:
    validate_competition_config(config, purpose="train")
    _require_b11_contract(config)
    seed = int(config.get("seed", 2026))
    seed_everything(seed + 7_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    pseudo_frame, pseudo_policy = _load_pseudo_artifacts(pseudo_root)
    all_uids, combined_y, combined_w, pseudo_w, b6_w, supervision = _reconstruct_combined_supervision(
        train, b6_frame, pseudo_frame
    )

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    all_index = build_series_index(series, all_uids, mode="dual")
    signature = _series_signature(all_index, all_uids)
    if signature != str(pseudo_policy.get("selected_series_signature")):
        raise ValueError("B11 historical series routing changed since pseudo-label generation")

    active = combined_w.sum(axis=1) > 0
    uids = [uid for uid, keep in zip(all_uids, active) if keep]
    targets = combined_y[active]
    weights = combined_w[active]
    pseudo_active = pseudo_w[active]
    b6_active = b6_w[active]
    index = {uid: all_index[uid] for uid in uids}
    has_mri = np.asarray([any(index.get(uid, {}).get(stream) for stream in DUAL_STREAMS) for uid in uids], dtype=bool)
    if not has_mri.all():
        raise ValueError(f"B11 active supervision includes {(~has_mri).sum()} study/studies without MRI")

    target_multiplier = target_balance_multipliers(b6_w)
    supervision.update({
        "training_studies": int(len(uids)),
        "training_combined_cells": int((weights > 0).sum()),
        "training_b6_cells": int((b6_active > 0).sum()),
        "training_pseudo_cells": int((pseudo_active > 0).sum()),
        "full_coverage_batches_per_epoch": int(math.ceil(len(uids) / 2)),
        "b6_target_balance_multiplier": {target: float(target_multiplier[j]) for j, target in enumerate(TARGETS)},
        "pseudo_policy_sha256": str(pseudo_policy.get("pseudo_labels_sha256")),
    })

    ds = KneeStudyDataset(
        uids,
        index,
        make_b7_dataset_config(config, root, train=True),
        targets=targets,
        weights=weights,
        train=True,
    )
    batch_size = int(config.get("b7_batch_size", 2))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **runtime.loader_kwargs(seed=seed + 7_100_000),
    )
    batches_per_epoch = int(math.ceil(len(ds) / batch_size))

    b5_path = Path(b5_checkpoint)
    b5_payload = load_b5_encoder_payload(b5_path)
    normalize_input = bool(b5_payload.get("config", {}).get("normalize_input", False))
    spec = b7_model_spec(config, normalize_input=normalize_input)
    model = build_b7_model(spec, encoder_state=b5_payload["encoder"]).to(runtime.device)

    encoder_lr = float(config.get("b7_encoder_lr", 1e-5))
    head_lr = float(config.get("b7_head_lr", 1e-4))
    encoder_params = list(model.encoder.parameters())
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": encoder_lr}, {"params": head_params, "lr": head_lr}],
        weight_decay=float(config.get("b7_weight_decay", 1e-4)),
    )
    epochs = int(config.get("b7_epochs", 4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=float(config.get("b7_min_lr", 1e-6))
    )
    scaler = make_scaler(runtime)
    clip = float(config.get("b7_grad_clip", 1.0))
    target_multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)

    outdir = Path(out_root)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "b11_model.pt"
    (outdir / "supervision_plan.json").write_text(json.dumps(supervision, indent=2), encoding="utf-8")
    policy_payload = {
        "experiment": B11_EXPERIMENT,
        "variant": B11_VARIANT,
        "status": "B11-v1 recipe frozen before first gold evaluation",
        "single_scientific_change": "conservative B7.1-teacher supervision on B6-unsupervised cells",
        "student_initialization": str(b5_path.resolve()),
        "student_initialization_variant": b5_payload.get("variant"),
        "teacher_checkpoint": pseudo_policy.get("teacher_checkpoint"),
        "teacher_checkpoint_sha256": pseudo_policy.get("teacher_checkpoint_sha256"),
        "pseudo_labels_sha256": pseudo_policy.get("pseudo_labels_sha256"),
        "pseudo_policy": pseudo_policy,
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "b6_policy": b6_policy,
        "routing_mode": "historical B7.1 dual routing",
        "preprocessing": "historical B7.1 legacy resize; no B10 physical normalization",
        "gold_labels_in_training_loss": False,
        "gold_labels_for_early_stopping": False,
        "gold_labels_used_to_choose_pseudo_cells": False,
        "fixed_epochs": epochs,
        "full_coverage_batches_per_epoch": batches_per_epoch,
        "model_spec": spec,
        "supervision": supervision,
    }
    (outdir / "policy.json").write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")

    history: list[dict] = []
    epoch_times: list[float] = []
    budget_exhausted = False
    for epoch in range(epochs):
        if epoch_times and not budget.can_start(float(np.median(epoch_times)) * 1.20):
            print("[budget] stopping B11 before next epoch")
            break
        start = time.monotonic()
        model.train()
        loss_sum = 0.0
        steps = study_draws = active_cells = positive_cells = negative_cells = 0
        for batch in loader:
            if not budget.can_start(120.0):
                budget_exhausted = True
                print("[budget] stopping B11 batches before wall-clock reserve")
                break
            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            target = batch["target"].to(runtime.device, non_blocking=True)
            weight = batch["weight"].to(runtime.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                logits = model(volumes, present)
                loss = target_balanced_weak_bce(logits, target, weight, target_multiplier_t)
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            active_cell = weight > 0
            loss_sum += float(loss.item())
            steps += 1
            study_draws += int(volumes.shape[0])
            active_cells += int(active_cell.sum().item())
            positive_cells += int((active_cell & (target > 0.5)).sum().item())
            negative_cells += int((active_cell & (target < 0.5)).sum().item())

        seconds = time.monotonic() - start
        epoch_times.append(seconds)
        if steps == 0:
            raise RuntimeError("B11 completed no training batches")
        scheduler.step()
        full_epoch = steps == batches_per_epoch and study_draws == len(ds)
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / steps,
            "encoder_lr": float(optimizer.param_groups[0]["lr"]),
            "head_lr": float(optimizer.param_groups[1]["lr"]),
            "epoch_seconds": float(seconds),
            "batches": int(steps),
            "expected_full_coverage_batches": int(batches_per_epoch),
            "study_draws": int(study_draws),
            "expected_full_coverage_studies": int(len(ds)),
            "active_supervision_cells_seen": int(active_cells),
            "positive_cells_seen": int(positive_cells),
            "negative_cells_seen": int(negative_cells),
            "b6_cells_expected_per_full_epoch": int((b6_active > 0).sum()),
            "pseudo_cells_expected_per_full_epoch": int((pseudo_active > 0).sum()),
            "full_coverage": bool(full_epoch),
            "budget_limited": bool(budget_exhausted),
        }
        history.append(row)
        print(row)
        torch.save(
            _checkpoint_payload(
                model=model, config=config, spec=spec, history=history,
                b5_checkpoint=b5_path, b5_payload=b5_payload, b6_root=Path(b6_root),
                b6_audit=b6_audit, supervision=supervision,
                target_multiplier=target_multiplier, metadata_stats=metadata_stats,
                pseudo_policy=pseudo_policy, pseudo_root=pseudo_root, budget=budget,
            ),
            checkpoint_path,
        )
        (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if budget_exhausted:
            break

    if len(history) != 4 or not all(bool(row.get("full_coverage")) for row in history):
        print("[warning] B11 did not complete the full four-pass contract; do not run frozen gold evaluation")
    return checkpoint_path


def load_b11_checkpoint(checkpoint: str | Path, *, device: torch.device | str = "cpu"):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("variant") != B11_VARIANT:
        raise ValueError(f"not a {B11_VARIANT} checkpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B11 checkpoint does not certify zero gold-gradient studies")
    if payload.get("gold_labels_used_to_choose_pseudo_cells") is not False:
        raise ValueError("B11 checkpoint does not certify label-free pseudo selection")
    spec = payload.get("model_spec")
    state = payload.get("model_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("B11 checkpoint missing model_spec/model_state")
    model = build_b7_model(spec)
    model.load_state_dict(state, strict=True)
    return model.to(device), payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b11")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b5-checkpoint", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--pseudo-root", required=True)
    parser.add_argument("--out-root", default="runs/b11_teacher_student")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = train_b11(
        config,
        b5_checkpoint=args.b5_checkpoint,
        b6_root=args.b6_root,
        pseudo_root=args.pseudo_root,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
