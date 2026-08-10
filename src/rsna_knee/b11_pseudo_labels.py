"""B11: frozen B7.1 teacher pseudo-label generation on B6-unsupervised cells.

This stage is label-free with respect to the 58 gold development studies.  It
uses the completed B7.1 MRI model as a fixed teacher over all 4,349 non-gold
competition studies.  B6-supervised cells are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .b7_weak_supervision import (
    _read_config,
    load_b7_checkpoint,
    load_frozen_b6_export,
    make_b7_dataset_config,
    prepare_b7_supervision,
    target_balance_multipliers,
)
from .constants import DUAL_STREAMS, TARGETS
from .data import backfill_series_metadata, build_series_index, gold_mask, load_series_csv, load_train_csv
from .dataset import KneeStudyDataset
from .runtime import autocast, resolve_runtime

B11_PSEUDO_POLICY = "b7_1_tta_confident_cells_v1"
B11_POSITIVE_THRESHOLD = 0.90
B11_NEGATIVE_THRESHOLD = 0.10
B11_MAX_TTA_RANGE = 0.05
B11_PSEUDO_BASE_WEIGHT = 0.20
B11_PSEUDO_MASS_CAP_FRACTION = 0.25
B11_MIN_TOTAL_PSEUDO_CELLS = 500
B11_MIN_PSEUDO_CELLS_PER_TARGET = 25
B11_TEACHER_TTA = (-1, 0, 1)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _series_signature(index: dict[str, dict[str, str | None]], uids: list[str]) -> str:
    rows = []
    for uid in uids:
        mapping = index.get(str(uid), {})
        rows.append([str(uid), *[str(mapping.get(stream) or "") for stream in DUAL_STREAMS]])
    raw = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_policy(config: dict) -> None:
    expected = {
        "b11_positive_threshold": B11_POSITIVE_THRESHOLD,
        "b11_negative_threshold": B11_NEGATIVE_THRESHOLD,
        "b11_max_tta_range": B11_MAX_TTA_RANGE,
        "b11_pseudo_base_weight": B11_PSEUDO_BASE_WEIGHT,
        "b11_pseudo_mass_cap_fraction": B11_PSEUDO_MASS_CAP_FRACTION,
        "b11_min_total_pseudo_cells": B11_MIN_TOTAL_PSEUDO_CELLS,
        "b11_min_pseudo_cells_per_target": B11_MIN_PSEUDO_CELLS_PER_TARGET,
    }
    for key, value in expected.items():
        got = config.get(key, value)
        if isinstance(value, int):
            if int(got) != value:
                raise ValueError(f"B11-v1 policy is frozen: {key} must be {value}, got {got}")
        elif not np.isclose(float(got), float(value), atol=1e-12, rtol=0):
            raise ValueError(f"B11-v1 policy is frozen: {key} must be {value}, got {got}")
    offsets = tuple(int(x) for x in config.get("b11_teacher_tta_offsets", list(B11_TEACHER_TTA)))
    if offsets != B11_TEACHER_TTA:
        raise ValueError(f"B11-v1 teacher TTA is frozen at {B11_TEACHER_TTA}, got {offsets}")


def combine_b6_and_teacher(
    b6_target: np.ndarray,
    b6_weight: np.ndarray,
    teacher_mean: np.ndarray,
    teacher_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Add conservative teacher labels only where B6 has zero supervision.

    Pseudo-label total base mass is independently capped for each target at 25%
    of that target's original B6 base supervision mass.
    """
    y = np.asarray(b6_target, dtype=np.float32).copy()
    w = np.asarray(b6_weight, dtype=np.float32).copy()
    mean = np.asarray(teacher_mean, dtype=np.float32)
    spread = np.asarray(teacher_range, dtype=np.float32)
    if y.shape != w.shape or y.shape != mean.shape or y.shape != spread.shape:
        raise ValueError("B11 supervision arrays must have identical shape")
    if y.ndim != 2 or y.shape[1] != len(TARGETS):
        raise ValueError("B11 supervision arrays must have shape [N,12]")

    unsupervised = w <= 0
    confidence = (mean >= B11_POSITIVE_THRESHOLD) | (mean <= B11_NEGATIVE_THRESHOLD)
    consistent = spread <= B11_MAX_TTA_RANGE
    accepted = unsupervised & confidence & consistent & np.isfinite(mean) & np.isfinite(spread)
    pseudo_weight = np.zeros_like(w, dtype=np.float32)
    per_target: dict[str, dict] = {}

    for j, target in enumerate(TARGETS):
        mask = accepted[:, j]
        count = int(mask.sum())
        b6_mass = float(w[:, j].sum())
        raw_mass = float(B11_PSEUDO_BASE_WEIGHT * count)
        cap_mass = float(B11_PSEUDO_MASS_CAP_FRACTION * b6_mass)
        scale = 0.0 if raw_mass <= 0 else min(1.0, cap_mass / raw_mass)
        applied_weight = float(B11_PSEUDO_BASE_WEIGHT * scale)
        if count:
            y[mask, j] = mean[mask, j]
            pseudo_weight[mask, j] = applied_weight
        per_target[target] = {
            "pseudo_cells": count,
            "pseudo_positive_cells": int((mask & (mean[:, j] >= B11_POSITIVE_THRESHOLD)).sum()),
            "pseudo_negative_cells": int((mask & (mean[:, j] <= B11_NEGATIVE_THRESHOLD)).sum()),
            "b6_base_weight_mass": b6_mass,
            "raw_pseudo_weight_mass": raw_mass,
            "pseudo_mass_cap": cap_mass,
            "pseudo_scale": float(scale),
            "applied_pseudo_cell_weight": applied_weight,
            "applied_pseudo_weight_mass": float(pseudo_weight[:, j].sum()),
        }

    combined_weight = w + pseudo_weight
    b6_active_study = w.sum(axis=1) > 0
    combined_active_study = combined_weight.sum(axis=1) > 0
    total_pseudo = int(accepted.sum())
    summary = {
        "policy": B11_PSEUDO_POLICY,
        "positive_threshold": B11_POSITIVE_THRESHOLD,
        "negative_threshold": B11_NEGATIVE_THRESHOLD,
        "max_tta_range": B11_MAX_TTA_RANGE,
        "pseudo_base_weight": B11_PSEUDO_BASE_WEIGHT,
        "pseudo_mass_cap_fraction": B11_PSEUDO_MASS_CAP_FRACTION,
        "b6_cells": int((w > 0).sum()),
        "pseudo_cells": total_pseudo,
        "combined_cells": int((combined_weight > 0).sum()),
        "b6_active_studies": int(b6_active_study.sum()),
        "combined_active_studies": int(combined_active_study.sum()),
        "newly_activated_studies": int((combined_active_study & ~b6_active_study).sum()),
        "per_target": per_target,
    }
    return y, combined_weight, pseudo_weight, summary


def _all_non_gold_b6(train: pd.DataFrame, b6_frame: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    non_gold = train.loc[~gold_mask(train), ["StudyInstanceUID"]].copy()
    uids = non_gold["StudyInstanceUID"].astype(str).tolist()
    active_uids, active_y, active_w, summary = prepare_b7_supervision(train, b6_frame)
    y = np.full((len(uids), len(TARGETS)), 0.5, dtype=np.float32)
    w = np.zeros((len(uids), len(TARGETS)), dtype=np.float32)
    pos = {uid: i for i, uid in enumerate(uids)}
    for uid, row_y, row_w in zip(active_uids, active_y, active_w):
        i = pos[str(uid)]
        y[i] = row_y
        w[i] = row_w
    return uids, y, w, summary


@torch.no_grad()
def predict_teacher_views(model, loader, runtime) -> tuple[list[str], np.ndarray]:
    """Return [N,V,12] probabilities so TTA consistency is auditable."""
    model.eval()
    uids: list[str] = []
    output: list[np.ndarray] = []
    for batch in loader:
        volumes = batch["volumes"]
        present = batch["present"].to(runtime.device, non_blocking=True)
        if volumes.ndim != 7:
            raise ValueError(f"B11 teacher requires TTA views, got {tuple(volumes.shape)}")
        view_probs = []
        for view in range(volumes.shape[1]):
            with autocast(runtime):
                logits = model(volumes[:, view].to(runtime.device, non_blocking=True), present)
            view_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        output.append(np.stack(view_probs, axis=1))
        uids.extend([str(x) for x in batch["study_uid"]])
    if not output:
        raise RuntimeError("B11 teacher produced no predictions")
    return uids, np.concatenate(output, axis=0)


def generate_b11_pseudo_labels(
    config: dict,
    *,
    teacher_checkpoint: str | Path,
    b6_root: str | Path,
    out_root: str | Path = "runs/b11_teacher_student/pseudo",
) -> dict:
    _require_policy(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, b6_audit = load_frozen_b6_export(b6_root)
    uids, b6_y, b6_w, b6_summary = _all_non_gold_b6(train, b6_frame)
    if len(uids) != 4349:
        raise ValueError(f"B11-v1 expects 4349 non-gold studies, got {len(uids)}")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_series_index(series, uids, mode="dual")
    if not all(any(index.get(uid, {}).get(stream) for stream in DUAL_STREAMS) for uid in uids):
        raise ValueError("B11-v1 requires at least one selected MRI stream for every non-gold study")

    teacher_path = Path(teacher_checkpoint)
    teacher, teacher_payload = load_b7_checkpoint(teacher_path, device=runtime.device)
    if int(teacher_payload.get("completed_epochs", -1)) != 4:
        raise ValueError("B11-v1 requires the completed four-epoch B7.1 teacher")
    teacher_name = str(teacher_payload.get("config", {}).get("b7_experiment_name", ""))
    if teacher_name != "B7.1_full_coverage":
        raise ValueError(f"B11-v1 requires B7.1_full_coverage teacher, got {teacher_name!r}")

    ds = KneeStudyDataset(
        uids,
        index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=B11_TEACHER_TTA),
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=max(1, int(config.get("b7_eval_batch_size", 2))),
        shuffle=False,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 11_100_000),
    )
    pred_uids, view_probs = predict_teacher_views(teacher, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("B11 teacher prediction order changed unexpectedly")
    teacher_mean = view_probs.mean(axis=1)
    teacher_range = view_probs.max(axis=1) - view_probs.min(axis=1)
    combined_y, combined_w, pseudo_w, summary = combine_b6_and_teacher(
        b6_y, b6_w, teacher_mean, teacher_range
    )

    if int(summary["pseudo_cells"]) < B11_MIN_TOTAL_PSEUDO_CELLS:
        raise ValueError(
            f"B11-v1 pseudo viability failed: {summary['pseudo_cells']} < {B11_MIN_TOTAL_PSEUDO_CELLS} total cells"
        )
    too_small = {
        target: int(summary["per_target"][target]["pseudo_cells"])
        for target in TARGETS
        if int(summary["per_target"][target]["pseudo_cells"]) < B11_MIN_PSEUDO_CELLS_PER_TARGET
    }
    if too_small:
        raise ValueError(f"B11-v1 pseudo viability failed per target: {too_small}")

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"StudyInstanceUID": uids})
    for j, target in enumerate(TARGETS):
        frame[f"{target}__teacher_mean"] = teacher_mean[:, j]
        frame[f"{target}__tta_range"] = teacher_range[:, j]
        frame[f"{target}__b6_weight"] = b6_w[:, j]
        frame[f"{target}__combined_target"] = combined_y[:, j]
        frame[f"{target}__combined_weight"] = combined_w[:, j]
        frame[f"{target}__pseudo_weight"] = pseudo_w[:, j]
    pseudo_csv = out / "pseudo_labels.csv"
    frame.to_csv(pseudo_csv, index=False)

    teacher_sha = _sha256_file(teacher_path)
    pseudo_sha = _sha256_file(pseudo_csv)
    target_multiplier = target_balance_multipliers(b6_w)
    policy = {
        "experiment": "B11_teacher_pseudo_label_generation",
        "status": "B11-v1 pseudo policy frozen before student training/gold evaluation",
        "policy": B11_PSEUDO_POLICY,
        "uses_gold_labels_to_choose_pseudo_cells": False,
        "teacher_checkpoint": str(teacher_path.resolve()),
        "teacher_checkpoint_sha256": teacher_sha,
        "teacher_experiment": teacher_name,
        "teacher_completed_epochs": teacher_payload.get("completed_epochs"),
        "teacher_tta_offsets": list(B11_TEACHER_TTA),
        "positive_threshold": B11_POSITIVE_THRESHOLD,
        "negative_threshold": B11_NEGATIVE_THRESHOLD,
        "max_tta_range": B11_MAX_TTA_RANGE,
        "pseudo_target": "teacher mean probability",
        "pseudo_base_weight": B11_PSEUDO_BASE_WEIGHT,
        "pseudo_mass_cap_fraction": B11_PSEUDO_MASS_CAP_FRACTION,
        "minimum_total_pseudo_cells": B11_MIN_TOTAL_PSEUDO_CELLS,
        "minimum_pseudo_cells_per_target": B11_MIN_PSEUDO_CELLS_PER_TARGET,
        "b6_version": b6_audit.get("b6_version"),
        "b6_cells": int((b6_w > 0).sum()),
        "b6_target_balance_multiplier": {target: float(target_multiplier[j]) for j, target in enumerate(TARGETS)},
        "non_gold_studies": len(uids),
        "routing_mode": "historical B7.1 dual routing",
        "preprocessing": "historical B7.1 legacy 224x224 resize; no B10 physical normalization",
        "selected_series_signature": _series_signature(index, uids),
        "pseudo_labels_csv": str(pseudo_csv.resolve()),
        "pseudo_labels_sha256": pseudo_sha,
        "metadata_repair": metadata_stats,
        "b6_summary": b6_summary,
        "pseudo_summary": summary,
    }
    (out / "pseudo_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (out / "pseudo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(pseudo_csv)
    print(out / "pseudo_policy.json")
    print(out / "pseudo_summary.json")
    return policy


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b11-pseudo")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--out-root", default="runs/b11_teacher_student/pseudo")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    generate_b11_pseudo_labels(
        config,
        teacher_checkpoint=args.teacher_checkpoint,
        b6_root=args.b6_root,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
