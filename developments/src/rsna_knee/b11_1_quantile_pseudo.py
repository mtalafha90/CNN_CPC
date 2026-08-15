"""B11.1: calibration-aware B7.1 teacher pseudo-label audit.

B11-v1 showed that a single absolute 0.10/0.90 confidence gate is unsuitable
because teacher probability ranges differ strongly by pathology. B11.1 remains
label-free with respect to the 58 gold studies and instead selects the bottom
and top 5% of B7.1 teacher probabilities separately for each target among
B6-unsupervised cells, while retaining the frozen TTA-stability requirement.

This module generates/audits pseudo labels only. Student training is enabled
only after the frozen B11.1 viability audit passes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .b11_pseudo_labels import (
    B11_TEACHER_TTA,
    _all_non_gold_b6,
    _series_signature,
    _sha256_file,
    predict_teacher_views,
)
from .b7_weak_supervision import (
    _read_config,
    load_b7_checkpoint,
    load_frozen_b6_export,
    make_b7_dataset_config,
    target_balance_multipliers,
)
from .constants import DUAL_STREAMS, TARGETS
from .data import backfill_series_metadata, build_series_index, load_series_csv, load_train_csv
from .dataset import KneeStudyDataset
from .runtime import resolve_runtime

B11_1_POLICY = "b7_1_tta_quantile_tails_v1"
B11_1_LOW_QUANTILE = 0.05
B11_1_HIGH_QUANTILE = 0.95
B11_1_MAX_TTA_RANGE = 0.05
B11_1_LOW_TARGET = 0.10
B11_1_HIGH_TARGET = 0.90
B11_1_PSEUDO_BASE_WEIGHT = 0.10
B11_1_PSEUDO_MASS_CAP_FRACTION = 0.15
B11_1_MIN_TOTAL_PSEUDO_CELLS = 2500
B11_1_MIN_PSEUDO_CELLS_PER_TARGET = 100
B11_1_MIN_CELLS_PER_TAIL = 50


def _require_policy(config: dict) -> None:
    expected = {
        "b11_1_low_quantile": B11_1_LOW_QUANTILE,
        "b11_1_high_quantile": B11_1_HIGH_QUANTILE,
        "b11_1_max_tta_range": B11_1_MAX_TTA_RANGE,
        "b11_1_low_target": B11_1_LOW_TARGET,
        "b11_1_high_target": B11_1_HIGH_TARGET,
        "b11_1_pseudo_base_weight": B11_1_PSEUDO_BASE_WEIGHT,
        "b11_1_pseudo_mass_cap_fraction": B11_1_PSEUDO_MASS_CAP_FRACTION,
        "b11_1_min_total_pseudo_cells": B11_1_MIN_TOTAL_PSEUDO_CELLS,
        "b11_1_min_pseudo_cells_per_target": B11_1_MIN_PSEUDO_CELLS_PER_TARGET,
        "b11_1_min_cells_per_tail": B11_1_MIN_CELLS_PER_TAIL,
    }
    for key, expected_value in expected.items():
        got = config.get(key, expected_value)
        if isinstance(expected_value, int):
            if int(got) != expected_value:
                raise ValueError(f"B11.1 policy frozen: {key} must be {expected_value}, got {got}")
        elif not np.isclose(float(got), float(expected_value), atol=1e-12, rtol=0):
            raise ValueError(f"B11.1 policy frozen: {key} must be {expected_value}, got {got}")
    offsets = tuple(int(x) for x in config.get("b11_1_teacher_tta_offsets", list(B11_TEACHER_TTA)))
    if offsets != B11_TEACHER_TTA:
        raise ValueError(f"B11.1 teacher TTA frozen at {B11_TEACHER_TTA}, got {offsets}")


def combine_b6_and_quantile_teacher(
    b6_target: np.ndarray,
    b6_weight: np.ndarray,
    teacher_mean: np.ndarray,
    teacher_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict[str, tuple[float, float]]]:
    """Add balanced target-wise teacher tails only on B6-unsupervised cells."""
    y = np.asarray(b6_target, dtype=np.float32).copy()
    w = np.asarray(b6_weight, dtype=np.float32).copy()
    mean = np.asarray(teacher_mean, dtype=np.float32)
    spread = np.asarray(teacher_range, dtype=np.float32)
    if y.shape != w.shape or y.shape != mean.shape or y.shape != spread.shape:
        raise ValueError("B11.1 arrays must have identical shape")
    if y.ndim != 2 or y.shape[1] != len(TARGETS):
        raise ValueError("B11.1 arrays must have shape [N,12]")

    pseudo_weight = np.zeros_like(w, dtype=np.float32)
    accepted_any = np.zeros_like(w, dtype=bool)
    thresholds: dict[str, tuple[float, float]] = {}
    per_target: dict[str, dict] = {}

    for j, target in enumerate(TARGETS):
        unsup = (w[:, j] <= 0) & np.isfinite(mean[:, j]) & np.isfinite(spread[:, j])
        values = mean[unsup, j]
        if values.size == 0:
            raise ValueError(f"B11.1 has no unsupervised teacher values for {target}")
        low_thr = float(np.quantile(values, B11_1_LOW_QUANTILE))
        high_thr = float(np.quantile(values, B11_1_HIGH_QUANTILE))
        if not low_thr < high_thr:
            raise ValueError(f"B11.1 degenerate quantile thresholds for {target}: {low_thr}, {high_thr}")
        thresholds[target] = (low_thr, high_thr)

        stable = spread[:, j] <= B11_1_MAX_TTA_RANGE
        low = unsup & stable & (mean[:, j] <= low_thr)
        high = unsup & stable & (mean[:, j] >= high_thr)
        accepted = low | high
        count = int(accepted.sum())
        low_count = int(low.sum())
        high_count = int(high.sum())

        b6_mass = float(w[:, j].sum())
        raw_mass = float(B11_1_PSEUDO_BASE_WEIGHT * count)
        cap_mass = float(B11_1_PSEUDO_MASS_CAP_FRACTION * b6_mass)
        scale = 0.0 if raw_mass <= 0 else min(1.0, cap_mass / raw_mass)
        applied = float(B11_1_PSEUDO_BASE_WEIGHT * scale)

        y[low, j] = B11_1_LOW_TARGET
        y[high, j] = B11_1_HIGH_TARGET
        pseudo_weight[accepted, j] = applied
        accepted_any[:, j] = accepted
        per_target[target] = {
            "unsupervised_cells": int(unsup.sum()),
            "low_quantile_threshold": low_thr,
            "high_quantile_threshold": high_thr,
            "pseudo_cells": count,
            "pseudo_low_cells": low_count,
            "pseudo_high_cells": high_count,
            "b6_base_weight_mass": b6_mass,
            "raw_pseudo_weight_mass": raw_mass,
            "pseudo_mass_cap": cap_mass,
            "pseudo_scale": float(scale),
            "applied_pseudo_cell_weight": applied,
            "applied_pseudo_weight_mass": float(pseudo_weight[:, j].sum()),
        }

    combined_weight = w + pseudo_weight
    b6_active = w.sum(axis=1) > 0
    combined_active = combined_weight.sum(axis=1) > 0
    total_pseudo = int(accepted_any.sum())
    failures = {}
    for target in TARGETS:
        row = per_target[target]
        reasons = []
        if int(row["pseudo_cells"]) < B11_1_MIN_PSEUDO_CELLS_PER_TARGET:
            reasons.append(f"pseudo_cells<{B11_1_MIN_PSEUDO_CELLS_PER_TARGET}")
        if int(row["pseudo_low_cells"]) < B11_1_MIN_CELLS_PER_TAIL:
            reasons.append(f"low_tail<{B11_1_MIN_CELLS_PER_TAIL}")
        if int(row["pseudo_high_cells"]) < B11_1_MIN_CELLS_PER_TAIL:
            reasons.append(f"high_tail<{B11_1_MIN_CELLS_PER_TAIL}")
        if reasons:
            failures[target] = reasons
    viability = total_pseudo >= B11_1_MIN_TOTAL_PSEUDO_CELLS and not failures

    summary = {
        "policy": B11_1_POLICY,
        "low_quantile": B11_1_LOW_QUANTILE,
        "high_quantile": B11_1_HIGH_QUANTILE,
        "max_tta_range": B11_1_MAX_TTA_RANGE,
        "low_pseudo_target": B11_1_LOW_TARGET,
        "high_pseudo_target": B11_1_HIGH_TARGET,
        "pseudo_base_weight": B11_1_PSEUDO_BASE_WEIGHT,
        "pseudo_mass_cap_fraction": B11_1_PSEUDO_MASS_CAP_FRACTION,
        "minimum_total_pseudo_cells": B11_1_MIN_TOTAL_PSEUDO_CELLS,
        "minimum_pseudo_cells_per_target": B11_1_MIN_PSEUDO_CELLS_PER_TARGET,
        "minimum_cells_per_tail": B11_1_MIN_CELLS_PER_TAIL,
        "b6_cells": int((w > 0).sum()),
        "pseudo_cells": total_pseudo,
        "combined_cells": int((combined_weight > 0).sum()),
        "b6_active_studies": int(b6_active.sum()),
        "combined_active_studies": int(combined_active.sum()),
        "newly_activated_studies": int((combined_active & ~b6_active).sum()),
        "viability_passed": bool(viability),
        "viability_failures": failures,
        "per_target": per_target,
    }
    return y, combined_weight, pseudo_weight, summary, thresholds


def generate_b11_1_pseudo_labels(
    config: dict,
    *,
    teacher_checkpoint: str | Path,
    b6_root: str | Path,
    out_root: str | Path = "runs/b11_1_quantile_teacher/pseudo",
) -> dict:
    _require_policy(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, b6_audit = load_frozen_b6_export(b6_root)
    uids, b6_y, b6_w, b6_summary = _all_non_gold_b6(train, b6_frame)
    if len(uids) != 4349:
        raise ValueError(f"B11.1 expects 4349 non-gold studies, got {len(uids)}")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_series_index(series, uids, mode="dual")
    if not all(any(index.get(uid, {}).get(stream) for stream in DUAL_STREAMS) for uid in uids):
        raise ValueError("B11.1 requires at least one selected MRI stream for every non-gold study")

    teacher_path = Path(teacher_checkpoint)
    teacher, teacher_payload = load_b7_checkpoint(teacher_path, device=runtime.device)
    if int(teacher_payload.get("completed_epochs", -1)) != 4:
        raise ValueError("B11.1 requires completed four-epoch B7.1 teacher")
    teacher_name = str(teacher_payload.get("config", {}).get("b7_experiment_name", ""))
    if teacher_name != "B7.1_full_coverage":
        raise ValueError(f"B11.1 requires B7.1_full_coverage teacher, got {teacher_name!r}")

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
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 11_110_000),
    )
    pred_uids, view_probs = predict_teacher_views(teacher, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("B11.1 teacher prediction order changed unexpectedly")
    teacher_mean = view_probs.mean(axis=1)
    teacher_range = view_probs.max(axis=1) - view_probs.min(axis=1)
    combined_y, combined_w, pseudo_w, summary, thresholds = combine_b6_and_quantile_teacher(
        b6_y, b6_w, teacher_mean, teacher_range
    )

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

    target_multiplier = target_balance_multipliers(b6_w)
    policy = {
        "experiment": "B11.1_quantile_teacher_pseudo_label_generation",
        "status": "B11.1 label-free pseudo policy frozen before any student training/gold evaluation",
        "policy": B11_1_POLICY,
        "viability_passed": bool(summary["viability_passed"]),
        "uses_gold_labels_to_choose_pseudo_cells": False,
        "motivation": "B11-v1 diagnostic showed target-specific probability calibration shifts with generally stable TTA predictions",
        "teacher_checkpoint": str(teacher_path.resolve()),
        "teacher_checkpoint_sha256": _sha256_file(teacher_path),
        "teacher_experiment": teacher_name,
        "teacher_completed_epochs": teacher_payload.get("completed_epochs"),
        "teacher_tta_offsets": list(B11_TEACHER_TTA),
        "low_quantile": B11_1_LOW_QUANTILE,
        "high_quantile": B11_1_HIGH_QUANTILE,
        "max_tta_range": B11_1_MAX_TTA_RANGE,
        "low_pseudo_target": B11_1_LOW_TARGET,
        "high_pseudo_target": B11_1_HIGH_TARGET,
        "pseudo_base_weight": B11_1_PSEUDO_BASE_WEIGHT,
        "pseudo_mass_cap_fraction": B11_1_PSEUDO_MASS_CAP_FRACTION,
        "thresholds_derived_without_labels": {
            target: {"low": float(thresholds[target][0]), "high": float(thresholds[target][1])}
            for target in TARGETS
        },
        "b6_version": b6_audit.get("b6_version"),
        "b6_cells": int((b6_w > 0).sum()),
        "b6_target_balance_multiplier": {target: float(target_multiplier[j]) for j, target in enumerate(TARGETS)},
        "non_gold_studies": len(uids),
        "routing_mode": "historical B7.1 dual routing",
        "preprocessing": "historical B7.1 legacy 224x224 resize; no B10 physical normalization",
        "selected_series_signature": _series_signature(index, uids),
        "pseudo_labels_csv": str(pseudo_csv.resolve()),
        "metadata_repair": metadata_stats,
        "b6_summary": b6_summary,
        "pseudo_summary": summary,
    }
    policy["pseudo_labels_sha256"] = _sha256_file(pseudo_csv)
    (out / "pseudo_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (out / "pseudo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["viability_passed"]:
        print("[B11.1] pseudo viability gate FAILED; inspect artifacts and do not train a student.")
    else:
        print("[B11.1] pseudo viability gate PASSED; artifacts are frozen for student implementation/training.")
    print(pseudo_csv)
    print(out / "pseudo_policy.json")
    print(out / "pseudo_summary.json")
    return policy


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b11-1-pseudo")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--out-root", default="runs/b11_1_quantile_teacher/pseudo")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    generate_b11_1_pseudo_labels(
        config,
        teacher_checkpoint=args.teacher_checkpoint,
        b6_root=args.b6_root,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
