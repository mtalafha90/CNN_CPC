"""Paired weak-v2 evaluation for matched B20-v2 control versus B21."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, load_frozen_b6_export, make_b7_dataset_config
from .b12_variable_series import build_variable_series_index, collate_variable_series
from .b12_1_gold_eval import predict_b12_1
from .b12_1_hierarchical import build_b12_1_model
from .b15_ssl import WEAK_V2_MANIFEST_SHA256, load_frozen_v2_manifest
from .b15_weak_eval import _holdout_supervision
from .b21_contract import require_b21_contract
from .b21_dataset import make_matched_crop_dataset
from .b21_protocol import B21_FIXED_EPOCHS, B21_WEAK_HOLDOUT_STUDIES
from .constants import TARGETS, TARGET_SLUGS
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .runtime import resolve_runtime
from .weak_validation import compare_on_weak_surface, evaluate_on_weak_surface


def _load_checkpoint(path: str | Path, mode: str, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("mode") != mode:
        raise ValueError(f"expected {mode!r} checkpoint, got {payload.get('mode')!r}")
    if int(payload.get("completed_epochs", -1)) != B21_FIXED_EPOCHS:
        raise ValueError("matched crop checkpoint must be the fixed epoch-2 endpoint")
    if payload.get("weak_holdout_manifest_sha256") != WEAK_V2_MANIFEST_SHA256:
        raise ValueError("checkpoint weak-v2 manifest SHA mismatch")
    if payload.get("gold_labels_used_for_development") is not False:
        raise ValueError("checkpoint does not certify zero gold development use")
    model = build_b12_1_model(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload


def _predict_mode(
    *,
    mode: str,
    model,
    config: dict,
    root: Path,
    uids,
    variable_index,
    runtime,
    crop_fraction: float,
):
    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B21 weak-v2 comparison freezes TTA [-1,0,1]")
    ds = make_matched_crop_dataset(
        mode,
        uids,
        variable_index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        crop_fraction=crop_fraction,
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=int(config.get("b7_eval_batch_size", 2)),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 21_500_000),
    )
    pred_uids, prediction = predict_b12_1(model, loader, runtime)
    if pred_uids != list(uids):
        raise RuntimeError(f"{mode} weak-v2 prediction order changed")
    return prediction


def compare_b20_b21_weak_v2(
    config: dict,
    *,
    control_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    b6_root: str | Path,
    weak_holdout_root: str | Path,
    out_root: str | Path,
) -> dict:
    crop_fraction = require_b21_contract(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    control_model, control_payload = _load_checkpoint(
        control_checkpoint, "control", runtime.device
    )
    candidate_model, candidate_payload = _load_checkpoint(
        candidate_checkpoint, "preresize", runtime.device
    )
    if control_payload.get("encoder_sha256_initial") != candidate_payload.get("encoder_sha256_initial"):
        raise ValueError("matched arms did not start from the same frozen encoder")

    weak_payload, manifest = load_frozen_v2_manifest(weak_holdout_root)
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, _ = load_frozen_b6_export(b6_root)
    uids, weak_targets, weak_weights, _ = _holdout_supervision(train, b6_frame, manifest)
    if len(uids) != B21_WEAK_HOLDOUT_STUDIES:
        raise ValueError("weak-v2 holdout count changed")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    variable_index = build_variable_series_index(series, uids)
    counts = [len(variable_index[uid]) for uid in uids]
    if any(count == 0 for count in counts):
        raise ValueError("weak-v2 holdout study has zero eligible series")

    print("[B21 weak-v2] predicting matched B20 post-resize control")
    pred_control = _predict_mode(
        mode="control",
        model=control_model,
        config=config,
        root=root,
        uids=uids,
        variable_index=variable_index,
        runtime=runtime,
        crop_fraction=crop_fraction,
    )
    print("[B21 weak-v2] predicting B21 pre-resize candidate")
    pred_candidate = _predict_mode(
        mode="preresize",
        model=candidate_model,
        config=config,
        root=root,
        uids=uids,
        variable_index=variable_index,
        runtime=runtime,
        crop_fraction=crop_fraction,
    )

    n_bootstrap = int(config.get("b7_n_bootstrap", 5000))
    seed = int(config.get("seed", 2026))
    control_eval = evaluate_on_weak_surface(
        weak_targets, pred_control, weak_weights,
        n_bootstrap=n_bootstrap, seed=seed + 211,
    )
    candidate_eval = evaluate_on_weak_surface(
        weak_targets, pred_candidate, weak_weights,
        n_bootstrap=n_bootstrap, seed=seed + 212,
    )
    paired = compare_on_weak_surface(
        weak_targets, pred_control, pred_candidate, weak_weights,
        n_bootstrap=n_bootstrap, seed=seed + 213,
    )

    result = {
        "variant": "b21_preresize_crop_weak_v2_paired_comparison_v1",
        "working_model_before_comparison": "B20_crop_only_joint_focus",
        "working_model_automatically_replaced": False,
        "comparison_surface": "weak_b6_holdout_v2",
        "measures": "agreement with frozen B6 report supervision, not expert truth",
        "n_holdout_studies": len(uids),
        "holdout_usable_cells": int((weak_weights > 0).sum()),
        "control": {
            "checkpoint": str(Path(control_checkpoint)),
            "crop_stage": "post_resize_224",
            "macro_auc": float(control_eval["macro_auc"]),
            "ci_lower": float(control_eval["ci_lower"]),
            "ci_upper": float(control_eval["ci_upper"]),
            "per_target_auc": control_eval["per_target_auc"],
        },
        "candidate": {
            "checkpoint": str(Path(candidate_checkpoint)),
            "crop_stage": "native_array_pre_resize",
            "macro_auc": float(candidate_eval["macro_auc"]),
            "ci_lower": float(candidate_eval["ci_lower"]),
            "ci_upper": float(candidate_eval["ci_upper"]),
            "per_target_auc": candidate_eval["per_target_auc"],
        },
        "paired_candidate_minus_control": paired,
        "crop_fraction": crop_fraction,
        "fixed_epoch": B21_FIXED_EPOCHS,
        "encoder_sha256": candidate_payload.get("encoder_sha256_initial"),
        "weak_holdout_manifest_sha256": WEAK_V2_MANIFEST_SHA256,
        "weak_holdout_metadata": weak_payload,
        "metadata_repair": metadata_stats,
        "holdout_series": {
            "total": int(sum(counts)),
            "min": int(min(counts)),
            "median": float(np.median(counts)),
            "max": int(max(counts)),
        },
        "interpretation": (
            "development ranking only. B21 is not promoted automatically; if the paired weak-v2 "
            "result is favorable, freeze the candidate before one predeclared gold acceptance check."
        ),
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"StudyInstanceUID": uids})
    for j, slug in enumerate(TARGET_SLUGS):
        frame[f"control_{slug}"] = pred_control[:, j]
        frame[f"candidate_{slug}"] = pred_candidate[:, j]
    frame.to_csv(out / "paired_predictions.csv", index=False)
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(out / "paired_predictions.csv")
    print(out / "comparison.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b21-weak-eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--weak-holdout-root", required=True)
    parser.add_argument("--out-root", default="runs/b21_preresize_crop/weak_v2_comparison")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    compare_b20_b21_weak_v2(
        config,
        control_checkpoint=args.control_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        b6_root=args.b6_root,
        weak_holdout_root=args.weak_holdout_root,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
