"""Frozen weak-holdout-v2 evaluation for matched B13-control and B15."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .b7_weak_supervision import (
    _read_config,
    load_frozen_b6_export,
    make_b7_dataset_config,
    prepare_b7_supervision,
)
from .b12_variable_series import (
    VariableSeriesKneeDataset,
    build_variable_series_index,
    collate_variable_series,
)
from .b12_1_gold_eval import predict_b12_1
from .b15_downstream import load_v2_downstream_checkpoint, require_v2_downstream_contract
from .b15_ssl import WEAK_V2_MANIFEST_SHA256, WEAK_V2_SURFACE, load_frozen_v2_manifest
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .runtime import resolve_runtime
from .weak_validation import evaluate_on_weak_surface


def _holdout_supervision(train, b6_frame, manifest):
    uids, y, w, summary = prepare_b7_supervision(train, b6_frame)
    row = {str(uid): i for i, uid in enumerate(uids)}
    holdout_uids = manifest.loc[
        manifest["split"] == "holdout", "StudyInstanceUID"
    ].astype(str).tolist()
    idx = np.asarray([row[uid] for uid in holdout_uids], dtype=int)
    if len(idx) != 623:
        raise ValueError("weak-v2 evaluator requires 623 holdout studies")
    return holdout_uids, y[idx], w[idx], summary


def evaluate_v2_weak(
    config: dict,
    *,
    checkpoint: str | Path,
    b6_root: str | Path,
    weak_holdout_root: str | Path,
    out_root: str | Path,
    expected_mode: str | None = None,
) -> dict:
    require_v2_downstream_contract(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, checkpoint_payload = load_v2_downstream_checkpoint(
        checkpoint, expected_mode=expected_mode, device=runtime.device
    )
    weak_payload, manifest = load_frozen_v2_manifest(weak_holdout_root)
    if checkpoint_payload.get("weak_holdout_manifest_sha256") != WEAK_V2_MANIFEST_SHA256:
        raise ValueError("checkpoint and evaluator v2 manifest differ")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, _ = load_frozen_b6_export(b6_root)
    uids, weak_targets, weak_weights, _ = _holdout_supervision(
        train, b6_frame, manifest
    )

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    variable_index = build_variable_series_index(series, uids)
    counts = [len(variable_index[uid]) for uid in uids]
    if any(count == 0 for count in counts):
        raise ValueError("weak-v2 holdout contains a study with zero eligible series")

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("weak-v2 evaluation freezes TTA [-1,0,1]")
    ds = VariableSeriesKneeDataset(
        uids,
        variable_index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=int(config.get("b7_eval_batch_size", 2)),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 16_500_000),
    )
    pred_uids, prediction = predict_b12_1(model, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("weak-v2 prediction order changed")

    result = evaluate_on_weak_surface(
        weak_targets,
        prediction,
        weak_weights,
        n_bootstrap=int(config.get("b7_n_bootstrap", 5000)),
        seed=int(config.get("seed", 2026)) + 151,
    )
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(prediction, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    frame.to_csv(out / "weak_predictions.csv", index=False)

    payload = dict(result)
    payload.update(
        {
            "experiment": (
                "B13_v2_control_weak_evaluation"
                if checkpoint_payload.get("mode") == "control"
                else "B15_weak_evaluation"
            ),
            "mode": checkpoint_payload.get("mode"),
            "variant": checkpoint_payload.get("variant"),
            "checkpoint": str(Path(checkpoint).resolve()),
            "training_experiment": checkpoint_payload.get("experiment"),
            "completed_epochs": checkpoint_payload.get("completed_epochs"),
            "weak_holdout_surface": WEAK_V2_SURFACE,
            "weak_holdout_manifest_sha256": WEAK_V2_MANIFEST_SHA256,
            "n_holdout_studies": len(uids),
            "holdout_usable_cells": int((weak_weights > 0).sum()),
            "tta_center_offsets": list(offsets),
            "holdout_series": {
                "total": int(sum(counts)),
                "min": int(min(counts)),
                "median": float(np.median(counts)),
                "max": int(max(counts)),
            },
            "metadata_repair": metadata_stats,
            "weak_holdout_metadata": weak_payload,
            "interpretation": (
                "biased teacher-agreement ranking surface only; not expert truth and not "
                "independent competition validation"
            ),
        }
    )
    (out / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"weak macro AUC {payload['macro_auc']:.4f} "
        f"[{payload['ci_lower']:.4f}, {payload['ci_upper']:.4f}] "
        f"strict={payload['n_valid_replicates']}/{payload['n_bootstrap']}"
    )
    print(out / "weak_predictions.csv")
    print(out / "eval.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b15-weak-eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--weak-holdout-root", required=True)
    parser.add_argument("--mode", choices=["control", "b15"], default=None)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    payload = evaluate_v2_weak(
        config,
        checkpoint=args.checkpoint,
        b6_root=args.b6_root,
        weak_holdout_root=args.weak_holdout_root,
        out_root=args.out_root,
        expected_mode=args.mode,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
