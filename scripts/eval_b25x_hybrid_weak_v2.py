#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_knee.b7_weak_supervision import (
    _read_config,
    load_frozen_b6_export,
    make_b7_dataset_config,
)
from rsna_knee.b12_variable_series import (
    build_variable_series_index,
    collate_variable_series,
)
from rsna_knee.b12_1_gold_eval import predict_b12_1
from rsna_knee.b12_1_hierarchical import build_b12_1_model
from rsna_knee.b15_ssl import (
    WEAK_V2_MANIFEST_SHA256,
    load_frozen_v2_manifest,
)
from rsna_knee.b15_weak_eval import _holdout_supervision
from rsna_knee.b21_dataset import make_matched_crop_dataset
from rsna_knee.constants import TARGETS
from rsna_knee.data import (
    backfill_series_metadata,
    load_series_csv,
    load_train_csv,
)
from rsna_knee.runtime import resolve_runtime
from rsna_knee.weak_validation import (
    compare_on_weak_surface,
    evaluate_on_weak_surface,
)


EXPERIMENT = "B25X_chatgpt_hybrid_training_v1"


def load_checkpoint(path: str | Path, expected_arm: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != EXPERIMENT:
        raise ValueError(
            f"{path}: experiment={payload.get('experiment')!r}, expected {EXPERIMENT!r}"
        )
    if payload.get("arm") != expected_arm:
        raise ValueError(
            f"{path}: arm={payload.get('arm')!r}, expected {expected_arm!r}"
        )
    if payload.get("mode") != f"b25x_{expected_arm}_exploratory":
        raise ValueError(f"{path}: unexpected mode {payload.get('mode')!r}")
    if payload.get("completed_epochs") != 2 or payload.get("fixed_endpoint") is not True:
        raise ValueError(f"{path}: B25X evaluation requires fixed E2 checkpoint")
    if payload.get("exploratory") is not True:
        raise ValueError(f"{path}: checkpoint is not marked exploratory")
    if payload.get("gold_acceptance_allowed") is not False:
        raise ValueError(f"{path}: gold acceptance must remain prohibited")
    return payload


def predict_checkpoint(payload, *, uids, variable_index, dataset_config, runtime, seed):
    model = build_b12_1_model(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model = model.to(runtime.device).eval()

    ds = make_matched_crop_dataset(
        "control",
        uids,
        variable_index,
        dataset_config,
        crop_fraction=float(payload["crop_fraction"]),
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=int(dataset_config.batch_size if hasattr(dataset_config, "batch_size") else 2),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=seed),
    )
    pred_uids, pred = predict_b12_1(model, loader, runtime)
    return [str(x) for x in pred_uids], pred


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate B25X B6 / ChatGPT-hybrid / B6+hybrid-fill on frozen weak-v2"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--hybrid-checkpoint", required=True)
    parser.add_argument("--fill-checkpoint", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--weak-holdout-root", required=True)
    parser.add_argument("--out-root", default="runs/b25x_hybrid/weak_v2_eval")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    config = _read_config(args.config)
    config["data_root"] = args.data_root

    control = load_checkpoint(args.control_checkpoint, "control")
    hybrid = load_checkpoint(args.hybrid_checkpoint, "hybrid")
    fill = load_checkpoint(args.fill_checkpoint, "fill")

    # Matched-training invariants.
    if not (control["study_uids"] == hybrid["study_uids"] == fill["study_uids"]):
        raise RuntimeError("B25X checkpoints were not trained on identical study order")
    if not (
        control["encoder_sha256_initial"]
        == hybrid["encoder_sha256_initial"]
        == fill["encoder_sha256_initial"]
    ):
        raise RuntimeError("B25X checkpoints do not share the same frozen encoder")
    if not (
        float(control["crop_fraction"])
        == float(hybrid["crop_fraction"])
        == float(fill["crop_fraction"])
    ):
        raise RuntimeError("B25X crop geometry differs across arms")

    fill_meta = fill.get("fill_ablation") or {}
    if int(fill_meta.get("b6_cells_dropped", -1)) != 0:
        raise RuntimeError("B25X fill checkpoint reports dropped B6 cells")
    if int(fill_meta.get("b6_cells_overridden", -1)) != 0:
        raise RuntimeError("B25X fill checkpoint reports overridden B6 cells")

    print("=" * 72)
    print("B25X CHECKPOINT VERIFICATION")
    print("=" * 72)
    print("training studies      :", len(control["study_uids"]))
    print("same study order      : True")
    print("same frozen encoder   : True")
    print("same crop             : True")
    print("fixed endpoint        : E2")
    print("fill B6 drops         :", fill_meta.get("b6_cells_dropped"))
    print("fill B6 overrides     :", fill_meta.get("b6_cells_overridden"))
    print("gold acceptance       : prohibited")

    weak_payload, manifest = load_frozen_v2_manifest(args.weak_holdout_root)
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, _ = load_frozen_b6_export(args.b6_root)
    uids, weak_targets, weak_weights, _ = _holdout_supervision(
        train, b6_frame, manifest
    )
    if len(uids) != 623:
        raise RuntimeError(f"frozen weak-v2 size changed: {len(uids)} != 623")

    train_set = set(map(str, control["study_uids"]))
    holdout_set = set(map(str, uids))
    overlap = train_set & holdout_set
    if overlap:
        raise RuntimeError(f"weak-v2 leakage detected: {len(overlap)} studies")

    print()
    print("Frozen weak-v2")
    print("  studies :", len(uids))
    print("  overlap :", len(overlap))
    print("  cells   :", int((weak_weights > 0).sum()))

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    variable_index = build_variable_series_index(series, uids)
    if any(len(variable_index[str(uid)]) == 0 for uid in uids):
        raise RuntimeError("a weak-v2 study has zero eligible series")

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise RuntimeError("B25X evaluation requires TTA [-1,0,1]")
    dataset_config = make_b7_dataset_config(
        config, root, train=False, tta_offsets=offsets
    )

    runtime = resolve_runtime(config)
    print()
    print(runtime.describe())

    expected_uids = [str(x) for x in uids]
    seed = int(config.get("seed", 2026))

    predictions = {}
    for i, (name, payload) in enumerate(
        (("control", control), ("hybrid", hybrid), ("fill", fill))
    ):
        print(f"\n[B25X] predicting {name} on frozen weak-v2...")
        pred_uids, pred = predict_checkpoint(
            payload,
            uids=uids,
            variable_index=variable_index,
            dataset_config=dataset_config,
            runtime=runtime,
            seed=seed + 25_100_000 + i,
        )
        if pred_uids != expected_uids:
            raise RuntimeError(f"{name} prediction order changed")
        predictions[name] = pred

    n_bootstrap = int(args.n_bootstrap)
    evals = {
        name: evaluate_on_weak_surface(
            weak_targets,
            pred,
            weak_weights,
            n_bootstrap=n_bootstrap,
            seed=seed + 25_200_000 + i,
        )
        for i, (name, pred) in enumerate(predictions.items())
    }

    comparisons = {
        "hybrid_minus_control": compare_on_weak_surface(
            weak_targets,
            predictions["control"],
            predictions["hybrid"],
            weak_weights,
            n_bootstrap=n_bootstrap,
            seed=seed + 25_300_001,
        ),
        "fill_minus_control": compare_on_weak_surface(
            weak_targets,
            predictions["control"],
            predictions["fill"],
            weak_weights,
            n_bootstrap=n_bootstrap,
            seed=seed + 25_300_002,
        ),
        "hybrid_minus_fill": compare_on_weak_surface(
            weak_targets,
            predictions["fill"],
            predictions["hybrid"],
            weak_weights,
            n_bootstrap=n_bootstrap,
            seed=seed + 25_300_003,
        ),
    }

    per_target = {}
    for target in TARGETS:
        c = float(evals["control"]["per_target_auc"][target])
        h = float(evals["hybrid"]["per_target_auc"][target])
        f = float(evals["fill"]["per_target_auc"][target])
        per_target[target] = {
            "control": c,
            "hybrid": h,
            "fill": f,
            "hybrid_minus_control": h - c,
            "fill_minus_control": f - c,
            "hybrid_minus_fill": h - f,
        }

    result = {
        "experiment": EXPERIMENT,
        "surface": "frozen_weak_b6_holdout_v2",
        "gold_used": False,
        "promotion_allowed": False,
        "n_training_studies": len(control["study_uids"]),
        "n_holdout_studies": len(uids),
        "holdout_usable_cells": int((weak_weights > 0).sum()),
        "weak_holdout_manifest_sha256": WEAK_V2_MANIFEST_SHA256,
        "weak_holdout_metadata": weak_payload,
        "metadata_repair": metadata_stats,
        "control_macro_auc": float(evals["control"]["macro_auc"]),
        "hybrid_macro_auc": float(evals["hybrid"]["macro_auc"]),
        "fill_macro_auc": float(evals["fill"]["macro_auc"]),
        "evaluations": evals,
        "comparisons": comparisons,
        "per_target": per_target,
        "interpretation": (
            "Exploratory B25X evaluation on frozen B6 weak-v2 teacher agreement. "
            "The ChatGPT hybrid source has mixed/unknown original provenance. "
            "No expert-gold evaluation or promotion is allowed."
        ),
    }

    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame({"StudyInstanceUID": expected_uids})
    for j, target in enumerate(TARGETS):
        for name in ("control", "hybrid", "fill"):
            frame[f"{target}__{name}"] = predictions[name][:, j]
    frame.to_csv(out / "three_arm_predictions.csv", index=False)
    (out / "comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    c_auc = result["control_macro_auc"]
    h_auc = result["hybrid_macro_auc"]
    f_auc = result["fill_macro_auc"]
    print()
    print("=" * 72)
    print("B25X FROZEN WEAK-V2 RESULT")
    print("=" * 72)
    print(f"B6 control : {c_auc:.10f}")
    print(f"Hybrid     : {h_auc:.10f}")
    print(f"B6+Fill    : {f_auc:.10f}")
    print()
    print(f"Hybrid - B6  : {h_auc-c_auc:+.10f}")
    print(f"Fill - B6    : {f_auc-c_auc:+.10f}")
    print(f"Hybrid - Fill: {h_auc-f_auc:+.10f}")

    for key, label in (
        ("hybrid_minus_control", "Hybrid - B6"),
        ("fill_minus_control", "Fill - B6"),
        ("hybrid_minus_fill", "Hybrid - Fill"),
    ):
        cmp = comparisons[key]
        print()
        print(label)
        print(f"  median : {cmp['median_difference']:+.10f}")
        print(f"  95% CI : [{cmp['ci_low']:+.10f}, {cmp['ci_high']:+.10f}]")
        print(f"  P(>0)  : {cmp['probability_b_better']:.4f}" if "probability_b_better" in cmp else f"  P(>0)  : {cmp['probability_candidate_better']:.4f}")

    print()
    print("Per-target")
    print("-" * 76)
    print(f"{'Target':18s} {'B6':>8s} {'Hybrid':>8s} {'Fill':>8s} {'H-B6':>8s} {'F-B6':>8s}")
    for target, row in per_target.items():
        print(
            f"{target:18s} {row['control']:8.4f} {row['hybrid']:8.4f} "
            f"{row['fill']:8.4f} {row['hybrid_minus_control']:+8.4f} "
            f"{row['fill_minus_control']:+8.4f}"
        )

    print()
    print("NO GOLD. NO PROMOTION.")
    print("saved:", out / "three_arm_predictions.csv")
    print("saved:", out / "comparison.json")


if __name__ == "__main__":
    main()
