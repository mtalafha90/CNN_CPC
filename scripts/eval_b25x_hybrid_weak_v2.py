#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_knee.b7_weak_supervision import _read_config, load_frozen_b6_export, make_b7_dataset_config
from rsna_knee.b12_variable_series import build_variable_series_index, collate_variable_series
from rsna_knee.b12_1_gold_eval import predict_b12_1
from rsna_knee.b12_1_hierarchical import build_b12_1_model
from rsna_knee.b15_ssl import WEAK_V2_MANIFEST_SHA256, load_frozen_v2_manifest
from rsna_knee.b15_weak_eval import _holdout_supervision
from rsna_knee.b21_dataset import make_matched_crop_dataset
from rsna_knee.constants import TARGETS
from rsna_knee.data import backfill_series_metadata, load_series_csv, load_train_csv
from rsna_knee.runtime import resolve_runtime
from rsna_knee.weak_validation import compare_on_weak_surface, evaluate_on_weak_surface

EXPERIMENT = "B25X_chatgpt_hybrid_training_v1"


def load_checkpoint(path: str | Path, arm: str) -> dict:
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("experiment") != EXPERIMENT:
        raise ValueError(f"{path}: wrong experiment {p.get('experiment')!r}")
    if p.get("arm") != arm or p.get("mode") != f"b25x_{arm}_exploratory":
        raise ValueError(f"{path}: wrong B25X arm/mode")
    if p.get("completed_epochs") != 2 or p.get("fixed_endpoint") is not True:
        raise ValueError(f"{path}: B25X requires fixed E2")
    if p.get("exploratory") is not True or p.get("gold_acceptance_allowed") is not False:
        raise ValueError(f"{path}: governance flags changed")
    return p


def predict_one(payload, uids, variable_index, dataset_config, runtime, batch_size, seed):
    model = build_b12_1_model(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model = model.to(runtime.device).eval()
    ds = make_matched_crop_dataset(
        "control", uids, variable_index, dataset_config,
        crop_fraction=float(payload["crop_fraction"]), train=False,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=seed),
    )
    puids, pred = predict_b12_1(model, loader, runtime)
    return [str(x) for x in puids], pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--control-checkpoint", required=True)
    ap.add_argument("--hybrid-checkpoint", required=True)
    ap.add_argument("--fill-checkpoint", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--weak-holdout-root", required=True)
    ap.add_argument("--out-root", default="runs/b25x_hybrid/weak_v2_eval")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()

    cfg = _read_config(args.config)
    cfg["data_root"] = args.data_root
    cps = {
        "control": load_checkpoint(args.control_checkpoint, "control"),
        "hybrid": load_checkpoint(args.hybrid_checkpoint, "hybrid"),
        "fill": load_checkpoint(args.fill_checkpoint, "fill"),
    }

    c, h, f = cps["control"], cps["hybrid"], cps["fill"]
    if not (c["study_uids"] == h["study_uids"] == f["study_uids"]):
        raise RuntimeError("B25X study order differs across arms")
    if not (c["encoder_sha256_initial"] == h["encoder_sha256_initial"] == f["encoder_sha256_initial"]):
        raise RuntimeError("B25X encoder differs across arms")
    if not (float(c["crop_fraction"]) == float(h["crop_fraction"]) == float(f["crop_fraction"])):
        raise RuntimeError("B25X crop differs across arms")
    fill_meta = f.get("fill_ablation") or {}
    if int(fill_meta.get("b6_cells_dropped", -1)) != 0 or int(fill_meta.get("b6_cells_overridden", -1)) != 0:
        raise RuntimeError("B25X fill arm did not preserve B6 exactly")

    root = Path(cfg["data_root"])
    train = load_train_csv(root / cfg.get("train_csv", "train.csv"))
    b6_frame, _, _ = load_frozen_b6_export(args.b6_root)
    weak_payload, manifest = load_frozen_v2_manifest(args.weak_holdout_root)
    uids, weak_y, weak_w, _ = _holdout_supervision(train, b6_frame, manifest)
    if len(uids) != 623:
        raise RuntimeError(f"weak-v2 size changed: {len(uids)}")
    overlap = set(map(str, c["study_uids"])) & set(map(str, uids))
    if overlap:
        raise RuntimeError(f"weak-v2 leakage: {len(overlap)} studies")

    series = load_series_csv(root / cfg.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    variable_index = build_variable_series_index(series, uids)
    if any(len(variable_index[str(uid)]) == 0 for uid in uids):
        raise RuntimeError("weak-v2 study with zero eligible series")

    offsets = tuple(int(x) for x in cfg.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise RuntimeError("B25X TTA must remain [-1,0,1]")
    ds_cfg = make_b7_dataset_config(cfg, root, train=False, tta_offsets=offsets)
    runtime = resolve_runtime(cfg)
    batch_size = int(cfg.get("b7_eval_batch_size", 2))
    seed = int(cfg.get("seed", 2026))
    expected_uids = [str(x) for x in uids]

    print("=" * 72)
    print("B25X CHECKPOINT / HOLDOUT VERIFICATION")
    print("=" * 72)
    print("training studies    :", len(c["study_uids"]))
    print("weak-v2 studies     :", len(uids))
    print("weak-v2 overlap     :", len(overlap))
    print("weak-v2 cells       :", int((weak_w > 0).sum()))
    print("same encoder/crop   : True")
    print("fixed endpoint      : E2")
    print("fill B6 drops       :", fill_meta.get("b6_cells_dropped"))
    print("fill B6 overrides   :", fill_meta.get("b6_cells_overridden"))
    print("gold                : prohibited")
    print()
    print(runtime.describe())

    preds = {}
    for i, name in enumerate(("control", "hybrid", "fill")):
        print(f"\n[B25X] predicting {name} on frozen weak-v2...")
        puids, pred = predict_one(
            cps[name], uids, variable_index, ds_cfg, runtime,
            batch_size, seed + 25_100_000 + i,
        )
        if puids != expected_uids:
            raise RuntimeError(f"{name} prediction order changed")
        preds[name] = pred

    nboot = int(args.n_bootstrap)
    evals = {
        name: evaluate_on_weak_surface(
            weak_y, preds[name], weak_w,
            n_bootstrap=nboot, seed=seed + 25_200_000 + i,
        )
        for i, name in enumerate(("control", "hybrid", "fill"))
    }
    comps = {
        "hybrid_minus_control": compare_on_weak_surface(
            weak_y, preds["control"], preds["hybrid"], weak_w,
            n_bootstrap=nboot, seed=seed + 25_300_001,
        ),
        "fill_minus_control": compare_on_weak_surface(
            weak_y, preds["control"], preds["fill"], weak_w,
            n_bootstrap=nboot, seed=seed + 25_300_002,
        ),
        "hybrid_minus_fill": compare_on_weak_surface(
            weak_y, preds["fill"], preds["hybrid"], weak_w,
            n_bootstrap=nboot, seed=seed + 25_300_003,
        ),
    }

    per_target = {}
    for target in TARGETS:
        cv = float(evals["control"]["per_target_auc"][target])
        hv = float(evals["hybrid"]["per_target_auc"][target])
        fv = float(evals["fill"]["per_target_auc"][target])
        per_target[target] = {
            "control": cv, "hybrid": hv, "fill": fv,
            "hybrid_minus_control": hv - cv,
            "fill_minus_control": fv - cv,
            "hybrid_minus_fill": hv - fv,
        }

    result = {
        "experiment": EXPERIMENT,
        "surface": "frozen_weak_b6_holdout_v2",
        "gold_used": False,
        "promotion_allowed": False,
        "n_training_studies": len(c["study_uids"]),
        "n_holdout_studies": len(uids),
        "holdout_usable_cells": int((weak_w > 0).sum()),
        "weak_holdout_manifest_sha256": WEAK_V2_MANIFEST_SHA256,
        "weak_holdout_metadata": weak_payload,
        "metadata_repair": metadata_stats,
        "control_macro_auc": float(evals["control"]["macro_auc"]),
        "hybrid_macro_auc": float(evals["hybrid"]["macro_auc"]),
        "fill_macro_auc": float(evals["fill"]["macro_auc"]),
        "evaluations": evals,
        "comparisons": comps,
        "per_target": per_target,
        "interpretation": (
            "Exploratory B25X evaluation on frozen B6 weak-v2 teacher agreement; "
            "hybrid source provenance is mixed/unknown; no gold evaluation or promotion."
        ),
    }

    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"StudyInstanceUID": expected_uids})
    for j, target in enumerate(TARGETS):
        for name in ("control", "hybrid", "fill"):
            frame[f"{target}__{name}"] = preds[name][:, j]
    frame.to_csv(out / "three_arm_predictions.csv", index=False)
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("B25X FROZEN WEAK-V2 RESULT")
    print("=" * 72)
    print(f"B6 control : {result['control_macro_auc']:.10f}")
    print(f"Hybrid     : {result['hybrid_macro_auc']:.10f}")
    print(f"B6+Fill    : {result['fill_macro_auc']:.10f}")

    for key, label in (
        ("hybrid_minus_control", "Hybrid - B6"),
        ("fill_minus_control", "Fill - B6"),
        ("hybrid_minus_fill", "Hybrid - Fill"),
    ):
        x = comps[key]
        print(f"\n{label}")
        print(f"  raw    : {x['raw_difference_b_minus_a']:+.10f}")
        print(f"  median : {x['median_difference']:+.10f}")
        print(f"  95% CI : [{x['ci_lower']:+.10f}, {x['ci_upper']:+.10f}]")
        print(f"  P(>0)  : {x['probability_b_better']:.4f}")

    print("\nPer-target")
    print("-" * 78)
    print(f"{'Target':18s} {'B6':>8s} {'Hybrid':>8s} {'Fill':>8s} {'H-B6':>8s} {'F-B6':>8s}")
    for target, row in per_target.items():
        print(
            f"{target:18s} {row['control']:8.4f} {row['hybrid']:8.4f} "
            f"{row['fill']:8.4f} {row['hybrid_minus_control']:+8.4f} "
            f"{row['fill_minus_control']:+8.4f}"
        )

    print("\nNO GOLD. NO PROMOTION.")
    print("saved:", out / "three_arm_predictions.csv")
    print("saved:", out / "comparison.json")


if __name__ == "__main__":
    main()
