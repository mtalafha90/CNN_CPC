"""Development-only frozen gold evaluation for B11.1 quantile teacher student."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .b11_1_teacher_student import B11_1_EXPERIMENT, B11_1_VARIANT, load_b11_1_checkpoint
from .b7_gold_eval import predict_b7
from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .constants import TARGETS
from .data import backfill_series_metadata, build_series_index, gold_mask, load_series_csv, load_train_csv
from .dataset import KneeStudyDataset
from .evaluation import bootstrap_macro_auc
from .runtime import resolve_runtime


def evaluate_b11_1_gold(
    config: dict,
    *,
    checkpoint: str | Path,
    out_root: str | Path = "runs/b11_1_quantile_teacher/gold_eval",
) -> dict:
    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, checkpoint_payload = load_b11_1_checkpoint(checkpoint, device=runtime.device)
    if int(checkpoint_payload.get("completed_epochs", -1)) != 4:
        raise ValueError("B11.1 gold evaluation requires the completed four-epoch checkpoint")
    history = checkpoint_payload.get("history", [])
    if len(history) != 4 or not all(bool(row.get("full_coverage")) for row in history):
        raise ValueError("B11.1 gold evaluation requires four complete full-coverage epochs")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != 58 or gold[TARGETS].isna().any().any():
        raise ValueError("B11.1 requires the complete 58-study gold development surface")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    uids = gold["StudyInstanceUID"].tolist()
    index = build_series_index(series, uids, mode="dual")
    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B11.1 gold TTA is frozen at [-1,0,1]")

    ds = KneeStudyDataset(
        uids,
        index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        targets=gold[TARGETS].to_numpy(np.float32),
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=max(1, int(config.get("b7_eval_batch_size", 2))),
        shuffle=False,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 11_510_000),
    )
    pred_uids, prediction = predict_b7(model, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("B11.1 gold prediction order changed unexpectedly")

    truth = gold[TARGETS].to_numpy(np.float64)
    result = bootstrap_macro_auc(
        truth,
        prediction,
        n_bootstrap=int(config.get("b7_n_bootstrap", 5000)),
        seed=int(config.get("seed", 2026)) + 77,
    )
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(prediction, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    frame.to_csv(out / "gold_predictions.csv", index=False)

    pseudo_policy = checkpoint_payload.get("pseudo_policy", {})
    payload = result.to_dict()
    payload.update({
        "experiment": "B11.1_gold_development_evaluation",
        "checkpoint": str(Path(checkpoint).resolve()),
        "b11_1_variant": B11_1_VARIANT,
        "training_experiment": B11_1_EXPERIMENT,
        "completed_epochs": checkpoint_payload.get("completed_epochs"),
        "n_gold_studies": int(len(gold)),
        "tta_center_offsets": list(offsets),
        "routing_mode": "historical B7.1 dual routing",
        "preprocessing": "historical B7.1 legacy resize; no B10 physical normalization",
        "gold_labels_used_in_b11_1_gradient": False,
        "gold_labels_used_for_b11_1_early_stopping": False,
        "gold_labels_used_to_choose_pseudo_cells": False,
        "teacher_checkpoint": pseudo_policy.get("teacher_checkpoint"),
        "teacher_checkpoint_sha256": pseudo_policy.get("teacher_checkpoint_sha256"),
        "pseudo_labels_sha256": pseudo_policy.get("pseudo_labels_sha256"),
        "pseudo_policy_name": pseudo_policy.get("policy"),
        "pseudo_summary": pseudo_policy.get("pseudo_summary"),
        "initialization": "B5 competition-only image-report encoder (same as B7.1)",
        "single_scientific_change_vs_b7_1": "frozen calibration-aware B7.1 teacher tail supervision on B6-unsupervised cells",
        "metadata_repair": metadata_stats,
        "interpretation": (
            "development estimate; pseudo selection used no gold labels, but the B7.1 teacher "
            "was retained after prior development on the same 58-study surface"
        ),
    })
    (out / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(result.summary())
    print(out / "gold_predictions.csv")
    print(out / "eval.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b11-1-eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b11_1_quantile_teacher/gold_eval")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    payload = evaluate_b11_1_gold(config, checkpoint=args.checkpoint, out_root=args.out_root)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
