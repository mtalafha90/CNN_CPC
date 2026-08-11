"""Development-only frozen gold evaluation for B13 ImageNet encoder protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import (
    VariableSeriesKneeDataset,
    build_variable_series_index,
    collate_variable_series,
)
from .b12_1_gold_eval import predict_b12_1
from .b13_training import (
    B13_EXPERIMENT,
    B13_INITIALIZATION,
    B13_INPUT_NORMALIZATION,
    B13_VARIANT,
    load_b13_checkpoint,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import bootstrap_macro_auc
from .runtime import resolve_runtime


def evaluate_b13_gold(
    config: dict,
    *,
    checkpoint: str | Path,
    out_root: str | Path = "runs/b13_imagenet/gold_eval",
) -> dict:
    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, checkpoint_payload = load_b13_checkpoint(checkpoint, device=runtime.device)

    if int(checkpoint_payload.get("completed_epochs", -1)) != 4:
        raise ValueError("B13 gold evaluation requires the completed four-epoch checkpoint")
    history = checkpoint_payload.get("history", [])
    if len(history) != 4 or not all(
        bool(row.get("full_coverage"))
        and bool(row.get("full_series_coverage"))
        and not bool(row.get("budget_limited"))
        for row in history
    ):
        raise ValueError("B13 gold evaluation requires four complete unbudgeted study/series epochs")
    if not all(int(row.get("series_instances_seen", -1)) == 17475 for row in history):
        raise ValueError("B13 gold evaluation requires 17,475 loaded series in every epoch")
    if checkpoint_payload.get("initialization") != B13_INITIALIZATION:
        raise ValueError("B13 checkpoint initialization metadata mismatch")
    if checkpoint_payload.get("input_normalization") != B13_INPUT_NORMALIZATION:
        raise ValueError("B13 checkpoint normalization metadata mismatch")
    if checkpoint_payload.get("external_pretrained") is not True:
        raise ValueError("B13 checkpoint does not certify ImageNet pretraining")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != 58 or gold[TARGETS].isna().any().any():
        raise ValueError("B13 requires the complete 58-study gold development surface")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    uids = gold["StudyInstanceUID"].tolist()
    variable_index = build_variable_series_index(series, uids)
    counts = [len(variable_index[uid]) for uid in uids]
    if any(count == 0 for count in counts):
        raise ValueError("B13 gold surface contains a study with zero eligible series")

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B13 gold TTA is frozen at [-1,0,1]")
    ds = VariableSeriesKneeDataset(
        uids,
        variable_index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        targets=gold[TARGETS].to_numpy(np.float32),
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=max(1, int(config.get("b7_eval_batch_size", 2))),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 12_500_000),
    )
    pred_uids, prediction = predict_b12_1(model, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("B13 gold prediction order changed unexpectedly")

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

    payload = result.to_dict()
    payload.update(
        {
            "experiment": "B13_gold_development_evaluation",
            "checkpoint": str(Path(checkpoint).resolve()),
            "b13_variant": B13_VARIANT,
            "training_experiment": B13_EXPERIMENT,
            "completed_epochs": checkpoint_payload.get("completed_epochs"),
            "n_gold_studies": int(len(gold)),
            "tta_center_offsets": list(offsets),
            "series_policy": "same frozen B12/B12.1 all-series surface; dynamic batch padding",
            "gold_series_count": {
                "total": int(sum(counts)),
                "min": int(min(counts)),
                "median": float(np.median(counts)),
                "max": int(max(counts)),
            },
            "aggregation": "same B12.1 hierarchical learned series-token aggregation",
            "initialization": B13_INITIALIZATION,
            "input_normalization": B13_INPUT_NORMALIZATION,
            "external_pretrained": True,
            "preprocessing": (
                "same MRI sampling and legacy 224x224 resize as B12.1; ImageNet mean/std "
                "normalization inside the pretrained encoder"
            ),
            "gold_labels_used_in_b13_gradient": False,
            "gold_labels_used_for_b13_early_stopping": False,
            "single_scientific_change_vs_b12_1": (
                "encoder initialization protocol changed from B5 competition-only SSL to "
                "torchvision ImageNet-1K V1 weights with standard ImageNet normalization"
            ),
            "metadata_repair": metadata_stats,
            "interpretation": (
                "development estimate on the repeatedly reused 58-study surface; no gold labels entered "
                "B13 optimization or checkpoint selection"
            ),
        }
    )
    (out / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(result.summary())
    print(out / "gold_predictions.csv")
    print(out / "eval.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b13-eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b13_imagenet/gold_eval")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    payload = evaluate_b13_gold(
        config,
        checkpoint=args.checkpoint,
        out_root=args.out_root,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
