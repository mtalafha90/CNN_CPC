"""Development-only gold evaluation for frozen B10 physical-scale normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .b7_gold_eval import predict_b7
from .b7_weak_supervision import make_b7_dataset_config
from .b10_physical_scale import B10_EXPERIMENT, B10_VARIANT, load_b10_checkpoint
from .constants import TARGETS
from .data import (
    backfill_series_metadata,
    build_series_index,
    gold_mask,
    load_series_csv,
    load_train_csv,
)
from .dataset import KneeStudyDataset
from .evaluation import bootstrap_macro_auc
from .physical_scale import B10_PHYSICAL_POLICY, physical_policy_digest
from .runtime import resolve_runtime


def _read_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def _dataset_config(config: dict, root: Path, *, policy: dict, offsets):
    ds_config = make_b7_dataset_config(
        config,
        root,
        train=False,
        tta_offsets=tuple(int(x) for x in offsets),
    )
    ds_config.physical_scale_policy = policy
    ds_config.__post_init__()
    return ds_config


@torch.no_grad()
def evaluate_b10_gold(
    config: dict,
    *,
    checkpoint: str | Path,
    out_root: str | Path = "runs/b10_physical_scale/gold_eval",
) -> dict:
    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, checkpoint_payload = load_b10_checkpoint(
        checkpoint, device=runtime.device
    )
    if int(checkpoint_payload.get("completed_epochs", -1)) != 4:
        raise ValueError("B10-v1 gold evaluation requires the completed four-epoch checkpoint")

    physical_policy = checkpoint_payload["physical_scale_policy"]
    if checkpoint_payload.get("physical_policy_sha256") != physical_policy_digest(
        physical_policy
    ):
        raise ValueError("B10 checkpoint physical policy changed before evaluation")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if gold.empty:
        raise ValueError("no gold studies found")
    if gold[TARGETS].isna().any().any():
        raise ValueError("B10 gold evaluation requires all 12 labels on every gold study")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    uids = gold["StudyInstanceUID"].tolist()
    index = build_series_index(series, uids, mode="dual")

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if not offsets:
        offsets = (0,)
    ds = KneeStudyDataset(
        uids,
        index,
        _dataset_config(config, root, policy=physical_policy, offsets=offsets),
        targets=gold[TARGETS].to_numpy(np.float32),
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=max(1, int(config.get("b7_eval_batch_size", 2))),
        shuffle=False,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 7_500_000),
    )
    pred_uids, prediction = predict_b7(model, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("B10 gold prediction order changed unexpectedly")

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
            "experiment": "B10_gold_development_evaluation",
            "checkpoint": str(Path(checkpoint).resolve()),
            "b10_variant": B10_VARIANT,
            "training_experiment": B10_EXPERIMENT,
            "physical_policy_name": B10_PHYSICAL_POLICY,
            "physical_policy_sha256": checkpoint_payload.get(
                "physical_policy_sha256"
            ),
            "physical_scale_policy": physical_policy,
            "completed_epochs": checkpoint_payload.get("completed_epochs"),
            "n_gold_studies": int(len(gold)),
            "tta_center_offsets": list(offsets),
            "routing_mode": "historical B7.1 dual routing",
            "gold_labels_used_in_b10_gradient": False,
            "gold_labels_used_for_b10_early_stopping": False,
            "gold_labels_used_to_choose_physical_policy": False,
            "b6_gold_audit_informed_global_policy": bool(
                checkpoint_payload.get(
                    "b6_gold_audit_informed_global_policy", False
                )
            ),
            "initialization": "B5 competition-only image-report encoder",
            "single_scientific_change_vs_b7_1": (
                "plane-specific in-plane PixelSpacing/FOV normalization before "
                "the unchanged 224x224 resize"
            ),
            "metadata_repair": metadata_stats,
            "interpretation": (
                "development estimate; the B10 physical-scale policy was derived "
                "without gold labels, while the broader 58-study set has been reused "
                "for sequential model development"
            ),
        }
    )
    (out / "eval.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(result.summary())
    print(out / "gold_predictions.csv")
    print(out / "eval.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b10-eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None, help="override data_root from YAML")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b10_physical_scale/gold_eval")
    args = parser.parse_args()

    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    payload = evaluate_b10_gold(
        config,
        checkpoint=args.checkpoint,
        out_root=args.out_root,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
