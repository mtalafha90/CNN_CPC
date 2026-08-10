"""Development-only gold evaluation for frozen B9 strict routing."""

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
from .b9_strict_routing import B9_EXPERIMENT, B9_VARIANT, load_b9_checkpoint
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .dataset import KneeStudyDataset
from .evaluation import bootstrap_macro_auc
from .runtime import resolve_runtime
from .strict_routing import STRICT_ROUTING_POLICY, build_strict_series_index, routing_audit


def _read_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


@torch.no_grad()
def evaluate_b9_gold(
    config: dict,
    *,
    checkpoint: str | Path,
    out_root: str | Path = "runs/b9_strict_routing/gold_eval",
) -> dict:
    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, checkpoint_payload = load_b9_checkpoint(checkpoint, device=runtime.device)
    if int(checkpoint_payload.get("completed_epochs", -1)) != 4:
        raise ValueError("B9-v1 gold evaluation requires the completed four-epoch checkpoint")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if gold.empty:
        raise ValueError("no gold studies found")
    if gold[TARGETS].isna().any().any():
        raise ValueError("B9 gold evaluation requires all 12 labels on every gold study")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    uids = gold["StudyInstanceUID"].tolist()
    route_audit = routing_audit(series, uids)
    index = build_strict_series_index(series, uids)

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if not offsets:
        offsets = (0,)
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
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 7_500_000),
    )
    pred_uids, prediction = predict_b7(model, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("B9 gold prediction order changed unexpectedly")

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
            "experiment": "B9_gold_development_evaluation",
            "checkpoint": str(Path(checkpoint).resolve()),
            "b9_variant": B9_VARIANT,
            "training_experiment": B9_EXPERIMENT,
            "routing_policy": STRICT_ROUTING_POLICY,
            "completed_epochs": checkpoint_payload.get("completed_epochs"),
            "n_gold_studies": int(len(gold)),
            "tta_center_offsets": list(offsets),
            "gold_labels_used_in_b9_gradient": False,
            "gold_labels_used_for_b9_early_stopping": False,
            "b6_gold_audit_informed_global_policy": bool(
                checkpoint_payload.get("b6_gold_audit_informed_global_policy", False)
            ),
            "initialization": "B5 competition-only image-report encoder",
            "single_scientific_change_vs_b7_1": (
                "strict Fluid_Sensitive routing with no cross-contrast substitution"
            ),
            "routing_audit": route_audit,
            "metadata_repair": metadata_stats,
            "interpretation": (
                "development estimate; B9 routing was chosen after prior development work, "
                "while gold labels did not enter B9 optimization or early stopping"
            ),
        }
    )
    (out / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(result.summary())
    print(out / "gold_predictions.csv")
    print(out / "eval.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b9-eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None, help="override data_root from YAML")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b9_strict_routing/gold_eval")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    payload = evaluate_b9_gold(config, checkpoint=args.checkpoint, out_root=args.out_root)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
