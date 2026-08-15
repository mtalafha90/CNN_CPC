"""Development-only gold evaluation for a frozen B8 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .b7_weak_supervision import make_b7_dataset_config
from .b8_anatomy_spatial import load_b8_checkpoint
from .constants import TARGETS
from .data import backfill_series_metadata, build_series_index, gold_mask, load_series_csv, load_train_csv
from .dataset import KneeStudyDataset
from .evaluation import bootstrap_macro_auc
from .runtime import autocast, resolve_runtime


def _read_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


@torch.no_grad()
def predict_b8(model, loader, runtime) -> tuple[list[str], np.ndarray]:
    model.eval()
    uids: list[str] = []
    probabilities: list[np.ndarray] = []
    for batch in loader:
        present = batch["present"].to(runtime.device, non_blocking=True)
        volumes = batch["volumes"]
        if volumes.ndim == 6:
            with autocast(runtime):
                logits = model(volumes.to(runtime.device, non_blocking=True), present)
            probs = torch.sigmoid(logits.float())
        elif volumes.ndim == 7:
            view_probs = []
            for view in range(volumes.shape[1]):
                with autocast(runtime):
                    logits = model(volumes[:, view].to(runtime.device, non_blocking=True), present)
                view_probs.append(torch.sigmoid(logits.float()))
            probs = torch.stack(view_probs, dim=0).mean(dim=0)
        else:
            raise ValueError(f"unexpected B8 evaluation volume shape: {tuple(volumes.shape)}")
        probabilities.append(probs.cpu().numpy())
        uids.extend([str(uid) for uid in batch["study_uid"]])
    if not probabilities:
        raise RuntimeError("B8 gold evaluation produced no predictions")
    return uids, np.concatenate(probabilities, axis=0)


def evaluate_b8_gold(
    config: dict,
    *,
    checkpoint: str | Path,
    out_root: str | Path = "runs/b8_spatial_anatomy/gold_eval",
) -> dict:
    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, checkpoint_payload = load_b8_checkpoint(checkpoint, device=runtime.device)

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if gold.empty:
        raise ValueError("no gold studies found")
    if gold[TARGETS].isna().any().any():
        raise ValueError("B8 gold evaluation requires all 12 labels on every gold study")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    uids = gold["StudyInstanceUID"].tolist()
    index = build_series_index(series, uids, mode="dual")

    offsets = tuple(int(x) for x in config.get("b8_eval_tta_offsets", [-1, 0, 1]))
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
        batch_size=max(1, int(config.get("b8_eval_batch_size", 2))),
        shuffle=False,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 8_500_000),
    )
    pred_uids, prediction = predict_b8(model, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("B8 gold prediction order changed unexpectedly")

    truth = gold[TARGETS].to_numpy(np.float64)
    result = bootstrap_macro_auc(
        truth,
        prediction,
        n_bootstrap=int(config.get("b8_n_bootstrap", 5000)),
        seed=int(config.get("seed", 2026)) + 88,
    )

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(prediction, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    frame.to_csv(out / "gold_predictions.csv", index=False)

    payload = result.to_dict()
    payload.update(
        {
            "experiment": "B8_gold_development_evaluation",
            "checkpoint": str(Path(checkpoint).resolve()),
            "b8_variant": checkpoint_payload.get("variant"),
            "completed_epochs": checkpoint_payload.get("completed_epochs"),
            "n_gold_studies": int(len(gold)),
            "tta_center_offsets": list(offsets),
            "gold_labels_used_in_b8_gradient": False,
            "gold_labels_used_for_b8_early_stopping": False,
            "b6_gold_audit_informed_global_policy": bool(
                checkpoint_payload.get("b6_gold_audit_informed_global_policy", False)
            ),
            "initialization_experiment": checkpoint_payload.get("initialization_experiment"),
            "interpretation": (
                "development estimate; B8 architecture was chosen after prior development results, "
                "while gold labels did not enter B8 optimization or early stopping"
            ),
            "metadata_repair": metadata_stats,
        }
    )
    (out / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(result.summary())
    print(out / "gold_predictions.csv")
    print(out / "eval.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b8-eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None, help="override data_root from YAML")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b8_spatial_anatomy/gold_eval")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    evaluate_b8_gold(config, checkpoint=args.checkpoint, out_root=args.out_root)


if __name__ == "__main__":
    main()
