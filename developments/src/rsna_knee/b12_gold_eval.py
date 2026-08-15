"""Development-only frozen gold evaluation for B12 variable-series model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_training import B12_EXPERIMENT, B12_VARIANT, load_b12_checkpoint
from .b12_variable_series import (
    VariableSeriesKneeDataset,
    build_variable_series_index,
    collate_variable_series,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import bootstrap_macro_auc
from .runtime import autocast, resolve_runtime


@torch.no_grad()
def predict_b12(model, loader, runtime) -> tuple[list[str], np.ndarray]:
    model.eval()
    uids: list[str] = []
    probabilities: list[np.ndarray] = []
    for batch in loader:
        present = batch["present"].to(runtime.device, non_blocking=True)
        series_meta = batch["series_meta"].to(runtime.device, non_blocking=True)
        volumes = batch["volumes"]
        if volumes.ndim == 6:
            with autocast(runtime):
                logits = model(
                    volumes.to(runtime.device, non_blocking=True), present, series_meta
                )
            probs = torch.sigmoid(logits.float())
        elif volumes.ndim == 7:
            view_probs = []
            for view in range(volumes.shape[1]):
                with autocast(runtime):
                    logits = model(
                        volumes[:, view].to(runtime.device, non_blocking=True),
                        present,
                        series_meta,
                    )
                view_probs.append(torch.sigmoid(logits.float()))
            probs = torch.stack(view_probs, dim=0).mean(dim=0)
        else:
            raise ValueError(f"unexpected B12 evaluation volume shape: {tuple(volumes.shape)}")
        probabilities.append(probs.cpu().numpy())
        uids.extend([str(uid) for uid in batch["study_uid"]])
    if not probabilities:
        raise RuntimeError("B12 gold evaluation produced no predictions")
    return uids, np.concatenate(probabilities, axis=0)


def evaluate_b12_gold(
    config: dict,
    *,
    checkpoint: str | Path,
    out_root: str | Path = "runs/b12_variable_series/gold_eval",
) -> dict:
    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, checkpoint_payload = load_b12_checkpoint(checkpoint, device=runtime.device)
    if int(checkpoint_payload.get("completed_epochs", -1)) != 4:
        raise ValueError("B12-v1 gold evaluation requires the completed four-epoch checkpoint")
    history = checkpoint_payload.get("history", [])
    if len(history) != 4 or not all(
        bool(row.get("full_coverage")) and bool(row.get("full_series_coverage")) for row in history
    ):
        raise ValueError("B12-v1 gold evaluation requires four complete study/series-coverage epochs")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != 58 or gold[TARGETS].isna().any().any():
        raise ValueError("B12-v1 requires the complete 58-study gold development surface")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    uids = gold["StudyInstanceUID"].tolist()
    variable_index = build_variable_series_index(series, uids)
    counts = [len(variable_index[uid]) for uid in uids]
    if any(count == 0 for count in counts):
        raise ValueError("B12 gold surface contains a study with zero eligible series")

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B12-v1 gold TTA is frozen at [-1,0,1]")
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
    pred_uids, prediction = predict_b12(model, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("B12 gold prediction order changed unexpectedly")

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
            "experiment": "B12_gold_development_evaluation",
            "checkpoint": str(Path(checkpoint).resolve()),
            "b12_variant": B12_VARIANT,
            "training_experiment": B12_EXPERIMENT,
            "completed_epochs": checkpoint_payload.get("completed_epochs"),
            "n_gold_studies": int(len(gold)),
            "tta_center_offsets": list(offsets),
            "series_policy": "all repaired sagittal/coronal/axial series; dynamic batch padding",
            "gold_series_count": {
                "total": int(sum(counts)),
                "min": int(min(counts)),
                "median": float(np.median(counts)),
                "max": int(max(counts)),
            },
            "preprocessing": "historical B7.1 legacy resize; no B10 physical normalization",
            "gold_labels_used_in_b12_gradient": False,
            "gold_labels_used_for_b12_early_stopping": False,
            "initialization": "B5 competition-only image-report encoder (same as B7.1)",
            "single_scientific_change_vs_b7_1": (
                "variable number of real MRI series with metadata embeddings instead of six selected semantic slots"
            ),
            "metadata_repair": metadata_stats,
            "interpretation": (
                "development estimate on the repeatedly reused 58-study surface; no gold labels entered "
                "B12 optimization or model-selection within B12-v1"
            ),
        }
    )
    (out / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(result.summary())
    print(out / "gold_predictions.csv")
    print(out / "eval.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b12-eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b12_variable_series/gold_eval")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    payload = evaluate_b12_gold(
        config, checkpoint=args.checkpoint, out_root=args.out_root
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
