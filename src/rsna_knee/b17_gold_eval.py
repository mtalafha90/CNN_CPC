"""Single reused-gold development look for B17 frozen-encoder training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import (
    VariableSeriesKneeDataset,
    build_variable_series_index,
    collate_variable_series,
)
from .b12_1_gold_eval import predict_b12_1
from .b17_training import (
    B17_EXPERIMENT,
    B17_VARIANT,
    load_b17_checkpoint,
    require_b17_contract,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import bootstrap_macro_auc, compare_runs
from .runtime import resolve_runtime

B16_REFERENCE_MACRO = 0.6349770242415445


def _read_reference_predictions(path: str | Path, ordered_uids: list[str]) -> np.ndarray:
    frame = pd.read_csv(path)
    required = {"StudyInstanceUID", *TARGETS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"B16 reference predictions missing columns: {missing}")
    frame = frame[["StudyInstanceUID", *TARGETS]].copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("B16 reference predictions contain duplicate UIDs")
    requested = [str(uid) for uid in ordered_uids]
    if set(frame["StudyInstanceUID"]) != set(requested):
        raise ValueError("B16 reference predictions do not match the exact 58-study gold surface")
    order = {uid: i for i, uid in enumerate(requested)}
    frame = frame.sort_values("StudyInstanceUID", key=lambda series: series.map(order))
    values = frame[TARGETS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("B16 reference predictions contain non-finite values")
    return values


def evaluate_b17_gold(
    config: dict,
    *,
    checkpoint: str | Path,
    b16_predictions: str | Path,
    out_root: str | Path = "runs/b17_frozen_encoder/gold_confirmation",
) -> dict:
    require_b17_contract(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, checkpoint_payload = load_b17_checkpoint(checkpoint, device=runtime.device)
    if checkpoint_payload.get("variant") != B17_VARIANT:
        raise ValueError("B17 gold checkpoint variant mismatch")
    if checkpoint_payload.get("experiment") != B17_EXPERIMENT:
        raise ValueError("B17 gold checkpoint experiment mismatch")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != 58 or gold[TARGETS].isna().any().any():
        raise ValueError("B17 gold evaluation requires the complete 58-study gold surface")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    uids = gold["StudyInstanceUID"].tolist()
    variable_index = build_variable_series_index(series, uids)
    counts = [len(variable_index[uid]) for uid in uids]
    if any(count == 0 for count in counts):
        raise ValueError("B17 gold surface contains a study with zero eligible series")

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B17 gold evaluation freezes TTA [-1,0,1]")
    ds = VariableSeriesKneeDataset(
        uids,
        variable_index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        targets=gold[TARGETS].to_numpy(np.float32),
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=int(config.get("b7_eval_batch_size", 2)),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 21_100_000),
    )
    pred_uids, prediction = predict_b12_1(model, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("B17 gold prediction order changed")

    truth = gold[TARGETS].to_numpy(np.float64)
    n_bootstrap = int(config.get("b7_n_bootstrap", 5000))
    result = bootstrap_macro_auc(
        truth,
        prediction,
        n_bootstrap=n_bootstrap,
        seed=int(config.get("seed", 2026)) + 211,
    )
    b16 = _read_reference_predictions(b16_predictions, uids)
    b16_result = bootstrap_macro_auc(
        truth,
        b16,
        n_bootstrap=n_bootstrap,
        seed=int(config.get("seed", 2026)) + 212,
    )
    b16_point = float(b16_result.macro_auc)
    if not np.isclose(b16_point, B16_REFERENCE_MACRO, atol=1e-10, rtol=0):
        raise ValueError(
            f"B17 expected the frozen B16 champion reference {B16_REFERENCE_MACRO:.10f}; "
            f"got {b16_point:.10f}"
        )
    paired = compare_runs(
        truth,
        b16,
        prediction,
        n_bootstrap=n_bootstrap,
        seed=int(config.get("seed", 2026)) + 213,
    )

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(prediction, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    frame.to_csv(out / "gold_predictions.csv", index=False)

    payload = result.to_dict()
    payload.update(
        {
            "experiment": "B17_one_look_reused_gold_development_confirmation",
            "variant": B17_VARIANT,
            "training_experiment": B17_EXPERIMENT,
            "checkpoint": str(Path(checkpoint).resolve()),
            "completed_epochs": checkpoint_payload.get("completed_epochs"),
            "n_gold_studies": int(len(gold)),
            "tta_center_offsets": list(offsets),
            "gold_series_count": {
                "total": int(sum(counts)),
                "min": int(min(counts)),
                "median": float(np.median(counts)),
                "max": int(max(counts)),
            },
            "initialization": checkpoint_payload.get("initialization"),
            "initialization_detail": checkpoint_payload.get("initialization_detail"),
            "input_normalization": checkpoint_payload.get("input_normalization"),
            "encoder_frozen": True,
            "encoder_sha256_initial": checkpoint_payload.get("encoder_sha256_initial"),
            "encoder_sha256_final": checkpoint_payload.get("encoder_sha256_final"),
            "gold_labels_used_in_gradient": False,
            "gold_labels_used_for_early_stopping": False,
            "gold_checkpoint_selection": False,
            "weak_v2_used_as_gate": False,
            "additional_label_smoothing": 0.0,
            "robust_loss": "none",
            "historical_b16_macro_auc_reproduced": b16_point,
            "raw_difference_B17_minus_B16": float(result.macro_auc - b16_point),
            "paired_B17_minus_B16": paired,
            "primary_decision_rule": (
                "global 12-target macro AUC only; B17 replaces B16 as the reused-gold development "
                "champion only if its global point estimate is higher; paired bootstrap quantifies "
                "uncertainty; no target-wise mixing or post-gold B17 tuning"
            ),
            "metadata_repair": metadata_stats,
            "interpretation": (
                "single reused-gold development look after a frozen no-gold B17 protocol; not "
                "independent validation and not eligible for B17 retuning"
            ),
        }
    )
    (out / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(result.summary())
    print(
        "B17-B16 raw delta",
        payload["raw_difference_B17_minus_B16"],
        "| paired median",
        paired["median_difference"],
        "| P(B17>B16)",
        paired["probability_b_better"],
    )
    print(out / "gold_predictions.csv")
    print(out / "eval.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b17-gold-eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--b16-predictions", required=True)
    parser.add_argument("--out-root", default="runs/b17_frozen_encoder/gold_confirmation")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    payload = evaluate_b17_gold(
        config,
        checkpoint=args.checkpoint,
        b16_predictions=args.b16_predictions,
        out_root=args.out_root,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
