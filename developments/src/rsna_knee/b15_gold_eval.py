"""One-look reused-gold development confirmation for B15.

This evaluator is intentionally gated by the frozen weak-holdout-v2 comparison.
It may be used only after B15 passes the predeclared weak-v2 gate.  The 58-study
surface is repeatedly reused development data, not independent validation.
"""
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
from .b15_downstream import (
    B15_EXPERIMENT,
    B15_VARIANT,
    load_v2_downstream_checkpoint,
    require_v2_downstream_contract,
)
from .b15_ssl import WEAK_V2_MANIFEST_SHA256, WEAK_V2_SURFACE
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import bootstrap_macro_auc
from .runtime import resolve_runtime


def load_passing_b15_gate(path: str | Path) -> dict:
    gate_path = Path(path)
    if not gate_path.is_file():
        raise FileNotFoundError(gate_path)
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    if payload.get("surface") != WEAK_V2_SURFACE:
        raise ValueError("B15 gold confirmation requires the frozen weak-v2 gate surface")
    if payload.get("weak_holdout_manifest_sha256") != WEAK_V2_MANIFEST_SHA256:
        raise ValueError("B15 gold confirmation gate manifest SHA mismatch")
    if payload.get("comparison") != "B15 minus B13-v2-control":
        raise ValueError("B15 gold confirmation requires the matched B15-vs-control gate")
    if payload.get("model_a") != "B13-v2-control" or payload.get("model_b") != "B15":
        raise ValueError("B15 gold confirmation gate model identities changed")
    if payload.get("strict_all_12_targets") is not True:
        raise ValueError("B15 gold confirmation requires strict all-12-target bootstrap")
    if int(payload.get("n_bootstrap", -1)) != 5000:
        raise ValueError("B15 gold confirmation requires the frozen 5000-replicate gate")
    rule = payload.get("predeclared_gate", {})
    if not (
        rule.get("raw_difference_positive") is True
        and rule.get("median_difference_positive") is True
        and rule.get("probability_b_better_at_least_0_95") is True
        and payload.get("passes_gate") is True
    ):
        raise ValueError("B15 did not pass the predeclared weak-v2 gate")
    if float(payload.get("raw_difference_b_minus_a", 0.0)) <= 0:
        raise ValueError("B15 weak-v2 raw delta is not positive")
    if float(payload.get("median_difference", 0.0)) <= 0:
        raise ValueError("B15 weak-v2 paired median delta is not positive")
    if float(payload.get("probability_b_better", 0.0)) < 0.95:
        raise ValueError("B15 weak-v2 probability gate is below 0.95")
    return payload


def evaluate_b15_gold(
    config: dict,
    *,
    checkpoint: str | Path,
    gate_json: str | Path,
    out_root: str | Path = "runs/b15_mri_ssl/gold_confirmation",
) -> dict:
    require_v2_downstream_contract(config)
    gate = load_passing_b15_gate(gate_json)

    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, checkpoint_payload = load_v2_downstream_checkpoint(
        checkpoint, expected_mode="b15", device=runtime.device
    )
    if checkpoint_payload.get("variant") != B15_VARIANT:
        raise ValueError("B15 gold confirmation checkpoint variant mismatch")
    if checkpoint_payload.get("experiment") != B15_EXPERIMENT:
        raise ValueError("B15 gold confirmation training experiment mismatch")
    if checkpoint_payload.get("weak_holdout_manifest_sha256") != WEAK_V2_MANIFEST_SHA256:
        raise ValueError("B15 checkpoint weak-v2 manifest SHA mismatch")
    if int(checkpoint_payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B15 checkpoint does not certify zero gold gradient use")
    if bool(checkpoint_payload.get("gold_labels_for_early_stopping", True)):
        raise ValueError("B15 checkpoint does not certify zero gold early stopping")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != 58 or gold[TARGETS].isna().any().any():
        raise ValueError("B15 gold confirmation requires the complete 58-study gold surface")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    uids = gold["StudyInstanceUID"].tolist()
    variable_index = build_variable_series_index(series, uids)
    counts = [len(variable_index[uid]) for uid in uids]
    if any(count == 0 for count in counts):
        raise ValueError("B15 gold surface contains a study with zero eligible series")

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B15 gold confirmation freezes TTA [-1,0,1]")
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
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 16_900_000),
    )
    pred_uids, prediction = predict_b12_1(model, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("B15 gold prediction order changed")

    truth = gold[TARGETS].to_numpy(np.float64)
    result = bootstrap_macro_auc(
        truth,
        prediction,
        n_bootstrap=int(config.get("b7_n_bootstrap", 5000)),
        seed=int(config.get("seed", 2026)) + 179,
    )

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(prediction, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    frame.to_csv(out / "gold_predictions.csv", index=False)

    payload = result.to_dict()
    payload.update(
        {
            "experiment": "B15_one_look_reused_gold_development_confirmation",
            "mode": "b15",
            "variant": B15_VARIANT,
            "checkpoint": str(Path(checkpoint).resolve()),
            "training_experiment": B15_EXPERIMENT,
            "completed_epochs": checkpoint_payload.get("completed_epochs"),
            "n_gold_studies": int(len(gold)),
            "tta_center_offsets": list(offsets),
            "gold_series_count": {
                "total": int(sum(counts)),
                "min": int(min(counts)),
                "median": float(np.median(counts)),
                "max": int(max(counts)),
            },
            "aggregation": "B13 hierarchical learned one-token-per-series aggregation",
            "initialization": checkpoint_payload.get("initialization"),
            "initialization_detail": checkpoint_payload.get("initialization_detail"),
            "input_normalization": checkpoint_payload.get("input_normalization"),
            "gold_labels_used_in_gradient": False,
            "gold_labels_used_for_early_stopping": False,
            "weak_v2_gate_json": str(Path(gate_json).resolve()),
            "weak_v2_gate": {
                "surface": gate.get("surface"),
                "manifest_sha256": gate.get("weak_holdout_manifest_sha256"),
                "raw_difference_b_minus_a": gate.get("raw_difference_b_minus_a"),
                "median_difference": gate.get("median_difference"),
                "ci_lower": gate.get("ci_lower"),
                "ci_upper": gate.get("ci_upper"),
                "probability_b_better": gate.get("probability_b_better"),
                "passes_gate": gate.get("passes_gate"),
            },
            "metadata_repair": metadata_stats,
            "interpretation": (
                "single development confirmation on the repeatedly reused 58-study gold surface; "
                "not independent validation and not eligible for further B15 tuning"
            ),
        }
    )
    (out / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(result.summary())
    print(out / "gold_predictions.csv")
    print(out / "eval.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b15-gold-eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--out-root", default="runs/b15_mri_ssl/gold_confirmation")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    payload = evaluate_b15_gold(
        config,
        checkpoint=args.checkpoint,
        gate_json=args.gate_json,
        out_root=args.out_root,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
