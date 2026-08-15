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
from rsna_knee.b24_protocol import B24_CROP_FRACTION
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


EXPERIMENT = "B24X_exploratory_pilot"


def load_b24x(path: str | Path, expected_formal_mode: str, device):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if payload.get("experiment") != EXPERIMENT:
        raise ValueError(
            f"{path} is not a B24X exploratory checkpoint"
        )

    if payload.get("formal_mode") != expected_formal_mode:
        raise ValueError(
            f"expected formal_mode={expected_formal_mode!r}, "
            f"got {payload.get('formal_mode')!r}"
        )

    expected_stored = expected_formal_mode + "_exploratory"
    if payload.get("mode") != expected_stored:
        raise ValueError(
            f"expected stored mode={expected_stored!r}, "
            f"got {payload.get('mode')!r}"
        )

    if payload.get("exploratory") is not True:
        raise ValueError("checkpoint is not marked exploratory")

    if payload.get("formal_b24_eligible") is not False:
        raise ValueError("checkpoint incorrectly claims formal B24 eligibility")

    if payload.get("gold_acceptance_allowed") is not False:
        raise ValueError("checkpoint incorrectly permits gold acceptance")

    if int(payload.get("completed_epochs", -1)) != 2:
        raise ValueError("B24X requires fixed epoch 2")

    if payload.get("fixed_endpoint") is not True:
        raise ValueError("B24X checkpoint is not a fixed endpoint")

    gate = payload.get("formal_b23_gate", {})
    if gate.get("passed") is not False:
        raise ValueError(
            "B24X checkpoint must retain the failed formal B23 gate"
        )

    model = build_b12_1_model(
        payload["model_spec"],
        pretrained_weights=False,
    )
    model.load_state_dict(
        payload["model_state"],
        strict=True,
    )

    return model.to(device).eval(), payload


def predict_model(
    model,
    *,
    config,
    root,
    uids,
    variable_index,
    runtime,
    crop_fraction,
):
    offsets = tuple(
        int(x)
        for x in config.get(
            "b7_eval_tta_offsets",
            [-1, 0, 1],
        )
    )

    if offsets != (-1, 0, 1):
        raise ValueError(
            "B24X weak-v2 evaluation freezes TTA [-1,0,1]"
        )

    dataset_config = make_b7_dataset_config(
        config,
        root,
        train=False,
        tta_offsets=offsets,
    )

    ds = make_matched_crop_dataset(
        "control",
        uids,
        variable_index,
        dataset_config,
        crop_fraction=crop_fraction,
        train=False,
    )

    loader = DataLoader(
        ds,
        batch_size=int(
            config.get("b7_eval_batch_size", 2)
        ),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(
            seed=int(config.get("seed", 2026))
            + 24_500_000
        ),
    )

    pred_uids, prediction = predict_b12_1(
        model,
        loader,
        runtime,
    )

    pred_uids = [str(x) for x in pred_uids]
    expected = [str(x) for x in uids]

    if pred_uids != expected:
        raise RuntimeError(
            "B24X weak-v2 prediction order changed"
        )

    return prediction


def main():
    parser = argparse.ArgumentParser(
        description="B24X exploratory paired weak-v2 evaluation"
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)

    parser.add_argument(
        "--control-checkpoint",
        required=True,
    )
    parser.add_argument(
        "--candidate-checkpoint",
        required=True,
    )

    parser.add_argument("--b6-root", required=True)
    parser.add_argument(
        "--weak-holdout-root",
        required=True,
    )

    parser.add_argument(
        "--out-root",
        default="runs/b24x_pilot/weak_v2_eval",
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
    )

    args = parser.parse_args()

    config = _read_config(args.config)
    config["data_root"] = args.data_root

    runtime = resolve_runtime(config)
    print(runtime.describe())

    # --------------------------------------------------------
    # Load both exploratory checkpoints
    # --------------------------------------------------------
    control, cp = load_b24x(
        args.control_checkpoint,
        "b6_control",
        runtime.device,
    )

    candidate, qp = load_b24x(
        args.candidate_checkpoint,
        "b23_candidate",
        runtime.device,
    )

    # --------------------------------------------------------
    # Matched-experiment invariants
    # --------------------------------------------------------
    if cp["study_uids"] != qp["study_uids"]:
        raise RuntimeError(
            "B24X arms did not train on identical study order"
        )

    if (
        cp["encoder_sha256_initial"]
        != qp["encoder_sha256_initial"]
    ):
        raise RuntimeError(
            "B24X arms did not start from identical encoder"
        )

    if cp["crop_fraction"] != qp["crop_fraction"]:
        raise RuntimeError(
            "B24X arms used different crop fractions"
        )

    if not np.isclose(
        float(cp["crop_fraction"]),
        float(B24_CROP_FRACTION),
    ):
        raise RuntimeError(
            "B24X crop fraction differs from frozen B24 value"
        )

    # --------------------------------------------------------
    # Frozen weak-v2 holdout
    # --------------------------------------------------------
    weak_payload, manifest = load_frozen_v2_manifest(
        args.weak_holdout_root
    )

    root = Path(config["data_root"])

    train = load_train_csv(
        root / config.get("train_csv", "train.csv")
    )

    b6_frame, _, _ = load_frozen_b6_export(
        args.b6_root
    )

    (
        uids,
        weak_targets,
        weak_weights,
        _,
    ) = _holdout_supervision(
        train,
        b6_frame,
        manifest,
    )

    if len(uids) != 623:
        raise RuntimeError(
            f"expected 623 weak-v2 holdout studies, got {len(uids)}"
        )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------
    train_uids = set(
        str(x)
        for x in cp["study_uids"]
    )
    holdout_uids = set(
        str(x)
        for x in uids
    )

    overlap = train_uids & holdout_uids

    print()
    print("B24X weak-v2 leakage check")
    print("  training studies :", len(train_uids))
    print("  holdout studies  :", len(holdout_uids))
    print("  overlap          :", len(overlap))

    if overlap:
        raise RuntimeError(
            f"{len(overlap)} weak-v2 holdout studies leaked into B24X training"
        )

    # --------------------------------------------------------
    # MRI series
    # --------------------------------------------------------
    series = load_series_csv(
        root
        / config.get(
            "train_series_csv",
            "train_series.csv",
        )
    )

    series, metadata_stats = backfill_series_metadata(
        series,
        root,
        split="train",
    )

    variable_index = build_variable_series_index(
        series,
        uids,
    )

    counts = [
        len(variable_index[str(uid)])
        for uid in uids
    ]

    if any(x == 0 for x in counts):
        raise RuntimeError(
            "weak-v2 holdout contains study with zero eligible series"
        )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------
    print()
    print(
        "[B24X] predicting B6-supervised control "
        "on frozen weak-v2..."
    )

    pred_control = predict_model(
        control,
        config=config,
        root=root,
        uids=uids,
        variable_index=variable_index,
        runtime=runtime,
        crop_fraction=float(cp["crop_fraction"]),
    )

    print(
        "[B24X] predicting Qwen/B23-supervised candidate "
        "on frozen weak-v2..."
    )

    pred_candidate = predict_model(
        candidate,
        config=config,
        root=root,
        uids=uids,
        variable_index=variable_index,
        runtime=runtime,
        crop_fraction=float(qp["crop_fraction"]),
    )

    # --------------------------------------------------------
    # Strict 12-target weak evaluation
    # --------------------------------------------------------
    n_bootstrap = int(args.n_bootstrap)
    seed = int(config.get("seed", 2026))

    control_eval = evaluate_on_weak_surface(
        weak_targets,
        pred_control,
        weak_weights,
        n_bootstrap=n_bootstrap,
        seed=seed + 241,
    )

    candidate_eval = evaluate_on_weak_surface(
        weak_targets,
        pred_candidate,
        weak_weights,
        n_bootstrap=n_bootstrap,
        seed=seed + 242,
    )

    paired = compare_on_weak_surface(
        weak_targets,
        pred_control,
        pred_candidate,
        weak_weights,
        n_bootstrap=n_bootstrap,
        seed=seed + 243,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    result = {
        "experiment": EXPERIMENT,
        "evaluation": "frozen_weak_b6_holdout_v2",
        "exploratory": True,
        "formal_b24_eligible": False,
        "gold_used": False,
        "promotion_allowed": False,

        "training_studies": len(cp["study_uids"]),
        "holdout_studies": len(uids),
        "training_holdout_overlap": 0,

        "control": {
            "supervision": "B6 v1.2.1",
            "checkpoint": str(
                Path(args.control_checkpoint)
            ),
            "macro_auc": float(
                control_eval["macro_auc"]
            ),
            "ci_lower": float(
                control_eval["ci_lower"]
            ),
            "ci_upper": float(
                control_eval["ci_upper"]
            ),
            "per_target_auc":
                control_eval["per_target_auc"],
        },

        "candidate": {
            "supervision": "B23/Qwen",
            "checkpoint": str(
                Path(args.candidate_checkpoint)
            ),
            "macro_auc": float(
                candidate_eval["macro_auc"]
            ),
            "ci_lower": float(
                candidate_eval["ci_lower"]
            ),
            "ci_upper": float(
                candidate_eval["ci_upper"]
            ),
            "per_target_auc":
                candidate_eval["per_target_auc"],
        },

        "paired_candidate_minus_control":
            paired,

        "holdout_usable_cells": int(
            (weak_weights > 0).sum()
        ),

        "holdout_series": {
            "total": int(sum(counts)),
            "min": int(min(counts)),
            "median": float(np.median(counts)),
            "max": int(max(counts)),
        },

        "encoder_sha256":
            cp["encoder_sha256_initial"],

        "crop_fraction":
            float(cp["crop_fraction"]),

        "fixed_epoch": 2,

        "weak_holdout_manifest_sha256":
            WEAK_V2_MANIFEST_SHA256,

        "weak_holdout_metadata":
            weak_payload,

        "metadata_repair":
            metadata_stats,

        "interpretation": (
            "Exploratory development evidence only. "
            "This surface is labelled by B6 and therefore "
            "favours the B6-supervised control by construction. "
            "A candidate win is informative cross-teacher evidence; "
            "a loss is not expert-truth evidence. "
            "No gold acceptance or model promotion is permitted."
        ),
    }

    out = Path(args.out_root)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_frame = pd.DataFrame(
        {
            "StudyInstanceUID": uids,
        }
    )

    for j, target in enumerate(TARGETS):
        prediction_frame[
            f"{target}__control"
        ] = pred_control[:, j]

        prediction_frame[
            f"{target}__candidate"
        ] = pred_candidate[:, j]

    prediction_frame.to_csv(
        out / "paired_predictions.csv",
        index=False,
    )

    (
        out / "comparison.json"
    ).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Human-readable result
    # --------------------------------------------------------
    print()
    print("=" * 72)
    print("B24X FROZEN WEAK-V2 RESULT")
    print("=" * 72)

    print(
        f"B6 control      : "
        f"{control_eval['macro_auc']:.10f} "
        f"[{control_eval['ci_lower']:.10f}, "
        f"{control_eval['ci_upper']:.10f}]"
    )

    print(
        f"B23/Qwen        : "
        f"{candidate_eval['macro_auc']:.10f} "
        f"[{candidate_eval['ci_lower']:.10f}, "
        f"{candidate_eval['ci_upper']:.10f}]"
    )

    print()
    print(
        f"raw B23 - B6    : "
        f"{paired['raw_difference_b_minus_a']:+.10f}"
    )

    print(
        f"paired median   : "
        f"{paired['median_difference']:+.10f}"
    )

    print(
        f"paired 95% CI   : "
        f"[{paired['ci_lower']:+.10f}, "
        f"{paired['ci_upper']:+.10f}]"
    )

    print(
        f"P(B23 > B6)     : "
        f"{paired['probability_b_better']:.4f}"
    )

    print(
        f"bootstrap       : "
        f"{paired['n_valid_replicates']}/"
        f"{paired['n_bootstrap']}"
    )

    print()
    print("NO GOLD USED.")
    print("NO PROMOTION DECISION.")

    print()
    print("saved:", out / "paired_predictions.csv")
    print("saved:", out / "comparison.json")


if __name__ == "__main__":
    main()
