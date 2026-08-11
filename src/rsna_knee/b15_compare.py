"""Strict paired weak-v2 comparison of B13-control (A) versus B15 (B)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .b7_weak_supervision import _read_config, load_frozen_b6_export, prepare_b7_supervision
from .b15_ssl import WEAK_V2_MANIFEST_SHA256, load_frozen_v2_manifest
from .constants import TARGETS
from .data import load_train_csv
from .weak_validation import compare_on_weak_surface


def _read_predictions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"StudyInstanceUID", *TARGETS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction file missing columns: {missing}")
    frame = frame[["StudyInstanceUID", *TARGETS]].copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("prediction file contains duplicate UIDs")
    return frame


def compare_b13_v2_vs_b15(
    config: dict,
    *,
    b6_root: str | Path,
    weak_holdout_root: str | Path,
    control_predictions: str | Path,
    b15_predictions: str | Path,
    out: str | Path,
) -> dict:
    weak_payload, manifest = load_frozen_v2_manifest(weak_holdout_root)
    if weak_payload.get("manifest_sha256") != WEAK_V2_MANIFEST_SHA256:
        raise ValueError("paired comparison manifest SHA mismatch")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, _ = load_frozen_b6_export(b6_root)
    active_uids, y, w, _ = prepare_b7_supervision(train, b6_frame)
    row = {str(uid): i for i, uid in enumerate(active_uids)}
    holdout_uids = manifest.loc[
        manifest["split"] == "holdout", "StudyInstanceUID"
    ].astype(str).tolist()
    idx = np.asarray([row[uid] for uid in holdout_uids], dtype=int)
    weak_targets = y[idx]
    weak_weights = w[idx]

    control = _read_predictions(control_predictions).set_index("StudyInstanceUID")
    candidate = _read_predictions(b15_predictions).set_index("StudyInstanceUID")
    expected = set(holdout_uids)
    if set(control.index) != expected:
        raise ValueError("control predictions do not exactly match frozen v2 holdout")
    if set(candidate.index) != expected:
        raise ValueError("B15 predictions do not exactly match frozen v2 holdout")
    pred_a = control.loc[holdout_uids, TARGETS].to_numpy(np.float64)
    pred_b = candidate.loc[holdout_uids, TARGETS].to_numpy(np.float64)

    payload = compare_on_weak_surface(
        weak_targets,
        pred_a,
        pred_b,
        weak_weights,
        n_bootstrap=int(config.get("b7_n_bootstrap", 5000)),
        seed=int(config.get("seed", 2026)) + 152,
    )
    payload.update(
        {
            "comparison": "B15 minus B13-v2-control",
            "model_a": "B13-v2-control",
            "model_b": "B15",
            "weak_holdout_manifest_sha256": WEAK_V2_MANIFEST_SHA256,
            "n_holdout_studies": len(holdout_uids),
            "holdout_usable_cells": int((weak_weights > 0).sum()),
            "predeclared_gate": {
                "raw_difference_positive": bool(payload["raw_difference_b_minus_a"] > 0),
                "median_difference_positive": bool(payload["median_difference"] > 0),
                "probability_b_better_at_least_0_95": bool(
                    payload["probability_b_better"] >= 0.95
                ),
            },
            "passes_gate": bool(
                payload["raw_difference_b_minus_a"] > 0
                and payload["median_difference"] > 0
                and payload["probability_b_better"] >= 0.95
            ),
            "decision_rule": (
                "B15 proceeds to one reused-gold development confirmation only if raw delta >0, "
                "paired median delta >0, and P(B15>B13-v2-control)>=0.95"
            ),
        }
    )
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b15-compare")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--weak-holdout-root", required=True)
    parser.add_argument("--control-predictions", required=True)
    parser.add_argument("--b15-predictions", required=True)
    parser.add_argument("--out", default="runs/b15_mri_ssl/weak_eval/b13_v2_vs_b15.json")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    compare_b13_v2_vs_b15(
        config,
        b6_root=args.b6_root,
        weak_holdout_root=args.weak_holdout_root,
        control_predictions=args.control_predictions,
        b15_predictions=args.b15_predictions,
        out=args.out,
    )


if __name__ == "__main__":
    main()
