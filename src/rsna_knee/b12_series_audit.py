"""Label-free B12 audit of the variable-series training surface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .b7_weak_supervision import _read_config, load_frozen_b6_export, prepare_b7_supervision
from .b12_variable_series import B12_SERIES_POLICY, audit_variable_series_surface
from .data import backfill_series_metadata, load_series_csv, load_train_csv

B12_AUDIT_EXPERIMENT = "B12_variable_series_audit"


def run_b12_series_audit(
    config: dict,
    *,
    b6_root: str | Path,
    out_root: str | Path = "runs/b12_variable_series/audit",
) -> dict:
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, b6_audit = load_frozen_b6_export(b6_root)
    uids, _, _, supervision = prepare_b7_supervision(train, b6_frame)
    if len(uids) != 3120 or int(supervision.get("usable_cells", -1)) != 14123:
        raise ValueError("B12-v1 must audit the exact retained B7.1 B6 supervision surface")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    summary, _ = audit_variable_series_surface(series, uids)

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": B12_AUDIT_EXPERIMENT,
        "status": "B12 label-free series policy frozen before training/gold evaluation",
        "policy": B12_SERIES_POLICY,
        "uses_gold_labels": False,
        "b6_version": b6_audit.get("b6_version"),
        "b6_active_studies": len(uids),
        "b6_usable_cells": int(supervision["usable_cells"]),
        "metadata_repair": metadata_stats,
        "series_summary": summary,
        "viability_passed": bool(summary["viability_passed"]),
        "single_scientific_change_vs_b7_1": (
            "retain all repaired sagittal/coronal/axial MRI series and represent them as "
            "variable-length series tokens instead of selecting six fixed semantic slots"
        ),
        "frozen_controls": {
            "supervision": "B6 v1.2.1 only; exact B7.1 active study/cell surface",
            "initialization": "B5 encoder",
            "preprocessing": "legacy direct 224x224 resize; no B10 physical normalization",
            "epochs": 4,
            "batch_size": 2,
            "optimizer_and_augmentation": "B7.1-equivalent",
            "gold_gradients": 0,
            "gold_early_stopping": 0,
        },
    }
    (out / "series_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "series_policy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["viability_passed"]:
        print("[B12] series viability PASSED; policy frozen for training implementation.")
    else:
        print("[B12] series viability FAILED; inspect audit and do not train B12-v1.")
    print(out / "series_audit.json")
    print(out / "series_policy.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b12-audit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--out-root", default="runs/b12_variable_series/audit")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    run_b12_series_audit(config, b6_root=args.b6_root, out_root=args.out_root)


if __name__ == "__main__":
    main()
