"""Validate the selected current-model checkpoint on the recorded expert surface.

This is development validation only. The 58 expert studies were reused during
model development and B20 epoch selection, so the result must not be described
as independent test performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from model.bootstrap import ensure_developments_source


def main() -> None:
    ensure_developments_source()

    from rsna_knee.b7_weak_supervision import _read_config
    from rsna_knee.b12_1_gold_eval import predict_b12_1
    from rsna_knee.b20_crop_focus import (
        _expert_loader,
        load_b20_checkpoint,
        require_b20_contract,
    )
    from rsna_knee.constants import TARGETS
    from rsna_knee.data import backfill_series_metadata, load_series_csv, load_train_csv
    from rsna_knee.evaluation import macro_auc_from_arrays
    from rsna_knee.runtime import resolve_runtime

    parser = argparse.ArgumentParser(
        description="Development validation for the active B20 checkpoint"
    )
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--checkpoint", default="runs/b20_crop_focus/b20_model.pt"
    )
    parser.add_argument("--out", default="runs/current_model/validation.json")
    args = parser.parse_args()

    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    root = Path(config["data_root"])

    runtime = resolve_runtime(config)
    print(runtime.describe())
    crop_policy = require_b20_contract(config)
    model, payload = load_b20_checkpoint(args.checkpoint, device=runtime.device)

    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    expert = _expert_loader(config, root, train, series, runtime, crop_policy)

    pred_uids, prediction = predict_b12_1(model, expert["loader"], runtime)
    if pred_uids != expert["uids"]:
        raise RuntimeError("expert validation order changed")

    macro_auc, per_target = macro_auc_from_arrays(expert["truth"], prediction)
    if not np.isfinite(macro_auc) or not np.isfinite(per_target).all():
        raise RuntimeError("all 12 expert AUCs must be defined")

    result = {
        "model": "B20_crop_only_joint_focus",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "selected_epoch": int(payload.get("selected_epoch", -1)),
        "evaluation_role": "reused expert development validation; not independent test evidence",
        "n_studies": len(expert["uids"]),
        "macro_auc": float(macro_auc),
        "per_target_auc": {
            target: float(value) for target, value in zip(TARGETS, per_target)
        },
        "crop_policy": crop_policy,
        "metadata_repair": metadata_stats,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(out)


if __name__ == "__main__":
    main()
