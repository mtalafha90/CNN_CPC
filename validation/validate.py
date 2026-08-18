"""Score a trained checkpoint on the expert-annotated studies.

These 58 studies were reused throughout development, so the result is a
development diagnostic and a plausibility check -- not independent test
performance.  At this sample size a paired difference below roughly 0.03 macro
AUC is not resolvable, so the number carries that limitation rather than being
used to rank models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.dataset import read_series, read_studies
from model._implementation import (
    expert_loader,
    macro_auc,
    predict,
    read_config,
    resolve_runtime,
    run_directory,
)
from model.architecture import TARGETS, load
from model.preprocessing import resolve_crop_policy

EVALUATION_ROLE = (
    "reused expert development diagnostic; not independent test evidence and "
    "not a promotion criterion"
)


def evaluate(config: dict, *, checkpoint: str, device: str | None = None) -> dict:
    """Predict the expert studies and report macro and per-target AUC."""
    root = Path(config["data_root"])
    runtime = resolve_runtime(config)
    print(runtime.describe())

    crop_policy = resolve_crop_policy(config)
    model, payload = load(checkpoint, device=device or runtime.device)

    studies = read_studies(root, config, split="train")
    series, metadata_repair = read_series(root, config, split="train")
    expert = expert_loader(config, root, studies, series, runtime, crop_policy)

    uids, prediction = predict(model, expert["loader"], runtime)
    if uids != expert["uids"]:
        raise RuntimeError("expert study order changed between loading and prediction")

    macro, per_target = macro_auc(expert["truth"], prediction)
    if not np.isfinite(macro) or not np.isfinite(per_target).all():
        raise RuntimeError("every one of the 12 expert AUCs must be defined")

    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "completed_epochs": int(payload.get("completed_epochs", -1)),
        "evaluation_role": EVALUATION_ROLE,
        "n_studies": len(expert["uids"]),
        "macro_auc": float(macro),
        "per_target_auc": {
            target: float(value) for target, value in zip(TARGETS, per_target)
        },
        "crop_policy": crop_policy,
        "metadata_repair": metadata_repair,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a checkpoint on the expert-annotated studies"
    )
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--experiment",
        required=True,
        help="run name; the report lands under runs/<experiment>/validate/",
    )
    parser.add_argument(
        "--name",
        help="report filename stem; defaults to the checkpoint's directory name",
    )
    args = parser.parse_args()

    config = read_config(args.config)
    config["data_root"] = str(Path(args.data_root).resolve())

    result = evaluate(config, checkpoint=args.checkpoint)

    stem = args.name or Path(args.checkpoint).resolve().parent.name
    out = run_directory(args.experiment, "validate") / f"{stem}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(out)


if __name__ == "__main__":
    main()
