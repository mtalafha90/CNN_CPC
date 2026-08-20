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
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from data.dataset import read_series, read_studies
from model._implementation import (
    expert_loader,
    macro_auc,
    predict,
    read_config,
    resolve_checkpoints,
    resolve_runtime,
    run_directory,
)
from model.architecture import TARGETS, load
from model.preprocessing import resolve_crop_policy

EVALUATION_ROLE = (
    "reused expert development diagnostic; not independent test evidence and "
    "not a promotion criterion"
)


def evaluate(
    config: dict,
    *,
    checkpoint: str | Path | Sequence[str | Path],
    device: str | None = None,
    loader=load,
) -> dict:
    """Predict the expert studies and report macro and per-target AUC.

    `loader` rebuilds the model from its checkpoint. It defaults to the
    supported one; a variant whose checkpoint is a different width supplies its
    own, so the rest of this stage stays shared rather than duplicated.
    """
    root = Path(config["data_root"])
    runtime = resolve_runtime(config)
    print(runtime.describe())

    crop_policy = resolve_crop_policy(config)
    paths = resolve_checkpoints(checkpoint)
    models, payloads = [], []
    for path in paths:
        built, built_payload = loader(path, device=device or runtime.device)
        built.eval()
        models.append(built)
        payloads.append(built_payload)
    model, payload = models[0], payloads[0]

    studies = read_studies(root, config, split="train")
    series, metadata_repair = read_series(root, config, split="train")
    expert = expert_loader(config, root, studies, series, runtime, crop_policy)

    predictions = []
    for member in models:
        uids, member_prediction = predict(member, expert["loader"], runtime)
        if uids != expert["uids"]:
            raise RuntimeError("expert study order changed between loading and prediction")
        predictions.append(member_prediction)
    prediction = np.mean(predictions, axis=0)

    macro, per_target = macro_auc(expert["truth"], prediction)
    if not np.isfinite(macro) or not np.isfinite(per_target).all():
        raise RuntimeError("every one of the 12 expert AUCs must be defined")

    return {
        "checkpoints": [str(p.resolve()) for p in paths],
        "ensemble_size": len(paths),
        "checkpoint": str(paths[0].resolve()),
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
    parser.add_argument(
        "--checkpoint",
        required=True,
        action="append",
        help="model to score; repeat the flag to average several",
    )
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

    default_stem = Path(args.checkpoint[0]).resolve().parent.name
    if len(args.checkpoint) > 1:
        default_stem = f"ensemble_{len(args.checkpoint)}"
    stem = args.name or default_stem
    out = run_directory(args.experiment, "validate") / f"{stem}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(out)


if __name__ == "__main__":
    main()
