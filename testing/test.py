"""Predict the competition test set and write a submission file.

Every study is scored with the same fixed slice offsets, and the averaged
probabilities are written in the submission's expected column order.  A
manifest is written alongside recording the checkpoint hash, the encoder
fingerprint and the series coverage of the test split, so a submitted file can
always be traced back to the run that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import (
    build_dataset,
    build_inference_config,
    collate_studies,
    index_series_by_study,
    read_series,
    read_studies,
)
from model._implementation import (
    autocast,
    inference_batch_size,
    inference_slice_offsets,
    read_config,
    resolve_runtime,
    run_directory,
    runtime_budget,
    validate_against_sample,
    validate_submission,
)
from model.architecture import TARGETS, load
from model.preprocessing import resolve_crop_policy

SLICE_OFFSETS = (-1, 0, 1)
LOADER_SEED_OFFSET = 90_100_000


def _file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def predict_test_set(
    config: dict,
    *,
    checkpoint: str | Path | Sequence[str | Path],
    out_path: str | Path = "submission.csv",
    loader=load,
) -> Path:
    """Score every test study and write the submission and its manifest.

    `checkpoint` may name several models. Their probabilities are averaged,
    which is the cheapest reliable way to gain a little accuracy: independent
    models make different mistakes, and averaging cancels some of them. Every
    model sees each study once, in one pass over the scans, so the expensive
    part -- reading and decoding MRI -- is paid once no matter how many models
    are averaged.

    `loader` rebuilds a model from its checkpoint. It defaults to the supported
    one; a variant whose checkpoint differs supplies its own, so inference stays
    shared rather than duplicated.
    """
    crop_policy = resolve_crop_policy(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())

    paths = (
        [Path(checkpoint).resolve()]
        if isinstance(checkpoint, (str, Path))
        else [Path(p).resolve() for p in checkpoint]
    )
    if not paths:
        raise ValueError("no checkpoint given")

    models, payloads = [], []
    for path in paths:
        built, payload = loader(path, device=runtime.device)
        built.eval()
        models.append(built)
        payloads.append(payload)
        print(f"[ensemble] loaded {path.name} from {path.parent.name}")
    model, payload = models[0], payloads[0]
    checkpoint = paths[0]
    if len(models) > 1:
        print(f"[ensemble] averaging {len(models)} models")

    root = Path(config["data_root"])
    test = read_studies(root, config, split="test")
    uids = test["StudyInstanceUID"].astype(str).tolist()
    if not uids:
        raise ValueError("the test table contains no studies")

    series, metadata_repair = read_series(root, config, split="test")
    index = index_series_by_study(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    empty = [uid for uid, count in zip(uids, counts) if count == 0]
    if empty:
        raise ValueError(f"{len(empty)} test study/studies have no eligible MRI series")

    offsets = inference_slice_offsets(config, SLICE_OFFSETS)
    if offsets != SLICE_OFFSETS:
        raise ValueError(f"inference slice offsets are fixed at {list(SLICE_OFFSETS)}")

    dataset = build_dataset(
        uids,
        index,
        build_inference_config(config, root, offsets),
        train=False,
        crop_policy=crop_policy,
    )
    loader = DataLoader(
        dataset,
        batch_size=inference_batch_size(config),
        shuffle=False,
        collate_fn=collate_studies,
        **runtime.loader_kwargs(
            seed=int(config.get("seed", 2026)) + LOADER_SEED_OFFSET
        ),
    )

    budget = runtime_budget(
        max_hours=min(float(config.get("runtime_budget_hours", 8.5)), 8.25),
        reserve_minutes=max(float(config.get("runtime_reserve_minutes", 10.0)), 30.0),
    )

    model.eval()
    scored_uids: list[str] = []
    probability_blocks: list[np.ndarray] = []
    batch_seconds: list[float] = []

    for batch_index, batch in enumerate(loader):
        if batch_seconds:
            remaining = max(1, len(loader) - batch_index)
            budget.require(
                float(np.mean(batch_seconds[-5:])) * remaining * 1.35,
                label="remaining test inference",
            )
        else:
            budget.require(180.0, label="first test batch")

        started = time.monotonic()
        present = batch["present"].to(runtime.device, non_blocking=True)
        series_meta = batch["series_meta"].to(runtime.device, non_blocking=True)
        volumes = batch["volumes"]
        if volumes.ndim != 7:
            raise RuntimeError(
                f"expected volumes shaped [B,V,K,S,C,H,W], got {tuple(volumes.shape)}"
            )
        if int(volumes.shape[1]) != len(offsets):
            raise RuntimeError("the number of slice-offset views changed")

        views = []
        for view in range(volumes.shape[1]):
            frame = volumes[:, view].to(runtime.device, non_blocking=True)
            for member in models:
                with autocast(runtime):
                    logits = member(frame, present, series_meta)
                views.append(torch.sigmoid(logits.float()))

        probability_blocks.append(torch.stack(views, dim=0).mean(dim=0).cpu().numpy())
        scored_uids.extend(str(uid) for uid in batch["study_uid"])
        batch_seconds.append(time.monotonic() - started)

    if scored_uids != uids:
        raise RuntimeError("test inference changed the study order")

    frame = pd.DataFrame(np.concatenate(probability_blocks, axis=0), columns=list(TARGETS))
    frame.insert(0, "StudyInstanceUID", uids)
    validate_submission(frame, uids)
    sample_check = validate_against_sample(root, frame)

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    manifest = {
        "checkpoints": [str(p) for p in paths],
        "ensemble_size": len(paths),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_digest(checkpoint),
        "checkpoint_sha256_all": [_file_digest(p) for p in paths],
        "submission_sha256": _file_digest(output),
        "completed_epochs": int(payload.get("completed_epochs", -1)),
        "fixed_endpoint": bool(payload.get("fixed_endpoint", False)),
        "encoder_frozen": bool(payload.get("encoder_frozen", False)),
        "encoder_sha256": payload.get("encoder_sha256_final"),
        "expert_labels_in_gradients": int(payload.get("gold_studies_used_in_gradient", 0)),
        "crop_policy": crop_policy,
        "slice_offsets": list(offsets),
        "test_studies": int(len(frame)),
        "test_series_total": int(sum(counts)),
        "test_series_min": int(min(counts)),
        "test_series_median": float(np.median(counts)),
        "test_series_max": int(max(counts)),
        "metadata_repair": metadata_repair,
        "elapsed_hours": float(budget.elapsed_seconds / 3600.0),
        **sample_check,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(output)
    print(manifest_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the competition test set with the working model"
    )
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        action="append",
        help="model to score with; repeat the flag to average several",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="run name; the submission lands under runs/<experiment>/test/",
    )
    parser.add_argument(
        "--name",
        help="submission filename stem; defaults to the checkpoint's directory name",
    )
    args = parser.parse_args()

    config = read_config(args.config)
    config["data_root"] = str(Path(args.data_root).resolve())

    default_stem = Path(args.checkpoint[0]).resolve().parent.name
    if len(args.checkpoint) > 1:
        default_stem = f"ensemble_{len(args.checkpoint)}"
    stem = args.name or default_stem
    out_path = run_directory(args.experiment, "test") / f"{stem}.csv"
    predict_test_set(config, checkpoint=args.checkpoint, out_path=out_path)


if __name__ == "__main__":
    main()
