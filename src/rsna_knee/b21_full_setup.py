from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from .b7_weak_supervision import (
    load_frozen_b6_export,
    make_b7_dataset_config,
    prepare_b7_supervision,
    target_balance_multipliers,
)
from .b12_training import _load_series_policy
from .b12_variable_series import audit_variable_series_surface, collate_variable_series
from .b13_training import B13_SERIES_SIGNATURE
from .b21_acceptance_protocol import (
    B21_CROP_FRACTION,
    B21_FULL_BATCHES,
    B21_FULL_NEGATIVE_CELLS,
    B21_FULL_POSITIVE_CELLS,
    B21_FULL_TRAIN_CELLS,
    B21_FULL_TRAIN_SERIES,
    B21_FULL_TRAIN_STUDIES,
)
from .b21_dataset import make_matched_crop_dataset
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv


def prepare_b21_full_surface(
    config: dict,
    *,
    b6_root: str | Path,
    series_policy_path: str | Path,
    runtime,
):
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    uids, targets, weights, supervision = prepare_b7_supervision(train, b6_frame)

    pos = int(((weights > 0) & (targets > 0.5)).sum())
    neg = int(((weights > 0) & (targets < 0.5)).sum())
    if len(uids) != B21_FULL_TRAIN_STUDIES:
        raise ValueError("B21 full active-study count changed")
    if int((weights > 0).sum()) != B21_FULL_TRAIN_CELLS:
        raise ValueError("B21 full usable-cell count changed")
    if pos != B21_FULL_POSITIVE_CELLS or neg != B21_FULL_NEGATIVE_CELLS:
        raise ValueError("B21 full positive/negative counts changed")

    gold = set(train.loc[gold_mask(train), "StudyInstanceUID"].astype(str))
    if len(gold) != 58:
        raise ValueError("B21 full expected 58 expert studies")
    if set(str(uid) for uid in uids).intersection(gold):
        raise RuntimeError("gold study leaked into B21 full gradients")

    series_policy = _load_series_policy(series_policy_path)
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    summary, index = audit_variable_series_surface(series, uids)
    if summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B21 full series SHA differs from frozen B13/B20 mapping")
    if int(summary.get("eligible_recognized_plane_series", -1)) != B21_FULL_TRAIN_SERIES:
        raise ValueError("B21 full requires exactly 17,475 eligible series")
    if summary.get("viability_passed") is not True:
        raise ValueError("B21 full series surface fails viability")
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("supplied series policy is not the frozen B12/B13 mapping")
    if any(not index.get(str(uid)) for uid in uids):
        raise ValueError("B21 full active study has zero eligible series")

    expected_series = int(sum(len(index[str(uid)]) for uid in uids))
    batch_size = int(config.get("b7_batch_size", 2))
    expected_batches = int(math.ceil(len(uids) / batch_size))
    if expected_series != B21_FULL_TRAIN_SERIES or expected_batches != B21_FULL_BATCHES:
        raise ValueError("B21 full series/batch coverage contract changed")

    target_multiplier = target_balance_multipliers(weights)
    ds = make_matched_crop_dataset(
        "preresize",
        uids,
        index,
        make_b7_dataset_config(config, root, train=True),
        crop_fraction=B21_CROP_FRACTION,
        targets=targets,
        weights=weights,
        train=True,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 19_100_000),
    )

    supervision = dict(supervision)
    supervision.update({
        "training_studies": len(uids),
        "training_cells": B21_FULL_TRAIN_CELLS,
        "training_positive_cells": pos,
        "training_negative_cells": neg,
        "training_series": expected_series,
        "batches_per_epoch": expected_batches,
        "gold_studies_in_gradient": 0,
    })
    return {
        "root": root,
        "train": train,
        "uids": uids,
        "targets": targets,
        "weights": weights,
        "supervision": supervision,
        "b6_policy": b6_policy,
        "b6_audit": b6_audit,
        "series_policy": series_policy,
        "metadata_stats": metadata_stats,
        "target_multiplier": target_multiplier,
        "dataset": ds,
        "loader": loader,
    }
