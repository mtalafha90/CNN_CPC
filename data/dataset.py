"""Study, series and slice loading for the working model.

A study contributes every one of its real MRI series; series count varies per
study, so batches are collated with padding and a presence mask rather than
being truncated to a fixed number of series.
"""

from __future__ import annotations

from pathlib import Path

from model._implementation import (
    collate,
    inference_dataset_config,
    load_series_table,
    load_study_table,
    load_test_table,
    repair_series_metadata,
    series_index,
    study_dataset,
)

collate_studies = collate


def read_studies(root: Path, config: dict, *, split: str = "train"):
    """Read the study table for a split."""
    if split == "train":
        return load_study_table(Path(root) / config.get("train_csv", "train.csv"))
    if split == "test":
        return load_test_table(Path(root) / config.get("test_csv", "test.csv"))
    raise ValueError("split must be 'train' or 'test'")


def read_series(root: Path, config: dict, *, split: str = "train"):
    """Read the series table for a split and repair missing acquisition metadata.

    Returns the series table and a record of what had to be recovered from the
    DICOM headers, which belongs in any run's audit trail.
    """
    key = "train_series_csv" if split == "train" else "test_series_csv"
    default = f"{split}_series.csv"
    series = load_series_table(Path(root) / config.get(key, default))
    return repair_series_metadata(series, Path(root), split=split)


def index_series_by_study(series, uids):
    """Map each study to its eligible MRI series."""
    return series_index(series, uids)


def build_dataset(uids, index, dataset_config, *, train: bool, crop_policy: dict):
    """Build a study-level dataset over variable numbers of real series."""
    return study_dataset(uids, index, dataset_config, train=train, policy=crop_policy)


def build_inference_config(config: dict, root: Path, offsets: tuple[int, ...]):
    """Dataset settings for inference: no augmentation, fixed slice offsets."""
    return inference_dataset_config(config, Path(root), offsets)


__all__ = [
    "build_dataset",
    "build_inference_config",
    "collate_studies",
    "index_series_by_study",
    "read_series",
    "read_studies",
]
