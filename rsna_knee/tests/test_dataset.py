"""Tests for the dataset's series routing, slice sampling and collation.

These need torch, so they live apart from the pure-NumPy DICOM tests: a
module-level skip would otherwise hide those too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("torch", reason="dataset tests need torch")

from rsnaknee.dataset import (  # noqa: E402
    NUM_SERIES_TYPES,
    DatasetConfig,
    KneeExamDataset,
    collate_exams,
    sample_slice_indices,
    select_series,
    series_type_code,
    stack_neighbours,
)


def test_slice_sampling_is_deterministic_in_evaluation() -> None:
    rng = np.random.default_rng(0)
    first = sample_slice_indices(40, 8, training=False, rng=rng)
    second = sample_slice_indices(40, 8, training=False, rng=rng)

    assert np.array_equal(first, second)
    assert first.min() >= 0 and first.max() < 40
    assert np.all(np.diff(first) > 0)  # spans the series in order


def test_slice_sampling_varies_during_training() -> None:
    rng = np.random.default_rng(1)
    draws = {tuple(sample_slice_indices(40, 8, True, rng)) for _ in range(5)}
    assert len(draws) > 1


def test_slice_sampling_handles_a_series_shorter_than_the_budget() -> None:
    indices = sample_slice_indices(3, 8, training=False, rng=np.random.default_rng(0))
    assert len(indices) == 8
    assert indices.max() < 3


def test_stack_neighbours_clamps_at_the_volume_edges() -> None:
    volume = np.arange(5 * 4 * 4).reshape(5, 4, 4)
    stacked = stack_neighbours(volume, np.array([0, 4]))

    assert stacked.shape == (2, 3, 4, 4)
    # At slice 0 the previous slice is slice 0 again, not a wrap to the end.
    assert np.array_equal(stacked[0, 0], volume[0])
    assert np.array_equal(stacked[1, 2], volume[4])


def test_series_type_codes_are_unique_and_in_range() -> None:
    codes = {
        series_type_code(plane, weighting, fat)
        for plane in ("sagittal", "coronal", "axial", "unknown")
        for weighting in ("t1", "pd", "t2", "stir", "unknown")
        for fat in (True, False)
    }
    assert len(codes) == 40
    assert max(codes) < NUM_SERIES_TYPES


def test_select_series_prefers_variety_over_duplicates() -> None:
    rows = pd.DataFrame(
        {
            "plane": ["sagittal", "sagittal", "coronal"],
            "weighting": ["pd", "pd", "pd"],
            "fat_saturated": [True, True, True],
            "cached_slices": [20, 30, 25],
            "cache_path": ["a.npy", "b.npy", "c.npy"],
        }
    )

    selected = select_series(rows, max_series=2)

    # The duplicated sagittal series collapses to the one with more slices,
    # leaving room for the coronal series.
    assert set(selected["cache_path"]) == {"b.npy", "c.npy"}


def _build_fake_cache(directory: Path) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for exam in ("exam1", "exam2"):
        (directory / exam).mkdir(parents=True, exist_ok=True)
        for series, (plane, weighting) in enumerate(
            [("sagittal", "pd"), ("coronal", "t2")]
        ):
            volume = rng.integers(0, 255, size=(12, 64, 64), dtype=np.uint8)
            np.save(directory / exam / f"s{series}.npy", volume)
            rows.append(
                {
                    "exam_id": exam,
                    "series_id": f"s{series}",
                    "plane": plane,
                    "weighting": weighting,
                    "fat_saturated": True,
                    "cached_slices": 12,
                    "cache_path": f"{exam}/s{series}.npy",
                }
            )
    return pd.DataFrame(rows)


def test_dataset_returns_the_expected_shapes(tmp_path: Path) -> None:
    manifest = _build_fake_cache(tmp_path)
    frame = pd.DataFrame({"exam_id": ["exam1", "exam2"], "acl_tear": [0, 1]})
    config = DatasetConfig(cache_dir=str(tmp_path), image_size=32, depth=6, max_series=2,
                           augment=False, series_dropout=0.0, random_erase=0.0)

    dataset = KneeExamDataset(frame, manifest, config, "exam_id", ["acl_tear"], training=False)
    sample = dataset[0]

    assert sample["pixels"].shape == (2, 6, 3, 32, 32)
    assert sample["series_type"].shape == (2,)
    assert sample["labels"].tolist() == [0.0]


def test_dataset_yields_blanks_for_an_exam_with_no_cached_series(tmp_path: Path) -> None:
    """A missing exam must not crash inference; it should predict from zeros."""
    manifest = _build_fake_cache(tmp_path)
    frame = pd.DataFrame({"exam_id": ["missing_exam"]})
    config = DatasetConfig(cache_dir=str(tmp_path), image_size=32, depth=4, max_series=2,
                           augment=False)

    dataset = KneeExamDataset(frame, manifest, config, "exam_id", training=False)
    sample = dataset[0]

    assert sample["pixels"].shape == (1, 4, 3, 32, 32)


def test_collate_pads_exams_with_different_series_counts(tmp_path: Path) -> None:
    manifest = _build_fake_cache(tmp_path)
    # Give exam2 only one series so the batch is ragged.
    manifest = manifest.drop(manifest.index[-1])
    frame = pd.DataFrame({"exam_id": ["exam1", "exam2"], "acl_tear": [0, 1]})
    config = DatasetConfig(cache_dir=str(tmp_path), image_size=32, depth=4, max_series=2,
                           augment=False, series_dropout=0.0)
    dataset = KneeExamDataset(frame, manifest, config, "exam_id", ["acl_tear"], training=False)

    batch = collate_exams([dataset[0], dataset[1]])

    assert batch["pixels"].shape == (2, 2, 4, 3, 32, 32)
    assert batch["series_mask"].tolist() == [[1.0, 1.0], [1.0, 0.0]]
    # The padded slot must be zeros so it cannot leak signal.
    assert batch["pixels"][1, 1].abs().sum().item() == 0.0
