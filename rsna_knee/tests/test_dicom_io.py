"""Tests for DICOM interpretation and the dataset's sampling logic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsnaknee.dicom_io import (  # noqa: E402
    normalise_volume,
    plane_from_orientation,
    weighting_from_parameters,
)


# Direction cosines as written by real scanners for each acquisition plane.
SAGITTAL = [0, 1, 0, 0, 0, -1]
CORONAL = [1, 0, 0, 0, 0, -1]
AXIAL = [1, 0, 0, 0, 1, 0]


@pytest.mark.parametrize(
    "orientation,expected",
    [(SAGITTAL, "sagittal"), (CORONAL, "coronal"), (AXIAL, "axial")],
)
def test_plane_detection(orientation: list[float], expected: str) -> None:
    assert plane_from_orientation(orientation) == expected


def test_plane_detection_tolerates_oblique_acquisitions() -> None:
    """A slightly tilted sagittal series is still sagittal."""
    tilted = [0.05, 0.99, 0.0, 0.0, 0.0, -1.0]
    assert plane_from_orientation(tilted) == "sagittal"


def test_plane_detection_handles_missing_orientation() -> None:
    assert plane_from_orientation(None) == "unknown"
    assert plane_from_orientation([1, 0, 0]) == "unknown"


@pytest.mark.parametrize(
    "echo,repetition,expected",
    [
        (10.0, 600.0, "t1"),   # short TE, short TR
        (30.0, 3000.0, "pd"),  # short TE, long TR
        (80.0, 4000.0, "t2"),  # long TE, long TR
    ],
)
def test_weighting_from_acquisition_parameters(
    echo: float, repetition: float, expected: str
) -> None:
    weighting, _ = weighting_from_parameters(echo, repetition)
    assert weighting == expected


def test_short_inversion_time_means_stir() -> None:
    weighting, fat_saturated = weighting_from_parameters(30.0, 4000.0, inversion_time=150.0)
    assert weighting == "stir"
    assert fat_saturated


def test_fat_saturation_is_read_from_scan_options() -> None:
    _, fat_saturated = weighting_from_parameters(30.0, 3000.0, scan_options="FS")
    assert fat_saturated
    _, plain = weighting_from_parameters(30.0, 3000.0, scan_options="")
    assert not plain


def test_missing_timing_gives_unknown_rather_than_a_guess() -> None:
    weighting, _ = weighting_from_parameters(None, None)
    assert weighting == "unknown"


def test_normalise_volume_is_robust_to_a_bright_artefact() -> None:
    """One hot pixel must not flatten the rest of the volume."""
    volume = np.random.default_rng(0).uniform(100, 200, size=(4, 16, 16))
    volume[0, 0, 0] = 1e6

    normalised = normalise_volume(volume)

    assert normalised.dtype == np.uint8
    # The genuine tissue range should still span most of the scale.
    assert np.ptp(normalised[1:]) > 200


def test_normalise_volume_handles_a_blank_series() -> None:
    assert normalise_volume(np.zeros((3, 8, 8))).max() == 0
