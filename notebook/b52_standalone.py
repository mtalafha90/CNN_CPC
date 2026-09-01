#!/usr/bin/env python3
"""B52 in one file: train the model, and write every result to a folder.

This is generated from `notebook/build_b52_script.py`. Do not edit it by hand --
edit the builder and regenerate, or the notebook and the script will disagree.

    python b52_standalone.py --data-root DIR --labels training_targets.csv --out DIR

## What B52 is

Every experiment in this line before it was measured on a model that had barely
been trained: the pixel encoder frozen at a learning rate of exactly zero, all
nine augmentations switched off, and two fixed epochs -- 3,120 optimiser steps
in total -- with no checkpoint selection. An architecture ablation measured that
way is measured through a floor.

B52 changes the training regime and nothing else:

    1. the encoder learns      the part that reads pixels, at a real rate
    2. augmentation is on      rotation, shift, scale, gamma, noise, dropout, bias
    3. a cosine that finishes,  and the best epoch is kept rather than the last

The geometry, the head, the labels and the loss are untouched, so anything that
changes is down to training.

## What one run writes

    config.json               the exact settings used
    labels_summary.json       how much supervision the reports actually gave
    history.json              per epoch: losses, hold-out AUC, gold AUC, gate, lr
    history.csv               the same, as a table
    per_target_auc.csv        every target's AUC at the best epoch
    holdout_predictions.csv   one probability row per held-out study
    gold_predictions.csv      the same for the expert-gold studies, if any
    test_predictions.csv      only when --test-root is given
    loss_curve.png            training and hold-out loss
    auc_curve.png             hold-out and gold macro AUC per epoch
    best_model.pt             the best epoch's weights, with its provenance
    summary.txt               the whole run in plain words

## What these numbers are worth

Nothing, in absolute terms. This trains a fresh compact model from random
weights on whatever subset you have. What transfers is the shape: whether the
training loss keeps falling, whether the hold-out score keeps rising, and which
epoch it peaks on. Do not compare the number it prints with a leaderboard score.
"""
from __future__ import annotations

# Figures are written to files, never shown, so the backend is fixed before
# pyplot is imported. Without this a machine with no display raises on import.
import matplotlib

matplotlib.use("Agg")

import argparse
import csv
import sys
from dataclasses import replace


# ==========================================================================
# Imports, labels, and reproducibility
# ==========================================================================


# Import dataclass helpers for clear configuration and experiment containers.
from dataclasses import asdict, dataclass, field


# Import a no-op context manager for CPU execution.
from contextlib import nullcontext


# Import Path for safe cross-platform file paths.
from pathlib import Path


# Import type names used in function annotations.
from typing import Iterable


# Import garbage collection for releasing large CPU tensors between epochs.
import gc


# Import JSON for readable configuration and history files.
import json


# Import math for sine/cosine position features and log-mean-exp pooling.
import math


# Import random for reproducible Python-level sampling.
import random


# Import Linux resource statistics so Colab preflight reports host-RAM pressure too.
import resource


# Import shutil for copying the two Drive archives to Colab's local SSD.
import shutil


# Import time for per-epoch timing.
import time


# Import zipfile for safe archive inspection and extraction.
import zipfile


# Import matplotlib for the loss curve and case-review figures.
import matplotlib.pyplot as plt


# Import NumPy for numerical arrays and deterministic splitting.
import numpy as np


# Import pandas for CSV tables and summary tables.
import pandas as pd


# Import PyTorch's main namespace.
import torch


# Import neural-network layers and functional operations.
import torch.nn.functional as F


from torch import nn


# Import the dataset and loader interfaces.
from torch.utils.data import DataLoader, Dataset


# Import activation checkpointing to trade extra compute for substantially lower GPU memory.
from torch.utils.checkpoint import checkpoint


# List target columns exactly as they appear in train.csv.
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


# Store the target count once so model classes do not repeat a magic number.
N_TARGETS = len(TARGETS)


# Convert normalized plane names into small categorical identifiers.
PLANE_TO_ID = {"Sagittal": 1, "Coronal": 2, "Axial": 3}


def set_seed(seed: int = 2026) -> None:
    """Make splitting and new-model initialization reproducible."""
    # Seed Python's random-number generator.
    random.seed(seed)
    # Seed NumPy's random-number generator.
    np.random.seed(seed)
    # Seed CPU PyTorch operations.
    torch.manual_seed(seed)
    # Seed every visible CUDA device when a GPU is present.
    torch.cuda.manual_seed_all(seed)
    # Prefer repeatability over cuDNN auto-tuned speed.
    torch.backends.cudnn.benchmark = False
    # Request deterministic cuDNN kernels when available.
    torch.backends.cudnn.deterministic = True


# Choose the GPU if Colab provides one; otherwise keep the notebook functional on CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class B52Reference:
    """What B52 has actually measured, recorded so a run can be read against it.

    These are the real runs, on the real data. They are here to give this
    notebook's numbers some context, and not to be compared with them: this
    notebook trains a fresh compact model on a subset.
    """

    # Name the regime this notebook reproduces.
    experiment: str = "B52 competition full fine-tune"
    # Record what B52 changed, since that is the whole experiment.
    changed: str = "encoder trains, augmentation on, cosine completes, best epoch kept"
    # Record the frozen control B52 is measured against.
    frozen_control_macro_auc: float = 0.763117
    # Record B52 on the gate's training rows, 1,447 studies.
    gate_split_macro_auc: float = 0.802666
    # Record B52 on the full training population, 3,801 studies.
    all_data_macro_auc: float = 0.834998
    # State the surface all three were measured on.
    evaluation: str = "548 unseen-scanner studies, report-derived labels"
    # State plainly what kind of number these are.
    caveat: str = "selection statistics: the best of several epochs on the surface used to pick the epoch"
    # Explain why this notebook's own numbers are not comparable.
    scope: str = "Reference only; this notebook trains fresh compact subset weights."


# Create the immutable reference used by the saved-result functions.
B52_REFERENCE = B52Reference()


# ==========================================================================
# Mount Drive, copy both archives locally, and define the run configuration
# ==========================================================================


@dataclass(frozen=True)
class DrivePaths:
    """Locations for local training data and persistent Drive outputs."""

    # Hold the local folder containing train.csv, train_series.csv, and training DICOM files.
    data_root: Path
    # Hold the Drive folder where this notebook writes persistent new results.
    output_root: Path

    @property
    def train_csv(self) -> Path:
        """Return the path of the study-level CSV file."""
        # Join the dataset root and the fixed study-table filename.
        return self.data_root / "train.csv"

    @property
    def series_csv(self) -> Path:
        """Return the path of the series-level CSV file."""
        # Join the dataset root and the fixed series-table filename.
        return self.data_root / "train_series.csv"


def make_paths(dataset_root: str | Path, output_root: str | Path | None = None) -> DrivePaths:
    """Build training-data and output paths from extracted local data and an optional Drive output folder."""
    # Convert either a string or Path into a Path object.
    root = Path(dataset_root)
    # Use a local outputs folder only when no persistent Drive output folder was supplied.
    results = root / "outputs" if output_root is None else Path(output_root)
    # Return the complete training input and output path bundle.
    return DrivePaths(data_root=root, output_root=results)


@dataclass(frozen=True)
class TestPaths:
    """Locations for the separately extracted test subset."""

    # Hold the local folder containing test.csv, test_series.csv, and test DICOM files.
    data_root: Path

    @property
    def test_csv(self) -> Path:
        """Return the test study table path."""
        # Join the test root and fixed test-table filename.
        return self.data_root / "test.csv"

    @property
    def series_csv(self) -> Path:
        """Return the test series metadata table path."""
        # Join the test root and fixed test-series table filename.
        return self.data_root / "test_series.csv"


def make_test_paths(dataset_root: str | Path) -> TestPaths:
    """Build test-subset paths from the extracted local folder."""
    # Convert either a string or Path into a Path object.
    root = Path(dataset_root)
    # Return the complete test input path bundle.
    return TestPaths(data_root=root)


@dataclass(frozen=True)
class RunConfig:
    """All user-adjustable choices for one standalone training run."""

    # Keep the high-resolution in-plane representation.
    image_size: int = 448
    # Use 32 deterministic slice centers per MRI series.
    slices_per_series: int = 32
    # Use immediate neighbors as the other two 2.5D channels.
    triplet_gap: int = 1
    # Retain the central 90 percent of each native-resolution image.
    crop_fraction: float = 0.90
    # Preserve each cropped matrix's aspect ratio while fitting it into the square canvas.
    resize_policy: str = "aspect_preserving_pad"
    # Use normalized black pixels for the symmetric margins added after resize-to-fit.
    pad_value: float = 0.0
    # Pool local encoder features into a 6 by 6 evidence grid.
    grid_size: int = 6
    # Retain the top eight local evidence tokens for every target.
    top_k: int = 8
    # Use a compact 128-dimensional model for practical Colab memory use.
    feature_dim: int = 128
    # Encode one 448-pixel triplet at a time to minimize the peak activation footprint.
    encoder_chunk_size: int = 1
    # Recompute encoder activations during backward instead of retaining every triplet's activations.
    gradient_checkpointing: bool = True
    # Keep four series by default; set a larger value only after that exact setting passes preflight.
    max_series_per_study: int = 4
    # Keep one study per batch to avoid padding and duplicate CPU allocations.
    batch_size: int = 1
    # Decode in the main process so worker processes cannot multiply host RAM use.
    num_workers: int = 0
    # Cap the deterministic DICOM-pixel sample used for memory-bounded percentile normalization.
    percentile_sample_cap: int = 262_144
    # Reserve twenty percent of usable labelled studies for validation.
    validation_fraction: float = 0.20
    # Two epochs is the inherited default and is the thing B52 replaces; see section 16.
    epochs: int = 2
    # Set the AdamW learning rate.
    learning_rate: float = 1e-4
    # Set the AdamW weight decay.
    weight_decay: float = 1e-4
    # Give the sparse local classifier a direct auxiliary loss.
    local_loss_weight: float = 1.0
    # Limit one optimizer update's gradient norm.
    grad_clip_norm: float = 1.0
    # Save the split and initialization randomness with this seed.
    seed: int = 2026
    # Raise on a bad DICOM when true; otherwise skip just that unreadable series.
    strict_dicom: bool = False


# Create the default conservative training configuration.
CONFIG = RunConfig()


# ==========================================================================
# Validate the tables and create the MRI series index
# ==========================================================================


# Define accepted text values that mean true in the metadata table.
TRUE_TOKENS = {"true", "t", "yes", "y", "1", "1.0"}


# Define accepted text values that mean false in the metadata table.
FALSE_TOKENS = {"false", "f", "no", "n", "0", "0.0"}


def parse_bool(value: object) -> int:
    """Convert a metadata flag to 0 unknown, 1 false, or 2 true."""
    # Preserve missing metadata as the unknown code.
    if pd.isna(value):
        return 0
    # Convert Python and NumPy booleans directly.
    if isinstance(value, (bool, np.bool_)):
        return 2 if bool(value) else 1
    # Convert numeric nonzero values into true and zero values into false.
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 2 if float(value) != 0 else 1
    # Normalize text before checking the accepted tokens.
    text = str(value).strip().lower()
    # Map accepted true tokens to code two.
    if text in TRUE_TOKENS:
        return 2
    # Map accepted false tokens to code one.
    if text in FALSE_TOKENS:
        return 1
    # Treat all other values as unknown.
    return 0


def normalise_plane(value: object) -> str:
    """Map common plane spelling variants to a standard anatomical-plane name."""
    # Define the accepted spelling variants.
    mapping = {
        "sagittal": "Sagittal", "sag": "Sagittal", "sagital": "Sagittal",
        "coronal": "Coronal", "cor": "Coronal",
        "axial": "Axial", "ax": "Axial", "transverse": "Axial",
    }
    # Return an empty string for a plane that cannot be recognized safely.
    return mapping.get(str(value).strip().lower(), "")


def validate_dataset(paths: DrivePaths) -> dict:
    """Validate the local training CSV schema and DICOM layout before using the GPU."""
    # List the two CSV files required by the standalone workflow.
    required_files = (paths.train_csv, paths.series_csv)
    # Collect every missing CSV filename so one error explains the whole problem.
    missing_files = [str(path) for path in required_files if not path.is_file()]
    # Stop before loading data if a required file is absent.
    if missing_files:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing_files))
    # Load the study-level table.
    train = pd.read_csv(paths.train_csv)
    # Load the series-level table.
    series = pd.read_csv(paths.series_csv)
    # Require a study UID and every classification target in the study table.
    train_required = {"StudyInstanceUID", *TARGETS}
    # Require series identifiers and routing metadata in the series table.
    series_required = {
        "StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive",
        "Fat_Suppression", "Anatomical_Plane",
    }
    # Find missing study-table columns.
    missing_train = sorted(train_required.difference(train.columns))
    # Find missing series-table columns.
    missing_series = sorted(series_required.difference(series.columns))
    # Explain an incomplete study table explicitly.
    if missing_train:
        raise ValueError(f"train.csv missing columns: {missing_train}")
    # Explain an incomplete series table explicitly.
    if missing_series:
        raise ValueError(f"train_series.csv missing columns: {missing_series}")
    # Copy tables before normalizing their identifiers.
    train = train.copy()
    # Convert study IDs into strings so numeric-looking UIDs do not lose leading digits.
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    # Convert series study IDs into strings for the same reason.
    series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
    # Convert series IDs into strings for safe directory lookup.
    series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
    # Reject duplicate studies because each study needs exactly one target row.
    if train["StudyInstanceUID"].duplicated().any():
        raise ValueError("train.csv contains duplicate StudyInstanceUID values")
    # Reject duplicate series entries because one directory should map to one metadata row.
    if series[["StudyInstanceUID", "SeriesInstanceUID"]].duplicated().any().any():
        raise ValueError("train_series.csv contains duplicate study/series rows")
    # Convert every target to numeric while preserving blank cells as NaN.
    labels = train[TARGETS].apply(pd.to_numeric, errors="coerce")
    # Record which target cells have a usable CSV label.
    known = labels.notna()
    # Identify labels that are neither zero nor one.
    invalid = known & ~labels.isin([0.0, 1.0])
    # Stop on invalid label values so the loss never silently interprets a bad code.
    if invalid.any().any():
        bad = invalid.sum()[invalid.sum() > 0].to_dict()
        raise ValueError(f"Target values must be 0, 1, or blank; invalid counts: {bad}")
    # Stop if the selected subset contains no supervised cells at all.
    if not known.any().any():
        raise ValueError("The subset has no known target labels in train.csv")
    # Normalize plane text before reporting how many MRI series can be used.
    plane = series["Anatomical_Plane"].map(normalise_plane)
    # Mark rows that have one of the three recognized anatomical planes.
    eligible = plane.isin(PLANE_TO_ID)
    # Accept either original-style DICOM root name.
    roots = [paths.data_root / "train_series", paths.data_root / "train_images"]
    # Stop with a clear path error if neither DICOM root exists.
    if not any(root.is_dir() for root in roots):
        raise FileNotFoundError(
            "Expected train_series/ or train_images/ under " f"{paths.data_root}"
        )
    # Build a concise data audit that is useful to save or screenshot.
    result = {
        "studies": int(len(train)),
        "series_rows": int(len(series)),
        "recognized_plane_series": int(eligible.sum()),
        "studies_with_any_label": int(known.any(axis=1).sum()),
        "known_label_cells": int(known.to_numpy().sum()),
        "data_root": str(paths.data_root),
    }
    # Print the audit in a readable JSON form.
    print(json.dumps(result, indent=2))
    # Return the audit for optional downstream use.
    return result


def validate_test_dataset(paths: TestPaths) -> dict:
    """Validate the extracted test CSV schema and DICOM layout before inference."""
    # List the two test CSV files required for prediction.
    required_files = (paths.test_csv, paths.series_csv)
    # Collect all missing test files for one actionable error message.
    missing_files = [str(path) for path in required_files if not path.is_file()]
    # Stop before model inference if the archive omitted a required file.
    if missing_files:
        raise FileNotFoundError("Missing required test file(s):\n" + "\n".join(missing_files))
    # Load the test study table.
    test = pd.read_csv(paths.test_csv)
    # Load the test series metadata table.
    series = pd.read_csv(paths.series_csv)
    # Require a unique study identifier in the test study table.
    test_required = {"StudyInstanceUID"}
    # Require the same MRI-routing metadata used by the training data.
    series_required = {
        "StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive",
        "Fat_Suppression", "Anatomical_Plane",
    }
    # Find missing test-study columns.
    missing_test = sorted(test_required.difference(test.columns))
    # Find missing test-series columns.
    missing_series = sorted(series_required.difference(series.columns))
    # Explain a malformed test study table.
    if missing_test:
        raise ValueError(f"test.csv missing columns: {missing_test}")
    # Explain a malformed test series table.
    if missing_series:
        raise ValueError(f"test_series.csv missing columns: {missing_series}")
    # Normalize identifiers before checking their uniqueness.
    test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    # Normalize test-series study identifiers.
    series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
    # Normalize test-series identifiers.
    series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
    # Reject duplicate test studies because predictions need one row per study.
    if test["StudyInstanceUID"].duplicated().any():
        raise ValueError("test.csv contains duplicate StudyInstanceUID values")
    # Reject duplicate test series metadata rows.
    if series[["StudyInstanceUID", "SeriesInstanceUID"]].duplicated().any().any():
        raise ValueError("test_series.csv contains duplicate study/series rows")
    # Find recognized anatomical plane metadata rows.
    eligible = series["Anatomical_Plane"].map(normalise_plane).isin(PLANE_TO_ID)
    # Accept either original-style test DICOM root name.
    roots = [paths.data_root / "test_series", paths.data_root / "test_images"]
    # Stop before inference if no test DICOM hierarchy was extracted.
    if not any(root.is_dir() for root in roots):
        raise FileNotFoundError(
            "Expected test_series/ or test_images/ under " f"{paths.data_root}"
        )
    # Build a concise test-subset audit.
    result = {
        "test_studies": int(len(test)),
        "test_series_rows": int(len(series)),
        "test_recognized_plane_series": int(eligible.sum()),
        "test_data_root": str(paths.data_root),
    }
    # Print the test audit in readable JSON.
    print(json.dumps(result, indent=2))
    # Return the test audit for optional later use.
    return result


def build_series_records(series: pd.DataFrame, config: RunConfig) -> dict[str, list[dict]]:
    """Create an ordered list of usable MRI series for each study."""
    # Work on a copy so callers keep their original table unchanged.
    work = series.copy()
    # Normalize study UIDs for dictionary keys.
    work["StudyInstanceUID"] = work["StudyInstanceUID"].astype(str)
    # Normalize series UIDs for directory lookup.
    work["SeriesInstanceUID"] = work["SeriesInstanceUID"].astype(str)
    # Normalize anatomical-plane text.
    work["plane"] = work["Anatomical_Plane"].map(normalise_plane)
    # Map each recognized plane to its categorical code.
    work["plane_id"] = work["plane"].map(PLANE_TO_ID).fillna(0).astype(int)
    # Map fluid sensitivity to its categorical code.
    work["fluid_id"] = work["Fluid_Sensitive"].map(parse_bool).astype(int)
    # Map fat suppression to its categorical code.
    work["fat_id"] = work["Fat_Suppression"].map(parse_bool).astype(int)
    # Discard unrecognized planes because their spatial orientation is unknown.
    work = work.loc[work["plane_id"] > 0].copy()
    # Prepare the final study-to-series mapping.
    result: dict[str, list[dict]] = {}
    # Build one deterministic record list per study.
    for uid, part in work.groupby("StudyInstanceUID", sort=False):
        # Convert selected metadata columns into plain dictionaries.
        rows = [
            {
                "series_uid": str(row.SeriesInstanceUID),
                "plane": str(row.plane),
                "plane_id": int(row.plane_id),
                "fluid_id": int(row.fluid_id),
                "fat_id": int(row.fat_id),
            }
            for row in part.itertuples(index=False)
        ]
        # Keep ordering reproducible and avoid an arbitrary filesystem order.
        rows.sort(
            key=lambda row: (
                row["plane_id"], row["fluid_id"], row["fat_id"], row["series_uid"]
            )
        )
        # Limit the number of series only when the configured limit is positive.
        if config.max_series_per_study > 0:
            rows = rows[: config.max_series_per_study]
        # Store only studies that retain at least one usable series.
        if rows:
            result[str(uid)] = rows
    # Return the mapping consumed by the dataset class.
    return result


# ==========================================================================
# DICOM decoding and 448×448 2.5D preparation
# ==========================================================================


# Accept normal DICOM suffixes and files without a suffix.
DICOM_SUFFIXES = {"", ".dcm", ".dicom", ".ima"}


def find_series_dir(data_root: Path, split: str, study_uid: str, series_uid: str) -> Path | None:
    """Locate one train or test DICOM series in either accepted directory hierarchy."""
    # Reject an unexpected split name before constructing a filesystem path.
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    # Check both accepted DICOM root names in a fixed order for this split.
    for root_name in (f"{split}_series", f"{split}_images"):
        # Build the expected study/series directory path.
        candidate = data_root / root_name / str(study_uid) / str(series_uid)
        # Return immediately when the expected directory exists.
        if candidate.is_dir():
            return candidate
    # Return None when no expected directory exists.
    return None


def dicom_sort_key(dataset) -> float:
    """Use patient-space slice position when available, else use instance number."""
    # Try the geometric DICOM ordering first.
    try:
        # Read the DICOM image position vector.
        position = np.asarray(dataset.ImagePositionPatient, dtype=float)
        # Read the DICOM row and column direction vectors.
        orientation = np.asarray(dataset.ImageOrientationPatient, dtype=float)
        # Project position onto the slice normal to get physical ordering.
        return float(np.dot(position, np.cross(orientation[:3], orientation[3:])))
    # Fall back safely when geometric fields are missing or malformed.
    except Exception:
        # Use InstanceNumber as the fallback ordering key.
        return float(getattr(dataset, "InstanceNumber", 0))


def center_pad_to_shape(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Centre-pad one unexpected mixed-matrix frame without discarding native pixels."""
    # Reject a target smaller than the source because this safety fallback never crops images.
    if image.shape[0] > target_shape[0] or image.shape[1] > target_shape[1]:
        raise ValueError(f"Cannot pad {image.shape} into smaller target {target_shape}")
    # Allocate NaN margins so later normalization can replace only synthetic padding.
    output = np.full(target_shape, np.nan, dtype=np.float32)
    # Keep every original source row.
    rows = image.shape[0]
    # Keep every original source column.
    cols = image.shape[1]
    # Centre the source rows inside the larger common matrix.
    dst_r = (target_shape[0] - rows) // 2
    # Centre the source columns inside the larger common matrix.
    dst_c = (target_shape[1] - cols) // 2
    # Copy all native pixels with no crop and no interpolation.
    output[dst_r : dst_r + rows, dst_c : dst_c + cols] = image
    # Return the padded single frame.
    return output


@dataclass(frozen=True)
class DicomFrameReference:
    """Point to one frame without retaining its DICOM pixel matrix in RAM."""

    # Store the DICOM file that contains this frame.
    path: Path
    # Store the zero-based frame index inside a multi-frame file.
    frame_index: int
    # Store the deterministic physical ordering key.
    sort_key: float
    # Store the header-declared native matrix shape.
    shape: tuple[int, int]


def list_dicom_frame_references(series_dir: Path) -> list[DicomFrameReference]:
    """Read DICOM headers only and return ordered frame references with bounded RAM use."""
    # Import pydicom inside the function so the notebook imports cleanly before installation.
    import pydicom
    # Collect eligible DICOM files in deterministic filename order.
    files = sorted(
        path for path in series_dir.iterdir()
        if path.is_file() and path.suffix.lower() in DICOM_SUFFIXES
    )
    # Prepare a compact list that holds metadata but no image arrays.
    references: list[DicomFrameReference] = []
    # Count header failures for an informative error message.
    failures = 0
    # Read each header without loading its potentially large PixelData element.
    for path in files:
        # Handle a malformed header independently so another valid slice can still be used.
        try:
            # Read only metadata; stop before the image payload to keep host RAM bounded.
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            # Read the geometry or instance-number sorting key.
            key = dicom_sort_key(dataset)
            # Read the header matrix dimensions required for the mixed-matrix safety fallback.
            shape = (int(dataset.Rows), int(dataset.Columns))
            # Read the number of frames while treating an ordinary DICOM as one frame.
            count = int(getattr(dataset, "NumberOfFrames", 1))
            # Reject invalid frame counts before constructing references.
            if count < 1:
                raise RuntimeError(f"Invalid NumberOfFrames={count}")
            # Add one lightweight reference for every frame in this DICOM file.
            for frame_index in range(count):
                # Offset equal-position multi-frame images into a deterministic within-file order.
                references.append(
                    DicomFrameReference(path, frame_index, key + frame_index * 1e-4, shape)
                )
        # Count the bad header and continue scanning the remaining files.
        except Exception:
            failures += 1
    # Stop if no candidate frame had a readable header.
    if not references:
        raise RuntimeError(
            f"No readable DICOM headers in {series_dir} "
            f"({len(files)} files, {failures} header failures)"
        )
    # Sort references along the physical MRI acquisition direction.
    references.sort(key=lambda reference: reference.sort_key)
    # Return only compact metadata, never an in-memory full MRI volume.
    return references


def decode_dicom_frame(reference: DicomFrameReference) -> np.ndarray:
    """Decode exactly one referenced DICOM frame as a native float32 image."""
    # Import pydicom locally so this helper is self-contained in Colab.
    import pydicom
    # Read the one DICOM file that contains the requested frame.
    dataset = pydicom.dcmread(str(reference.path), force=True)
    # Decode its pixel payload; a multi-frame file is still decoded only when encountered.
    decoded = np.asarray(dataset.pixel_array)
    # Select a normal single-frame image when the file contains one two-dimensional matrix.
    if decoded.ndim == 2:
        # Guard against a malformed reference that asks for a non-existent second frame.
        if reference.frame_index != 0:
            raise RuntimeError(f"Single-frame DICOM requested frame {reference.frame_index}")
        # Keep the two-dimensional decoded matrix.
        pixels = decoded
    # Select the requested frame from a three-dimensional multi-frame DICOM.
    elif decoded.ndim == 3:
        # Guard against a DICOM whose header and decoded frame count disagree.
        if reference.frame_index >= decoded.shape[0]:
            raise RuntimeError(f"Multi-frame DICOM has only {decoded.shape[0]} frames")
        # Keep exactly the requested two-dimensional frame.
        pixels = decoded[reference.frame_index]
    # Refuse unsupported DICOM pixel dimensionality explicitly.
    else:
        raise RuntimeError(f"Unsupported decoded DICOM shape: {decoded.shape}")
    # Convert the retained native frame to float32 before intensity rescaling.
    pixels = np.asarray(pixels, dtype=np.float32)
    # Apply the DICOM rescale slope when present.
    pixels *= float(getattr(dataset, "RescaleSlope", 1.0))
    # Apply the DICOM rescale intercept when present.
    pixels += float(getattr(dataset, "RescaleIntercept", 0.0))
    # Invert MONOCHROME1 images so bright tissue remains bright.
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        # Create the conventional bright-tissue orientation without changing the source files.
        pixels = float(np.nanmax(pixels)) - pixels
    # Return a contiguous float32 matrix suitable for NumPy and PyTorch operations.
    return np.ascontiguousarray(pixels, dtype=np.float32)


def deterministic_pixel_sample(image: np.ndarray, count: int) -> np.ndarray:
    """Take a bounded, evenly spaced finite-pixel sample from one native MRI frame."""
    # Flatten only this one frame instead of a whole MRI series.
    flat = np.asarray(image, dtype=np.float32).reshape(-1)
    # Exclude NaNs and infinities before percentile estimation.
    finite = flat[np.isfinite(flat)]
    # Return all values when this frame is already smaller than the requested sample count.
    if finite.size <= count:
        return finite
    # Choose evenly spaced source indices so the sampling is deterministic and reproducible.
    index = np.linspace(0, finite.size - 1, num=count, dtype=np.int64)
    # Return only the bounded representative sample.
    return finite[index]


def streaming_percentile_bounds(
    references: list[DicomFrameReference],
    sample_cap: int,
    retained_indices: set[int],
) -> tuple[float, float, dict[int, np.ndarray]]:
    """Estimate robust series percentiles while retaining only selected native frames."""
    # Reject a nonsensical sample budget before reading any DICOM pixel payload.
    if sample_cap < 1:
        raise ValueError("percentile_sample_cap must be positive")
    # Limit the number of source frames sampled when an unusual series has more frames than the budget.
    sample_count = min(len(references), sample_cap)
    # Choose deterministic frame locations that span the whole acquired series.
    sample_indices = set(
        np.linspace(0, len(references) - 1, num=sample_count, dtype=np.int64).tolist()
    )
    # Allocate the bounded number of pixels contributed by every sampled frame.
    per_frame = max(1, min(4096, sample_cap // max(len(sample_indices), 1)))
    # Keep compact sample fragments rather than a full [frames, height, width] volume.
    samples: list[np.ndarray] = []
    # Keep only the selected frames needed by the later 32 triplets and their neighbors.
    retained_frames: dict[int, np.ndarray] = {}
    # Decode only sampled or later-selected frames one at a time so host RAM stays bounded.
    for index, reference in enumerate(references):
        # Skip frames that neither contribute to normalization nor to a final 2.5D triplet.
        if index not in sample_indices and index not in retained_indices:
            continue
        # Allow an unreadable non-selected frame to be skipped during percentile sampling.
        try:
            # Decode this one native frame.
            frame = decode_dicom_frame(reference)
            # Save its small deterministic intensity sample only when this frame is part of the sample plan.
            if index in sample_indices:
                samples.append(deterministic_pixel_sample(frame, per_frame))
            # Keep the full matrix only when the later 2.5D construction needs it.
            if index in retained_indices:
                retained_frames[index] = frame
            # Release every non-selected image before reading the next DICOM file.
            else:
                del frame
        # Ignore an unreadable frame here; selected frames are checked again before use.
        except Exception:
            continue
    # Stop when no DICOM frame supplied any finite intensity values.
    if not samples:
        raise RuntimeError("No finite DICOM pixels were available for percentile normalization")
    # Join only the bounded representative samples for robust percentile estimation.
    pooled = np.concatenate(samples).astype(np.float32, copy=False)
    # Stop when all decoded images happened to contain only non-finite values.
    if pooled.size == 0:
        raise RuntimeError("DICOM frames contained no finite pixels")
    # Estimate the familiar 1st and 99th percentile bounds without a full-volume allocation.
    low, high = np.percentile(pooled, [1, 99])
    # Ensure a constant-valued series still has a valid nonzero normalization range.
    high = max(float(high), float(low) + 1e-6)
    # Return scalar bounds plus the few native frames that must be reused for triplets.
    return float(low), float(high), retained_frames


def normalize_native_frame(
    frame: np.ndarray,
    low: float,
    high: float,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Normalize one selected native frame using the series-level streaming bounds."""
    # Pad an unexpected mixed matrix only after percentiles were estimated from real pixels.
    if frame.shape != target_shape:
        frame = center_pad_to_shape(frame, target_shape)
    # Replace synthetic NaNs and unexpected non-finite values with the nearest valid bound.
    normalized = np.nan_to_num(frame, nan=low, posinf=high, neginf=low, copy=True)
    # Move the lower robust intensity bound to zero.
    normalized -= low
    # Scale the robust intensity interval into the zero-to-one range.
    normalized /= max(high - low, 1e-6)
    # Clip remaining outliers into the same normalized support.
    np.clip(normalized, 0.0, 1.0, out=normalized)
    # Return a contiguous float32 native image.
    return np.ascontiguousarray(normalized, dtype=np.float32)


def sample_centers(n_frames: int, n_samples: int, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic slice centres and normalized through-plane positions."""
    # Reject invalid sampling settings before calculating slice indices.
    if n_frames < 1 or n_samples < 1 or gap < 1:
        raise ValueError("frames, samples, and gap must all be positive")
    # Avoid edge centers when the series is long enough for a full 2.5D neighborhood.
    low, high = (gap, n_frames - 1 - gap) if n_frames > 2 * gap else (0, n_frames - 1)
    # Spread the requested centers evenly across the usable MRI range.
    centres = np.round(np.linspace(low, high, n_samples)).astype(np.int64)
    # Normalize the chosen centers into the zero-to-one through-plane coordinate range.
    positions = centres.astype(np.float32) / float(max(n_frames - 1, 1))
    # Return both integer frame indices and continuous positions.
    return centres, positions


def native_center_crop(triplets: np.ndarray, fraction: float) -> np.ndarray:
    """Crop every 2.5D triplet before its one high-resolution resize-to-fit."""
    # Validate the requested retained image fraction.
    if not 0 < fraction <= 1:
        raise ValueError("crop_fraction must be in the interval (0, 1]")
    # Read the native height and width from the final two array dimensions.
    height, width = triplets.shape[-2:]
    # Round the requested native crop height while keeping at least two pixels.
    crop_h = max(2, min(height, int(round(height * fraction))))
    # Round the requested native crop width while keeping at least two pixels.
    crop_w = max(2, min(width, int(round(width * fraction))))
    # Calculate the centered top edge.
    top = (height - crop_h) // 2
    # Calculate the centered left edge.
    left = (width - crop_w) // 2
    # Return the centered native-resolution crop.
    return triplets[..., top : top + crop_h, left : left + crop_w]


def resize_triplets_aspect_preserving_pad(
    triplets: np.ndarray,
    image_size: int,
    pad_value: float,
) -> torch.Tensor:
    """Resize a [triplets, channels, H, W] batch once, then centre-pad it square."""
    # Require the expected 2.5D tensor layout before reading spatial dimensions.
    if triplets.ndim != 4:
        raise ValueError(f"Expected [triplets, channels, height, width], got {triplets.shape}")
    # Require a useful positive target canvas.
    if image_size < 2:
        raise ValueError("image_size must be at least two pixels")
    # Read the retained native crop dimensions.
    height, width = triplets.shape[-2:]
    # Choose one common scale so neither resized side exceeds the square canvas.
    scale = min(image_size / float(height), image_size / float(width))
    # Round the fitted height while keeping it inside the target canvas.
    resized_h = max(1, min(image_size, int(round(height * scale))))
    # Round the fitted width while keeping it inside the target canvas.
    resized_w = max(1, min(image_size, int(round(width * scale))))
    # Convert the contiguous crop into the [batch, channels, H, W] PyTorch layout.
    tensor = torch.from_numpy(np.ascontiguousarray(triplets))
    # Resize every triplet once with antialiased bilinear interpolation.
    resized = F.interpolate(
        tensor,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    # Allocate the fixed square model canvas using normalized black margins.
    output = resized.new_full(
        (resized.shape[0], resized.shape[1], image_size, image_size),
        float(pad_value),
    )
    # Calculate the symmetric vertical margin before copying the resized crop.
    top = (image_size - resized_h) // 2
    # Calculate the symmetric horizontal margin before copying the resized crop.
    left = (image_size - resized_w) // 2
    # Copy the complete resized crop into the centre without a second interpolation.
    output[..., top : top + resized_h, left : left + resized_w] = resized
    # Return the fixed 448-by-448-style tensor with the retained aspect ratio intact.
    return output


def prepare_series_tensor_from_dicom(
    series_dir: Path,
    config: RunConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream one series into aspect-preserving [slices, 3, 448, 448] triplets."""
    # Reject an accidental return to direct square stretching before any expensive decoding.
    if config.resize_policy != "aspect_preserving_pad":
        raise ValueError(
            "resize_policy must be 'aspect_preserving_pad' for this native-geometry notebook"
        )
    # List DICOM frame metadata without retaining a full MRI volume in host memory.
    references = list_dicom_frame_references(series_dir)
    # Select deterministic centers and their normalized slice positions from header counts.
    centres, positions = sample_centers(
        len(references), config.slices_per_series, config.triplet_gap
    )
    # Define the previous, central, and next frame offsets for each 2.5D input.
    offsets = np.asarray([-config.triplet_gap, 0, config.triplet_gap], dtype=np.int64)
    # Clip all neighbor indices to valid DICOM frame locations.
    index = np.clip(centres[:, None] + offsets[None, :], 0, len(references) - 1)
    # Record the unique frames that the final 32 triplets will need.
    retained_indices = {int(value) for value in index.reshape(-1)}
    # Estimate series normalization bounds while caching only the selected native frames.
    low, high, cache = streaming_percentile_bounds(
        references,
        config.percentile_sample_cap,
        retained_indices,
    )
    # Build a safety shape only for unexpected mixed-matrix series.
    target_shape = (
        max(reference.shape[0] for reference in references),
        max(reference.shape[1] for reference in references),
    )
    # Allocate only the final fixed-size image stack, not a full native-resolution volume.
    images = torch.empty(
        config.slices_per_series,
        3,
        config.image_size,
        config.image_size,
        dtype=torch.float32,
    )
    # Build one native triplet and one 448-pixel output at a time.
    for output_index, frame_indices in enumerate(index):
        # Prepare the three normalized native frames for this one 2.5D triplet.
        channels: list[np.ndarray] = []
        # Decode or reuse each of the previous, center, and next source frames.
        for frame_index in frame_indices:
            # Convert NumPy's integer type into a normal dictionary key.
            frame_index = int(frame_index)
            # Reuse the bounded cache when this frame was retained during percentile sampling.
            frame = cache.get(frame_index)
            # Decode a selected frame again only when it failed to enter the cache.
            if frame is None:
                frame = decode_dicom_frame(references[frame_index])
            # Normalize this one frame using the same series-level robust intensity bounds.
            channels.append(normalize_native_frame(frame, low, high, target_shape))
        # Stack only the three native channels required for this output triplet.
        triplet = np.stack(channels, axis=0)[None, ...]
        # Crop the triplet in native pixels before its one permitted high-resolution resize.
        cropped = native_center_crop(triplet, config.crop_fraction)
        # Resize-to-fit and center-pad this one triplet into the fixed model canvas.
        resized = resize_triplets_aspect_preserving_pad(
            cropped,
            config.image_size,
            config.pad_value,
        )
        # Copy the completed triplet into its final compact output slot.
        images[output_index].copy_(resized[0])
        # Release the temporary native arrays before the next triplet is constructed.
        del channels, triplet, cropped, resized
    # Drop cached native matrices before the caller starts processing the next MRI series.
    cache.clear()
    # Ask Python to collect temporary DICOM and NumPy objects before returning.
    gc.collect()
    # Return the final 448-pixel image stack and its through-plane coordinates.
    return images, torch.from_numpy(positions)


# ==========================================================================
# Dataset and batch collation classes
# ==========================================================================


class KneeMRIDataset(Dataset):
    """Decode one train or test study as a variable number of high-resolution MRI series."""

    def __init__(
        self,
        frame: pd.DataFrame,
        series_records: dict[str, list[dict]],
        paths: DrivePaths | TestPaths,
        config: RunConfig,
        split: str,
        include_targets: bool,
    ) -> None:
        # Store file locations for lazy DICOM loading.
        self.paths = paths
        # Store image and error-handling settings.
        self.config = config
        # Store whether DICOM folders and CSV rows belong to the train or test subset.
        self.split = split
        # Store whether this split carries known training labels.
        self.include_targets = include_targets
        # Copy rows so filtering cannot mutate a caller's data frame.
        work = frame.copy()
        # Normalize UID strings to match the series-record dictionary keys.
        work["StudyInstanceUID"] = work["StudyInstanceUID"].astype(str)
        # Keep only studies that have at least one recognized-plane MRI series.
        work = work.loc[work["StudyInstanceUID"].isin(series_records)].reset_index(drop=True)
        # Stop with an actionable error when no usable study remains.
        if work.empty:
            raise ValueError("No studies have a recognized-plane MRI series")
        # Store deterministic study order for DataLoader indexing.
        self.study_uids = work["StudyInstanceUID"].tolist()
        # Store training targets as float32 only for the labelled training split.
        self.targets = (
            work[TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
            if self.include_targets else None
        )
        # Store the metadata records used to find and describe each MRI series.
        self.series_records = series_records

    def __len__(self) -> int:
        """Return the number of studies in this split."""
        # Let PyTorch know how many valid integer indices exist.
        return len(self.study_uids)

    def _zero_series(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Create a shape-compatible placeholder for one unreadable series."""
        # Make a zero image tensor with the same shape as a prepared MRI series.
        images = torch.zeros(
            self.config.slices_per_series,
            3,
            self.config.image_size,
            self.config.image_size,
            dtype=torch.float32,
        )
        # Make zero positions for the unreadable placeholder.
        positions = torch.zeros(self.config.slices_per_series, dtype=torch.float32)
        # Return the placeholder with present=0 so the model masks it out.
        return images, positions, 0.0

    def _load_series(self, study_uid: str, record: dict) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Read and preprocess one MRI series, or return a masked placeholder."""
        # Locate the expected DICOM directory for this study and series.
        series_dir = find_series_dir(
            self.paths.data_root, self.split, study_uid, record["series_uid"]
        )
        # Handle a completely missing series directory.
        if series_dir is None:
            # Raise immediately only when strict DICOM behavior is requested.
            if self.config.strict_dicom:
                raise FileNotFoundError(f"Missing series {study_uid}/{record['series_uid']}")
            # Otherwise let the model ignore a zero placeholder.
            return self._zero_series()
        # Try to decode and prepare the available DICOM directory.
        try:
            # Stream DICOM frames into high-resolution triplets without retaining a full native volume.
            images, positions = prepare_series_tensor_from_dicom(series_dir, self.config)
            # Mark this series readable so the model uses it.
            return images, positions, 1.0
        # Handle a DICOM decode or preprocessing failure.
        except Exception:
            # Raise the original error in strict mode to find the bad input quickly.
            if self.config.strict_dicom:
                raise
            # Otherwise ignore only this series and continue with the study.
            return self._zero_series()

    def __getitem__(self, index: int) -> dict:
        """Load one variable-series study on demand."""
        # Resolve the study identifier from the dataset's deterministic order.
        study_uid = self.study_uids[index]
        # Prepare lists for every real or placeholder MRI series.
        images, positions, present, metadata = [], [], [], []
        # Load every selected series in this study.
        for record in self.series_records[study_uid]:
            # Decode one series and obtain its readable flag.
            image, position, readable = self._load_series(study_uid, record)
            # Append the prepared 2.5D image stack.
            images.append(image)
            # Append matching through-plane positions.
            positions.append(position)
            # Append the model mask flag.
            present.append(readable)
            # Append plane, fluid, and fat-suppression categorical metadata.
            metadata.append([record["plane_id"], record["fluid_id"], record["fat_id"]])
        # Stop if no selected MRI series could be decoded for this study.
        if not any(present):
            raise RuntimeError(f"Study {study_uid} has no readable MRI series")
        # Return tensors with a variable first dimension equal to the series count.
        item = {
            "study_uid": study_uid,
            "volumes": torch.stack(images),
            "slice_position": torch.stack(positions),
            "present": torch.tensor(present, dtype=torch.float32),
            "series_meta": torch.tensor(metadata, dtype=torch.long),
        }
        # Attach labels only when this is the labelled training subset.
        if self.targets is not None:
            item["target"] = torch.from_numpy(self.targets[index])
        # Return the train or test sample dictionary.
        return item


def collate_studies(batch: list[dict]) -> dict:
    """Pad only the variable series dimension when combining studies into a batch."""
    # Reject the impossible empty batch case explicitly.
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    # Use a zero-copy unsqueeze path for the safe default batch size of one.
    if len(batch) == 1:
        # Extract the only sample dictionary.
        item = batch[0]
        # Add a batch dimension without duplicating the large image tensor.
        result = {
            "study_uid": [item["study_uid"]],
            "volumes": item["volumes"].unsqueeze(0),
            "slice_position": item["slice_position"].unsqueeze(0),
            "present": item["present"].unsqueeze(0),
            "series_meta": item["series_meta"].unsqueeze(0),
        }
        # Add a label tensor only for a labelled training or validation sample.
        if "target" in item:
            result["target"] = item["target"].unsqueeze(0)
        # Return the memory-safe one-study batch.
        return result
    # Find the largest series count in this multi-study batch.
    max_series = max(int(item["present"].shape[0]) for item in batch)
    # Read image shape information from the first study.
    first = batch[0]["volumes"]
    # Read the actual batch size.
    size = len(batch)
    # Unpack one series tensor shape.
    _, slices, channels, height, width = first.shape
    # Allocate padded image storage.
    volumes = first.new_zeros((size, max_series, slices, channels, height, width))
    # Allocate padded position storage.
    positions = torch.zeros((size, max_series, slices), dtype=torch.float32)
    # Allocate padded readable-series flags.
    present = torch.zeros((size, max_series), dtype=torch.float32)
    # Allocate padded categorical metadata.
    metadata = torch.zeros((size, max_series, 3), dtype=torch.long)
    # Copy each study's variable number of series into the padded tensors.
    for row, item in enumerate(batch):
        # Read this study's real series count.
        count = int(item["present"].shape[0])
        # Copy image triplets.
        volumes[row, :count] = item["volumes"]
        # Copy through-plane positions.
        positions[row, :count] = item["slice_position"]
        # Copy readable-series flags.
        present[row, :count] = item["present"]
        # Copy categorical metadata.
        metadata[row, :count] = item["series_meta"]
    # Build a normal multi-study batch dictionary.
    result = {
        "study_uid": [item["study_uid"] for item in batch],
        "volumes": volumes,
        "slice_position": positions,
        "present": present,
        "series_meta": metadata,
    }
    # Stack labels only if every sample in the batch provides them.
    if all("target" in item for item in batch):
        result["target"] = torch.stack([item["target"] for item in batch])
    # Return the padded train or test batch.
    return result


# ==========================================================================
# Model classes: encoder, sparse evidence head, and study model
# ==========================================================================


class ConvNormAct(nn.Module):
    """A convolution, group-normalization, and GELU activation block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        # Initialize the parent PyTorch module.
        super().__init__()
        # Choose a group count that divides each channel count used below.
        groups = max(1, min(8, out_channels // 8))
        # Build the spatial feature-extraction sequence.
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation."""
        # Pass images through the three-layer block.
        return self.layers(x)


class ResidualBlock(nn.Module):
    """A compact residual block that refines features without changing resolution."""

    def __init__(self, channels: int) -> None:
        # Initialize the parent PyTorch module.
        super().__init__()
        # Choose a valid group-normalization group count.
        groups = max(1, min(8, channels // 8))
        # Build the residual transform branch.
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        # Create the activation after adding the skip connection.
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the learned refinement to the input and activate it."""
        # Apply the residual branch, add the identity path, and activate the sum.
        return self.activation(x + self.block(x))


class SliceEncoder(nn.Module):
    """Encode one 448×448 triplet into global and 6×6 local image features."""

    def __init__(self, feature_dim: int, grid_size: int) -> None:
        # Initialize the parent PyTorch module.
        super().__init__()
        # Remember the final feature size for output-shape checks.
        self.feature_dim = int(feature_dim)
        # Remember the requested square local-grid dimension.
        self.grid_size = int(grid_size)
        # Downsample 448 to 224 while creating initial features.
        self.stem = ConvNormAct(3, 32, stride=2)
        # Downsample 224 to 112 and refine features.
        self.stage1 = nn.Sequential(ConvNormAct(32, 48, stride=2), ResidualBlock(48))
        # Downsample 112 to 56 and refine features.
        self.stage2 = nn.Sequential(ConvNormAct(48, 72, stride=2), ResidualBlock(72))
        # Downsample 56 to 28 and refine features.
        self.stage3 = nn.Sequential(ConvNormAct(72, 96, stride=2), ResidualBlock(96))
        # Downsample 28 to 14 and produce final image features.
        self.stage4 = nn.Sequential(ConvNormAct(96, self.feature_dim, stride=2), ResidualBlock(self.feature_dim))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one global vector and one local feature map for each triplet."""
        # Apply every encoder stage in spatial order.
        features = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(images)))))
        # Average the full feature map into one global image representation.
        global_feature = F.adaptive_avg_pool2d(features, 1).flatten(1)
        # Pool the same feature map into the requested local evidence grid.
        local_feature = F.adaptive_avg_pool2d(features, self.grid_size)
        # Return both complementary representations.
        return global_feature, local_feature


def position_basis(position: torch.Tensor) -> torch.Tensor:
    """Create an eight-dimensional continuous representation of slice position."""
    # Clamp positions defensively into the valid zero-to-one range.
    z = position.float().clamp(0.0, 1.0)
    # Stack polynomial and sinusoidal position features along the last axis.
    return torch.stack(
        [
            z,
            z.square(),
            torch.sin(math.pi * z),
            torch.cos(math.pi * z),
            torch.sin(2 * math.pi * z),
            torch.cos(2 * math.pi * z),
            torch.sin(4 * math.pi * z),
            torch.cos(4 * math.pi * z),
        ],
        dim=-1,
    )


@dataclass
class SparseMILOutput:
    """Store combined, global-only, and local-only logits from the model."""

    # Hold the fused prediction logits used for case classifications.
    logits: torch.Tensor
    # Hold logits from global study-level image features.
    global_logits: torch.Tensor
    # Hold logits from sparse local evidence features.
    local_logits: torch.Tensor


class SparseEvidenceHead(nn.Module):
    """Score local MRI tokens per target and pool only the strongest top-k tokens."""

    def __init__(self, feature_dim: int, grid_size: int, top_k: int) -> None:
        # Initialize the parent PyTorch module.
        super().__init__()
        # Store the local feature-channel count.
        self.feature_dim = int(feature_dim)
        # Store the local grid side length.
        self.grid_size = int(grid_size)
        # Store the number of strongest tokens retained per target.
        self.top_k = int(top_k)
        # Compute the number of local regions per slice.
        self.n_regions = self.grid_size * self.grid_size
        # Learn how continuous slice coordinates affect local evidence features.
        self.position_projection = nn.Linear(8, self.feature_dim, bias=False)
        # Learn an embedding for each two-dimensional local region.
        self.region_embedding = nn.Parameter(torch.zeros(self.n_regions, self.feature_dim))
        # Learn a small embedding for anatomical plane.
        self.plane_embedding = nn.Embedding(4, self.feature_dim, padding_idx=0)
        # Learn a small embedding for fluid sensitivity.
        self.fluid_embedding = nn.Embedding(3, self.feature_dim, padding_idx=0)
        # Learn a small embedding for fat suppression.
        self.fat_embedding = nn.Embedding(3, self.feature_dim, padding_idx=0)
        # Learn one evidence direction for every target.
        self.evidence_weight = nn.Parameter(torch.empty(N_TARGETS, self.feature_dim))
        # Learn one evidence bias for every target.
        self.evidence_bias = nn.Parameter(torch.zeros(N_TARGETS))
        # Initialize target evidence directions with small nonzero values.
        nn.init.normal_(self.evidence_weight, mean=0.0, std=0.02)

    def forward(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> torch.Tensor:
        """Return one sparse pooled local logit per target for every study."""
        # Unpack the local-feature tensor shape.
        batch, series, slices, regions, feature_dim = spatial.shape
        # Check that the model and input grid definitions agree.
        if regions != self.n_regions or feature_dim != self.feature_dim:
            raise ValueError("Sparse feature shape does not match the head")
        # Check that the readable-series mask corresponds to the same studies and series.
        if present.shape != (batch, series):
            raise ValueError("present mask shape mismatch")
        # Normalize every local token feature independently.
        tokens = F.layer_norm(spatial.float(), (feature_dim,)).to(spatial.dtype)
        # Project continuous slice coordinates into feature space.
        position = self.position_projection(position_basis(slice_position)).to(tokens.dtype)
        # Look up the three categorical series metadata embeddings.
        metadata = (
            self.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
            + self.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
            + self.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        ).to(tokens.dtype)
        # Add through-plane position information to every local region in a slice.
        tokens = tokens + position[:, :, :, None, :]
        # Add series-level metadata to every slice and region in that series.
        tokens = tokens + metadata[:, :, None, None, :]
        # Add the learned two-dimensional region embedding.
        tokens = tokens + self.region_embedding.to(tokens.dtype)[None, None, None, :, :]
        # Flatten series, slice, and region into a single candidate-token dimension.
        tokens = tokens.reshape(batch, series * slices * regions, feature_dim)
        # Expand the readable-series mask so every token from an unreadable series is invalid.
        invalid = (
            (present <= 0)[:, :, None, None]
            .expand(batch, series, slices, regions)
            .reshape(batch, series * slices * regions)
        )
        # Count valid local tokens in every study.
        valid_count = (~invalid).sum(dim=1)
        # Require enough valid tokens for the requested top-k operation.
        if int(valid_count.min().item()) < self.top_k:
            raise RuntimeError("There are fewer valid local-MIL tokens than top_k")
        # Compute every target's evidence score at every local token.
        score = torch.einsum("btd,nd->bnt", tokens, self.evidence_weight.to(tokens.dtype))
        # Add one target-specific scalar bias to every token score.
        score = score + self.evidence_bias.to(tokens.dtype)[None, :, None]
        # Exclude tokens from unreadable series from the top-k selection.
        score = score.masked_fill(invalid[:, None, :], float("-inf"))
        # Keep the strongest evidence values per target and study.
        top_values = torch.topk(score, k=self.top_k, dim=-1).values
        # Smoothly pool selected values while preserving their logit scale.
        return torch.logsumexp(top_values.float(), dim=-1) - math.log(float(self.top_k))


class HighResolutionSparseMIL(nn.Module):
    """Train a global and sparse-local MRI classifier from randomly initialized weights."""

    def __init__(self, config: RunConfig) -> None:
        # Initialize the parent PyTorch module.
        super().__init__()
        # Store configuration because encoding is chunked using its memory setting.
        self.config = config
        # Create the image encoder shared by global and local branches.
        self.encoder = SliceEncoder(config.feature_dim, config.grid_size)
        # Transform averaged global study features before classification.
        self.global_projection = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        # Map global study features to one logit per target.
        self.global_classifier = nn.Linear(config.feature_dim, N_TARGETS)
        # Create the target-specific sparse local evidence branch.
        self.sparse_head = SparseEvidenceHead(config.feature_dim, config.grid_size, config.top_k)
        # Start fusion at global-only and let training learn target-wise local contribution.
        self.fusion_gate = nn.Parameter(torch.zeros(N_TARGETS))

    def _encode_active_series(
        self, volumes: torch.Tensor, present: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode only readable series and restore them to padded study shapes."""
        # Unpack the padded high-resolution batch shape.
        batch, series, slices, channels, height, width = volumes.shape
        # Verify that 2.5D inputs still have their three intended channels.
        if channels != 3:
            raise ValueError("The model expects 2.5D triplets with three channels")
        # Flatten batch and series so the readable-series mask can select complete series.
        flat_series = volumes.reshape(batch * series, slices, channels, height, width)
        # Find the flattened rows that represent readable MRI series.
        active_index = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        # Stop if every series in a batch is unreadable.
        if active_index.numel() == 0:
            raise RuntimeError("The batch has no readable MRI series")
        # Keep only real MRI series before expensive image encoding.
        active = flat_series.index_select(0, active_index)
        # Flatten active series and their 32 slice triplets into image rows.
        images = active.reshape(-1, channels, height, width)
        # Prepare lists to hold chunk-wise global features.
        global_blocks = []
        # Prepare lists to hold chunk-wise local grids.
        local_blocks = []
        # Encode only a few 448-pixel images at a time to control memory use.
        for image_chunk in images.split(self.config.encoder_chunk_size, dim=0):
            # Recompute encoder activations during backward when training to reduce GPU memory.
            if self.training and self.config.gradient_checkpointing:
                # Use non-reentrant checkpointing so encoder parameters receive gradients even from input images.
                global_feature, local_feature = checkpoint(
                    self.encoder,
                    image_chunk,
                    use_reentrant=False,
                )
            # Use the ordinary direct encoder path during evaluation or when checkpointing is disabled.
            else:
                # Encode this chunk into one global vector and one local grid per image.
                global_feature, local_feature = self.encoder(image_chunk)
            # Remember the global vectors for concatenation.
            global_blocks.append(global_feature)
            # Remember the local grids for concatenation.
            local_blocks.append(local_feature)
        # Restore global vectors to [active_series, slices, feature_dim].
        global_active = torch.cat(global_blocks, dim=0).reshape(
            active.shape[0], slices, self.config.feature_dim
        )
        # Restore local maps to [active_series, slices, feature_dim, grid, grid].
        local_active = torch.cat(local_blocks, dim=0).reshape(
            active.shape[0], slices, self.config.feature_dim,
            self.config.grid_size, self.config.grid_size,
        )
        # Allocate padded global features for every batch/series row.
        global_all = global_active.new_zeros((batch * series, slices, self.config.feature_dim))
        # Insert active global features into their original padded positions.
        global_all.index_copy_(0, active_index, global_active)
        # Allocate padded local features for every batch/series row.
        local_all = local_active.new_zeros(
            (batch * series, slices, self.config.feature_dim,
             self.config.grid_size, self.config.grid_size)
        )
        # Insert active local features into their original padded positions.
        local_all.index_copy_(0, active_index, local_active)
        # Restore global vectors to [batch, series, slices, feature_dim].
        global_feature = global_all.reshape(batch, series, slices, self.config.feature_dim)
        # Restore and reorder local maps to [batch, series, slices, regions, feature_dim].
        spatial_feature = local_all.reshape(
            batch, series, slices, self.config.feature_dim,
            self.config.grid_size, self.config.grid_size,
        ).permute(0, 1, 2, 4, 5, 3).reshape(
            batch, series, slices, self.config.grid_size * self.config.grid_size,
            self.config.feature_dim,
        )
        # Return the global and local representations.
        return global_feature, spatial_feature

    def forward(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> SparseMILOutput:
        """Classify one padded batch of variable-series MRI studies."""
        # Encode all readable MRI slice triplets.
        global_feature, spatial_feature = self._encode_active_series(volumes, present)
        # Expand the readable-series mask across slices and features.
        mask = present[:, :, None, None].to(global_feature.dtype)
        # Sum only global features from readable MRI series and slices.
        study_feature = (global_feature * mask).sum(dim=(1, 2))
        # Count the readable series and all 32 sampled slice features in each study.
        feature_count = present.sum(dim=1, keepdim=True).to(global_feature.dtype)
        # Multiply readable-series count by the number of slices to form a true feature mean.
        feature_count = feature_count * float(global_feature.shape[2])
        # Divide by the readable slice-feature count to form the global study mean.
        study_feature = study_feature / feature_count.clamp_min(1.0)
        # Produce logits from the global study representation.
        global_logits = self.global_classifier(self.global_projection(study_feature))
        # Produce logits from top-k sparse local evidence.
        local_logits = self.sparse_head(spatial_feature, present, series_meta, slice_position)
        # Fuse local logits through a target-wise gate initialized at zero.
        logits = global_logits + torch.tanh(self.fusion_gate)[None, :] * local_logits
        # Return all three logit forms so training can supervise both branches.
        return SparseMILOutput(logits=logits, global_logits=global_logits, local_logits=local_logits)


# ==========================================================================
# Loss, metrics, preflight, training, and plotting functions
# ==========================================================================


def binary_auc(target: np.ndarray, probability: np.ndarray) -> float | None:
    """Compute AUC without an additional package; return None when one class is absent."""
    # Convert labels into integer binary values.
    target = np.asarray(target, dtype=np.int64)
    # Convert probabilities into a stable floating-point array.
    probability = np.asarray(probability, dtype=np.float64)
    # Count positive examples.
    positive = int(target.sum())
    # Count negative examples.
    negative = int(len(target) - positive)
    # AUC is undefined when either class is absent in a small validation subset.
    if positive == 0 or negative == 0:
        return None
    # Sort probabilities stably so equal predictions can receive tied ranks.
    order = np.argsort(probability, kind="mergesort")
    # Allocate a rank array in original input order.
    ranks = np.empty(len(probability), dtype=np.float64)
    # Fill one-based ranks in sorted order.
    ranks[order] = np.arange(1, len(probability) + 1, dtype=np.float64)
    # Read sorted probabilities for tie detection.
    sorted_probability = probability[order]
    # Start scanning the first tie group.
    start = 0
    # Process every contiguous tie group.
    while start < len(sorted_probability):
        # Start with a group containing one element.
        end = start + 1
        # Extend the group while probabilities are exactly tied.
        while end < len(sorted_probability) and sorted_probability[end] == sorted_probability[start]:
            end += 1
        # Replace tied ranks with their mean rank.
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
        # Continue at the next tie group.
        start = end
    # Return the Mann-Whitney form of ROC AUC.
    return float((ranks[target == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def move_model_inputs(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move model inputs to the selected device for either train or test batches."""
    # Keep non-blocking transfers disabled because the loader intentionally avoids pinned memory.
    non_blocking = False
    # Return only the tensors required by the MRI model forward pass.
    return (
        batch["volumes"].to(DEVICE, non_blocking=non_blocking),
        batch["present"].to(DEVICE, non_blocking=non_blocking),
        batch["series_meta"].to(DEVICE, non_blocking=non_blocking),
        batch["slice_position"].to(DEVICE, non_blocking=non_blocking),
    )


def move_batch(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move one labelled training batch to the selected device without pinned-memory pressure."""
    # Move the four MRI model-input tensors first.
    volumes, present, metadata, position = move_model_inputs(batch)
    # Move the labelled target matrix required by the training loss.
    target = batch["target"].to(DEVICE, non_blocking=False)
    # Return tensors in the order expected by the model and loss functions.
    return (
        volumes,
        present,
        metadata,
        position,
        target,
    )


def autocast_context():
    """Use fp16 mixed precision on CUDA and ordinary precision on CPU."""
    # Create CUDA fp16 autocasting only when a CUDA GPU exists.
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    # Return a no-op context manager for CPU execution.
    return nullcontext()


@dataclass
class Experiment:
    """Hold all newly created objects for one standalone subset run."""

    # Keep the Drive paths used to read inputs and write results.
    paths: DrivePaths
    # Keep the exact image, model, and optimization settings.
    config: RunConfig
    # Keep the newly initialized high-resolution sparse-MIL model.
    model: HighResolutionSparseMIL
    # Keep the optimizer for this new model.
    optimizer: torch.optim.Optimizer
    # Keep the mixed-precision gradient scaler.
    scaler: object
    # Keep the training DataLoader.
    train_loader: DataLoader
    # Keep the optional validation DataLoader.
    validation_loader: DataLoader | None
    # Keep target-wise positive class weights derived from training rows.
    positive_weight: torch.Tensor
    # Keep one summary dictionary per completed epoch.
    history: list[dict] = field(default_factory=list)


def build_test_loader(paths: TestPaths, config: RunConfig = CONFIG) -> DataLoader:
    """Build a no-label DataLoader from the extracted test subset."""
    # Validate the test tables and DICOM hierarchy before creating a loader.
    validate_test_dataset(paths)
    # Read the test study table.
    test_table = pd.read_csv(paths.test_csv)
    # Read the test series metadata table.
    series_table = pd.read_csv(paths.series_csv)
    # Normalize study identifiers for record matching.
    test_table["StudyInstanceUID"] = test_table["StudyInstanceUID"].astype(str)
    # Build deterministic test-series record lists using the same memory limit as training.
    records = build_series_records(series_table, config)
    # Keep test studies that have at least one recognized-plane MRI metadata record.
    test_table = test_table.loc[test_table["StudyInstanceUID"].isin(records)].reset_index(drop=True)
    # Stop before inference if no test study has a usable MRI metadata record.
    if test_table.empty:
        raise ValueError("No test studies remain after matching test CSV rows to MRI metadata")
    # Create a dataset that omits target tensors because test labels are unavailable.
    test_dataset = KneeMRIDataset(
        test_table, records, paths, config, split="test", include_targets=False
    )
    # Build a memory-safe, deterministic test loader.
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
        collate_fn=collate_studies,
    )
    # Report the usable test-study count.
    print(f"test studies={len(test_dataset)}")
    # Return the ready-to-run prediction loader.
    return test_loader


def predict_test_set(experiment: Experiment, test_loader: DataLoader) -> pd.DataFrame:
    """Run the new model on every extracted test study and return probability columns."""
    # Put the model in evaluation mode so dropout is disabled.
    experiment.model.eval()
    # Prepare one table fragment per test batch.
    fragments: list[pd.DataFrame] = []
    # Disable gradients because test prediction never updates the model.
    with torch.no_grad():
        # Process every test batch in deterministic study order.
        for batch in test_loader:
            # Preserve study identifiers on CPU for the prediction table.
            study_uids = list(batch["study_uid"])
            # Move only model-input tensors because the test batch has no targets.
            volumes, present, metadata, position = move_model_inputs(batch)
            # Run one no-update mixed-precision inference pass.
            with autocast_context():
                output = experiment.model(volumes, present, metadata, position)
            # Convert the fused logits into CPU probabilities.
            probability = torch.sigmoid(output.logits).float().cpu().numpy()
            # Build one prediction table for this batch using the original target column names.
            fragment = pd.DataFrame(probability, columns=TARGETS)
            # Insert study IDs as the first column.
            fragment.insert(0, "StudyInstanceUID", study_uids)
            # Add a thresholded human-readable classification summary.
            fragment["predicted_positive"] = [
                format_positive_predictions(row) for row in probability
            ]
            # Keep the completed batch table.
            fragments.append(fragment)
            # Release large GPU references before decoding the next test batch.
            del batch, volumes, present, metadata, position, output
    # Join all batch fragments into one test prediction table.
    predictions = pd.concat(fragments, ignore_index=True)
    # Print the number of generated test predictions.
    print(f"Generated predictions for {len(predictions)} test studies")
    # Return probabilities and thresholded classes for saving or review.
    return predictions


def save_results(
    experiment: Experiment,
    test_predictions: pd.DataFrame | None = None,
    run_name: str = "standalone_sparse_mil",
) -> Path:
    """Save this notebook's new model, history, configuration, and optional test predictions to Drive."""
    # Create a unique output directory name under the selected data folder.
    run_root = experiment.paths.output_root / run_name
    # Create missing output parent folders safely.
    run_root.mkdir(parents=True, exist_ok=True)
    # Save weights and the exact target ordering needed to use this new model later.
    torch.save(
        {
            "model_state": experiment.model.state_dict(),
            "config": asdict(experiment.config),
            "targets": TARGETS,
            "b52_reference": asdict(B52_REFERENCE),
        },
        run_root / "trained_model.pt",
    )
    # Save readable training history beside the model weights.
    (run_root / "history.json").write_text(json.dumps(experiment.history, indent=2), encoding="utf-8")
    # Save readable configuration beside the model weights.
    (run_root / "config.json").write_text(json.dumps(asdict(experiment.config), indent=2), encoding="utf-8")
    # Save the reference separately so it cannot be mistaken for a subset result.
    (run_root / "b52_reference.json").write_text(
        json.dumps(asdict(B52_REFERENCE), indent=2),
        encoding="utf-8",
    )
    # Save test-subset probabilities and classifications only when inference was requested.
    if test_predictions is not None:
        # Write one prediction row per test study in CSV form.
        test_predictions.to_csv(run_root / "test_predictions.csv", index=False)
    # Print the Drive location for the user.
    print("Saved new run to:", run_root)
    # Return the output directory for optional follow-up code.
    return run_root


# ==========================================================================
# Twelve-case classification review
# ==========================================================================


def format_positive_predictions(probability: np.ndarray, threshold: float = 0.50) -> str:
    """Format all target probabilities at or above the classification threshold."""
    # Build short target/probability strings for positive classifications.
    selected = [
        f"{name}: {value:.2f}"
        for name, value in zip(TARGETS, probability)
        if value >= threshold
    ]
    # Return a clear sentence when no target reaches the threshold.
    return "none at or above 0.50" if not selected else "; ".join(selected)


# ==========================================================================
# Read the report labels
# ==========================================================================


REPORT_LABELS_FILENAME = "training_targets.csv"


def report_label_columns() -> list[str]:
    """The exact columns b23_llm_labels.py and b6_report_labels.py write."""
    columns = ["StudyInstanceUID"]
    for target in TARGETS:
        columns.extend([target, f"{target}__confidence", f"{target}__state"])
    return columns


def load_report_labels(path: Path) -> pd.DataFrame:
    """Read the label export and refuse anything that is not the agreed shape."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No report labels at {path}.\n"
            "Export them with b23_llm_labels.py and copy training_targets.csv "
            "into the same Drive folder as train.csv."
        )

    frame = pd.read_csv(path)
    missing = [name for name in report_label_columns() if name not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing[:6]}")

    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{path.name} lists the same study more than once")

    for target in TARGETS:
        confidence = pd.to_numeric(frame[f"{target}__confidence"], errors="coerce")
        if confidence.isna().any() or float(confidence.min()) < 0 or float(confidence.max()) > 1:
            raise ValueError(f"{target}__confidence must be a number between 0 and 1")
    return frame


def weak_targets_and_confidence(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split the export into a target matrix and a confidence matrix.

    A cell with zero confidence is blanked to NaN as well as being given zero
    weight. Either alone would be enough; doing both means a later change to the
    loss cannot accidentally start training on report silence.
    """
    targets = np.full((len(frame), len(TARGETS)), np.nan, dtype=np.float32)
    confidence = np.zeros((len(frame), len(TARGETS)), dtype=np.float32)

    for index, target in enumerate(TARGETS):
        column = pd.to_numeric(frame[target], errors="coerce").to_numpy(np.float32)
        weight = pd.to_numeric(frame[f"{target}__confidence"], errors="coerce").to_numpy(np.float32)
        used = weight > 0
        targets[used, index] = column[used]
        confidence[used, index] = weight[used]
    return targets, confidence


def split_gold_and_report_only(train_table: pd.DataFrame) -> tuple:
    """A study is 'gold' exactly when train.csv already carries a label for it."""
    written = train_table[TARGETS].apply(pd.to_numeric, errors="coerce")
    is_gold = written.notna().any(axis=1)
    return (
        train_table.loc[is_gold].reset_index(drop=True),
        train_table.loc[~is_gold].reset_index(drop=True),
    )


def select_report_training_studies(
    train_table: pd.DataFrame, labels: pd.DataFrame, records
) -> tuple:
    """Choose what trains and what is held back, and refuse a gold leak.

    Kept apart from model building so it can be checked on its own. A mistake
    here would not crash: it would quietly put expert-gold studies into training
    and make every score the notebook prints look better than it is.
    """
    gold_frame, report_only = split_gold_and_report_only(train_table)

    leaked = sorted(set(labels["StudyInstanceUID"]) & set(gold_frame["StudyInstanceUID"]))
    if leaked:
        raise ValueError(
            f"the label export contains {len(leaked)} expert-gold studies "
            f"(for example {leaked[0]}); it must hold report-only studies only"
        )

    keep = labels["StudyInstanceUID"].isin(set(report_only["StudyInstanceUID"])) & labels[
        "StudyInstanceUID"
    ].isin(set(records))
    train_frame = labels.loc[keep].reset_index(drop=True)
    if train_frame.empty:
        raise ValueError(
            "No study is in the export, in train.csv as report-only, and in your "
            "extracted DICOM subset at once. Check that the export covers the "
            "studies you actually downloaded."
        )

    gold_usable = gold_frame.loc[
        gold_frame["StudyInstanceUID"].isin(set(records))
    ].reset_index(drop=True)
    return train_frame, gold_usable


def describe_report_labels(confidence: np.ndarray) -> dict:
    """How much supervision the reports actually provide, per target."""
    used = confidence > 0
    return {
        "studies": int(confidence.shape[0]),
        "cells_total": int(used.size),
        "cells_used": int(used.sum()),
        "coverage": float(used.mean()),
        "per_target_cells": {
            target: int(used[:, index].sum()) for index, target in enumerate(TARGETS)
        },
    }


# ==========================================================================
# A loss that respects confidence
# ==========================================================================


def target_balance_multipliers(confidence: np.ndarray) -> np.ndarray:
    """Give every target the same total say, whatever the reports talked about."""
    confidence = np.asarray(confidence, dtype=np.float64)
    if confidence.ndim != 2 or confidence.shape[1] != len(TARGETS):
        raise ValueError(f"confidence must have shape [N,{len(TARGETS)}]")

    mass = confidence.sum(axis=0)
    if not (mass > 0).all():
        empty = [TARGETS[index] for index in np.flatnonzero(mass <= 0)]
        raise ValueError(f"the reports gave no usable supervision for: {empty}")
    return (float(mass.mean()) / mass).astype(np.float32)


def report_weighted_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    multiplier: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy weighted by per-cell confidence and per-target balance."""
    if logits.shape != target.shape or logits.shape != confidence.shape:
        raise ValueError("logits, target and confidence must have the same shape")

    # A blank target is unusable whatever its confidence claims.
    known = torch.isfinite(target).float()
    effective = confidence.float() * known * multiplier.to(logits.device)[None, :]

    denominator = effective.sum()
    if float(denominator.detach().cpu()) <= 0:
        # No usable cell in this batch. Return a real zero that still has a
        # gradient path, so the training step stays well defined.
        return logits.sum() * 0.0

    safe_target = torch.nan_to_num(target, nan=0.0)
    cell = F.binary_cross_entropy_with_logits(
        logits.float(), safe_target.float(), reduction="none"
    )
    return (cell * effective).sum() / denominator.clamp_min(1e-8)


class ReportSupervision:
    """Per-study confidence, looked up by study UID rather than by row number.

    The dataset filters and reindexes the frame it is given, so a confidence
    array addressed by position would silently drift out of step with the studies
    the loader actually yields. Addressing by UID cannot drift.
    """

    def __init__(self, confidence_by_study: dict, multiplier: np.ndarray) -> None:
        self.confidence_by_study = confidence_by_study
        self.multiplier = torch.tensor(multiplier, dtype=torch.float32, device=DEVICE)

    def batch(self, study_uids: list) -> torch.Tensor:
        """The confidence rows for one batch, in the batch's own order."""
        missing = [uid for uid in study_uids if uid not in self.confidence_by_study]
        if missing:
            raise KeyError(f"no confidence recorded for {missing[:3]}")
        rows = np.stack([self.confidence_by_study[uid] for uid in study_uids])
        return torch.tensor(rows, dtype=torch.float32, device=DEVICE)


# ==========================================================================
# Change 2 — turn augmentation on
# ==========================================================================


@dataclass(frozen=True)
class AugmentationPolicy:
    """How hard to distort a training study. Zero everywhere means off."""

    # Rotate the image in plane, up to this many degrees either way.
    rotation_deg: float = 8.0
    # Shift the image, as a fraction of its width and height.
    translate_frac: float = 0.05
    # Zoom in or out by up to this fraction.
    scale_jitter: float = 0.10
    # Brighten or darken the mid-tones, as an exponent around 1.0.
    gamma_jitter: float = 0.20
    # Add Gaussian noise of this standard deviation, in normalised units.
    noise_std: float = 0.02
    # Blank this fraction of slices, so no single slice can carry a study.
    slice_dropout: float = 0.10
    # Multiply by a smooth field of this strength, imitating coil shading.
    bias_field_strength: float = 0.10


AUGMENTATION = AugmentationPolicy()


NO_AUGMENTATION = AugmentationPolicy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def augment_series(
    series: torch.Tensor, policy: AugmentationPolicy, generator: torch.Generator
) -> torch.Tensor:
    """Distort one prepared MRI series of shape [slices, 3, height, width].

    Every draw comes from the generator passed in, so nothing here touches the
    global random state and two runs with the same seed agree exactly.
    """
    if series.ndim != 4:
        raise ValueError(f"expected [slices,3,H,W], got {tuple(series.shape)}")

    def uniform(low: float, high: float) -> float:
        if high <= low:
            return low
        drawn = torch.rand((), generator=generator, dtype=torch.float32)
        return float(low + (high - low) * drawn)

    slices, channels, height, width = series.shape
    out = series.float()

    # --- rotation, translation and scale, as one warp ----------------------
    # Doing them together means one interpolation rather than three, so the
    # image is blurred once instead of three times.
    if policy.rotation_deg > 0 or policy.translate_frac > 0 or policy.scale_jitter > 0:
        angle = math.radians(uniform(-policy.rotation_deg, policy.rotation_deg))
        scale = 1.0 + uniform(-policy.scale_jitter, policy.scale_jitter)
        shift_x = uniform(-policy.translate_frac, policy.translate_frac) * 2.0
        shift_y = uniform(-policy.translate_frac, policy.translate_frac) * 2.0

        cosine, sine = math.cos(angle) / scale, math.sin(angle) / scale
        theta = torch.tensor(
            [[cosine, -sine, shift_x], [sine, cosine, shift_y]], dtype=torch.float32
        ).expand(slices, 2, 3)
        grid = F.affine_grid(theta, list(out.shape), align_corners=False)
        # Zero padding matches the notebook's pad_value, so a rotated corner
        # looks like the padding the geometry policy already produces.
        out = F.grid_sample(
            out, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

    # --- gamma -------------------------------------------------------------
    # Applied on the positive part only. These images are percentile-normalised
    # and can hold small negatives, and a fractional power of a negative number
    # is not a real number.
    if policy.gamma_jitter > 0:
        gamma = math.exp(uniform(-policy.gamma_jitter, policy.gamma_jitter))
        positive = out.clamp_min(0.0)
        out = positive.pow(gamma) + (out - positive)

    # --- smooth bias field -------------------------------------------------
    # A coarse 4x4 grid stretched up to full size: slow shading across the
    # image, which is what an imperfect receive coil actually produces.
    if policy.bias_field_strength > 0:
        coarse = torch.rand((1, 1, 4, 4), generator=generator, dtype=torch.float32)
        field = F.interpolate(
            coarse, size=(height, width), mode="bilinear", align_corners=False
        )
        field = 1.0 + policy.bias_field_strength * (2.0 * field - 1.0)
        out = out * field

    # --- noise -------------------------------------------------------------
    if policy.noise_std > 0:
        noise = torch.randn(out.shape, generator=generator, dtype=torch.float32)
        out = out + policy.noise_std * noise

    # --- slice dropout -----------------------------------------------------
    # Never drop every slice: a study with nothing left in it would be a
    # blank input carrying a real label, which teaches the wrong thing.
    if policy.slice_dropout > 0 and slices > 1:
        keep = torch.rand(slices, generator=generator, dtype=torch.float32)
        drop = keep < policy.slice_dropout
        if bool(drop.all()):
            drop[int(torch.argmax(keep))] = False
        out = out * (~drop).float().view(slices, 1, 1, 1)

    return out


class AugmentedKneeMRIDataset(KneeMRIDataset):
    """The inherited dataset, with B52's augmentation on the training split.

    Subclassed rather than edited so the validation and test paths keep using
    the original, untouched decoding. An augmented validation set would make
    every score noisier and none of them comparable.
    """

    def __init__(self, *args, policy: AugmentationPolicy = NO_AUGMENTATION, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Give the next pass a different draw. The training loop calls this."""
        self.epoch = int(epoch)

    def _generator(self, index: int) -> torch.Generator:
        """A generator fixed by run seed, epoch and study position.

        Not shared state: a DataLoader worker holds a copy of the dataset, so a
        single shared generator would give every worker the same numbers.
        """
        generator = torch.Generator()
        generator.manual_seed(
            (int(self.config.seed) * 1_000_003 + self.epoch * 9_176 + index) % (2**31 - 1)
        )
        return generator

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        if self.policy == NO_AUGMENTATION:
            return item

        generator = self._generator(index)
        volumes, present = item["volumes"], item["present"]
        augmented = []
        for position in range(volumes.shape[0]):
            # A masked series is a zero placeholder the model ignores. Warping
            # it would only cost time.
            if float(present[position]) <= 0:
                augmented.append(volumes[position])
            else:
                augmented.append(augment_series(volumes[position], self.policy, generator))
        item["volumes"] = torch.stack(augmented)
        return item


def describe_augmentation(policy: AugmentationPolicy) -> dict:
    """Which augmentations are actually switched on, so it can be checked."""
    active = {
        name: float(getattr(policy, name))
        for name in (
            "rotation_deg",
            "translate_frac",
            "scale_jitter",
            "gamma_jitter",
            "noise_std",
            "slice_dropout",
            "bias_field_strength",
        )
        if float(getattr(policy, name)) > 0
    }
    return {"active": active, "count": len(active)}


# ==========================================================================
# Change 3 — a held-out split, a schedule, and keeping the best epoch
# ==========================================================================


def split_report_studies(
    study_uids: list, validation_fraction: float, seed: int
) -> tuple[list, list]:
    """Divide the report-labelled studies into a training and a hold-out part.

    Sorted before shuffling so the split depends on the seed alone, not on the
    order pandas happened to read the file in.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    ordered = sorted(str(uid) for uid in study_uids)
    if len(ordered) != len(set(ordered)):
        raise ValueError("the same study appears twice in the split input")

    shuffled = list(np.random.default_rng(int(seed)).permutation(ordered))
    held = max(1, int(round(len(shuffled) * float(validation_fraction))))
    if held >= len(shuffled):
        raise ValueError(
            f"{len(shuffled)} studies cannot give both a training and a hold-out "
            "part at this fraction; use more studies or a smaller fraction"
        )

    validation = [str(uid) for uid in shuffled[:held]]
    training = [str(uid) for uid in shuffled[held:]]
    overlap = set(training) & set(validation)
    if overlap:
        raise ValueError(f"{len(overlap)} studies ended up on both sides of the split")
    return training, validation


def build_cosine_schedule(optimizer, epochs: int):
    """Fall from the full learning rate to nearly zero across exactly `epochs`.

    `T_max` equal to the epochs actually run is the whole point. A longer T_max
    leaves the run stopping while the rate is still high, which is what two fixed
    epochs of a long schedule did.
    """
    if int(epochs) < 1:
        raise ValueError("a schedule needs at least one epoch")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs), eta_min=0.0
    )


def evaluate_weak_predictions(target: np.ndarray, probability: np.ndarray) -> dict:
    """Macro AUC against report labels, where the truth is "above 0.5".

    The inherited `evaluate_predictions` casts the target with `.astype(int)`.
    That is right for the expert-gold studies, whose labels really are 0 or 1,
    and quietly wrong here: a report label of `0.97` and one of `0.03` both
    truncate to `0`, so every target looks one-class, every AUC comes back
    undefined, and no epoch can ever be chosen.

    Binarising at 0.5 is what the real pipeline does. It also works unchanged on
    hard 0/1 labels, so this one function scores both surfaces.
    """
    per_target: dict = {}
    for index, name in enumerate(TARGETS):
        known = np.isfinite(target[:, index])
        truth = (target[known, index] > 0.5).astype(int)
        # An AUC needs one of each class. One-class is a fact about this split,
        # not an error, so the target is reported as undefined rather than raising.
        per_target[name] = (
            fast_auc_or_none(truth, probability[known, index])
            if known.any() and 0 < truth.sum() < len(truth) else None
        )

    defined = [value for value in per_target.values() if value is not None]
    return {
        "mean_auc": None if not defined else float(np.mean(defined)),
        "per_target_auc": per_target,
        "targets_defined": len(defined),
        "known_cells": int(np.isfinite(target).sum()),
    }


def fast_auc_or_none(truth: np.ndarray, score: np.ndarray) -> float | None:
    """Rank-based AUC for one target, or None when it is not defined."""
    if len(truth) < 2 or truth.sum() in (0, len(truth)):
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)

    # Tied scores must share a rank, or identical predictions would be counted
    # as if the model had ordered them.
    sorted_scores = score[order]
    start = 0
    for position in range(1, len(sorted_scores) + 1):
        if position == len(sorted_scores) or sorted_scores[position] != sorted_scores[start]:
            if position - start > 1:
                ranks[order[start:position]] = ranks[order[start:position]].mean()
            start = position

    positives = float(truth.sum())
    negatives = float(len(truth) - positives)
    return float((ranks[truth == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


class BestEpoch:
    """Remember the weights of the best epoch, rather than the last one.

    The comparison is deliberately strict: on a tie the earlier epoch is kept,
    because a later epoch that only matched it is not evidence of improvement.
    """

    def __init__(self) -> None:
        self.epoch: int | None = None
        self.score: float | None = None
        self.weights: dict | None = None

    def offer(self, epoch: int, score: float | None, model: nn.Module) -> bool:
        """Keep this epoch if it beats the best so far. Returns whether it did."""
        if score is None or not np.isfinite(score):
            return False
        if self.score is not None and float(score) <= self.score:
            return False
        self.epoch = int(epoch)
        self.score = float(score)
        self.weights = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        return True

    def restore(self, model: nn.Module) -> int:
        """Put the best epoch's weights back into the model."""
        if self.weights is None:
            raise RuntimeError("no epoch was ever scored, so there is nothing to restore")
        model.load_state_dict(self.weights)
        return int(self.epoch)


# ==========================================================================
# Change 1 — the encoder learns, and building the run
# ==========================================================================


HIERARCHY_PREFIXES = ("global_projection.", "global_classifier.")


HIERARCHY_LR_SCALE = 0.05  # the value B50 measured and B52 kept unchanged


def hierarchy_parameter_names(model: nn.Module) -> list[str]:
    """Name every parameter that belongs to the study hierarchy."""
    return [
        name
        for name, _ in model.named_parameters()
        if name.startswith(HIERARCHY_PREFIXES)
    ]


def build_parameter_groups(model: nn.Module, head_lr: float) -> list[dict]:
    """Encoder and head at full rate, the study hierarchy at a reduced one."""
    hierarchy_names = set(hierarchy_parameter_names(model))
    head, hierarchy = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (hierarchy if name in hierarchy_names else head).append(parameter)

    if not head:
        raise RuntimeError("nothing outside the hierarchy is trainable; B52 trains the encoder")

    groups = [{"params": head, "lr": float(head_lr), "name": "encoder_and_head"}]
    if hierarchy:
        groups.append(
            {
                "params": hierarchy,
                "lr": float(head_lr) * HIERARCHY_LR_SCALE,
                "name": "study_hierarchy",
            }
        )
    return groups


def describe_trainable(model: nn.Module) -> dict:
    """What is actually learning, so a setting can be checked rather than assumed."""
    hierarchy_names = set(hierarchy_parameter_names(model))
    counts = {"encoder": 0, "hierarchy": 0, "head_and_rest": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            counts["encoder"] += parameter.numel()
        elif name in hierarchy_names:
            counts["hierarchy"] += parameter.numel()
        else:
            counts["head_and_rest"] += parameter.numel()
    return counts


def read_fusion_gate(model: nn.Module) -> np.ndarray:
    """tanh(g): how much of the local branch reaches the score, per target."""
    return torch.tanh(model.fusion_gate.detach()).cpu().numpy()


# ==========================================================================
# Build the B52 run
# ==========================================================================


@dataclass
class B52Run:
    """Everything one B52 run needs, kept in one place."""

    experiment: Experiment
    supervision: ReportSupervision
    scheduler: object
    gold_loader: DataLoader | None
    train_dataset: AugmentedKneeMRIDataset
    best: BestEpoch


def build_b52_run(
    paths: DrivePaths,
    config: RunConfig = CONFIG,
    *,
    policy: AugmentationPolicy = AUGMENTATION,
    labels_path: Path | None = None,
) -> B52Run:
    """Assemble B52's regime: augmentation on, a real schedule, a hold-out split."""
    set_seed(config.seed)
    validate_dataset(paths)

    train_table = pd.read_csv(paths.train_csv)
    train_table["StudyInstanceUID"] = train_table["StudyInstanceUID"].astype(str)
    series_table = pd.read_csv(paths.series_csv)
    records = build_series_records(series_table, config)

    labels = load_report_labels(labels_path or paths.data_root / REPORT_LABELS_FILENAME)
    labelled, gold_usable = select_report_training_studies(train_table, labels, records)

    targets, confidence = weak_targets_and_confidence(labelled)

    # A report that mentions none of the twelve findings supervises nothing. Such
    # a study would cost a full DICOM decode per epoch and teach nothing, and the
    # inherited preflight refuses a batch with no usable label at all.
    usable = confidence.sum(axis=1) > 0
    if not usable.all():
        print(f"skipping {int((~usable).sum())} studies whose report mentions no finding")
        labelled = labelled.loc[usable].reset_index(drop=True)
        targets, confidence = targets[usable], confidence[usable]
    if labelled.empty:
        raise ValueError("no report in the export mentions any of the twelve findings")

    for index, target in enumerate(TARGETS):
        labelled[target] = targets[:, index]

    confidence_by_study = {
        uid: confidence[row] for row, uid in enumerate(labelled["StudyInstanceUID"])
    }
    # Gold labels are real, so every known gold cell carries full confidence.
    for uid in gold_usable["StudyInstanceUID"]:
        confidence_by_study[uid] = np.ones(len(TARGETS), dtype=np.float32)

    train_uids, holdout_uids = split_report_studies(
        list(labelled["StudyInstanceUID"]), config.validation_fraction, config.seed
    )
    train_frame = labelled.loc[
        labelled["StudyInstanceUID"].isin(set(train_uids))
    ].reset_index(drop=True)
    holdout_frame = labelled.loc[
        labelled["StudyInstanceUID"].isin(set(holdout_uids))
    ].reset_index(drop=True)

    # Augmentation on the training split only. The hold-out and the gold studies
    # are decoded exactly as the inherited notebook decodes them, so their scores
    # stay comparable from epoch to epoch.
    train_dataset = AugmentedKneeMRIDataset(
        train_frame, records, paths, config,
        split="train", include_targets=True, policy=policy,
    )
    holdout_dataset = KneeMRIDataset(
        holdout_frame, records, paths, config, split="train", include_targets=True
    )
    gold_dataset = (
        KneeMRIDataset(gold_usable, records, paths, config, split="train", include_targets=True)
        if not gold_usable.empty else None
    )

    # Balance is measured over the studies the training loader will really yield.
    used_confidence = np.stack([confidence_by_study[uid] for uid in train_dataset.study_uids])
    supervision = ReportSupervision(
        confidence_by_study, target_balance_multipliers(used_confidence)
    )

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": False,
        "collate_fn": collate_studies,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    holdout_loader = DataLoader(holdout_dataset, shuffle=False, **loader_kwargs)
    gold_loader = (
        DataLoader(gold_dataset, shuffle=False, **loader_kwargs)
        if gold_dataset is not None else None
    )

    model = HighResolutionSparseMIL(config).to(DEVICE)
    optimizer = torch.optim.AdamW(
        build_parameter_groups(model, config.learning_rate),
        weight_decay=config.weight_decay,
    )

    experiment = Experiment(
        paths=paths,
        config=config,
        model=model,
        optimizer=optimizer,
        scaler=torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda"),
        train_loader=train_loader,
        validation_loader=holdout_loader,
        # Balance is handled per target by the multiplier, so this stays neutral.
        positive_weight=torch.ones(len(TARGETS), dtype=torch.float32, device=DEVICE),
    )

    summary = describe_report_labels(used_confidence)
    augmentation = describe_augmentation(policy)
    print(f"training studies (reports) : {len(train_dataset)}")
    print(f"hold-out studies (reports) : {len(holdout_dataset)}   <- the epoch is chosen on these")
    print(f"gold studies (read only)   : {0 if gold_dataset is None else len(gold_dataset)}")
    print(f"report cells used          : {summary['cells_used']:,} of {summary['cells_total']:,} "
          f"({summary['coverage']:.1%})")
    print(f"augmentations on           : {augmentation['count']} -> {augmentation['active']}")
    print(f"epochs / cosine T_max      : {config.epochs}")
    print(f"trainable                  : {describe_trainable(model)}")
    print(f"optimiser groups           : {[group['name'] for group in optimizer.param_groups]}")
    if gold_dataset is None:
        print("note: no expert-gold study is in your subset, so the gold column stays blank.")

    return B52Run(
        experiment=experiment,
        supervision=supervision,
        scheduler=build_cosine_schedule(optimizer, config.epochs),
        gold_loader=gold_loader,
        train_dataset=train_dataset,
        best=BestEpoch(),
    )


# ==========================================================================
# Train, and keep the best epoch
# ==========================================================================


def run_b52_epoch(
    experiment: Experiment,
    loader: DataLoader,
    supervision: ReportSupervision,
    training: bool,
) -> dict:
    """One pass, using each cell's confidence instead of treating all cells alike."""
    experiment.model.train(training)
    losses: list[float] = []
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    for batch in loader:
        # Read the UIDs before move_batch drops them.
        confidence = supervision.batch(list(batch["study_uid"]))
        volumes, present, metadata, position, target = move_batch(batch)
        del batch

        if training:
            experiment.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training), autocast_context():
            output = experiment.model(volumes, present, metadata, position)
            combined_loss = report_weighted_bce(
                output.logits, target, confidence, supervision.multiplier
            )
            local_loss = report_weighted_bce(
                output.local_logits, target, confidence, supervision.multiplier
            )
            loss = combined_loss + experiment.config.local_loss_weight * local_loss

        if training:
            experiment.scaler.scale(loss).backward()
            experiment.scaler.unscale_(experiment.optimizer)
            torch.nn.utils.clip_grad_norm_(
                experiment.model.parameters(), experiment.config.grad_clip_norm
            )
            experiment.scaler.step(experiment.optimizer)
            experiment.scaler.update()

        losses.append(float(loss.detach().cpu()))
        targets.append(target.detach().cpu().numpy())
        probabilities.append(torch.sigmoid(output.logits).detach().cpu().numpy())
        del volumes, present, metadata, position, target, output
        del loss, combined_loss, local_loss

    return {
        "loss": float(np.mean(losses)),
        "target": np.concatenate(targets, axis=0),
        "probability": np.concatenate(probabilities, axis=0),
    }


def train_b52(run: B52Run) -> list[dict]:
    """Train under B52's regime and leave the best epoch's weights in the model."""
    experiment = run.experiment
    if experiment.validation_loader is None:
        raise RuntimeError("B52 chooses an epoch on a hold-out split, so one is required")

    for epoch in range(1, experiment.config.epochs + 1):
        started = time.time()
        # A different augmentation draw each epoch, reproducible from the seed.
        run.train_dataset.set_epoch(epoch)
        rate = experiment.optimizer.param_groups[0]["lr"]

        train_result = run_b52_epoch(
            experiment, experiment.train_loader, run.supervision, training=True
        )
        holdout = run_b52_epoch(
            experiment, experiment.validation_loader, run.supervision, training=False
        )
        holdout_scores = evaluate_weak_predictions(holdout["target"], holdout["probability"])

        row = {
            "epoch": epoch,
            "learning_rate": float(rate),
            "train_loss": train_result["loss"],
            "validation_loss": holdout["loss"],
            "holdout_macro_auc": holdout_scores["mean_auc"],
            "per_target_auc": holdout_scores["per_target_auc"],
            "gate": float(np.abs(read_fusion_gate(experiment.model)).mean()),
            "seconds": round(time.time() - started, 1),
        }

        if run.gold_loader is not None:
            gold = run_b52_epoch(experiment, run.gold_loader, run.supervision, training=False)
            # Read only. Choosing on the 58 expert studies is exactly what
            # section 14 explains this notebook does not do.
            row["gold_macro_auc"] = evaluate_weak_predictions(
                gold["target"], gold["probability"]
            )["mean_auc"]

        kept = run.best.offer(epoch, row["holdout_macro_auc"], experiment.model)
        row["kept"] = kept
        experiment.history.append(row)
        run.scheduler.step()

        def shown(value) -> str:
            return f"{value:.5f}" if value is not None else "  n/a  "

        print(
            f"epoch {epoch:>2} | lr {rate:.2e} | train {row['train_loss']:.5f} | "
            f"holdout {shown(row['holdout_macro_auc'])} | "
            f"gold {shown(row.get('gold_macro_auc'))} | "
            f"|gate| {row['gate']:.5f} | {row['seconds']}s"
            f"{'  <- best so far' if kept else ''}"
        )

    best_epoch = run.best.restore(experiment.model)
    print()
    print(f"restored epoch {best_epoch}, hold-out macro AUC {run.best.score:.6f}")
    print("the model now holds the best epoch's weights, not the last epoch's")
    return experiment.history


# ==========================================================================
# Running B52 end to end
# ==========================================================================
#
# Everything above is shared with the notebook. Everything below exists only
# because this is a script: it turns command-line arguments into one run, and
# turns that run into files on disk.


# The augmentation the real B53 run uses, taken from
# config/b42_constant_area_aspect_sparse.yaml rather than chosen here. It is
# milder than this notebook's own defaults on every geometric setting, so a
# subset run with --augment-preset b53 is a rehearsal for the full-data run
# rather than a differently-tuned experiment.
B53_AUGMENTATION = AugmentationPolicy(
    rotation_deg=5.0,
    translate_frac=0.03,
    scale_jitter=0.05,
    gamma_jitter=0.12,
    noise_std=0.02,
    slice_dropout=0.08,
    bias_field_strength=0.08,
)

AUGMENT_PRESETS = {
    "notebook": AUGMENTATION,
    "b53": B53_AUGMENTATION,
    "off": NO_AUGMENTATION,
}


def resolve_paths(data_root: Path, out_dir: Path) -> DrivePaths:
    """Point the inherited path bundle at a plain folder rather than at Drive."""
    data_root = Path(data_root).expanduser().resolve()
    for name in ("train.csv", "train_series.csv"):
        if not (data_root / name).is_file():
            raise FileNotFoundError(f"no {name} under {data_root}; check --data-root")
    return make_paths(data_root, Path(out_dir).expanduser().resolve())


def limit_labels(labels_path: Path, max_studies: int, scratch: Path) -> Path:
    """Write a shortened copy of the label export, for a quick trial run.

    A copy rather than an edit: the original export is the record of what the
    reports said, and a run that trims it in place would destroy that.
    """
    frame = pd.read_csv(labels_path)
    if max_studies <= 0 or max_studies >= len(frame):
        return Path(labels_path)

    scratch.mkdir(parents=True, exist_ok=True)
    trimmed = scratch / "training_targets_limited.csv"
    frame.head(int(max_studies)).to_csv(trimmed, index=False)
    print(f"[B52] limited to the first {max_studies} of {len(frame)} labelled studies")
    return trimmed


def b52_preflight(run: B52Run) -> dict:
    """One forward and backward pass, with B52's own loss and no update.

    It answers the question B52 exists to ask: does a gradient actually reach
    the encoder? A silent no here is the frozen baseline wearing B52's name, and
    it would look like an ordinary disappointing result rather than a bug.
    """
    print("preflight: one forward and backward pass, no optimiser step")
    experiment = run.experiment
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    experiment.model.train()
    experiment.model.zero_grad(set_to_none=True)

    batch = next(iter(experiment.train_loader))
    confidence = run.supervision.batch(list(batch["study_uid"]))
    volumes, present, metadata, position, target = move_batch(batch)
    del batch

    with autocast_context():
        output = experiment.model(volumes, present, metadata, position)
        combined = report_weighted_bce(
            output.logits, target, confidence, run.supervision.multiplier
        )
        local = report_weighted_bce(
            output.local_logits, target, confidence, run.supervision.multiplier
        )
        total = combined + experiment.config.local_loss_weight * local

    experiment.scaler.scale(total).backward()

    def moved(module: nn.Module) -> int:
        return sum(
            1
            for parameter in module.parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and torch.count_nonzero(parameter.grad).item() > 0
        )

    report = {
        "loss": float(total.detach().cpu()),
        "encoder_tensors_with_gradient": moved(experiment.model.encoder),
        "hierarchy_tensors_with_gradient": moved(experiment.model.global_classifier),
        "head_tensors_with_gradient": moved(experiment.model.sparse_head),
        "trainable": describe_trainable(experiment.model),
    }
    if DEVICE.type == "cuda":
        report["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)

    experiment.model.zero_grad(set_to_none=True)

    if report["encoder_tensors_with_gradient"] == 0:
        raise RuntimeError(
            "preflight FAILED: no gradient reached the encoder. B52 is the "
            "experiment in which the encoder learns, so this run would be the "
            "frozen baseline under a different name."
        )

    for name, value in report.items():
        print(f"  {name:<32} {value}")
    print("preflight PASS")
    return report


def score_split(run: B52Run, loader, name: str, out_dir: Path) -> dict:
    """Score one split with the final weights and write a prediction row per study."""
    result = run_b52_epoch(run.experiment, loader, run.supervision, training=False)
    uids = list(loader.dataset.study_uids)
    probability = result["probability"]
    if len(uids) != len(probability):
        raise RuntimeError(
            f"{name}: {len(uids)} studies but {len(probability)} predictions; "
            "the loader and the dataset are out of step"
        )

    frame = pd.DataFrame(probability, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    frame["predicted_positive"] = [format_positive_predictions(row) for row in probability]
    frame.to_csv(out_dir / f"{name}_predictions.csv", index=False)

    scores = evaluate_weak_predictions(result["target"], result["probability"])
    scores["loss"] = result["loss"]
    scores["studies"] = len(uids)
    return scores


def write_history(history: list, out_dir: Path) -> None:
    """The epoch table, as JSON for exactness and CSV for reading."""
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    columns = [
        "epoch",
        "learning_rate",
        "train_loss",
        "validation_loss",
        "holdout_macro_auc",
        "gold_macro_auc",
        "gate",
        "seconds",
        "kept",
    ]
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def write_per_target(scores: dict, out_dir: Path) -> None:
    """Every target's AUC, which is where a macro average hides its detail."""
    rows = [
        {"target": target, "auc": scores["per_target_auc"].get(target)}
        for target in TARGETS
    ]
    pd.DataFrame(rows).to_csv(out_dir / "per_target_auc.csv", index=False)


def plot_curves(history: list, out_dir: Path) -> None:
    """Two figures: whether it is learning, and whether that is helping."""
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="training loss")
    plt.plot(
        epochs, [row["validation_loss"] for row in history], marker="o", label="hold-out loss"
    )
    plt.xlabel("epoch")
    plt.ylabel("confidence-weighted BCE")
    plt.title("B52: training and hold-out loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=140)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(
        epochs,
        [row.get("holdout_macro_auc") for row in history],
        marker="o",
        label="hold-out macro AUC (the epoch is chosen on this)",
    )
    if any(row.get("gold_macro_auc") is not None for row in history):
        plt.plot(
            epochs,
            [row.get("gold_macro_auc") for row in history],
            marker="s",
            label="expert-gold macro AUC (read only)",
        )
    plt.xlabel("epoch")
    plt.ylabel("macro AUC")
    plt.title("B52: score per epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "auc_curve.png", dpi=140)
    plt.close()


def write_summary(
    run: B52Run,
    history: list,
    holdout: dict,
    gold: dict | None,
    policy: AugmentationPolicy,
    out_dir: Path,
    minutes: float,
) -> None:
    """The whole run in plain words, for whoever opens the folder next."""
    best_epoch = run.best.epoch
    config = run.experiment.config
    augmentation = describe_augmentation(policy)

    def line(value) -> str:
        return f"{value:.6f}" if isinstance(value, float) else str(value)

    text = [
        "B52 -- actually train the model",
        "=" * 74,
        "",
        "What B52 changes, and nothing else:",
        "  1. the encoder learns          (it was frozen, at a rate of exactly 0.0)",
        "  2. augmentation is on          (nine settings existed and were all zeroed)",
        "  3. the cosine finishes and     (it was two fixed epochs, and whatever",
        "     the best epoch is kept       epoch 2 produced was the answer)",
        "",
        "Settings",
        "-" * 74,
        f"  epochs                  {config.epochs}",
        f"  learning rate           {config.learning_rate}",
        f"  hierarchy rate          {config.learning_rate * HIERARCHY_LR_SCALE} ({HIERARCHY_LR_SCALE}x)",
        f"  image size              {config.image_size}",
        f"  slices per series       {config.slices_per_series}",
        f"  batch size              {config.batch_size}",
        f"  seed                    {config.seed}",
        f"  device                  {DEVICE}",
        f"  augmentations on        {augmentation['count']}",
        *(f"    {name:<22}{value}" for name, value in augmentation["active"].items()),
        "",
        "Studies",
        "-" * 74,
        f"  training (reports)      {len(run.experiment.train_loader.dataset)}",
        f"  hold-out (reports)      {holdout['studies']}   <- the epoch is chosen on these",
        f"  expert gold (read only) {gold['studies'] if gold else 0}",
        "",
        "Result",
        "-" * 74,
        f"  best epoch              {best_epoch} of {config.epochs}",
        f"  hold-out macro AUC      {line(run.best.score)}",
        f"  expert-gold macro AUC   {line(gold['mean_auc']) if gold else 'n/a'}",
        f"  wall clock              {minutes:.1f} minutes",
        "",
    ]

    if best_epoch == config.epochs:
        text += [
            "  The last epoch was the best, so the hold-out score was still climbing",
            "  when the run stopped. More epochs are worth trying.",
        ]
    else:
        text += [
            f"  The run peaked at epoch {best_epoch} and did not improve after it,",
            "  so more epochs would not have helped.",
        ]

    text += [
        "",
        "What this number is worth",
        "-" * 74,
        "  Nothing, in absolute terms. This is a fresh compact model trained from",
        "  random weights on a subset. Read the shape -- is the loss falling, is the",
        "  hold-out score rising, which epoch does it peak on -- and not the value.",
        "  It is not comparable with any leaderboard score.",
        "",
        "  The hold-out number is also a selection statistic: it is the best of",
        f"  {config.epochs} epochs on the very surface used to choose the epoch, so it is",
        "  optimistically biased by construction.",
        "",
        "For context, what the real B52 measured on the real data",
        "-" * 74,
        f"  frozen control          {B52_REFERENCE.frozen_control_macro_auc:.6f}",
        f"  B52, 1,447 studies      {B52_REFERENCE.gate_split_macro_auc:.6f}",
        f"  B52, 3,801 studies      {B52_REFERENCE.all_data_macro_auc:.6f}",
        f"  measured on             {B52_REFERENCE.evaluation}",
        "",
    ]

    (out_dir / "summary.txt").write_text("\n".join(text), encoding="utf-8")
    print()
    print("\n".join(text))


def run_b52(arguments: argparse.Namespace) -> Path:
    """One complete B52 run, from paths to a folder full of results."""
    started = time.time()
    out_dir = Path(arguments.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_paths(arguments.data_root, out_dir)
    labels_path = Path(arguments.labels).expanduser().resolve()
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"no label export at {labels_path}.\n"
            "Export it with b23_llm_labels.py (or b6_report_labels.py) and pass "
            "it with --labels."
        )
    labels_path = limit_labels(labels_path, arguments.max_studies, out_dir / "scratch")

    config = replace(
        CONFIG,
        epochs=int(arguments.epochs),
        seed=int(arguments.seed),
        batch_size=int(arguments.batch_size),
        num_workers=int(arguments.num_workers),
        learning_rate=float(arguments.learning_rate),
        validation_fraction=float(arguments.validation_fraction),
        image_size=int(arguments.image_size),
        slices_per_series=int(arguments.slices_per_series),
    )
    policy = NO_AUGMENTATION if arguments.no_augment else AUGMENT_PRESETS[arguments.augment_preset]

    print("=" * 74)
    print("B52: building the run")
    print("=" * 74)
    run = build_b52_run(paths, config, policy=policy, labels_path=labels_path)

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "augmentation": asdict(policy),
                "augmentation_preset": "off" if arguments.no_augment else arguments.augment_preset,
                "augmentations_on": describe_augmentation(policy),
                "data_root": str(paths.data_root),
                "labels": str(labels_path),
                "device": str(DEVICE),
                "hierarchy_lr_scale": HIERARCHY_LR_SCALE,
                "trainable_parameters": describe_trainable(run.experiment.model),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    b52_preflight(run)
    if arguments.preflight_only:
        print()
        print("--preflight-only: stopping before training")
        return out_dir

    print()
    print("=" * 74)
    print(f"B52: training for {config.epochs} epochs")
    print("=" * 74)
    history = train_b52(run)
    write_history(history, out_dir)

    print()
    print("scoring the best epoch")
    holdout = score_split(run, run.experiment.validation_loader, "holdout", out_dir)
    write_per_target(holdout, out_dir)
    gold = (
        score_split(run, run.gold_loader, "gold", out_dir)
        if run.gold_loader is not None else None
    )

    if arguments.test_root:
        test_paths = make_test_paths(Path(arguments.test_root).expanduser().resolve())
        predictions = predict_test_set(run.experiment, build_test_loader(test_paths, config))
        predictions.to_csv(out_dir / "test_predictions.csv", index=False)

    plot_curves(history, out_dir)

    torch.save(
        {
            "model_state": run.experiment.model.state_dict(),
            "config": asdict(config),
            "augmentation": asdict(policy),
            "targets": TARGETS,
            "history": history,
            "selected_epoch": run.best.epoch,
            "selection_metric": "macro AUC on a held-out report-labelled split",
            "selection_value": run.best.score,
            "gold_labels_used": False,
            "b52_reference": asdict(B52_REFERENCE),
            "governance": (
                "The selection value is the best of several epochs on the surface "
                "used to choose the epoch, so it is optimistically biased by "
                "construction. It is not an effect size, and a subset run's "
                "absolute value is not comparable with any leaderboard score."
            ),
        },
        out_dir / "best_model.pt",
    )

    describe = describe_report_labels(
        np.stack(
            [
                run.supervision.confidence_by_study[uid]
                for uid in run.experiment.train_loader.dataset.study_uids
            ]
        )
    )
    (out_dir / "labels_summary.json").write_text(json.dumps(describe, indent=2), encoding="utf-8")

    write_summary(
        run, history, holdout, gold, policy, out_dir, (time.time() - started) / 60.0
    )
    print()
    print(f"every result is in {out_dir}")
    return out_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="b52_standalone.py",
        description="Run B52 once and write every result to a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", required=True,
        help="folder holding train.csv, train_series.csv and the DICOM directories",
    )
    parser.add_argument(
        "--labels", required=True,
        help="the report label export (training_targets.csv)",
    )
    parser.add_argument("--out", required=True, help="where to write the results")
    parser.add_argument(
        "--epochs", type=int, default=6,
        help="two is the inherited default and is the thing B52 replaces",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="raise it to feed a fast GPU; each worker is a process and costs host RAM",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--validation-fraction", type=float, default=0.20,
        help="share of report-labelled studies held out to choose the epoch",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="turn augmentation off -- this removes one of B52's three changes",
    )
    parser.add_argument(
        "--augment-preset", choices=sorted(AUGMENT_PRESETS), default="notebook",
        help=(
            "which augmentation strengths to use. 'b53' takes them from the "
            "frozen config, so a subset run rehearses the full-data B53 run; "
            "'notebook' is this file's own, stronger, defaults"
        ),
    )
    parser.add_argument(
        "--test-root", default=None,
        help="optional separate folder with test.csv and test_series.csv",
    )
    parser.add_argument(
        "--max-studies", type=int, default=0,
        help="use only the first N labelled studies, for a quick trial run",
    )
    parser.add_argument(
        "--image-size", type=int, default=CONFIG.image_size,
        help="the geometry every experiment in this line holds fixed; lower it "
             "only for a quick trial run, never for a result",
    )
    parser.add_argument(
        "--slices-per-series", type=int, default=CONFIG.slices_per_series,
        help="also part of the fixed geometry; same warning as --image-size",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="one forward and backward pass, then stop",
    )
    return parser


def main(argv: list | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.image_size != CONFIG.image_size or arguments.slices_per_series != CONFIG.slices_per_series:
        print(
            f"WARNING: geometry changed to {arguments.image_size}px x "
            f"{arguments.slices_per_series} slices, from the fixed "
            f"{CONFIG.image_size}px x {CONFIG.slices_per_series}. Fine for a "
            "trial run; the result is not comparable with anything."
        )
    if arguments.no_augment:
        print(
            "WARNING: --no-augment removes one of the three changes that make this "
            "B52. The run is still valid; it is just not the full regime."
        )
    run_b52(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
