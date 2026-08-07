"""Datasets, series routing and augmentation.

The interesting decisions live in two places.

**Series routing.** An exam may hold anything from two to a dozen series and
the protocol differs by site. Rather than demanding a fixed protocol we score
every series by how useful its (plane, weighting, fat-saturation) combination
is for knee pathology, keep the best few while forcing variety across planes,
and hand the model an integer code describing each one. Exams with unusual
protocols therefore still train, and the model learns what each series type is
worth.

**Augmentation.** Note what is *not* here: horizontal flipping is off by
default. Flipping a coronal knee series swaps medial and lateral, which turns a
medial meniscal tear into a lateral one — it corrupts the label. This is the
single easiest way to lose accuracy on this dataset, so the flag exists but
defaults to ``False``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dicom_io import PLANES, WEIGHTINGS
from .utils import get_logger

LOGGER = get_logger()

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# How valuable each series type is for knee findings, highest first. Fat
# saturated fluid-sensitive sequences show oedema, effusion and synovitis;
# sagittal and coronal proton density show the menisci and cruciate ligaments;
# T1 mainly contributes anatomy and marrow signal.
SERIES_PRIORITY = (
    ("sagittal", "pd", True),
    ("sagittal", "t2", True),
    ("coronal", "pd", True),
    ("coronal", "t2", True),
    ("sagittal", "pd", False),
    ("axial", "pd", True),
    ("axial", "t2", True),
    ("coronal", "pd", False),
    ("sagittal", "t2", False),
    ("coronal", "stir", True),
    ("sagittal", "t1", False),
    ("coronal", "t1", False),
    ("axial", "pd", False),
)


def series_type_code(plane: str, weighting: str, fat_saturated: bool) -> int:
    """Map a series description to a small integer the model can embed."""
    plane_index = PLANES.index(plane) if plane in PLANES else len(PLANES)
    weight_index = WEIGHTINGS.index(weighting) if weighting in WEIGHTINGS else len(WEIGHTINGS) - 1
    return plane_index * (len(WEIGHTINGS) * 2) + weight_index * 2 + int(bool(fat_saturated))


NUM_SERIES_TYPES = (len(PLANES) + 1) * len(WEIGHTINGS) * 2


def series_priority_score(plane: str, weighting: str, fat_saturated: bool) -> float:
    """Lower is better. Unlisted combinations sort after every listed one."""
    for index, (p, w, f) in enumerate(SERIES_PRIORITY):
        if p == plane and w == weighting and f == bool(fat_saturated):
            return float(index)
    return float(len(SERIES_PRIORITY) + (0 if plane in PLANES else 1))


def select_series(rows: pd.DataFrame, max_series: int) -> pd.DataFrame:
    """Pick the most useful series for an exam, favouring variety.

    Duplicated series types (a repeated acquisition, or the same sequence saved
    twice) are collapsed to the one with the most slices, then the highest
    priority types are kept until the budget is full.
    """
    rows = rows.copy()
    rows["priority"] = [
        series_priority_score(r.plane, r.weighting, r.fat_saturated)
        for r in rows.itertuples()
    ]
    rows["type_key"] = [
        f"{r.plane}_{r.weighting}_{int(bool(r.fat_saturated))}" for r in rows.itertuples()
    ]
    rows = rows.sort_values(["priority", "cached_slices"], ascending=[True, False])
    deduplicated = rows.drop_duplicates("type_key", keep="first")

    selected = deduplicated.head(max_series)
    if len(selected) < max_series:
        # Backfill with the repeats we dropped, best first.
        remainder = rows.drop(selected.index, errors="ignore")
        selected = pd.concat([selected, remainder.head(max_series - len(selected))])
    return selected


def sample_slice_indices(
    num_slices: int, depth: int, training: bool, rng: np.random.Generator
) -> np.ndarray:
    """Choose ``depth`` slice positions spanning the series.

    Evaluation takes an even spread so results are deterministic. Training
    jitters each position within its own bin, which acts as a cheap
    through-plane augmentation and means repeated epochs see different slices.
    """
    if num_slices <= 0:
        return np.zeros(depth, dtype=np.int64)
    edges = np.linspace(0, num_slices, depth + 1)
    if training:
        offsets = rng.random(depth)
    else:
        offsets = np.full(depth, 0.5)
    positions = edges[:-1] + offsets * np.diff(edges)
    return np.clip(positions.astype(np.int64), 0, num_slices - 1)


def stack_neighbours(volume: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Build 3-channel slices from each index and its two neighbours.

    This is what makes the model "2.5D": the middle channel is the slice of
    interest and the outer channels give it through-plane context, at no extra
    backbone cost.
    """
    last = volume.shape[0] - 1
    previous = np.clip(indices - 1, 0, last)
    following = np.clip(indices + 1, 0, last)
    return np.stack([volume[previous], volume[indices], volume[following]], axis=1)


@dataclass
class DatasetConfig:
    cache_dir: str = "cache"
    image_size: int = 224
    depth: int = 24
    max_series: int = 5
    augment: bool = True
    horizontal_flip: bool = False  # See the module docstring: this flips medial/lateral.
    rotate_degrees: float = 10.0
    scale_jitter: float = 0.1
    intensity_jitter: float = 0.2
    noise_std: float = 0.01
    series_dropout: float = 0.1
    random_erase: float = 0.15


class KneeExamDataset(Dataset):
    """Yields one exam as ``[S, D, 3, H, W]`` together with its labels."""

    def __init__(
        self,
        frame: pd.DataFrame,
        manifest: pd.DataFrame,
        config: DatasetConfig,
        id_column: str,
        label_columns: Sequence[str] | None = None,
        training: bool = True,
        teacher_columns: Sequence[str] | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.id_column = id_column
        self.label_columns = list(label_columns or [])
        self.teacher_columns = list(teacher_columns or [])
        self.training = training
        self.cache_dir = Path(config.cache_dir)

        manifest = manifest.copy()
        if "fat_saturated" in manifest:
            manifest["fat_saturated"] = manifest["fat_saturated"].astype(bool)
        self.series_by_exam: dict[str, pd.DataFrame] = {
            str(exam): rows for exam, rows in manifest.groupby("exam_id")
        }
        missing = [
            exam for exam in self.frame[id_column].astype(str) if exam not in self.series_by_exam
        ]
        if missing:
            LOGGER.warning(
                "%d of %d exams have no cached series and will yield blanks (e.g. %s)",
                len(missing),
                len(self.frame),
                missing[:3],
            )

    def __len__(self) -> int:
        return len(self.frame)

    def _load_series(self, cache_path: str, rng: np.random.Generator) -> np.ndarray:
        volume = np.load(self.cache_dir / cache_path, mmap_mode="r")
        indices = sample_slice_indices(volume.shape[0], self.config.depth, self.training, rng)
        stack = stack_neighbours(np.asarray(volume), indices).astype(np.float32) / 255.0
        return self._transform(stack, rng)

    def _transform(self, stack: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Resize to the model's input size and apply augmentation."""
        size = self.config.image_size
        stack = _resize_stack(stack, size)

        if self.training and self.config.augment:
            stack = _augment(stack, self.config, rng)

        stack = (stack - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
        return stack.astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        exam_id = str(row[self.id_column])
        seed = abs(hash((exam_id, index, int(self.training)))) % (2**31)
        rng = np.random.default_rng(seed if self.training else 12345)

        config = self.config
        series_rows = self.series_by_exam.get(exam_id)
        volumes: list[np.ndarray] = []
        type_codes: list[int] = []

        if series_rows is not None and len(series_rows):
            selected = select_series(series_rows, config.max_series)
            for series in selected.itertuples():
                if (
                    self.training
                    and config.series_dropout > 0
                    and len(volumes) > 0
                    and rng.random() < config.series_dropout
                ):
                    continue  # Teaches the model to cope with a missing sequence.
                try:
                    volumes.append(self._load_series(series.cache_path, rng))
                except Exception as error:  # pragma: no cover
                    LOGGER.warning("Could not load %s: %s", series.cache_path, error)
                    continue
                type_codes.append(
                    series_type_code(series.plane, series.weighting, bool(series.fat_saturated))
                )

        if not volumes:
            blank = np.zeros(
                (config.depth, 3, config.image_size, config.image_size), dtype=np.float32
            )
            volumes, type_codes = [blank], [NUM_SERIES_TYPES - 1]

        pixels = torch.from_numpy(np.stack(volumes))
        sample = {
            "pixels": pixels,
            "series_type": torch.tensor(type_codes, dtype=torch.long),
            "series_mask": torch.ones(len(volumes), dtype=torch.float32),
            "exam_id": exam_id,
        }
        if self.label_columns:
            sample["labels"] = torch.tensor(
                row[self.label_columns].to_numpy(dtype=np.float32), dtype=torch.float32
            )
        if self.teacher_columns:
            sample["teacher"] = torch.tensor(
                row[self.teacher_columns].to_numpy(dtype=np.float32), dtype=torch.float32
            )
        return sample


def collate_exams(batch: list[dict]) -> dict:
    """Pad a batch to the largest number of series it contains."""
    max_series = max(item["pixels"].shape[0] for item in batch)
    depth, channels, height, width = batch[0]["pixels"].shape[1:]

    pixels = torch.zeros(len(batch), max_series, depth, channels, height, width)
    series_type = torch.zeros(len(batch), max_series, dtype=torch.long)
    series_mask = torch.zeros(len(batch), max_series)

    for index, item in enumerate(batch):
        count = item["pixels"].shape[0]
        pixels[index, :count] = item["pixels"]
        series_type[index, :count] = item["series_type"]
        series_mask[index, :count] = 1.0

    output = {
        "pixels": pixels,
        "series_type": series_type,
        "series_mask": series_mask,
        "exam_id": [item["exam_id"] for item in batch],
    }
    for key in ("labels", "teacher"):
        if key in batch[0]:
            output[key] = torch.stack([item[key] for item in batch])
    return output


def _resize_stack(stack: np.ndarray, size: int) -> np.ndarray:
    """Resize ``[D, 3, H, W]`` to ``[D, 3, size, size]``."""
    if stack.shape[-2:] == (size, size):
        return stack
    try:
        import cv2

        depth = stack.shape[0]
        # cv2 resizes HWC images, so move the channel axis and back.
        moved = np.transpose(stack, (0, 2, 3, 1))
        resized = np.stack(
            [cv2.resize(moved[i], (size, size), interpolation=cv2.INTER_LINEAR) for i in range(depth)]
        )
        return np.transpose(resized, (0, 3, 1, 2))
    except ImportError:
        rows = np.linspace(0, stack.shape[2] - 1, size).round().astype(int)
        columns = np.linspace(0, stack.shape[3] - 1, size).round().astype(int)
        return stack[:, :, rows][:, :, :, columns]


def _augment(stack: np.ndarray, config: DatasetConfig, rng: np.random.Generator) -> np.ndarray:
    """Apply geometric and intensity augmentation consistently across a series.

    One transform is drawn per series rather than per slice: applying different
    rotations to neighbouring slices would destroy the through-plane continuity
    the slice transformer relies on.
    """
    if config.horizontal_flip and rng.random() < 0.5:
        stack = stack[..., ::-1].copy()

    angle = float(rng.uniform(-config.rotate_degrees, config.rotate_degrees))
    scale = 1.0 + float(rng.uniform(-config.scale_jitter, config.scale_jitter))
    shift_x = float(rng.uniform(-0.05, 0.05)) * stack.shape[-1]
    shift_y = float(rng.uniform(-0.05, 0.05)) * stack.shape[-2]

    if abs(angle) > 0.1 or abs(scale - 1.0) > 0.01 or abs(shift_x) > 1 or abs(shift_y) > 1:
        stack = _affine(stack, angle, scale, shift_x, shift_y)

    if config.intensity_jitter > 0:
        gain = 1.0 + float(rng.uniform(-config.intensity_jitter, config.intensity_jitter))
        bias = float(rng.uniform(-config.intensity_jitter, config.intensity_jitter)) * 0.5
        stack = np.clip(stack * gain + bias, 0.0, 1.0)
        # A gamma shift mimics the contrast differences between vendors.
        gamma = math.exp(float(rng.uniform(-0.3, 0.3)))
        stack = np.power(stack, gamma, dtype=np.float32)

    if config.noise_std > 0:
        stack = np.clip(
            stack + rng.normal(0.0, config.noise_std, size=stack.shape).astype(np.float32), 0.0, 1.0
        )

    if config.random_erase > 0 and rng.random() < config.random_erase:
        height, width = stack.shape[-2:]
        box_h = int(height * rng.uniform(0.1, 0.25))
        box_w = int(width * rng.uniform(0.1, 0.25))
        top = int(rng.integers(0, max(1, height - box_h)))
        left = int(rng.integers(0, max(1, width - box_w)))
        stack[..., top : top + box_h, left : left + box_w] = float(stack.mean())

    return stack


def _affine(stack: np.ndarray, angle: float, scale: float, shift_x: float, shift_y: float) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return stack
    depth, channels, height, width = stack.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y
    moved = np.transpose(stack, (0, 2, 3, 1))
    warped = np.stack(
        [
            cv2.warpAffine(moved[i], matrix, (width, height), borderMode=cv2.BORDER_CONSTANT)
            for i in range(depth)
        ]
    )
    if warped.ndim == 3:  # OpenCV drops a trailing axis of size one.
        warped = warped[..., None]
    return np.transpose(warped, (0, 3, 1, 2)).astype(np.float32)
