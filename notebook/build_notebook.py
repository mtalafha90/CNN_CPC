"""Generate the standalone Colab notebook for the working knee-MRI model.

The notebook is written from here rather than edited directly, because a
`.ipynb` is JSON: hand-editing it invites broken escaping and silent
duplicate cell ids, and a diff of it is unreadable. Everything below is the
source of truth; run this file to regenerate `knee_mri_model.ipynb`.

Naming rule for the notebook, and the reason for it: the research code names
its parts after the experiment that introduced them -- `b12_1_hierarchical`,
`b29_complementary_series_pool`, `phase9_matched_supervision_training`. That is
useful in a lab archive where each name pins a frozen comparison, and useless to
a reader meeting the model for the first time. In the notebook every function is
named for what it does. Where a name would otherwise carry a finding, the
comment carries it instead.
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


def build(path: Path) -> Path:
    cells = []
    for kind, text in CELLS:
        lines = text.splitlines(keepends=True)
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": lines,
                }
            )
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return path


markdown(r"""
# Knee MRI abnormality detection — the working model

One notebook, no repository imports, no experiment numbers. Every cell is a
function or a class named for what it does, with the comment explaining *why*
it is that way.

**What this reproduces.** The configuration that scored **0.694 macro ROC AUC**
on the competition's hidden test: a frozen-then-partly-unfrozen ConvNeXt slice
encoder, attention pooling from slices to series, a small transformer across a
study's series, and twelve pathology queries that read out one probability per
finding.

**What it learns from.** Radiology reports, not expert image labels. A rule
parser turns each report into twelve states, and those become soft targets. The
expert-annotated studies never enter the gradient — they exist only to measure.

**Running it.** Set `STUDY_LIMIT` in the config cell. A few hundred studies runs
end to end on a Colab GPU in minutes and prints every intermediate shape;
`None` is the full run and needs the dataset on fast local disk, which Colab is
not well suited to — 24,371 series across 819,078 files.

Two things my audit of this model found, marked in the cells where they live, so
you can experiment from a baseline rather than trust a black box:

- the learning-rate schedule is written for five epochs and stopped at two, so
  it never anneals (`train_model`);
- the encoder average-pools away *where* in the slice anything was, and eight of
  the twelve findings are focal (`SliceFeatureExtractor`).
""")

markdown("## 1. Runtime")

code(r'''
def check_runtime(seed: int = 2026) -> str:
    """Report the GPU and pin every random source we can reach.

    Seeding matters more than usual here: the model trains for a fixed two
    epochs with no checkpoint selection, so a run is decided entirely by its
    starting point and its data order. Two runs with different seeds are
    genuinely different models -- which is what makes averaging them worthwhile
    and comparing them directly misleading.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU        : {device} ({total:.1f} GB)")
    else:
        device = "cpu"
        print("GPU        : none -- this will be far too slow for real training")

    print(f"torch      : {torch.__version__}")
    print(f"seed       : {seed}")
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = check_runtime()
''')

code(r'''
# Colab has torch and torchvision already; pydicom is the one thing missing.
try:
    import pydicom  # noqa: F401
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydicom"], check=True)
    import pydicom  # noqa: F401

import math
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
print("imports ready")
''')

markdown("## 2. Configuration")

code(r'''
# The twelve findings, in the exact order the submission file expects. Changing
# this order silently scrambles every prediction, so it is defined once.
FINDINGS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)


@dataclass
class Config:
    """Every constant in one place, named for what it controls."""

    # --- where the data is -------------------------------------------------
    data_root: str = "/content/rsna-knee-abnormality-detection"
    # None trains on everything. A number keeps the notebook runnable on Colab:
    # the full set is 24,371 series across 819,078 DICOM files.
    study_limit: int | None = 200

    # --- how a study becomes a tensor --------------------------------------
    slices_per_series: int = 16     # evenly spaced positions through the volume
    image_size: int = 224
    triplet_gap: int = 1            # a slice plus its neighbours, as 3 channels
    crop_fraction: float = 0.90     # centre crop, then resize back to 224

    # --- training-time augmentation ----------------------------------------
    train_gap_choices: tuple[int, ...] = (1, 2)
    centre_jitter: int = 2
    rotation_degrees: float = 5.0
    translate_fraction: float = 0.03
    scale_jitter: float = 0.05
    gamma_jitter: float = 0.12
    bias_field_strength: float = 0.08
    noise_std: float = 0.02
    slice_dropout: float = 0.08

    # --- the model ---------------------------------------------------------
    feature_width: int = 768        # ConvNeXt-Tiny's output width
    attention_heads: int = 8
    study_layers: int = 2           # transformer layers across a study's series
    query_layers: int = 1           # self-attention among the 12 queries
    feedforward_multiplier: float = 2.0
    dropout: float = 0.25
    slices_per_encoder_batch: int = 24

    # --- optimisation ------------------------------------------------------
    epochs: int = 2                 # see the note in train_model
    schedule_length: int = 5        # the cosine is written for five
    batch_size: int = 2             # studies per step; series count varies
    head_lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    encoder_lr_fraction: float = 0.05   # unfrozen encoder blocks learn slowly
    trainable_encoder_blocks: int = 1   # the setting that reached 0.694

    # --- how report states become targets ----------------------------------
    # A report saying a finding is present is not proof it is present, so the
    # target sits below 1.0 and the positive half is weighted down. Auditing the
    # parser against expert labels measured 69% agreement for "yes" and 96% for
    # "no", which is the evidence behind the asymmetry -- though 0.85 is higher
    # than that 69% would justify, and lowering it is an open experiment.
    min_label_confidence: float = 0.75
    positive_target: float = 0.85
    negative_target: float = 0.05
    positive_weight: float = 0.50
    negative_weight: float = 1.00

    # --- inference ---------------------------------------------------------
    # Test-time augmentation: the whole comb of slice positions is shifted by
    # one slice each way and the probabilities averaged. With a median stride of
    # 1.9 slices this is a very mild augmentation.
    tta_offsets: tuple[int, ...] = (-1, 0, 1)

    seed: int = 2026


CONFIG = Config()
print(f"{len(FINDINGS)} findings, {CONFIG.slices_per_series} slices per series, "
      f"{CONFIG.image_size}px")
''')


markdown("## 3. Reading the tables")

code(r'''
_TRUE_WORDS = {"true", "t", "yes", "y", "1", "1.0"}
_FALSE_WORDS = {"false", "f", "no", "n", "0", "0.0"}
_PLANE_WORDS = {
    "sagittal": "Sagittal", "sag": "Sagittal", "sagital": "Sagittal",
    "coronal": "Coronal", "cor": "Coronal",
    "axial": "Axial", "ax": "Axial", "transverse": "Axial",
}


def _read_flag(values: pd.Series) -> pd.Series:
    """Parse a yes/no column while keeping 'not stated' distinct from 'no'.

    This distinction is load-bearing. The model receives fluid-sensitive and
    fat-suppression as small categorical embeddings, and 'unknown' is its own
    category. Collapsing missing values to False would tell the model something
    the data never said.
    """
    out = pd.Series(pd.NA, index=values.index, dtype="boolean")
    if pd.api.types.is_bool_dtype(values):
        known = values.notna()
        out.loc[known] = values.loc[known].astype(bool)
    elif pd.api.types.is_numeric_dtype(values):
        known = values.notna()
        out.loc[known] = values.loc[known].astype(float).ne(0.0)
    else:
        text = values.astype("string").str.strip().str.lower()
        out.loc[text.isin(_TRUE_WORDS)] = True
        out.loc[text.isin(_FALSE_WORDS)] = False
    return out


def _read_plane(values: pd.Series) -> pd.Series:
    """Map the many spellings of a scan plane onto three names."""
    text = values.astype("string").str.strip().str.lower()
    return text.map(_PLANE_WORDS).fillna("").astype(str)


def read_study_table(data_root: str, split: str = "train") -> pd.DataFrame:
    """One row per study: its id, its report, and (for 58 studies) expert labels.

    Only a handful of studies carry expert labels. Those never enter training --
    they are the measuring stick, and a measuring stick you have trained on
    measures nothing.
    """
    frame = pd.read_csv(Path(data_root) / f"{split}.csv")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if "Report" not in frame.columns:
        frame["Report"] = ""
    return frame


def read_series_table(data_root: str, split: str = "train") -> pd.DataFrame:
    """One row per MRI series, with its plane and sequence flags normalised."""
    frame = pd.read_csv(Path(data_root) / f"{split}_series.csv")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    frame["SeriesInstanceUID"] = frame["SeriesInstanceUID"].astype(str)
    frame["Fluid_Sensitive"] = _read_flag(frame["Fluid_Sensitive"])
    frame["Fat_Suppression"] = _read_flag(frame["Fat_Suppression"])
    frame["Anatomical_Plane"] = _read_plane(frame["Anatomical_Plane"])
    return frame


def has_expert_labels(studies: pd.DataFrame) -> pd.Series:
    """A study is 'expert-labelled' if any of the twelve columns is filled in."""
    return studies[list(FINDINGS)].notna().any(axis=1)
''')

markdown("## 4. Choosing which series to read")

code(r'''
_PLANE_IDS = {"Sagittal": 1, "Coronal": 2, "Axial": 3}


def _flag_id(value) -> int:
    """0 = not stated, 1 = no, 2 = yes. Zero is also the padding index."""
    if pd.isna(value):
        return 0
    return 2 if bool(value) else 1


def select_usable_series(series: pd.DataFrame, study_uids) -> dict[str, list[dict]]:
    """Every series with a recognisable plane, in a stable order.

    Deliberately *not* a selection: earlier work tried picking one 'best' series
    per plane and per finding, and it lost. All eligible series go in and the
    model decides which to attend to. The only series dropped are those whose
    plane could not be identified at all.

    The sort is for reproducibility rather than meaning -- the model has no
    series-position embedding, so it cannot tell which came first.
    """
    by_study = {uid: part for uid, part in series.groupby("StudyInstanceUID", sort=False)}
    empty = series.iloc[0:0]

    chosen: dict[str, list[dict]] = {}
    for study in study_uids:
        uid = str(study)
        records: list[dict] = []
        for _, row in by_study.get(uid, empty).iterrows():
            plane_id = _PLANE_IDS.get(str(row.get("Anatomical_Plane", "")), 0)
            if plane_id == 0:
                continue
            records.append(
                {
                    "series_uid": str(row["SeriesInstanceUID"]),
                    "plane_id": plane_id,
                    "fluid_id": _flag_id(row.get("Fluid_Sensitive")),
                    "fat_id": _flag_id(row.get("Fat_Suppression")),
                }
            )
        records.sort(key=lambda r: (r["plane_id"], r["fluid_id"], r["fat_id"], r["series_uid"]))
        chosen[uid] = records
    return chosen
''')

markdown("## 5. From DICOM files to a volume")

code(r'''
_DICOM_SUFFIXES = {"", ".dcm", ".dicom", ".ima"}


def find_series_directory(data_root: str, split: str, study: str, series: str) -> Path | None:
    """Locate a series folder, trying the layouts this dataset ships in."""
    root = Path(data_root)
    for candidate in (
        root / f"{split}_series" / str(study) / str(series),
        root / f"{split}_images" / str(study) / str(series),
        root / str(study) / str(series),
    ):
        if candidate.is_dir():
            return candidate
    return None


def _slice_order_key(dataset) -> float:
    """Sort slices along the direction the scanner actually stepped through.

    File names and instance numbers both lie often enough to matter. Projecting
    the patient-space position onto the slice normal gives the true through-plane
    order, so neighbouring slices really are neighbours -- which the three-channel
    triplet below depends on entirely.
    """
    try:
        orientation = np.asarray(dataset.ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        return float(np.dot(np.asarray(dataset.ImagePositionPatient, dtype=float), normal))
    except Exception:
        return float(getattr(dataset, "InstanceNumber", 0))


def _fit_to_shape(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Centre a differently-sized slice inside the series' dominant shape."""
    out = np.zeros(target, dtype=image.dtype)
    rows = min(image.shape[0], target[0])
    cols = min(image.shape[1], target[1])
    sr, sc = (image.shape[0] - rows) // 2, (image.shape[1] - cols) // 2
    tr, tc = (target[0] - rows) // 2, (target[1] - cols) // 2
    out[tr:tr + rows, tc:tc + cols] = image[sr:sr + rows, sc:sc + cols]
    return out


def read_dicom_volume(series_dir: Path) -> np.ndarray:
    """Read one MRI series into a [slices, height, width] float array.

    Rescale slope/intercept are applied, and MONOCHROME1 series are inverted so
    that bright always means high signal. A file that will not decode is skipped
    rather than failing the study -- with 819,078 files across the training set,
    a hard failure on one would stop a run for no clinical reason.
    """
    frames: list[tuple[float, np.ndarray]] = []
    for path in sorted(p for p in series_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in _DICOM_SUFFIXES):
        try:
            dataset = pydicom.dcmread(str(path), force=True)
            pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
            pixels = pixels * float(getattr(dataset, "RescaleSlope", 1.0)) \
                + float(getattr(dataset, "RescaleIntercept", 0.0))
            if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
                pixels = pixels.max() - pixels

            position = _slice_order_key(dataset)
            if pixels.ndim == 2:
                frames.append((position, pixels))
            elif pixels.ndim == 3:   # a multi-frame file holds a whole stack
                frames.extend((position + i * 1e-4, f) for i, f in enumerate(pixels))
        except Exception:
            continue

    if not frames:
        raise RuntimeError(f"no readable DICOM pixels in {series_dir}")

    frames.sort(key=lambda item: item[0])
    images = [image for _, image in frames]
    shapes = {image.shape for image in images}
    if len(shapes) > 1:
        target = max(shapes, key=lambda s: s[0] * s[1])
        images = [_fit_to_shape(image, target) for image in images]
    return np.stack(images).astype(np.float32, copy=False)


def scale_intensities(volume: np.ndarray) -> np.ndarray:
    """Map one series onto [0, 1] using its own 1st and 99th percentiles.

    MRI has no absolute intensity scale -- the same tissue reads differently on
    different scanners and sequences, so a fixed window is meaningless. The
    percentiles are taken over the whole volume rather than per slice, which
    keeps slices comparable to each other; clipping at 1/99 stops a single
    metal artefact from compressing everything else into a narrow band.
    """
    volume = np.asarray(volume, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise RuntimeError("volume contains no finite pixels")
    low, high = np.percentile(finite, [1, 99])
    volume = np.nan_to_num(volume, nan=float(low), posinf=float(high), neginf=float(low))
    volume = np.clip(volume, low, high)
    return ((volume - low) / max(float(high - low), 1e-6)).astype(np.float32, copy=False)
''')

markdown("## 6. Choosing slices and stacking neighbours")

code(r'''
def choose_slice_positions(
    n_frames: int,
    n_positions: int,
    gap: int,
    *,
    offset: int = 0,
    jitter: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Spread sixteen sampling positions evenly through the volume.

    Sixteen positions covers a typical series (median 30 slices) almost
    completely. It does *not* cover the long tail: a few hundred series hold
    over 200 slices, and there a structure a few slices thick can fall between
    two samples entirely. The two weakest findings, ACL and MCL, are exactly
    such structures -- worth remembering before blaming the model.

    `offset` shifts the whole comb (test-time augmentation); `jitter` moves each
    position independently (training only). They are never used together.
    """
    if n_frames < 1:
        raise ValueError("a series needs at least one slice")
    low, high = (gap, n_frames - 1 - gap) if n_frames > 2 * gap else (0, n_frames - 1)
    positions = np.round(np.linspace(low, high, n_positions)).astype(int) + int(offset)
    if jitter > 0:
        rng = rng or np.random.default_rng()
        positions = positions + rng.integers(-int(jitter), int(jitter) + 1, size=n_positions)
    return np.clip(positions, 0, n_frames - 1)


def build_slice_triplets(
    volume: np.ndarray,
    *,
    n_positions: int,
    image_size: int,
    gap: int,
    offset: int = 0,
    jitter: int = 0,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Turn a volume into [positions, 3, size, size] of neighbouring slices.

    Each sampled position becomes three channels: the slice before, the slice
    itself, and the slice after. The encoder is a 2D network pretrained on
    photographs, so this is how a little through-plane context reaches it
    without moving to a 3D architecture -- cheap, and it keeps the pretrained
    weights usable.
    """
    if gap < 1:
        raise ValueError("the neighbour gap must be at least one slice")
    volume = scale_intensities(volume)
    centres = choose_slice_positions(
        len(volume), n_positions, gap, offset=offset, jitter=jitter, rng=rng
    )
    neighbours = np.asarray([-gap, 0, gap], dtype=int)
    index = np.clip(centres[:, None] + neighbours[None, :], 0, len(volume) - 1)
    stacked = torch.from_numpy(volume[index].astype(np.float32, copy=False))
    return F.interpolate(stacked, (image_size, image_size), mode="bilinear", align_corners=False)
''')

markdown("## 7. The centre crop")

code(r'''
def crop_centre_and_resize(volume: torch.Tensor, crop_fraction: float) -> torch.Tensor:
    """Keep the middle 90% of the field of view, then resize back to 224.

    Knee MRI is framed with the joint near the centre and a margin of air, coil
    and soft tissue around it. Trimming that margin makes the joint fill more of
    the input.

    Worth knowing before you build on this: the crop happens *after* the resize
    to 224, so the effective support is 202x202 upsampled back to 224, and the
    output carries no detail the 202-pixel crop did not. Cropping at native
    resolution first was tried and lost on the expert surface, but in-plane
    resolution itself has never been varied.
    """
    if volume.ndim < 4:
        raise ValueError(f"expected [..., C, H, W], got {tuple(volume.shape)}")
    height, width = int(volume.shape[-2]), int(volume.shape[-1])
    channels = int(volume.shape[-3])
    original = tuple(volume.shape)
    flat = volume.reshape(-1, channels, height, width)

    crop_h = max(2, min(height, int(round(height * crop_fraction))))
    crop_w = max(2, min(width, int(round(width * crop_fraction))))
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    cropped = flat[:, :, top:top + crop_h, left:left + crop_w]
    if (crop_h, crop_w) != (height, width):
        cropped = F.interpolate(
            cropped, size=(height, width), mode="bilinear", align_corners=False
        )
    return cropped.reshape(original)
''')

markdown("## 8. Training-time augmentation")

code(r'''
def augment_training_volume(volume: torch.Tensor, config: "Config") -> torch.Tensor:
    """Mild acquisition-like distortion, drawn once and shared across a series.

    One draw per series, not per slice: the slices of a series were acquired in
    one go, so rotating them independently would create a physically impossible
    volume and teach the model to ignore through-plane consistency.

    Every transform imitates something a scanner or a patient actually does --
    slight repositioning, receiver-coil shading, thermal noise. Nothing here
    flips the image, because left and right knees are not interchangeable.
    """
    import torchvision.transforms.functional as TVF
    from torchvision.transforms import InterpolationMode

    angle = float(torch.empty(1).uniform_(-config.rotation_degrees, config.rotation_degrees))
    max_shift = int(round(config.translate_fraction * config.image_size))
    translate = [
        int(torch.randint(-max_shift, max_shift + 1, (1,)).item()) if max_shift else 0,
        int(torch.randint(-max_shift, max_shift + 1, (1,)).item()) if max_shift else 0,
    ]
    scale = float(torch.empty(1).uniform_(1 - config.scale_jitter, 1 + config.scale_jitter))
    volume = TVF.affine(
        volume, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
    )

    if config.gamma_jitter > 0:      # overall brightness curve
        gamma = float(torch.empty(1).uniform_(1 - config.gamma_jitter, 1 + config.gamma_jitter))
        volume = volume.clamp(0, 1).pow(gamma)

    if config.bias_field_strength > 0:   # smooth shading, as a coil produces
        height, width = volume.shape[-2:]
        yy = torch.linspace(-1, 1, height, device=volume.device).view(1, 1, height, 1)
        xx = torch.linspace(-1, 1, width, device=volume.device).view(1, 1, 1, width)
        ax = float(torch.empty(1).uniform_(-config.bias_field_strength, config.bias_field_strength))
        ay = float(torch.empty(1).uniform_(-config.bias_field_strength, config.bias_field_strength))
        volume = (volume * (1 + ax * xx + ay * yy).clamp(0.8, 1.2)).clamp(0, 1)

    if config.noise_std > 0:
        volume = (volume + torch.randn_like(volume) * config.noise_std).clamp(0, 1)

    if config.slice_dropout > 0:
        # Blank a few sampled positions outright, so the study head cannot come
        # to depend on any single slice being present.
        volume = volume.clone()
        volume[torch.rand(volume.shape[0]) < config.slice_dropout] = 0
    return volume
''')


markdown("## 9. One study as tensors")

code(r'''
class KneeStudyDataset(Dataset):
    """One item is one study: all of its series, stacked.

    Studies differ in how many series they have -- three to fourteen, median
    five -- so an item has a ragged first dimension and the collate function
    below pads it. A series that fails to read comes back as zeros with its
    presence flag off, so one unreadable folder cannot stop a run.
    """

    def __init__(self, study_uids, series_by_study, config: "Config", *,
                 split="train", targets=None, weights=None, train=False,
                 tta_offsets=()):
        self.study_uids = [str(u) for u in study_uids]
        self.series_by_study = series_by_study
        self.config = config
        self.split = split
        self.targets = targets
        self.weights = weights
        self.train = bool(train)
        self.tta_offsets = tuple(int(o) for o in tta_offsets)

    def __len__(self) -> int:
        return len(self.study_uids)

    def _blank(self) -> torch.Tensor:
        shape = (self.config.slices_per_series, 3, self.config.image_size, self.config.image_size)
        blank = torch.zeros(shape, dtype=torch.float32)
        if self.tta_offsets:
            return blank.unsqueeze(0).repeat(len(self.tta_offsets), 1, 1, 1, 1)
        return blank

    def _read_one_series(self, study: str, series: str):
        directory = find_series_directory(self.config.data_root, self.split, study, series)
        if directory is None:
            return self._blank(), 0.0
        try:
            volume = read_dicom_volume(directory)
        except Exception:
            return self._blank(), 0.0

        if self.train:
            # A random neighbour gap and jittered positions, so the model never
            # sees exactly the same sixteen slices of a series twice.
            gap = int(self.config.train_gap_choices[
                int(torch.randint(len(self.config.train_gap_choices), (1,)).item())])
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            sampled = build_slice_triplets(
                volume, n_positions=self.config.slices_per_series,
                image_size=self.config.image_size, gap=gap,
                jitter=self.config.centre_jitter, rng=rng,
            )
            return augment_training_volume(sampled, self.config), 1.0

        if self.tta_offsets:
            views = [
                build_slice_triplets(
                    volume, n_positions=self.config.slices_per_series,
                    image_size=self.config.image_size, gap=self.config.triplet_gap,
                    offset=offset,
                )
                for offset in self.tta_offsets
            ]
            return torch.stack(views, dim=0), 1.0

        return build_slice_triplets(
            volume, n_positions=self.config.slices_per_series,
            image_size=self.config.image_size, gap=self.config.triplet_gap,
        ), 1.0

    def __getitem__(self, index: int) -> dict:
        uid = self.study_uids[index]
        volumes, present, metadata = [], [], []
        for record in self.series_by_study[uid]:
            volume, flag = self._read_one_series(uid, record["series_uid"])
            volumes.append(volume)
            present.append(flag)
            metadata.append([record["plane_id"], record["fluid_id"], record["fat_id"]])

        stacked = torch.stack(volumes)
        if self.tta_offsets:
            stacked = stacked.permute(1, 0, 2, 3, 4, 5).contiguous()   # views first

        # The crop is applied here, once, to the whole study at once.
        stacked = crop_centre_and_resize(stacked, self.config.crop_fraction)

        item = {
            "study_uid": uid,
            "volumes": stacked,
            "present": torch.tensor(present, dtype=torch.float32),
            "series_meta": torch.tensor(metadata, dtype=torch.long),
        }
        if self.targets is not None:
            item["target"] = torch.from_numpy(np.asarray(self.targets[index], dtype=np.float32))
        if self.weights is not None:
            item["weight"] = torch.from_numpy(np.asarray(self.weights[index], dtype=np.float32))
        return item
''')

markdown("## 10. Padding studies into a batch")

code(r'''
def pad_studies_into_batch(items: list[dict]) -> dict:
    """Pad to the largest series count *in this batch*, not to a global maximum.

    Study series counts run from three to fourteen. Padding everything to
    fourteen would waste most of the encoder's work on zeros, so the batch is
    padded only as far as its own widest member -- which is also why the batch
    size is two: with a bigger batch, one fourteen-series study drags every
    other study in it up to fourteen.
    """
    if not items:
        raise ValueError("cannot collate an empty batch")

    widest = max(int(item["present"].shape[0]) for item in items)
    first = items[0]["volumes"]
    n = len(items)

    if first.ndim == 5:                                 # [K, S, C, H, W]
        _, s, c, h, w = first.shape
        volumes = first.new_zeros((n, widest, s, c, h, w))
        for i, item in enumerate(items):
            k = item["volumes"].shape[0]
            volumes[i, :k] = item["volumes"]
    elif first.ndim == 6:                               # [V, K, S, C, H, W]
        v, _, s, c, h, w = first.shape
        volumes = first.new_zeros((n, v, widest, s, c, h, w))
        for i, item in enumerate(items):
            k = item["volumes"].shape[1]
            volumes[i, :, :k] = item["volumes"]
    else:
        raise ValueError(f"unexpected volume shape {tuple(first.shape)}")

    present = torch.zeros((n, widest), dtype=torch.float32)
    metadata = torch.zeros((n, widest, 3), dtype=torch.long)   # zero == padding index
    for i, item in enumerate(items):
        k = item["present"].shape[0]
        present[i, :k] = item["present"]
        metadata[i, :k] = item["series_meta"]

    batch = {
        "study_uid": [str(item["study_uid"]) for item in items],
        "volumes": volumes,
        "present": present,
        "series_meta": metadata,
    }
    if all("target" in item for item in items):
        batch["target"] = torch.stack([item["target"] for item in items])
    if all("weight" in item for item in items):
        batch["weight"] = torch.stack([item["weight"] for item in items])
    return batch
''')

markdown("## 11. The slice encoder")

code(r'''
class SliceFeatureExtractor(nn.Module):
    """Turn one 224x224 three-channel slice into a 768-number description.

    A ConvNeXt-Tiny pretrained on photographs. Natural-image pretraining
    transfers here because early layers learn edges and textures, which knee MRI
    also has; the later layers are what needs adapting, which is why fine-tuning
    starts from the output end.

    **The known limitation.** The final 7x7x768 feature map is average-pooled to
    a single 768-vector, so *where* in the slice something appeared is discarded
    entirely. Eight of the twelve findings are focal -- ligaments, menisci,
    compartment-specific arthritis, fracture -- and for those, "somewhere in this
    slice" is close to the least useful summary available. Keeping a 2x2 or 3x3
    grid instead of one vector is the most promising untried change to this
    model.
    """

    def __init__(self, *, pretrained_weights: bool = False, normalise_input: bool = True):
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained_weights else None
        network = convnext_tiny(weights=weights)
        self.features = network.features
        self.avgpool = network.avgpool
        self.pre_classifier = nn.Sequential(*list(network.classifier.children())[:-1])
        self.out_dim = int(network.classifier[-1].in_features)      # 768
        self.normalise_input = bool(normalise_input)

        # The three channels are adjacent anatomical slices, not red/green/blue,
        # yet they are normalised with per-channel ImageNet statistics. That is
        # inherited from the pretrained weights and kept deliberately: changing
        # it would invalidate every stored checkpoint.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("input_mean", mean, persistent=False)
        self.register_buffer("input_std", std, persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.normalise_input:
            images = (images - self.input_mean.to(images.dtype)) / self.input_std.to(images.dtype)
        features = self.features(images)          # [N, 768, 7, 7]
        pooled = self.avgpool(features)           # [N, 768, 1, 1]
        return self.pre_classifier(pooled)        # [N, 768]
''')

markdown("## 12. The whole model")

code(r'''
class KneeAbnormalityModel(nn.Module):
    """Slices -> series -> study -> twelve findings.

    Four stages, each collapsing one level of structure:

    1. every slice of every real series becomes a 768-vector;
    2. a series' sixteen slice vectors become one series vector, by two
       different pooling routes blended through a learned gate;
    3. a small transformer lets a study's series see each other;
    4. twelve learned queries read out one logit per finding.

    Two details that look like ornament and are not:

    * The gate starts at zero, so the second pooling route contributes exactly
      nothing at initialisation and only enters if training pulls it in.
    * A small depthwise convolution gives each slice a little context from its
      neighbours **during training only**, and is skipped exactly at evaluation.
      It behaves as a scaffold: it shapes what gets learned without changing the
      scoring function you eventually deploy.
    """

    def __init__(self, config: "Config", *, pretrained_encoder: bool = False):
        super().__init__()
        self.config = config
        self.encoder = SliceFeatureExtractor(pretrained_weights=pretrained_encoder)
        width = self.encoder.out_dim                         # 768
        heads = config.attention_heads
        feedforward = int(width * config.feedforward_multiplier)

        # --- what a slice knows about itself besides its pixels -------------
        self.slice_position = nn.Parameter(torch.randn(config.slices_per_series, width) * 0.02)
        # Index 0 means "not stated" and is held at zero -- the model is told the
        # difference between "no fat suppression" and "nobody recorded it".
        self.plane_embedding = nn.Embedding(4, width, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, width, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, width, padding_idx=0)

        # --- route A: one learned query attends over the sixteen slices ------
        self.pool_query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pool_attention = nn.MultiheadAttention(
            width, heads, dropout=config.dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(width)
        self.pool_dropout = nn.Dropout(config.dropout)

        # --- route B: a plain softmax summary, blended in by a zero gate -----
        self.summary_query = nn.Parameter(torch.randn(width) * 0.02)
        self.summary_gate = nn.Parameter(torch.zeros(width))

        # --- the training-only neighbour context ----------------------------
        self.local_context = nn.Conv1d(width, width, kernel_size=3, padding=1,
                                       groups=width, bias=False)
        nn.init.zeros_(self.local_context.weight)

        # --- the study transformer ------------------------------------------
        study_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=feedforward,
            dropout=config.dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.study_context = nn.TransformerEncoder(
            study_layer, num_layers=config.study_layers,
            norm=nn.LayerNorm(width), enable_nested_tensor=False)

        # --- the twelve pathology queries ------------------------------------
        self.finding_queries = nn.Parameter(torch.randn(len(FINDINGS), width) * 0.02)
        query_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=feedforward,
            dropout=config.dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.query_context = nn.TransformerEncoder(
            query_layer, num_layers=config.query_layers,
            norm=nn.LayerNorm(width), enable_nested_tensor=False)

        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=config.dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(config.dropout)

        # Each finding reads only its own query row -- a block-diagonal head
        # rather than a shared linear layer, so the twelve outputs cannot
        # borrow each other's features at the very last step.
        self.finding_weight = nn.Parameter(torch.empty(len(FINDINGS), width))
        self.finding_bias = nn.Parameter(torch.zeros(len(FINDINGS)))
        nn.init.xavier_uniform_(self.finding_weight)

    # ---------------------------------------------------------------- stage 1
    def _describe_slices(self, volumes, present, series_meta):
        """[B,K,S,3,H,W] -> [B,K,S,768], encoding only the series that exist."""
        b, k, s, c, h, w = volumes.shape
        width = self.encoder.out_dim
        flat = volumes.reshape(b * k, s, c, h, w)
        real = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        if real.numel() == 0:
            return volumes.new_zeros((b, k, s, width)), real

        active = flat.index_select(0, real)
        slices = active.reshape(-1, c, h, w)
        # Chunked so a study with many series cannot exhaust GPU memory.
        encoded = torch.cat(
            [self.encoder(chunk)
             for chunk in slices.split(self.config.slices_per_encoder_batch, dim=0)],
            dim=0,
        ).reshape(active.shape[0], s, width)

        features = encoded.new_zeros((b * k, s, width)).index_copy(0, real, encoded)
        features = features.reshape(b, k, s, width)

        metadata = (self.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
                    + self.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
                    + self.fat_embedding(series_meta[:, :, 2].clamp(0, 2)))
        mask = present[:, :, None, None].to(features.dtype)
        described = (features
                     + self.slice_position[None, None, :, :]
                     + metadata[:, :, None, :]) * mask
        return described, real

    # ---------------------------------------------------------------- stage 2
    def _summarise_series(self, described, present, real):
        """[B,K,S,768] -> [B,K,768], one vector per series."""
        b, k, s, width = described.shape
        flat = described.reshape(b * k, s, width)
        if real.numel() == 0:
            return described.new_zeros((b, k, width))

        slices = flat.index_select(0, real)                        # [N, S, 768]

        # Route A: a learned query attends over the slices.
        query = self.pool_query.expand(slices.shape[0], -1, -1)
        attended, _ = self.pool_attention(query, slices, slices, need_weights=False)
        route_a = self.pool_dropout(self.pool_norm(query + attended)).squeeze(1)

        # The scaffold: neighbour context while training, exact identity at eval.
        if self.training:
            normed = F.layer_norm(slices.float(), (width,)).to(slices.dtype)
            scored_on = slices + self.local_context(
                normed.transpose(1, 2)).transpose(1, 2)
        else:
            scored_on = slices

        # Route B: softmax weights from `scored_on`, but the values summed are
        # always the original slices -- context decides *what to look at*, never
        # *what is reported*. That is what makes the eval bypass exact.
        scores = torch.matmul(scored_on, self.summary_query.to(scored_on.dtype)) \
            / math.sqrt(float(width))
        attention = torch.softmax(scores.float(), dim=1).to(slices.dtype)
        summed = torch.sum(attention[:, :, None] * slices, dim=1)
        route_b = F.layer_norm(summed.float(), (width,)).to(slices.dtype)

        gate = torch.tanh(self.summary_gate).to(route_a.dtype)
        blended = route_a + gate[None, :] * (route_b - route_a)
        return blended.new_zeros((b * k, width)).index_copy(
            0, real, blended).reshape(b, k, width)

    # ---------------------------------------------------------------- forward
    def forward(self, volumes, present, series_meta):
        described, real = self._describe_slices(volumes, present, series_meta)
        series = self._summarise_series(described, present, real)

        padding = present <= 0
        empty = padding.all(dim=1)
        # Attention over an entirely masked sequence is undefined, so a study
        # with no readable series is given one zero key to attend to and its
        # output is replaced by the learned bias at the very end.
        safe_padding = padding.clone()
        if empty.any():
            safe_padding[empty, 0] = False
            series = series.clone()
            series[empty, 0] = 0

        memory = self.study_context(series, src_key_padding_mask=safe_padding)
        memory = memory.masked_fill(padding[:, :, None], 0.0)

        b = volumes.shape[0]
        queries = self.finding_queries[None, :, :].expand(b, -1, -1)
        queries = self.query_context(queries)
        attended, _ = self.cross_attention(
            queries, memory, memory, key_padding_mask=safe_padding, need_weights=False)
        queries = self.dropout(self.query_norm(queries + attended))

        logits = (queries * self.finding_weight[None, :, :]).sum(dim=-1) + self.finding_bias
        return torch.where(empty[:, None], self.finding_bias[None, :], logits)
''')


markdown("## 13. Report states into training targets")

code(r'''
def build_supervision_targets(studies: pd.DataFrame, labels: pd.DataFrame, config: "Config"):
    """Turn parsed report states into soft targets and per-cell weights.

    The label file holds two columns per finding: a state (`positive`,
    `negated`, `uncertain`, `unmentioned`) and a confidence. Only definite
    states above the confidence threshold are supervised at all -- roughly a
    quarter of all cells. The rest carry weight zero and contribute nothing,
    because a report not mentioning a finding is not the same as a report
    denying it.

    The asymmetry is measured, not assumed. Against expert labels the parser's
    "yes" agrees 69% of the time and its "no" 96%, so a "yes" is aimed at 0.85
    and counted at half weight, while a "no" is aimed at 0.05 at full weight.
    (0.85 is still more confident than 69% would justify -- `positive_target` is
    exposed in the config precisely so that can be tested.)
    """
    labelled = studies.loc[~has_expert_labels(studies), ["StudyInstanceUID"]].copy()
    ordered = labelled.merge(labels, on="StudyInstanceUID", how="left", validate="one_to_one")

    n = len(ordered)
    targets = np.full((n, len(FINDINGS)), 0.5, dtype=np.float32)   # never supervised
    weights = np.zeros((n, len(FINDINGS)), dtype=np.float32)

    for j, finding in enumerate(FINDINGS):
        state = ordered.get(f"{finding}__state", pd.Series([""] * n)).fillna("").astype(str).to_numpy()
        confidence = pd.to_numeric(
            ordered.get(f"{finding}__confidence", pd.Series([0.0] * n)), errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)

        confident = confidence >= config.min_label_confidence
        says_yes = (state == "positive") & confident
        says_no = (state == "negated") & confident

        targets[says_yes, j] = config.positive_target
        weights[says_yes, j] = config.positive_weight
        targets[says_no, j] = config.negative_target
        weights[says_no, j] = config.negative_weight

    supervised = int((weights > 0).sum())
    print(f"{n} studies, {supervised:,} supervised cells "
          f"({supervised / (n * len(FINDINGS)) * 100:.1f}% of the label surface)")
    return ordered["StudyInstanceUID"].astype(str).tolist(), targets, weights
''')

markdown("## 14. The loss")

code(r'''
def finding_balance_multipliers(weights: np.ndarray) -> np.ndarray:
    """Make every finding contribute equally, whatever its supervision count.

    Effusion has five times the supervised cells of Synovitis. Without this,
    the loss would be dominated by whichever findings radiologists happen to
    mention most, and the twelve-way average the competition scores would be
    optimised unevenly. Each finding is scaled by (mean mass / its own mass), so
    all twelve carry a twelfth of the gradient.

    One consequence worth knowing: a finding with very few cells gets each of
    them inflated heavily. Synovitis has 434 positives to 17 negatives, so that
    inflation is applied to a signal that is 92% "yes" -- the model is told
    loudly to predict Synovitis present.
    """
    mass = np.asarray(weights, dtype=np.float64).sum(axis=0)
    if not (mass > 0).all():
        missing = [FINDINGS[j] for j in np.flatnonzero(mass <= 0)]
        raise ValueError(f"no supervision at all for: {missing}")
    return (float(mass.mean()) / mass).astype(np.float32)


def weighted_weak_bce(logits, targets, weights, multipliers):
    """Cross-entropy over supervised cells only, averaged within the batch.

    Dividing by this batch's own total weight rather than a fixed constant keeps
    the loss the same size whether a batch happens to contain two supervised
    cells or twenty, so the gradient does not swing with label density.

    The zero guard matters: studies whose report yielded nothing are kept in the
    loader so that every model sees the same scans, and a batch of two such
    studies has no supervised cells at all. Returning `logits.sum() * 0` keeps
    the graph intact and produces exact-zero gradients, where a plain 0/0 would
    produce NaN and poison the weights.
    """
    multiplier = torch.as_tensor(multipliers, dtype=logits.dtype, device=logits.device)
    effective = weights * multiplier[None, :]
    denominator = effective.sum()
    if float(denominator.detach().item()) <= 0:
        return logits.sum() * 0.0
    per_cell = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (per_cell * effective).sum() / denominator.clamp_min(1e-8)
''')

markdown("## 15. Which weights are allowed to learn")

code(r'''
# ConvNeXt's stages, listed from the output backwards. Unfreezing works from
# this end: early layers see edges and textures, which transfer from
# photographs perfectly well, while the late layers carry object-level meaning,
# which is where knee MRI differs most.
ENCODER_BLOCKS_FROM_OUTPUT = (
    ("pre_classifier", "features.7"),   # last stage plus its output norm
    ("features.6",),                    # the downsample feeding it
    ("features.5",),
    ("features.4",),
    ("features.3",),
)


def freeze_encoder(model: KneeAbnormalityModel) -> None:
    """Hold the encoder still and keep it in eval mode."""
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()


def unfreeze_last_encoder_blocks(model: KneeAbnormalityModel, blocks: int) -> int:
    """Let the last N encoder blocks learn. Returns how many weights were freed.

    Two restraints, both deliberate:

    * The encoder stays in `eval()` even when unfrozen. ConvNeXt normalises with
      LayerNorm, which behaves the same either way, but eval also disables
      stochastic depth -- so the forward pass keeps exactly the shape it had and
      only the flow of gradients changes.
    * The freed weights get their own, much smaller learning rate. Pretrained
      features are worth more than a randomly initialised head and are easily
      destroyed by the head's step size.

    Unfreezing one block is what took the hidden-test score from 0.688 to 0.694.
    That is a smaller gap than a single submission can resolve, so treat it as
    a direction worth probing rather than a settled result.
    """
    if not 0 <= blocks <= len(ENCODER_BLOCKS_FROM_OUTPUT):
        raise ValueError(f"blocks must be 0..{len(ENCODER_BLOCKS_FROM_OUTPUT)}")
    if blocks == 0:
        return 0

    prefixes = tuple(name for block in ENCODER_BLOCKS_FROM_OUTPUT[:blocks] for name in block)
    freed = 0
    for name, parameter in model.encoder.named_parameters():
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            parameter.requires_grad_(True)
            freed += parameter.numel()
    if freed == 0:
        raise RuntimeError(f"nothing matched {prefixes}; the encoder layout changed")
    model.encoder.eval()
    print(f"freed {freed:,} encoder weights in the last {blocks} block(s)")
    return freed


def build_parameter_groups(model: KneeAbnormalityModel, config: "Config") -> list[dict]:
    """Two learning rates: the head's, and a much gentler one for the encoder."""
    head, encoder = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (encoder if name.startswith("encoder.") else head).append(parameter)
    groups = [{"params": head, "lr": config.head_lr, "name": "head"}]
    if encoder:
        groups.append({
            "params": encoder,
            "lr": config.head_lr * config.encoder_lr_fraction,
            "name": "encoder",
        })
    print(f"trainable: {sum(p.numel() for p in head):,} head"
          + (f" + {sum(p.numel() for p in encoder):,} encoder" if encoder else ""))
    return groups
''')

markdown("## 16. Training")

code(r'''
def train_model(model, loader, config: "Config", device: str):
    """Train to a fixed endpoint, with no checkpoint chosen by looking at a score.

    The endpoint is fixed on purpose. With only 58 expert-labelled studies to
    measure against, picking "the best epoch" would mostly be picking the
    luckiest noise, and the score would then flatter itself. Deciding the number
    of epochs before seeing any result costs a little performance and buys an
    honest number.

    **The known oddity.** The cosine schedule is written for five epochs and the
    loop runs two, so the learning rate never anneals -- the two epochs train at
    100% and 90.5% of peak, and the low-rate refinement phase where a cosine
    normally delivers its final gain simply does not happen. It is kept here
    because it is what produced 0.694 and changing it changes the model. Setting
    `schedule_length = 2` is the cheapest experiment available on this notebook:
    identical cost, properly annealed.

    Training longer was tried and made things worse: the loss kept falling while
    accuracy fell with it, which is what happens when a model gets better at
    reproducing labels that are partly wrong. Longer training is worth retrying
    only on better labels.
    """
    model.to(device)
    optimiser = torch.optim.AdamW(
        build_parameter_groups(model, config), weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config.schedule_length, eta_min=config.min_lr)

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    trainable = [p for p in model.parameters() if p.requires_grad]
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        model.encoder.eval()     # re-asserted every epoch; see the note above
        running, steps = 0.0, 0

        for step, batch in enumerate(loader, start=1):
            volumes = batch["volumes"].to(device, non_blocking=True)
            present = batch["present"].to(device, non_blocking=True)
            metadata = batch["series_meta"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            weights = batch["weight"].to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(volumes, present, metadata)
                loss = weighted_weak_bce(logits, targets, weights, model.balance_multipliers)

            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            scaler.step(optimiser)
            scaler.update()

            running += float(loss.item())
            steps += 1
            if step % 100 == 0:
                print(f"  epoch {epoch}  step {step}/{len(loader)}  "
                      f"loss {running / steps:.4f}")

        scheduler.step()      # once per epoch, not per batch
        mean_loss = running / max(steps, 1)
        head_lr = optimiser.param_groups[0]["lr"]
        history.append({"epoch": epoch, "loss": mean_loss, "head_lr": head_lr})
        print(f"epoch {epoch}: loss {mean_loss:.4f}, next head lr {head_lr:.2e}")

    return history
''')

markdown("## 17. Scoring")

code(r'''
def binary_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve for one finding, with ties handled properly.

    Written out rather than imported so the notebook has no sklearn dependency,
    and so the tie handling is visible: equal scores share a mid-rank. A finding
    with no positives or no negatives has no defined AUC and returns NaN.
    """
    truth = np.asarray(truth, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    usable = np.isfinite(truth) & np.isfinite(score)
    truth, score = truth[usable], score[usable]

    positive = truth == 1
    n_pos, n_neg = int(positive.sum()), int(positive.size - positive.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1, dtype=np.float64)
    sorted_scores = score[order]
    edges = np.flatnonzero(np.diff(sorted_scores)) + 1
    for start, stop in zip(np.concatenate(([0], edges)),
                           np.concatenate((edges, [sorted_scores.size]))):
        if stop - start > 1:
            ranks[order[start:stop]] = ranks[order[start:stop]].mean()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_roc_auc(truth: np.ndarray, score: np.ndarray):
    """The competition metric: the mean AUC across the twelve findings.

    Undefined findings are dropped from the mean rather than scored 0.5, so on a
    small evaluation set the macro may average fewer than twelve numbers -- worth
    checking before comparing two runs.
    """
    per_finding = np.array(
        [binary_auc(truth[:, j], score[:, j]) for j in range(truth.shape[1])],
        dtype=np.float64,
    )
    usable = per_finding[np.isfinite(per_finding)]
    return (float(usable.mean()) if usable.size else float("nan")), per_finding
''')

markdown("## 18. Prediction")

code(r'''
@torch.no_grad()
def predict_with_augmentation(models, loader, device: str):
    """Score every study, averaging over slice offsets and over models.

    Each study is read once and scored under three slice-sampling offsets; the
    probabilities are averaged. Reading the scans is by far the expensive part,
    so several models are scored in the same pass rather than one pass each.

    Two details that change the numbers if you get them wrong:

    * `model.eval()` is what switches off the training-only neighbour context.
      Forgetting it silently scores with a different function than intended.
    * Averaging happens in probability space, after a float32 sigmoid. Averaging
      logits instead gives different -- and worse-calibrated -- answers.
    """
    for model in models:
        model.to(device).eval()

    uids, blocks = [], []
    for batch in loader:
        present = batch["present"].to(device, non_blocking=True)
        metadata = batch["series_meta"].to(device, non_blocking=True)
        volumes = batch["volumes"]

        views = []
        n_views = volumes.shape[1] if volumes.ndim == 7 else 1
        for view in range(n_views):
            frame = (volumes[:, view] if volumes.ndim == 7 else volumes).to(
                device, non_blocking=True)
            for model in models:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits = model(frame, present, metadata)
                views.append(torch.sigmoid(logits.float()))

        blocks.append(torch.stack(views, dim=0).mean(dim=0).cpu().numpy())
        uids.extend(str(uid) for uid in batch["study_uid"])

    return uids, np.concatenate(blocks, axis=0)
''')

markdown("## 19. Writing the submission")

code(r'''
def write_submission(uids, probabilities, path="submission.csv") -> pd.DataFrame:
    """Write the thirteen-column file the competition expects, and check it.

    Row order must match `test.csv` exactly and the column names are the display
    names, apostrophe and all. The checks below are cheap and catch the two
    failures that produce a plausible-looking file with meaningless contents:
    a reordered study list, and probabilities that escaped [0, 1].
    """
    frame = pd.DataFrame(np.asarray(probabilities, dtype=np.float64), columns=list(FINDINGS))
    frame.insert(0, "StudyInstanceUID", [str(u) for u in uids])

    assert list(frame.columns) == ["StudyInstanceUID", *FINDINGS], "column order changed"
    assert not frame["StudyInstanceUID"].duplicated().any(), "a study appears twice"
    values = frame[list(FINDINGS)].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all(), "non-finite probability"
    assert (values >= 0).all() and (values <= 1).all(), "probability outside [0, 1]"

    frame.to_csv(path, index=False)
    print(f"wrote {path}: {len(frame)} studies x {len(FINDINGS)} findings")

    # Every column should vary across studies. One that does not means the model
    # is giving every knee the same answer for that finding.
    spread = (frame[list(FINDINGS)].max() - frame[list(FINDINGS)].min()).sort_values()
    print("\nsmallest spreads across studies:")
    print(spread.head(3).to_string())
    if spread.max() < 0.01:
        print("\nWARNING: every column is nearly constant -- do not submit this")
    return frame
''')


markdown("""
## 20. Running it

The cells below are the only ones that touch your data. Everything above is
definitions, so you can read the whole model before running anything.
""")

code(r'''
# --- point these at your data -------------------------------------------------
CONFIG.data_root = "/content/rsna-knee-abnormality-detection"
LABEL_FILE = "/content/training_targets.csv"   # the parsed report states
CONFIG.study_limit = 200                       # None for the full population

studies = read_study_table(CONFIG.data_root, "train")
series = read_series_table(CONFIG.data_root, "train")
print(f"{len(studies)} studies, {len(series)} series")
print(f"{int(has_expert_labels(studies).sum())} carry expert labels "
      f"(these never enter training)")
''')

code(r'''
labels = pd.read_csv(LABEL_FILE)
labels["StudyInstanceUID"] = labels["StudyInstanceUID"].astype(str)

train_uids, targets, weights = build_supervision_targets(studies, labels, CONFIG)

if CONFIG.study_limit is not None:
    keep = min(CONFIG.study_limit, len(train_uids))
    train_uids, targets, weights = train_uids[:keep], targets[:keep], weights[:keep]
    print(f"limited to {keep} studies for this run")

series_by_study = select_usable_series(series, train_uids)

# Studies whose report yielded nothing are kept: they still show the encoder
# real anatomy, and the loss simply ignores them. Studies with no readable
# series at all are dropped, because there is nothing to look at.
usable = [uid for uid in train_uids if series_by_study.get(uid)]
if len(usable) != len(train_uids):
    index = {uid: i for i, uid in enumerate(train_uids)}
    rows = [index[uid] for uid in usable]
    targets, weights = targets[rows], weights[rows]
    train_uids = usable
    print(f"dropped {len(index) - len(usable)} studies with no eligible series")

counts = [len(series_by_study[uid]) for uid in train_uids]
print(f"{len(train_uids)} studies, {sum(counts)} series "
      f"(min {min(counts)}, median {int(np.median(counts))}, max {max(counts)})")
''')

code(r'''
train_dataset = KneeStudyDataset(
    train_uids, series_by_study, CONFIG,
    split="train", targets=targets, weights=weights, train=True,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG.batch_size,
    shuffle=True,
    collate_fn=pad_studies_into_batch,
    num_workers=2,
    pin_memory=(DEVICE == "cuda"),
)

# Look at one batch before training on all of them.
sample = next(iter(train_loader))
for key in ("volumes", "present", "series_meta", "target", "weight"):
    print(f"{key:12} {tuple(sample[key].shape)}")
''')

code(r'''
model = KneeAbnormalityModel(CONFIG, pretrained_encoder=True)
freeze_encoder(model)
unfreeze_last_encoder_blocks(model, CONFIG.trainable_encoder_blocks)

# The loss needs these, and they are a property of the training surface rather
# than of the model, so they are attached once here.
model.balance_multipliers = torch.from_numpy(
    finding_balance_multipliers(weights)).to(DEVICE)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total:,} parameters, {trainable:,} trainable ({trainable / total * 100:.0f}%)")
''')

code(r'''
history = train_model(model, train_loader, CONFIG, DEVICE)

torch.save(
    {
        "model_state": model.state_dict(),
        "config": CONFIG.__dict__,
        "findings": list(FINDINGS),
        "history": history,
        "trained_studies": len(train_uids),
    },
    "knee_model.pt",
)
print("saved knee_model.pt")
''')

markdown("### Scoring against the expert-labelled studies")

code(r'''
# These studies never entered the gradient, so this is an honest read -- but
# there are only 58 of them, and at that size a difference below about 0.03
# macro AUC cannot be told from noise. Use it to catch a broken run, not to
# choose between two reasonable ones.
expert = studies.loc[has_expert_labels(studies)].reset_index(drop=True)
expert_uids = expert["StudyInstanceUID"].astype(str).tolist()
expert_series = select_usable_series(series, expert_uids)
expert_uids = [uid for uid in expert_uids if expert_series.get(uid)]
truth = expert.set_index("StudyInstanceUID").loc[expert_uids, list(FINDINGS)].to_numpy(float)

expert_loader = DataLoader(
    KneeStudyDataset(expert_uids, expert_series, CONFIG, split="train",
                     train=False, tta_offsets=CONFIG.tta_offsets),
    batch_size=CONFIG.batch_size, shuffle=False,
    collate_fn=pad_studies_into_batch, num_workers=2,
)

scored_uids, predictions = predict_with_augmentation([model], expert_loader, DEVICE)
assert scored_uids == expert_uids, "study order changed during prediction"

macro, per_finding = macro_roc_auc(truth, predictions)
print(f"macro ROC AUC over {len(expert_uids)} expert studies: {macro:.4f}\n")
for name, value in sorted(zip(FINDINGS, per_finding), key=lambda x: -x[1]):
    print(f"  {name:<18} {value:.4f}")
''')

markdown("### Predicting the test set")

code(r'''
test_studies = read_study_table(CONFIG.data_root, "test")
test_series = read_series_table(CONFIG.data_root, "test")
test_uids = test_studies["StudyInstanceUID"].astype(str).tolist()
test_by_study = select_usable_series(test_series, test_uids)

test_loader = DataLoader(
    KneeStudyDataset(test_uids, test_by_study, CONFIG, split="test",
                     train=False, tta_offsets=CONFIG.tta_offsets),
    batch_size=CONFIG.batch_size, shuffle=False,
    collate_fn=pad_studies_into_batch, num_workers=2,
)

predicted_uids, probabilities = predict_with_augmentation([model], test_loader, DEVICE)
assert predicted_uids == test_uids, "test order changed -- the submission would be scrambled"
submission = write_submission(predicted_uids, probabilities)
submission.head()
''')

markdown(r"""
## Where to go from here

Three things this notebook makes easy to try, in the order I would try them.

**1. Let the schedule finish.** Set `CONFIG.schedule_length = 2`. Identical
cost, and the learning rate actually anneals. Currently the two epochs run at
100% and 90.5% of peak and stop there.

**2. Keep some spatial detail.** `SliceFeatureExtractor` average-pools a 7x7
grid down to one vector. Returning a 2x2 or 3x3 grid instead, and letting the
series pooling attend over slices *and* positions, is the largest untried change
to this architecture -- and eight of the twelve findings are focal, so there is a
clear reason to expect it to matter.

**3. Better labels, not more of them.** The parser answers about a quarter of
the label cells and declines the rest, and where it does answer, its "yes" is
right 69% of the time. Removing 500 training studies changed the hidden score by
nothing; improving what the remaining labels *say* is the direction with real
evidence behind it.

What I would not spend time on, because the record here already rules it out:
more study-transformer capacity, different activation functions, and adding more
unlabelled studies. Eight architecture variants moved the score by about 0.015
in total, with every confidence interval crossing zero.
""")


if __name__ == "__main__":
    out = build(Path(__file__).with_name("knee_mri_model.ipynb"))
    print(f"{out}  ({len(CELLS)} cells)")
