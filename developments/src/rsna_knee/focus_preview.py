"""Shared lightweight preprocessing preview utilities for B19/B20.

A preview is deliberately cheaper than a model pass: it reads one deterministic
representative DICOM from at most one sagittal, coronal and axial series, applies
the requested in-plane focus transform, and writes a compact before/after PNG.
It never instantiates the full B12 study dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .dicom import DICOM_SUFFIXES, find_series_dir

FocusTransform = Callable[[torch.Tensor, dict], torch.Tensor]
PLANES = ("Sagittal", "Coronal", "Axial")


def _dicom_candidates(path: Path) -> list[Path]:
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES
    )


def representative_image(series_dir: Path) -> tuple[np.ndarray, Path, int]:
    """Decode one filesystem-middle DICOM image from ``series_dir``.

    This is only a preprocessing sanity check, not an anatomical slice-selection
    routine. Filename ordering is deterministic and avoids scanning every DICOM
    header in a series.
    """
    import pydicom

    candidates = _dicom_candidates(series_dir)
    if not candidates:
        raise RuntimeError(f"no DICOM candidates in {series_dir}")

    path = candidates[len(candidates) // 2]
    ds = pydicom.dcmread(str(path), force=True)
    arr = np.asarray(ds.pixel_array, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[len(arr) // 2]
    if arr.ndim != 2:
        raise RuntimeError(f"unsupported representative pixel shape {arr.shape} in {path}")

    arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(
        getattr(ds, "RescaleIntercept", 0.0)
    )
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise RuntimeError(f"representative DICOM has no finite pixels: {path}")
    lo, hi = np.percentile(finite, [1, 99])
    arr = np.nan_to_num(arr, nan=float(lo), posinf=float(hi), neginf=float(lo))
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / max(float(hi - lo), 1e-6)
    return arr.astype(np.float32, copy=False), path, len(candidates)


def resize_224(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image).view(1, 1, image.shape[0], image.shape[1])
    return F.interpolate(
        tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )


def _to_pil_gray(image: np.ndarray) -> Image.Image:
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return Image.fromarray(np.round(array * 255.0).astype(np.uint8)).convert("RGB")


def select_preview_study(
    config: dict,
    root: Path,
    *,
    requested_uid: str | None,
    require_expert_uid: bool,
) -> tuple[str, object, dict]:
    """Resolve one training UID and repair metadata only for that study."""
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    train_uids = train["StudyInstanceUID"].astype(str)
    expert_uids = set(train.loc[gold_mask(train), "StudyInstanceUID"].astype(str))

    if requested_uid is None:
        if not expert_uids:
            raise ValueError("no expert-labelled study is available for the default preview")
        # Preserve train.csv order while choosing a deterministic expert default.
        uid = next(value for value in train_uids if value in expert_uids)
    else:
        uid = str(requested_uid)
        if uid not in set(train_uids):
            raise ValueError(f"unknown training StudyInstanceUID {uid}")
        if require_expert_uid and uid not in expert_uids:
            raise ValueError("--uid must identify one of the 58 expert-labelled studies")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series = series.loc[series["StudyInstanceUID"].astype(str).eq(uid)].copy()
    if series.empty:
        raise ValueError("selected study has no rows in train_series.csv")
    series, repair = backfill_series_metadata(series, root, split="train")
    return uid, series, repair


def _save_montage(
    output: Path,
    *,
    uid: str,
    title: str,
    subtitle: str,
    transformed_label: str,
    panels: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    cell = 224
    label_h = 28
    header_h = 46
    gap = 8
    cols = len(panels)
    width = cols * cell + max(cols - 1, 0) * gap
    height = header_h + 2 * (label_h + cell) + gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), f"{title} | UID={uid}", fill="black")
    draw.text((6, 24), subtitle, fill="black")

    for col, (plane, original, transformed) in enumerate(panels):
        x = col * (cell + gap)
        y0 = header_h
        draw.text((x + 4, y0 + 6), f"{plane} - original", fill="black")
        canvas.paste(_to_pil_gray(original), (x, y0 + label_h))

        y1 = y0 + label_h + cell + gap
        draw.text((x + 4, y1 + 6), f"{plane} - {transformed_label}", fill="black")
        canvas.paste(_to_pil_gray(transformed), (x, y1 + label_h))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=False)


def run_focus_preview(
    *,
    root: Path,
    uid: str,
    series,
    repair: dict,
    transform: FocusTransform,
    policy: dict,
    title: str,
    subtitle: str,
    transformed_label: str,
    output: Path,
) -> None:
    """Read one image per available plane, apply ``transform``, save montage."""
    selected: list[tuple[str, str]] = []
    for plane in PLANES:
        part = series.loc[series["Anatomical_Plane"].eq(plane)].sort_values(
            "SeriesInstanceUID"
        )
        if not part.empty:
            selected.append((plane, str(part.iloc[0]["SeriesInstanceUID"])))
    if not selected:
        raise RuntimeError("no sagittal/coronal/axial series available for preview")

    repairs = int(repair.get("repaired_plane", 0)) + int(repair.get("repaired_fluid", 0)) + int(
        repair.get("repaired_fat_suppression", 0)
    )
    print(f"preview UID={uid} | series={len(series)} | metadata repairs={repairs}", flush=True)

    panels: list[tuple[str, np.ndarray, np.ndarray]] = []
    for plane, series_uid in selected:
        series_dir = find_series_dir(root, "train", uid, series_uid)
        if series_dir is None:
            raise FileNotFoundError(f"missing training series {uid}/{series_uid}")
        raw, dicom_path, n_files = representative_image(series_dir)
        original = resize_224(raw)
        transformed = transform(original, policy)
        panels.append(
            (
                plane,
                original[0, 0].numpy(),
                transformed[0, 0].numpy(),
            )
        )
        print(
            f"  {plane}: {series_uid} | files={n_files} | representative={dicom_path.name}",
            flush=True,
        )

    _save_montage(
        output,
        uid=uid,
        title=title,
        subtitle=subtitle,
        transformed_label=transformed_label,
        panels=panels,
    )
    print(output, flush=True)
