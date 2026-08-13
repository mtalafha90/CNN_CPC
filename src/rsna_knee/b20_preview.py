"""Preview B20 crop-only preprocessing before training.

This preview is intentionally lightweight: it decodes only one representative
DICOM image from at most one sagittal, coronal and axial series. It does not
instantiate the full B12 study dataset or decode every slice/series in a study.

The preview montage is written with Pillow rather than Matplotlib so a local
font-cache/backend problem cannot block a simple preprocessing sanity check.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .b7_weak_supervision import _read_config
from .b20_crop_focus import b20_crop_focus_policy
from .crop_focus import apply_crop_focus
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .dicom import DICOM_SUFFIXES, find_series_dir


def _dicom_candidates(path: Path) -> list[Path]:
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES
    )


def _representative_image(series_dir: Path) -> np.ndarray:
    """Decode one approximately central DICOM image from a series."""
    import pydicom

    candidates = _dicom_candidates(series_dir)
    if not candidates:
        raise RuntimeError(f"no DICOM candidates in {series_dir}")

    ordered: list[tuple[float, Path]] = []
    for ordinal, path in enumerate(candidates):
        try:
            ds = pydicom.dcmread(
                str(path),
                force=True,
                stop_before_pixels=True,
                specific_tags=["InstanceNumber"],
            )
            key = float(getattr(ds, "InstanceNumber", ordinal))
        except Exception:
            key = float(ordinal)
        ordered.append((key, path))
    ordered.sort(key=lambda item: item[0])

    path = ordered[len(ordered) // 2][1]
    print(
        {
            "decoding_representative_dicom": str(path),
            "series_files": len(candidates),
        },
        flush=True,
    )
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
    return arr.astype(np.float32, copy=False)


def _resize_224(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image).view(1, 1, image.shape[0], image.shape[1])
    return F.interpolate(
        tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )


def _to_pil_gray(image: np.ndarray) -> Image.Image:
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="L").convert("RGB")


def _save_montage(
    output: Path,
    *,
    uid: str,
    policy: dict,
    panels: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    """Save a compact 2xN PNG without invoking Matplotlib."""
    cell = 224
    label_h = 28
    header_h = 44
    gap = 8
    cols = len(panels)
    width = cols * cell + max(cols - 1, 0) * gap
    height = header_h + 2 * (label_h + cell) + gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (6, 6),
        f"B20 crop-only preview | UID={uid}",
        fill="black",
    )
    draw.text(
        (6, 23),
        f"crop_fraction={float(policy['crop_fraction']):.2f} | no vignette / no black mask",
        fill="black",
    )

    for col, (plane, original, crop) in enumerate(panels):
        x = col * (cell + gap)
        y0 = header_h
        draw.text((x + 4, y0 + 6), f"{plane} - original", fill="black")
        canvas.paste(_to_pil_gray(original), (x, y0 + label_h))

        y1 = y0 + label_h + cell + gap
        draw.text((x + 4, y1 + 6), f"{plane} - B20 crop-only", fill="black")
        canvas.paste(_to_pil_gray(crop), (x, y1 + label_h))

    output.parent.mkdir(parents=True, exist_ok=True)
    print({"preview_save": str(output), "writer": "Pillow", "status": "saving"}, flush=True)
    canvas.save(output, format="PNG", optimize=False)
    print({"preview_save": str(output), "writer": "Pillow", "status": "done"}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b20-preview")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--uid", default=None)
    parser.add_argument("--out", default="runs/b20_crop_focus/crop_focus_preview.png")
    args = parser.parse_args()

    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    policy = b20_crop_focus_policy(config)
    root = Path(config["data_root"])

    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    if args.uid:
        uid = str(args.uid)
        if uid not in set(train["StudyInstanceUID"].astype(str)):
            raise ValueError(f"unknown training StudyInstanceUID {uid}")
    else:
        expert = train.loc[gold_mask(train), "StudyInstanceUID"].astype(str)
        if expert.empty:
            raise ValueError("no expert-labelled study available for default B20 preview")
        uid = str(expert.iloc[0])

    # Subset before metadata backfill. A preview must never audit all series.
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series = series.loc[series["StudyInstanceUID"].astype(str).eq(uid)].copy()
    if series.empty:
        raise ValueError("selected study has no rows in train_series.csv")
    print({"preview_uid": uid, "series_rows_before_backfill": int(len(series))}, flush=True)
    series, repair = backfill_series_metadata(series, root, split="train")
    print({"metadata_repair": repair}, flush=True)

    selected: list[tuple[str, str]] = []
    for plane in ("Sagittal", "Coronal", "Axial"):
        part = series.loc[series["Anatomical_Plane"].eq(plane)]
        if not part.empty:
            selected.append((plane, str(part.iloc[0]["SeriesInstanceUID"])))
    if not selected:
        raise RuntimeError("no sagittal/coronal/axial series available for preview")

    panels: list[tuple[str, np.ndarray, np.ndarray]] = []
    for plane, series_uid in selected:
        series_dir = find_series_dir(root, "train", uid, series_uid)
        if series_dir is None:
            raise FileNotFoundError(f"missing training series {uid}/{series_uid}")
        print(
            {"plane": plane, "series_uid": series_uid, "status": "reading_one_image"},
            flush=True,
        )
        raw = _representative_image(series_dir)
        original = _resize_224(raw)
        crop = apply_crop_focus(original, policy)
        original_np = original[0, 0].numpy()
        crop_np = crop[0, 0].numpy()
        panels.append((plane, original_np, crop_np))
        print(
            {
                "plane": plane,
                "series_uid": series_uid,
                "status": "done",
                "original_mean": float(original_np.mean()),
                "crop_mean": float(crop_np.mean()),
            },
            flush=True,
        )

    output = Path(args.out)
    _save_montage(output, uid=uid, policy=policy, panels=panels)
    print(output, flush=True)


if __name__ == "__main__":
    main()
