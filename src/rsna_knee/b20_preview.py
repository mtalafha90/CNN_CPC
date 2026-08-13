"""Preview B20 crop-only preprocessing before training.

This preview is intentionally lightweight: it decodes only one representative
DICOM image from at most one sagittal, coronal and axial series. It does not
instantiate the full B12 study dataset or decode every slice/series in a study.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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
    """Decode one approximately central DICOM image from a series.

    We intentionally avoid decoding the complete volume because B20's preview
    only needs to demonstrate the in-plane crop transform.
    """
    import pydicom

    candidates = _dicom_candidates(series_dir)
    if not candidates:
        raise RuntimeError(f"no DICOM candidates in {series_dir}")

    # Header-only reads are cheap and let InstanceNumber choose an approximate
    # anatomical middle without touching pixel data for every file.
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
    # Default to an expert-labelled study, matching the B19 preview policy and
    # avoiding an arbitrary first report-only study that may contain unusual
    # acquisition files. --uid can still select any training study explicitly.
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

    fig, axes = plt.subplots(2, len(selected), figsize=(5 * len(selected), 8), squeeze=False)
    for col, (plane, series_uid) in enumerate(selected):
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

        axes[0, col].imshow(original_np, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{plane} — original")
        axes[1, col].imshow(crop_np, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"{plane} — B20 crop-only")
        for row in range(2):
            axes[row, col].axis("off")
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

    fig.suptitle(f"B20 crop-only preview | UID={uid} | policy={policy}", fontsize=11)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output, flush=True)


if __name__ == "__main__":
    main()
