"""Convert the raw DICOM tree into a compact cache of uint8 volumes.

Decoding DICOM is the slowest part of training by a wide margin, so we pay
that cost once. Each series becomes a single ``.npy`` file of shape
``[slices, size, size]`` in uint8, plus one manifest row describing its plane
and weighting. A typical exam shrinks from hundreds of megabytes to a few, and
epochs then run at disk speed.

Run it once before training::

    python -m rsnaknee.preprocess --dicom-dir /data/train_images --out-dir /cache
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .dicom_io import SeriesInfo, read_series, resize_volume
from .utils import get_logger, timed

LOGGER = get_logger()

DICOM_SUFFIXES = {".dcm", ".dicom", ""}


def group_dicom_files(dicom_dir: str | Path) -> dict[tuple[str, str], list[Path]]:
    """Group every DICOM file under ``dicom_dir`` into ``(exam, series)`` buckets.

    The usual RSNA layout is ``<root>/<study>/<series>/<instance>.dcm``, which
    we detect from the directory structure. Flatter or deeper layouts fall back
    to grouping by the parent directory, which is correct for any layout that
    keeps one series per folder.
    """
    dicom_dir = Path(dicom_dir)
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in dicom_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in DICOM_SUFFIXES:
            continue
        relative = path.relative_to(dicom_dir)
        parts = relative.parts
        if len(parts) >= 3:
            exam_id, series_id = parts[0], parts[1]
        elif len(parts) == 2:
            exam_id, series_id = parts[0], path.stem
        else:
            exam_id, series_id = path.stem, path.stem
        groups[(exam_id, series_id)].append(path)
    return dict(groups)


def process_series(
    exam_id: str,
    series_id: str,
    paths: list[Path],
    out_dir: Path,
    size: int,
    max_slices: int,
) -> dict | None:
    """Decode, resize and cache one series. Returns its manifest row."""
    try:
        volume, info = read_series(paths)
    except Exception as error:
        LOGGER.warning("Failed to read %s/%s: %s", exam_id, series_id, error)
        return None

    if volume.shape[0] > max_slices:
        # Keep the central slices: knee pathology sits in the middle of the stack
        # and the outer slices are mostly air and soft tissue.
        keep = np.linspace(0, volume.shape[0] - 1, max_slices).round().astype(int)
        volume = volume[keep]

    volume = resize_volume(volume, size)

    # Directory names are the ground truth for identity: DICOM tags are
    # sometimes anonymised inconsistently between sites.
    info.exam_id = exam_id
    info.series_id = series_id

    destination = out_dir / exam_id
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / f"{series_id}.npy", volume)

    row = info.to_dict()
    row["cached_slices"] = int(volume.shape[0])
    row["cache_path"] = str(Path(exam_id) / f"{series_id}.npy")
    return row


def build_cache(
    dicom_dir: str | Path,
    out_dir: str | Path,
    size: int = 256,
    max_slices: int = 48,
    workers: int = 8,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Build the whole cache and write ``series_manifest.csv`` beside it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "series_manifest.csv"

    if manifest_path.exists() and not overwrite:
        LOGGER.info("Manifest already exists at %s; pass --overwrite to rebuild", manifest_path)
        return pd.read_csv(manifest_path)

    with timed("Scanning DICOM tree", LOGGER):
        groups = group_dicom_files(dicom_dir)
    LOGGER.info("Found %d series across %d exams", len(groups), len({k[0] for k in groups}))

    rows: list[dict] = []
    if workers <= 1:
        for (exam_id, series_id), paths in groups.items():
            row = process_series(exam_id, series_id, paths, out_dir, size, max_slices)
            if row:
                rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(process_series, exam, series, paths, out_dir, size, max_slices): exam
                for (exam, series), paths in groups.items()
            }
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                if row:
                    rows.append(row)
                if index % 500 == 0:
                    LOGGER.info("Cached %d / %d series", index, len(futures))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    LOGGER.info("Cached %d series to %s", len(manifest), out_dir)
    if not manifest.empty:
        LOGGER.info(
            "Series mix:\n%s",
            manifest.groupby(["plane", "weighting"]).size().to_string(),
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache knee MRI DICOM series as uint8 volumes")
    parser.add_argument("--dicom-dir", required=True, help="Root of the raw DICOM tree")
    parser.add_argument("--out-dir", required=True, help="Where to write the cache")
    parser.add_argument("--size", type=int, default=256, help="In-plane resolution of the cache")
    parser.add_argument("--max-slices", type=int, default=48, help="Slice cap per series")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_cache(
        args.dicom_dir,
        args.out_dir,
        size=args.size,
        max_slices=args.max_slices,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
