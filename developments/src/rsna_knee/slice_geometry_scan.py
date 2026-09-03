"""How thick a slice is, how long a series is, and what the triplet actually spans.

The inventory established that 16.4% of the frames present are never read. It
could not say **which** frames, because it knew only how many files sat in a
folder. This scan adds the missing axis: the physical distance between one
slice and the next, read from the DICOM headers.

Three questions, all of them label-free:

```text
1  the triplet          the 2.5D input is [-gap, centre, +gap] in *index*
                        units, so its physical depth is 2 x gap x spacing.
                        A 4 mm 2D series gives the encoder an 8 mm sandwich;
                        a 0.5 mm 3D series gives it 1 mm. Same three channels,
                        very different amounts of knee
2  where the loss is    of the frames never read, how many sit in thin 3D
                        volumes (where neighbouring slices are near-duplicates
                        and losing them costs little) versus thick 2D stacks
                        (where every frame is distinct anatomy)
3  length vs thickness  the long series are *assumed* to be the thin ones.
                        This measures it instead of assuming it
```

## It also corrects the inventory's arithmetic

`data_inventory` counted the frames read as `min(frames, centres)` — one frame
per centre. That is the right count for a plain volume, and too pessimistic for
the 2.5D path this project actually runs: each centre pulls three frames, and on
a long series those triplets do not overlap. Thirty-two centres on a 320-frame
series touch 96 distinct frames, not 32. The loss is real but smaller than the
inventory reported, and this module reports both figures side by side so the
difference is visible rather than silently swapped.

## What it costs to run

A few headers per series, not all of them: first, middle and last by filename.
That is enough to read the thickness tags and to estimate the true slice pitch
from `ImagePositionPatient`, while keeping the scan to roughly three reads per
series instead of thirty-four. Pixel data is never decoded.

## What it does not do

It reads no labels and changes no training artefact. It is a measurement.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PLANES, backfill_series_metadata, load_series_csv
from .dicom import DICOM_SUFFIXES, _centers, find_series_dir

SCAN_VERSION = "slice_geometry_v1"

# What the loader takes from every series, and the 2.5D index gap it uses.
DEFAULT_CENTRES = 32
DEFAULT_GAP = 1

# Headers opened per series: first, middle, last.
DEFAULT_SAMPLES = 3

# A slice pitch outside this range is a broken header, not a knee.
PLAUSIBLE_SPACING_MM = (0.1, 20.0)

GEOMETRY_TAGS = [
    "SliceThickness",
    "SpacingBetweenSlices",
    "MRAcquisitionType",
    "PixelSpacing",
    "Rows",
    "Columns",
    "SeriesDescription",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "NumberOfFrames",
]

# Bands are stated once, here, so the report and the tests cannot drift apart.
SPAN_BANDS = (0.0, 2.0, 4.0, 6.0, 8.0, 12.0, np.inf)
SPAN_LABELS = ("<2 mm", "2-4 mm", "4-6 mm", "6-8 mm", "8-12 mm", ">=12 mm")

SPACING_BANDS = (0.0, 1.5, 3.0, 4.5, np.inf)
SPACING_LABELS = ("<1.5 mm", "1.5-3 mm", "3-4.5 mm", ">=4.5 mm")

LENGTH_BANDS = (0, 32, 48, 80, 160, np.inf)
LENGTH_LABELS = ("<=32", "33-48", "49-80", "81-160", ">160")

MISSING = "<missing>"


def _float(value) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _slice_coordinate(ds) -> float | None:
    """Project `ImagePositionPatient` onto the slice normal.

    This mirrors the ordering `dicom._sort_key` uses, so the pitch measured here
    is the pitch along the axis the loader stacks. `ImagePositionPatient[2]` on
    its own would be wrong for sagittal and coronal acquisitions, which is most
    of this dataset.
    """
    try:
        position = np.asarray(ds.ImagePositionPatient, dtype=float).reshape(-1)
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float).reshape(-1)
    except Exception:
        return None
    if position.size < 3 or orientation.size < 6:
        return None
    normal = np.cross(orientation[:3], orientation[3:6])
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    value = float(np.dot(position[:3], normal / norm))
    return value if np.isfinite(value) else None


def _sample_positions(n_files: int, samples: int) -> list[int]:
    """First, middle and last — or fewer, on a short series."""
    if n_files < 1:
        return []
    samples = max(1, min(int(samples), n_files))
    return sorted({int(round(x)) for x in np.linspace(0, n_files - 1, samples)})


def _pitch_from_positions(points: list[tuple[int, float]]) -> tuple[float, bool]:
    """Slice pitch in mm from sampled positions, and whether they were ordered.

    `points` are `(file index, coordinate)` pairs. The pitch is the coordinate
    span divided by the number of files between the two ends. That is only the
    real pitch if filename order follows geometric order, so any sample between
    the ends is checked for lying between them; if one does not, the estimate is
    reported as unordered and the caller falls back to the tags.
    """
    if len(points) < 2:
        return float("nan"), False
    points = sorted(points)
    (first_index, first_value), (last_index, last_value) = points[0], points[-1]
    steps = last_index - first_index
    span = abs(last_value - first_value)
    if steps <= 0 or not np.isfinite(span) or span <= 0:
        return float("nan"), False

    low, high = sorted((first_value, last_value))
    ordered = all(low - 1e-6 <= value <= high + 1e-6 for _, value in points[1:-1])
    return span / float(steps), ordered


def _choose_spacing(row: dict) -> tuple[float, str]:
    """The pitch to believe, and where it came from.

    Measured positions first: they are the distance the encoder actually steps.
    `SpacingBetweenSlices` next, since it means the same thing when present.
    `SliceThickness` last — it ignores any gap between slices, so it understates
    the pitch on an interleaved 2D stack, but it is better than nothing.
    """
    low, high = PLAUSIBLE_SPACING_MM
    measured = row.get("pitch_from_positions_mm", float("nan"))
    if row.get("positions_ordered") and np.isfinite(measured) and low <= measured <= high:
        return float(measured), "image_positions"
    for column, source in (
        ("spacing_between_slices_mm", "SpacingBetweenSlices"),
        ("slice_thickness_mm", "SliceThickness"),
    ):
        value = _float(row.get(column))
        if np.isfinite(value) and low <= value <= high:
            return float(value), source
    return float("nan"), "unavailable"


def _missing_series() -> dict:
    """The row a series gets when its folder is not on disk."""
    return {
        "series_dir_found": False,
        "frames": 0,
        "headers_read": 0,
        "header_failures": 0,
        "slice_thickness_mm": float("nan"),
        "spacing_between_slices_mm": float("nan"),
        "pitch_from_positions_mm": float("nan"),
        "positions_ordered": False,
        "acquisition_type": "",
        "pixel_spacing_mm": float("nan"),
        "rows": 0,
        "columns": 0,
        "series_description": "",
        "slice_spacing_mm": float("nan"),
        "spacing_source": "unavailable",
    }


def geometry_from_headers(series_dir: str | Path, *, samples: int = DEFAULT_SAMPLES) -> dict:
    """Read a handful of headers from one series and describe its geometry."""
    import pydicom

    path = Path(series_dir)
    blank = _missing_series()
    if not path.is_dir():
        return blank

    files = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES
    )
    result = dict(blank, series_dir_found=True, frames=len(files))

    thicknesses: list[float] = []
    between: list[float] = []
    pixel_spacings: list[float] = []
    points: list[tuple[int, float]] = []

    for index in _sample_positions(len(files), samples):
        try:
            ds = pydicom.dcmread(
                str(files[index]),
                force=True,
                stop_before_pixels=True,
                specific_tags=GEOMETRY_TAGS,
            )
        except Exception:
            result["header_failures"] += 1
            continue
        result["headers_read"] += 1

        value = _float(getattr(ds, "SliceThickness", None))
        if value > 0:
            thicknesses.append(value)
        value = _float(getattr(ds, "SpacingBetweenSlices", None))
        if value > 0:
            between.append(value)
        try:
            spacing = np.asarray(ds.PixelSpacing, dtype=float).reshape(-1)
            if spacing.size >= 1 and np.isfinite(spacing[0]) and spacing[0] > 0:
                pixel_spacings.append(float(spacing[0]))
        except Exception:
            pass
        if not result["acquisition_type"]:
            result["acquisition_type"] = str(
                getattr(ds, "MRAcquisitionType", "") or ""
            ).strip().upper()
        if not result["series_description"]:
            result["series_description"] = str(
                getattr(ds, "SeriesDescription", "") or ""
            ).strip()
        if not result["rows"]:
            result["rows"] = int(getattr(ds, "Rows", 0) or 0)
            result["columns"] = int(getattr(ds, "Columns", 0) or 0)

        # One position locates one single-frame instance. An enhanced
        # multi-frame object carries one position for the whole stack, so its
        # pitch cannot be measured this way.
        try:
            n_frames = max(1, int(getattr(ds, "NumberOfFrames", 1) or 1))
        except Exception:
            n_frames = 1
        if n_frames == 1:
            coordinate = _slice_coordinate(ds)
            if coordinate is not None:
                points.append((index, coordinate))

    if thicknesses:
        result["slice_thickness_mm"] = float(np.median(thicknesses))
    if between:
        result["spacing_between_slices_mm"] = float(np.median(between))
    if pixel_spacings:
        result["pixel_spacing_mm"] = float(np.median(pixel_spacings))
    pitch, ordered = _pitch_from_positions(points)
    result["pitch_from_positions_mm"] = pitch
    result["positions_ordered"] = bool(ordered)

    spacing, source = _choose_spacing(result)
    result["slice_spacing_mm"] = spacing
    result["spacing_source"] = source
    return result


def frames_read(
    n_frames: int, *, centres: int = DEFAULT_CENTRES, gap: int = DEFAULT_GAP
) -> int:
    """Distinct frames one deterministic 2.5D pass touches.

    Each centre pulls `[-gap, centre, +gap]`. On a short series the centres
    crowd together and the triplets overlap heavily; on a long one they spread
    out and every triplet is disjoint, so the count reaches `3 x centres`.
    """
    if n_frames < 1:
        return 0
    positions = _centers(n_frames, int(centres), int(gap))
    offsets = np.asarray([-int(gap), 0, int(gap)], dtype=int)
    touched = np.clip(positions[:, None] + offsets[None, :], 0, n_frames - 1)
    return int(np.unique(touched).size)


def series_table(
    *,
    data_root: str | Path,
    split: str = "train",
    centres: int = DEFAULT_CENTRES,
    gap: int = DEFAULT_GAP,
    samples: int = DEFAULT_SAMPLES,
    workers: int = 8,
    max_series: int | None = None,
) -> pd.DataFrame:
    """One row per model-eligible series, with its geometry and its losses."""
    root = Path(data_root).resolve()
    series = load_series_csv(root / "train_series.csv")
    series, _ = backfill_series_metadata(series, root, split=split)
    series = series.loc[series["Anatomical_Plane"].isin(PLANES)].reset_index(drop=True)
    if max_series is not None:
        series = series.head(int(max_series)).reset_index(drop=True)

    directories = [
        find_series_dir(root, split, str(row.StudyInstanceUID), str(row.SeriesInstanceUID))
        for row in series.itertuples(index=False)
    ]

    def _one(directory: Path | None) -> dict:
        if directory is None:
            return _missing_series()
        return geometry_from_headers(directory, samples=samples)

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            rows = list(pool.map(_one, directories))
    else:
        rows = [_one(directory) for directory in directories]

    frame = pd.concat(
        [
            series[["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"]],
            pd.DataFrame(rows, index=series.index),
        ],
        axis=1,
    )

    frame["triplet_span_mm"] = 2.0 * int(gap) * frame["slice_spacing_mm"]
    frame["frames_read"] = [
        frames_read(int(n), centres=centres, gap=gap) for n in frame["frames"]
    ]
    frame["frames_read_centres_only"] = np.minimum(
        frame["frames"].to_numpy(dtype=np.int64), int(centres)
    )
    frame["frames_never_read"] = frame["frames"] - frame["frames_read"]
    frame["acquisition"] = (
        frame["acquisition_type"].replace("", "unknown").fillna("unknown")
    )
    return frame


def _quantiles(values: pd.Series) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not clean.size:
        return {"n": 0}
    keys = ("p0", "p05", "p25", "p50", "p75", "p95", "p99", "p100")
    points = np.quantile(clean, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {"n": int(clean.size), **{k: float(v) for k, v in zip(keys, points)}}


def _banded(frame: pd.DataFrame, column: str, bands, labels) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    banded = pd.cut(values, bins=list(bands), labels=list(labels), right=False)
    return banded.cat.add_categories([MISSING]).fillna(MISSING)


def _group_rows(frame: pd.DataFrame, key: pd.Series, order: list[str]) -> list[dict]:
    """Series, frames and lost frames for each band, in a fixed order."""
    total_lost = max(int(frame["frames_never_read"].sum()), 1)
    grouped = frame.groupby(key, observed=False)
    rows = []
    for name in order:
        if name not in grouped.groups:
            continue
        part = frame.loc[grouped.groups[name]]
        lost = int(part["frames_never_read"].sum())
        rows.append(
            {
                "band": str(name),
                "series": int(len(part)),
                "frames": int(part["frames"].sum()),
                "frames_never_read": lost,
                "share_of_frames_never_read": float(lost / total_lost),
                # None rather than NaN: the summary is written as JSON, and NaN
                # is not JSON, so it would not survive a round trip.
                "median_frames": float(part["frames"].median()) if len(part) else None,
                "median_spacing_mm": (
                    float(part["slice_spacing_mm"].median())
                    if part["slice_spacing_mm"].notna().any()
                    else None
                ),
            }
        )
    return rows


def summarise(frame: pd.DataFrame, *, centres: int, gap: int) -> dict:
    """Turn the per-series table into the three answers the scan exists for."""
    total_frames = int(frame["frames"].sum())
    read_triplets = int(frame["frames_read"].sum())
    read_centres = int(frame["frames_read_centres_only"].sum())
    known = frame.loc[frame["slice_spacing_mm"].notna()]

    span_band = _banded(frame, "triplet_span_mm", SPAN_BANDS, SPAN_LABELS)
    spacing_band = _banded(frame, "slice_spacing_mm", SPACING_BANDS, SPACING_LABELS)
    length_band = _banded(frame, "frames", LENGTH_BANDS, LENGTH_LABELS)

    correlation = None
    if len(known) > 2:
        value = float(known["frames"].corr(known["slice_spacing_mm"], method="spearman"))
        correlation = value if np.isfinite(value) else None

    acquisition_order = sorted(frame["acquisition"].astype(str).unique())
    return {
        "version": SCAN_VERSION,
        "centres_taken_per_series": int(centres),
        "triplet_gap": int(gap),
        "series": int(len(frame)),
        "series_with_a_known_spacing": int(len(known)),
        "series_dirs_not_found": int((~frame["series_dir_found"]).sum()),
        "header_failures": int(frame["header_failures"].sum()),
        "spacing_source": {
            str(k): int(v) for k, v in frame["spacing_source"].value_counts().items()
        },
        "slice_spacing_mm": _quantiles(frame["slice_spacing_mm"]),
        "triplet_span_mm": _quantiles(frame["triplet_span_mm"]),
        "triplet_span_bands": _group_rows(
            frame, span_band, list(SPAN_LABELS) + [MISSING]
        ),
        "acquisition_type": _group_rows(
            frame, frame["acquisition"].astype(str), acquisition_order
        ),
        "spacing_bands": _group_rows(
            frame, spacing_band, list(SPACING_LABELS) + [MISSING]
        ),
        "length_bands": _group_rows(
            frame, length_band, list(LENGTH_LABELS) + [MISSING]
        ),
        "spearman_frames_vs_spacing": correlation,
        "reading_loss": {
            "frames_total": total_frames,
            "frames_read_triplets": read_triplets,
            "frames_never_read": total_frames - read_triplets,
            "fraction_never_read": (
                float((total_frames - read_triplets) / total_frames)
                if total_frames
                else 0.0
            ),
            "frames_read_centres_only": read_centres,
            "frames_never_read_centres_only": total_frames - read_centres,
            "fraction_never_read_centres_only": (
                float((total_frames - read_centres) / total_frames)
                if total_frames
                else 0.0
            ),
        },
    }


def scan(
    *,
    data_root: str | Path,
    split: str = "train",
    centres: int = DEFAULT_CENTRES,
    gap: int = DEFAULT_GAP,
    samples: int = DEFAULT_SAMPLES,
    workers: int = 8,
    max_series: int | None = None,
    out_csv: str | Path | None = None,
    out_json: str | Path | None = None,
) -> dict:
    frame = series_table(
        data_root=data_root,
        split=split,
        centres=centres,
        gap=gap,
        samples=samples,
        workers=workers,
        max_series=max_series,
    )
    result = summarise(frame, centres=centres, gap=gap)
    result["data_root"] = str(Path(data_root).resolve())

    if out_csv is not None:
        path = Path(out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        result["series_csv"] = str(path)
    if out_json is not None:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _band_table(title: str, rows: list[dict], *, spacing: bool = False) -> None:
    print()
    print(f"  {title}")
    print(
        f"    {'band':<12}{'series':>9}{'frames':>12}{'never read':>13}"
        f"{'share':>9}{'med mm' if spacing else 'med len':>10}"
    )
    for row in rows:
        last = row["median_spacing_mm"] if spacing else row["median_frames"]
        text = "-" if last is None else f"{last:.2f}"
        print(
            f"    {row['band']:<12}{row['series']:>9,}{row['frames']:>12,}"
            f"{row['frames_never_read']:>13,}"
            f"{row['share_of_frames_never_read'] * 100:>8.1f}%{text:>10}"
        )


def _report(result: dict) -> None:
    loss = result["reading_loss"]
    print()
    print(f"  series measured                   {result['series']:>10,}")
    print(f"  with a known slice spacing        {result['series_with_a_known_spacing']:>10,}")
    print()
    print("  where the spacing came from")
    for source, count in result["spacing_source"].items():
        print(f"    {source:<30}{count:>10,}")

    spacing, span = result["slice_spacing_mm"], result["triplet_span_mm"]
    if spacing.get("n"):
        print()
        print("  slice spacing, mm")
        print(
            f"    p05 {spacing['p05']:.2f}   p25 {spacing['p25']:.2f}   "
            f"p50 {spacing['p50']:.2f}   p75 {spacing['p75']:.2f}   "
            f"p95 {spacing['p95']:.2f}   max {spacing['p100']:.2f}"
        )
        print()
        print(f"  what one 2.5D triplet spans, mm (gap {result['triplet_gap']})")
        print(
            f"    p05 {span['p05']:.2f}   p25 {span['p25']:.2f}   "
            f"p50 {span['p50']:.2f}   p75 {span['p75']:.2f}   "
            f"p95 {span['p95']:.2f}   max {span['p100']:.2f}"
        )

    _band_table("triplet depth", result["triplet_span_bands"])
    _band_table("acquisition type", result["acquisition_type"], spacing=True)
    _band_table("slice spacing", result["spacing_bands"], spacing=True)
    _band_table("series length", result["length_bands"], spacing=True)

    correlation = result["spearman_frames_vs_spacing"]
    print()
    print(
        "  Spearman, series length vs slice spacing   "
        + ("not enough series" if correlation is None else f"{correlation:+.3f}")
    )
    print("    negative means the long series really are the thin ones")

    print()
    print(f"  reading loss, {result['centres_taken_per_series']} centres per series")
    print(f"    frames present                  {loss['frames_total']:>10,}")
    print(
        f"    read, one frame per centre      {loss['frames_read_centres_only']:>10,}"
        f"   lost {loss['fraction_never_read_centres_only'] * 100:5.1f}%"
    )
    print(
        f"    read, 2.5D triplets             {loss['frames_read_triplets']:>10,}"
        f"   lost {loss['fraction_never_read'] * 100:5.1f}%"
    )
    print(
        "    The triplet figure is the real one: each centre pulls three frames,\n"
        "    and on a long series those triplets do not overlap."
    )

    if result["series_dirs_not_found"]:
        print()
        print(f"  WARNING {result['series_dirs_not_found']:,} series folders were not found")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        "Measure slice thickness against series length, and what the triplet spans"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--centres", type=int, default=DEFAULT_CENTRES)
    parser.add_argument("--gap", type=int, default=DEFAULT_GAP)
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="headers opened per series: first, middle, last",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    _report(
        scan(
            data_root=args.data_root,
            split=args.split,
            centres=args.centres,
            gap=args.gap,
            samples=args.samples,
            workers=args.workers,
            max_series=args.max_series,
            out_csv=args.out_csv,
            out_json=args.out_json,
        )
    )


if __name__ == "__main__":
    main()
