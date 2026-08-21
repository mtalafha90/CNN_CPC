"""Audit native knee-MRI geometry before choosing a no-resize input policy.

The audit is intentionally header-only: it never decompresses PixelData and never
changes any training artifact.  Every series listed in train_series.csv is located,
all of its DICOM headers are inspected, and a series-level table is written with
native matrix, PixelSpacing, physical FOV, slice geometry and scanner metadata.

The report separately summarizes the model-eligible repaired-plane series and asks
one concrete question: after a fixed 90% native center crop, what square canvas
would preserve the original cropped pixels by padding only (no interpolation)?

Padding feasibility is *not* physical-scale equivalence.  PixelSpacing variability
is reported explicitly so a 512x512 acquisition at 0.33 mm/pixel is not treated as
scientifically interchangeable with another 512x512 acquisition at a different
physical sampling.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import PLANES, backfill_series_metadata, load_series_csv
from .dicom import DICOM_SUFFIXES, find_series_dir

AUDIT_VERSION = "native_resolution_geometry_audit_v1"
DEFAULT_CROP_FRACTION = 0.90
DEFAULT_CANVASES = (288, 320, 384, 448, 464, 512, 576, 640)

HEADER_TAGS = [
    "Rows",
    "Columns",
    "PixelSpacing",
    "ImagerPixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "SeriesDescription",
    "SequenceName",
    "NumberOfFrames",
    "PhotometricInterpretation",
    "BitsAllocated",
    "BitsStored",
]


def _dicom_files(path: Path) -> list[Path]:
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES
    )


def _float(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _spacing(ds) -> tuple[float, float] | None:
    for name in ("PixelSpacing", "ImagerPixelSpacing"):
        try:
            values = np.asarray(getattr(ds, name), dtype=float).reshape(-1)
        except Exception:
            continue
        if len(values) >= 2 and np.isfinite(values[:2]).all() and np.all(values[:2] > 0):
            return float(values[0]), float(values[1])
    return None


def _mode_pair(values: list[tuple[int, int]]) -> tuple[int | None, int | None]:
    if not values:
        return None, None
    pair, _ = Counter(values).most_common(1)[0]
    return int(pair[0]), int(pair[1])


def _median_pair(values: list[tuple[float, float]]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    med = np.median(arr, axis=0)
    return float(med[0]), float(med[1])


def _clean_text(value) -> str:
    return str(value or "").strip()


def inspect_series(
    data_root: str | Path,
    split: str,
    study_uid: str,
    series_uid: str,
) -> dict:
    """Inspect every header in one series without reading PixelData."""
    import pydicom

    directory = find_series_dir(data_root, split, study_uid, series_uid)
    base = {
        "StudyInstanceUID": str(study_uid),
        "SeriesInstanceUID": str(series_uid),
        "series_dir_found": directory is not None,
        "candidate_files": 0,
        "readable_headers": 0,
        "header_failures": 0,
    }
    if directory is None:
        return base

    files = _dicom_files(directory)
    base["candidate_files"] = len(files)
    dims: list[tuple[int, int]] = []
    spacings: list[tuple[float, float]] = []
    thicknesses: list[float] = []
    between: list[float] = []
    manufacturers: list[str] = []
    models: list[str] = []
    fields: list[float] = []
    descriptions: list[str] = []
    sequences: list[str] = []
    frames = 0
    photometric: list[str] = []
    bits_allocated: list[int] = []
    bits_stored: list[int] = []

    for path in files:
        try:
            ds = pydicom.dcmread(
                str(path),
                force=True,
                stop_before_pixels=True,
                specific_tags=HEADER_TAGS,
            )
            base["readable_headers"] += 1
            rows = int(getattr(ds, "Rows", 0) or 0)
            cols = int(getattr(ds, "Columns", 0) or 0)
            if rows > 0 and cols > 0:
                dims.append((rows, cols))
            spacing = _spacing(ds)
            if spacing is not None:
                spacings.append(spacing)
            value = _float(getattr(ds, "SliceThickness", None))
            if value is not None and value > 0:
                thicknesses.append(value)
            value = _float(getattr(ds, "SpacingBetweenSlices", None))
            if value is not None and value > 0:
                between.append(value)
            manufacturer = _clean_text(getattr(ds, "Manufacturer", ""))
            model = _clean_text(getattr(ds, "ManufacturerModelName", ""))
            description = _clean_text(getattr(ds, "SeriesDescription", ""))
            sequence = _clean_text(getattr(ds, "SequenceName", ""))
            photo = _clean_text(getattr(ds, "PhotometricInterpretation", ""))
            if manufacturer:
                manufacturers.append(manufacturer)
            if model:
                models.append(model)
            if description:
                descriptions.append(description)
            if sequence:
                sequences.append(sequence)
            if photo:
                photometric.append(photo)
            value = _float(getattr(ds, "MagneticFieldStrength", None))
            if value is not None and value > 0:
                fields.append(value)
            try:
                frames += max(1, int(getattr(ds, "NumberOfFrames", 1) or 1))
            except Exception:
                frames += 1
            try:
                bits_allocated.append(int(getattr(ds, "BitsAllocated")))
            except Exception:
                pass
            try:
                bits_stored.append(int(getattr(ds, "BitsStored")))
            except Exception:
                pass
        except Exception:
            base["header_failures"] += 1

    rows, cols = _mode_pair(dims)
    row_spacing, col_spacing = _median_pair(spacings)
    unique_dims = sorted(set(dims))
    rounded_spacings = sorted(set((round(a, 6), round(b, 6)) for a, b in spacings))

    base.update(
        {
            "decoded_frames_from_headers": int(frames),
            "rows": rows,
            "columns": cols,
            "matrix": f"{rows}x{cols}" if rows and cols else "",
            "unique_matrix_count_within_series": len(unique_dims),
            "matrix_consistent_within_series": len(unique_dims) <= 1,
            "matrix_values": ";".join(f"{r}x{c}" for r, c in unique_dims),
            "pixel_spacing_row_mm": row_spacing,
            "pixel_spacing_col_mm": col_spacing,
            "unique_spacing_count_within_series": len(rounded_spacings),
            "spacing_consistent_within_series": len(rounded_spacings) <= 1,
            "pixel_spacing_values_mm": ";".join(f"{r:g}x{c:g}" for r, c in rounded_spacings),
            "fov_row_mm": float(rows * row_spacing) if rows and row_spacing else None,
            "fov_col_mm": float(cols * col_spacing) if cols and col_spacing else None,
            "slice_thickness_mm": float(np.median(thicknesses)) if thicknesses else None,
            "spacing_between_slices_mm": float(np.median(between)) if between else None,
            "manufacturer": Counter(manufacturers).most_common(1)[0][0] if manufacturers else "",
            "manufacturer_model": Counter(models).most_common(1)[0][0] if models else "",
            "magnetic_field_strength_t": float(np.median(fields)) if fields else None,
            "series_description": Counter(descriptions).most_common(1)[0][0] if descriptions else "",
            "sequence_name": Counter(sequences).most_common(1)[0][0] if sequences else "",
            "photometric_interpretation": Counter(photometric).most_common(1)[0][0] if photometric else "",
            "bits_allocated": int(np.median(bits_allocated)) if bits_allocated else None,
            "bits_stored": int(np.median(bits_stored)) if bits_stored else None,
        }
    )
    return base


def _quantiles(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if not len(values):
        return {"n": 0}
    q = np.quantile(values, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "n": int(len(values)),
        "min": float(q[0]),
        "p01": float(q[1]),
        "p05": float(q[2]),
        "p25": float(q[3]),
        "median": float(q[4]),
        "p75": float(q[5]),
        "p95": float(q[6]),
        "p99": float(q[7]),
        "max": float(q[8]),
    }


def _counts(series: pd.Series, n: int = 25) -> list[dict]:
    values = series.fillna("<missing>").astype(str).replace("", "<missing>")
    total = max(len(values), 1)
    return [
        {"value": str(value), "series": int(count), "fraction": float(count / total)}
        for value, count in values.value_counts(dropna=False).head(n).items()
    ]


def _subset_summary(frame: pd.DataFrame, *, crop_fraction: float, canvases: Iterable[int]) -> dict:
    valid = frame.loc[
        frame["rows"].notna() & frame["columns"].notna()
    ].copy()
    valid["crop_rows"] = np.rint(valid["rows"].astype(float) * crop_fraction).astype(int)
    valid["crop_columns"] = np.rint(valid["columns"].astype(float) * crop_fraction).astype(int)
    valid["crop_max_dim"] = valid[["crop_rows", "crop_columns"]].max(axis=1)

    canvas_rows = []
    for canvas in canvases:
        fits = (valid["crop_rows"] <= int(canvas)) & (valid["crop_columns"] <= int(canvas))
        canvas_rows.append(
            {
                "canvas": int(canvas),
                "fits_series": int(fits.sum()),
                "valid_geometry_series": int(len(valid)),
                "coverage": float(fits.mean()) if len(valid) else 0.0,
                "pixel_area_ratio_vs_224": float((int(canvas) / 224.0) ** 2),
            }
        )

    spacing_row = _quantiles(valid["pixel_spacing_row_mm"])
    spacing_col = _quantiles(valid["pixel_spacing_col_mm"])
    spacing_ratio = None
    if spacing_row.get("p05") and spacing_row.get("p95"):
        spacing_ratio = float(spacing_row["p95"] / spacing_row["p05"])
    col_ratio = None
    if spacing_col.get("p05") and spacing_col.get("p95"):
        col_ratio = float(spacing_col["p95"] / spacing_col["p05"])

    smallest_99 = next((x["canvas"] for x in canvas_rows if x["coverage"] >= 0.99), None)
    physical_heterogeneity = max(x for x in (spacing_ratio, col_ratio) if x is not None) if any(
        x is not None for x in (spacing_ratio, col_ratio)
    ) else None

    if smallest_99 is None:
        decision = (
            "No tested <=640 square canvas preserves the fixed 90% native crop for >=99% "
            "of readable series; a pure padding-only policy needs either a larger canvas, "
            "a different crop contract, or selective resampling."
        )
    elif physical_heterogeneity is not None and physical_heterogeneity > 1.25:
        decision = (
            f"Padding-only is geometrically feasible for >=99% at canvas {smallest_99}, "
            "but PixelSpacing is materially heterogeneous (p95/p05 >1.25). Padding preserves "
            "sampled pixels but does not make physical anatomy scale comparable across scans."
        )
    else:
        decision = (
            f"Padding-only is geometrically feasible for >=99% at canvas {smallest_99}; "
            "PixelSpacing variability should still be reviewed before freezing a native-input experiment."
        )

    return {
        "series": int(len(frame)),
        "readable_geometry_series": int(len(valid)),
        "matrix_distribution": _counts(valid["matrix"]),
        "rows": _quantiles(valid["rows"]),
        "columns": _quantiles(valid["columns"]),
        "pixel_spacing_row_mm": spacing_row,
        "pixel_spacing_col_mm": spacing_col,
        "pixel_spacing_p95_p05_ratio_row": spacing_ratio,
        "pixel_spacing_p95_p05_ratio_col": col_ratio,
        "fov_row_mm": _quantiles(valid["fov_row_mm"]),
        "fov_col_mm": _quantiles(valid["fov_col_mm"]),
        "slice_thickness_mm": _quantiles(valid["slice_thickness_mm"]),
        "spacing_between_slices_mm": _quantiles(valid["spacing_between_slices_mm"]),
        "manufacturer_distribution": _counts(valid["manufacturer"]),
        "manufacturer_model_distribution": _counts(valid["manufacturer_model"]),
        "field_strength_t_distribution": _counts(valid["magnetic_field_strength_t"].round(3).astype("string")),
        "crop_fraction": float(crop_fraction),
        "crop_max_dimension": _quantiles(valid["crop_max_dim"]),
        "padding_canvas_feasibility": canvas_rows,
        "smallest_tested_canvas_covering_99pct": smallest_99,
        "decision_note": decision,
    }


def _markdown(summary: dict) -> str:
    eligible = summary["model_eligible"]
    lines = [
        "# Native DICOM resolution audit",
        "",
        f"Audit version: `{summary['audit_version']}`",
        "",
        "This is a header-only geometry audit. PixelData was not decompressed or modified.",
        "",
        "## Coverage",
        "",
        f"- Series rows: **{summary['series_rows']}**",
        f"- Model-eligible repaired-plane series: **{summary['model_eligible_series']}**",
        f"- Missing series directories: **{summary['missing_series_directories']}**",
        f"- Header files inspected: **{summary['header_files_inspected']}**",
        f"- Header read failures: **{summary['header_failures']}**",
        f"- Series with internally inconsistent matrix: **{summary['series_with_matrix_variation']}**",
        f"- Series with internally inconsistent PixelSpacing: **{summary['series_with_spacing_variation']}**",
        "",
        "## Model-eligible native geometry",
        "",
        "### Matrix distribution (top entries)",
        "",
        "| Matrix | Series | Fraction |",
        "|---|---:|---:|",
    ]
    for row in eligible["matrix_distribution"]:
        lines.append(f"| {row['value']} | {row['series']} | {row['fraction']:.4f} |")

    lines += [
        "",
        "### Pixel spacing",
        "",
        f"- Row spacing median: **{eligible['pixel_spacing_row_mm'].get('median')} mm/pixel**",
        f"- Row spacing p05–p95: **{eligible['pixel_spacing_row_mm'].get('p05')}–{eligible['pixel_spacing_row_mm'].get('p95')} mm/pixel**",
        f"- Column spacing median: **{eligible['pixel_spacing_col_mm'].get('median')} mm/pixel**",
        f"- Column spacing p05–p95: **{eligible['pixel_spacing_col_mm'].get('p05')}–{eligible['pixel_spacing_col_mm'].get('p95')} mm/pixel**",
        "",
        "### 90% native crop + padding-only feasibility",
        "",
        "| Square canvas | Fits series | Coverage | Pixel area vs 224 |",
        "|---:|---:|---:|---:|",
    ]
    for row in eligible["padding_canvas_feasibility"]:
        lines.append(
            f"| {row['canvas']} | {row['fits_series']} | {row['coverage']:.4f} | {row['pixel_area_ratio_vs_224']:.2f}x |"
        )
    lines += [
        "",
        "## Audit interpretation",
        "",
        eligible["decision_note"],
        "",
        "A padding-only canvas preserves the retained source pixels exactly, but it does not standardize physical scale. "
        "The final B37/native decision must consider both matrix-size coverage and PixelSpacing/FOV variability.",
        "",
    ]
    return "\n".join(lines)


def run_audit(
    *,
    data_root: str | Path,
    series_csv: str | Path | None = None,
    split: str = "train",
    out_root: str | Path = "runs/native_resolution_audit",
    workers: int = 4,
    crop_fraction: float = DEFAULT_CROP_FRACTION,
    canvases: Iterable[int] = DEFAULT_CANVASES,
    max_series: int | None = None,
) -> dict:
    root = Path(data_root).resolve()
    series_path = Path(series_csv) if series_csv else root / f"{split}_series.csv"
    series = load_series_csv(series_path)
    series, repair_stats = backfill_series_metadata(series, root, split=split)
    if max_series is not None:
        series = series.iloc[: int(max_series)].copy()

    rows = list(series.itertuples(index=False))
    started = time.monotonic()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        future_to_meta = {
            pool.submit(
                inspect_series,
                root,
                split,
                str(row.StudyInstanceUID),
                str(row.SeriesInstanceUID),
            ): row
            for row in rows
        }
        for done, future in enumerate(as_completed(future_to_meta), start=1):
            row = future_to_meta[future]
            result = future.result()
            result["Anatomical_Plane"] = str(row.Anatomical_Plane)
            result["Fluid_Sensitive"] = row.Fluid_Sensitive
            result["Fat_Suppression"] = row.Fat_Suppression
            result["model_eligible_recognized_plane"] = str(row.Anatomical_Plane) in PLANES
            results.append(result)
            if done % 250 == 0 or done == len(rows):
                elapsed = (time.monotonic() - started) / 60.0
                print(f"[native-audit] {done}/{len(rows)} series | {elapsed:.1f} min", flush=True)

    frame = pd.DataFrame(results)
    frame = frame.sort_values(["StudyInstanceUID", "SeriesInstanceUID"]).reset_index(drop=True)
    for column in ("rows", "columns"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    eligible = frame.loc[frame["model_eligible_recognized_plane"].astype(bool)].copy()
    summary = {
        "audit_version": AUDIT_VERSION,
        "data_root": str(root),
        "series_csv": str(series_path.resolve()),
        "split": split,
        "header_only": True,
        "pixel_data_decompressed": False,
        "crop_fraction_for_padding_feasibility": float(crop_fraction),
        "tested_square_canvases": [int(x) for x in canvases],
        "series_rows": int(len(frame)),
        "model_eligible_series": int(len(eligible)),
        "missing_series_directories": int((~frame["series_dir_found"].astype(bool)).sum()),
        "header_files_inspected": int(frame["readable_headers"].fillna(0).sum()),
        "header_failures": int(frame["header_failures"].fillna(0).sum()),
        "series_with_matrix_variation": int((~frame["matrix_consistent_within_series"].fillna(True)).sum()),
        "series_with_spacing_variation": int((~frame["spacing_consistent_within_series"].fillna(True)).sum()),
        "metadata_repair": repair_stats,
        "all_series": _subset_summary(frame, crop_fraction=crop_fraction, canvases=canvases),
        "model_eligible": _subset_summary(eligible, crop_fraction=crop_fraction, canvases=canvases),
        "elapsed_minutes": float((time.monotonic() - started) / 60.0),
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "series_geometry.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "series": summary["series_rows"],
        "model_eligible": summary["model_eligible_series"],
        "missing_dirs": summary["missing_series_directories"],
        "smallest_99pct_canvas": summary["model_eligible"]["smallest_tested_canvas_covering_99pct"],
        "decision_note": summary["model_eligible"]["decision_note"],
        "out_root": str(out),
    }, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser("Audit native RSNA knee DICOM matrix and physical sampling")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--series-csv")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-root", default="runs/native_resolution_audit")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crop-fraction", type=float, default=DEFAULT_CROP_FRACTION)
    ap.add_argument("--canvases", type=int, nargs="+", default=list(DEFAULT_CANVASES))
    ap.add_argument("--max-series", type=int)
    args = ap.parse_args()
    if not 0.5 <= float(args.crop_fraction) <= 1.0:
        ap.error("--crop-fraction must be in [0.5,1.0]")
    run_audit(
        data_root=args.data_root,
        series_csv=args.series_csv,
        split=args.split,
        out_root=args.out_root,
        workers=args.workers,
        crop_fraction=args.crop_fraction,
        canvases=args.canvases,
        max_series=args.max_series,
    )


if __name__ == "__main__":
    main()
