"""Phase-3 descriptive DICOM-header audit for the knee MRI dataset.

This module reads one representative DICOM header per listed training series with
``stop_before_pixels=True``.  It does not decode image pixels, train a model,
change B6 supervision, or use PV1/PV2 outcomes.  Its purpose is to characterize
scanner/protocol/resolution/orientation heterogeneity and explain the long
slice-count tail identified by the Phase-2 dataset-contract audit.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_series_csv
from .dicom import DICOM_SUFFIXES, find_series_dir

HEADER_AUDIT_VERSION = "official_dataset_header_audit_v1"
TAIL_THRESHOLDS = (78, 100, 200)


def _finite_float(value):
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _finite_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _quantiles(values) -> dict:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    out = {"n": int(arr.size), "mean": float(arr.mean())}
    for q in (0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0):
        out[f"q{int(round(100*q)):02d}"] = float(np.quantile(arr, q))
    return out


def orientation_from_iop(value) -> dict:
    """Return patient-axis normal/closest anatomical plane for one IOP value."""
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return {"normal_plane": "", "obliquity_deg": None}
    if arr.size < 6 or not np.isfinite(arr[:6]).all():
        return {"normal_plane": "", "obliquity_deg": None}
    row = arr[:3]
    col = arr[3:6]
    normal = np.cross(row, col)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1e-8:
        return {"normal_plane": "", "obliquity_deg": None}
    normal = normal / norm
    absn = np.abs(normal)
    axis = int(np.argmax(absn))
    plane = ("Sagittal", "Coronal", "Axial")[axis]
    cosine = float(np.clip(absn[axis], 0.0, 1.0))
    return {
        "normal_plane": plane,
        "obliquity_deg": float(np.degrees(np.arccos(cosine))),
        "normal_x": float(normal[0]),
        "normal_y": float(normal[1]),
        "normal_z": float(normal[2]),
    }


def _dicom_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES
    )


def inspect_series_header(data_root: Path, record: pd.Series) -> dict:
    import pydicom

    study_uid = str(record["StudyInstanceUID"])
    series_uid = str(record["SeriesInstanceUID"])
    folder = find_series_dir(data_root, "train", study_uid, series_uid)
    base = {
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "Anatomical_Plane": str(record["Anatomical_Plane"]),
        "Fluid_Sensitive": record["Fluid_Sensitive"],
        "Fat_Suppression": record["Fat_Suppression"],
        "directory_exists": bool(folder is not None),
        "dicom_files": 0,
        "header_read_ok": False,
        "header_error": "",
    }
    if folder is None:
        base["header_error"] = "missing_series_directory"
        return base
    files = _dicom_files(folder)
    base["dicom_files"] = int(len(files))
    if not files:
        base["header_error"] = "no_dicom_files"
        return base

    try:
        ds = pydicom.dcmread(str(files[0]), force=True, stop_before_pixels=True)
    except Exception as exc:
        base["header_error"] = f"{type(exc).__name__}: {exc}"[:240]
        return base

    base["header_read_ok"] = True
    try:
        transfer = _text(ds.file_meta.TransferSyntaxUID)
    except Exception:
        transfer = ""

    spacing = getattr(ds, "PixelSpacing", None)
    ps_row = ps_col = None
    try:
        vals = np.asarray(spacing, dtype=float).reshape(-1)
        if vals.size >= 2 and np.isfinite(vals[:2]).all() and np.all(vals[:2] > 0):
            ps_row, ps_col = float(vals[0]), float(vals[1])
    except Exception:
        pass

    rows = _finite_int(getattr(ds, "Rows", None))
    cols = _finite_int(getattr(ds, "Columns", None))
    orient = orientation_from_iop(getattr(ds, "ImageOrientationPatient", None))
    base.update({
        "transfer_syntax_uid": transfer,
        "manufacturer": _text(getattr(ds, "Manufacturer", None)),
        "manufacturer_model": _text(getattr(ds, "ManufacturerModelName", None)),
        "magnetic_field_strength_t": _finite_float(getattr(ds, "MagneticFieldStrength", None)),
        "mr_acquisition_type": _text(getattr(ds, "MRAcquisitionType", None)),
        "scanning_sequence": _text(getattr(ds, "ScanningSequence", None)),
        "sequence_variant": _text(getattr(ds, "SequenceVariant", None)),
        "scan_options": _text(getattr(ds, "ScanOptions", None)),
        "rows": rows,
        "columns": cols,
        "pixel_spacing_row_mm": ps_row,
        "pixel_spacing_col_mm": ps_col,
        "fov_row_mm": float(rows * ps_row) if rows is not None and ps_row is not None else None,
        "fov_col_mm": float(cols * ps_col) if cols is not None and ps_col is not None else None,
        "slice_thickness_mm": _finite_float(getattr(ds, "SliceThickness", None)),
        "spacing_between_slices_mm": _finite_float(getattr(ds, "SpacingBetweenSlices", None)),
        "number_of_frames": _finite_int(getattr(ds, "NumberOfFrames", 1)) or 1,
        "photometric_interpretation": _text(getattr(ds, "PhotometricInterpretation", None)),
        "bits_allocated": _finite_int(getattr(ds, "BitsAllocated", None)),
        "bits_stored": _finite_int(getattr(ds, "BitsStored", None)),
        "pixel_representation": _finite_int(getattr(ds, "PixelRepresentation", None)),
        **orient,
    })
    base["orientation_matches_supplied_plane"] = bool(
        base.get("normal_plane") and base.get("normal_plane") == base["Anatomical_Plane"]
    )
    return base


def _slice_bin(n: int) -> str:
    if n > 200:
        return ">200"
    if n > 100:
        return "101-200"
    if n > 78:
        return "79-100"
    if n > 48:
        return "49-78"
    return "<=48"


def summarize_headers(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ok = frame.loc[frame["header_read_ok"].astype(bool)].copy()
    summary = {
        "series": int(len(frame)),
        "header_read_ok": int(len(ok)),
        "header_read_failures": int(len(frame) - len(ok)),
        "multi_frame_representative_headers": int((pd.to_numeric(ok["number_of_frames"], errors="coerce").fillna(1) > 1).sum()),
        "orientation_available": int(ok["normal_plane"].fillna("").astype(str).ne("").sum()),
        "orientation_matches_supplied_plane": int(ok["orientation_matches_supplied_plane"].fillna(False).astype(bool).sum()),
        "orientation_mismatches_supplied_plane": int((ok["normal_plane"].fillna("").astype(str).ne("") & ~ok["orientation_matches_supplied_plane"].fillna(False).astype(bool)).sum()),
        "obliquity_deg": _quantiles(ok["obliquity_deg"]),
        "magnetic_field_strength_t": _quantiles(ok["magnetic_field_strength_t"]),
        "pixel_spacing_row_mm": _quantiles(ok["pixel_spacing_row_mm"]),
        "pixel_spacing_col_mm": _quantiles(ok["pixel_spacing_col_mm"]),
        "fov_row_mm": _quantiles(ok["fov_row_mm"]),
        "fov_col_mm": _quantiles(ok["fov_col_mm"]),
        "slice_thickness_mm": _quantiles(ok["slice_thickness_mm"]),
        "spacing_between_slices_mm": _quantiles(ok["spacing_between_slices_mm"]),
        "rows": _quantiles(ok["rows"]),
        "columns": _quantiles(ok["columns"]),
        "dicom_files": _quantiles(frame["dicom_files"]),
    }

    categorical_fields = (
        "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression",
        "manufacturer", "manufacturer_model", "magnetic_field_strength_t",
        "mr_acquisition_type", "transfer_syntax_uid", "photometric_interpretation",
        "bits_allocated", "bits_stored", "pixel_representation", "normal_plane",
    )
    cat_rows = []
    for field in categorical_fields:
        if field not in ok.columns:
            continue
        values = ok[field].astype("string").fillna("<missing>")
        for value, n in values.value_counts(dropna=False).items():
            cat_rows.append({"field": field, "value": str(value), "series": int(n)})
    categorical = pd.DataFrame(cat_rows)

    tail = frame.copy()
    tail["slice_bin"] = tail["dicom_files"].astype(int).map(_slice_bin)
    tail_group_fields = [
        "slice_bin", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression",
        "mr_acquisition_type", "manufacturer", "manufacturer_model", "magnetic_field_strength_t",
        "normal_plane",
    ]
    tail_profile = (
        tail.groupby(tail_group_fields, dropna=False).size().reset_index(name="series")
        .sort_values(["slice_bin", "series"], ascending=[True, False])
    )

    orientation = (
        ok.groupby(["Anatomical_Plane", "normal_plane"], dropna=False)
        .agg(series=("SeriesInstanceUID", "size"), median_obliquity_deg=("obliquity_deg", "median"),
             p95_obliquity_deg=("obliquity_deg", lambda x: float(np.nanquantile(pd.to_numeric(x, errors="coerce"), 0.95)) if pd.to_numeric(x, errors="coerce").notna().any() else np.nan))
        .reset_index()
    )
    return summary, categorical, tail_profile, orientation


def run_header_audit(*, data_root: str | Path, out_root: str | Path) -> dict:
    root = Path(data_root).resolve()
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    series = load_series_csv(root / "train_series.csv")

    rows = []
    total = len(series)
    for i, (_, record) in enumerate(series.iterrows(), start=1):
        rows.append(inspect_series_header(root, record))
        if i % 1000 == 0 or i == total:
            print(f"[header audit] {i}/{total} series")
    frame = pd.DataFrame(rows)
    summary, categorical, tail_profile, orientation = summarize_headers(frame)
    payload = {
        "audit_version": HEADER_AUDIT_VERSION,
        "purpose": "descriptive scanner/protocol/resolution/orientation audit after slice-count Phase 2",
        "representative_header_policy": "lexicographically first DICOM file per listed training series; stop_before_pixels=True",
        "tail_thresholds": list(TAIL_THRESHOLDS),
        "summary": summary,
        "governance": (
            "Descriptive only. Do not define target-specific architecture changes, B35, or B6 changes directly "
            "from this audit. Interpret manufacturer/model/protocol categories as acquisition metadata, not patient/site identity."
        ),
    }
    frame.to_csv(out / "header_by_series.csv", index=False)
    categorical.to_csv(out / "header_categorical_counts.csv", index=False)
    tail_profile.to_csv(out / "slice_tail_header_profile.csv", index=False)
    orientation.to_csv(out / "orientation_vs_supplied_plane.csv", index=False)
    (out / "header_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "audit_version": HEADER_AUDIT_VERSION,
        "series": summary["series"],
        "header_read_ok": summary["header_read_ok"],
        "header_read_failures": summary["header_read_failures"],
        "multi_frame_representative_headers": summary["multi_frame_representative_headers"],
        "orientation_mismatches_supplied_plane": summary["orientation_mismatches_supplied_plane"],
        "summary": str(out / "header_summary.json"),
    }, indent=2))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser("Audit one representative DICOM header per training series")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", default="runs/dataset_header_audit")
    args = ap.parse_args()
    run_header_audit(data_root=args.data_root, out_root=args.out_root)


if __name__ == "__main__":
    main()
