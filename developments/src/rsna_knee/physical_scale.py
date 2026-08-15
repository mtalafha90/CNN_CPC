"""Label-free physical geometry audit and in-plane MRI normalization for B10."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .constants import DUAL_STREAMS
from .dicom import DICOM_SUFFIXES, find_series_dir

B10_PHYSICAL_POLICY = "inplane_median_spacing_fov_v1"
B10_ROUTING_MODE = "dual"
B10_MISSING_SPACING_ACTION = "legacy_resize"
B10_MIN_GEOMETRY_COVERAGE = 0.95


def _plane_from_stream(stream: str) -> str:
    plane = str(stream).split("_", 1)[0].strip().lower()
    mapping = {"sagittal": "Sagittal", "coronal": "Coronal", "axial": "Axial"}
    if plane not in mapping:
        raise ValueError(f"unknown MRI stream plane: {stream!r}")
    return mapping[plane]


def selected_series_signature(
    series_index: dict[str, dict[str, str | None]],
    studies: Iterable[str],
) -> str:
    rows: list[str] = []
    for uid in sorted(str(x) for x in studies):
        mapping = series_index.get(uid, {})
        for stream in DUAL_STREAMS:
            series_uid = mapping.get(stream)
            rows.append(f"{uid}|{stream}|{'' if series_uid is None else str(series_uid)}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _iter_dicom_files(path: Path) -> list[Path]:
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES
    )


def inspect_series_geometry(path: str | Path) -> dict:
    """Read representative header geometry without decoding pixel data."""
    import pydicom

    path = Path(path)
    candidates = _iter_dicom_files(path)
    errors = 0
    for file_path in candidates:
        try:
            ds = pydicom.dcmread(str(file_path), force=True, stop_before_pixels=True)
            spacing = None
            try:
                values = np.asarray(getattr(ds, "PixelSpacing"), dtype=float).reshape(-1)
                if (
                    len(values) >= 2
                    and np.isfinite(values[:2]).all()
                    and np.all(values[:2] > 0)
                ):
                    spacing = [float(values[0]), float(values[1])]
            except Exception:
                spacing = None

            rows = int(getattr(ds, "Rows", 0) or 0)
            columns = int(getattr(ds, "Columns", 0) or 0)
            fov = None
            if spacing is not None and rows > 0 and columns > 0:
                fov = [float(rows * spacing[0]), float(columns * spacing[1])]

            def _positive_float(name: str):
                try:
                    value = float(getattr(ds, name))
                    if np.isfinite(value) and value > 0:
                        return value
                except Exception:
                    pass
                return None

            return {
                "header_file": str(file_path),
                "pixel_spacing_mm": spacing,
                "rows": rows if rows > 0 else None,
                "columns": columns if columns > 0 else None,
                "fov_mm": fov,
                "slice_thickness_mm": _positive_float("SliceThickness"),
                "spacing_between_slices_mm": _positive_float("SpacingBetweenSlices"),
                "manufacturer": str(getattr(ds, "Manufacturer", "")).strip() or None,
                "manufacturer_model": (
                    str(getattr(ds, "ManufacturerModelName", "")).strip() or None
                ),
                "magnetic_field_strength_t": _positive_float("MagneticFieldStrength"),
                "candidate_files": len(candidates),
                "header_read_failures_before_success": errors,
            }
        except Exception:
            errors += 1

    return {
        "header_file": None,
        "pixel_spacing_mm": None,
        "rows": None,
        "columns": None,
        "fov_mm": None,
        "slice_thickness_mm": None,
        "spacing_between_slices_mm": None,
        "manufacturer": None,
        "manufacturer_model": None,
        "magnetic_field_strength_t": None,
        "candidate_files": len(candidates),
        "header_read_failures_before_success": errors,
    }


def _quantiles(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([x for x in values if np.isfinite(x) and x > 0], dtype=float)
    if finite.size == 0:
        return {"p05": None, "p25": None, "p50": None, "p75": None, "p95": None}
    q = np.quantile(finite, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "p05": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
    }


def derive_policy_from_geometry(
    geometry: pd.DataFrame,
    *,
    source_study_count: int,
    selected_series_signature_value: str,
    min_geometry_coverage: float = B10_MIN_GEOMETRY_COVERAGE,
) -> dict:
    required = {
        "StudyInstanceUID",
        "stream",
        "plane",
        "SeriesInstanceUID",
        "row_spacing_mm",
        "col_spacing_mm",
        "row_fov_mm",
        "col_fov_mm",
    }
    missing = required.difference(geometry.columns)
    if missing:
        raise ValueError(f"geometry table missing columns: {sorted(missing)}")

    selected_count = int(len(geometry))
    valid = (
        pd.to_numeric(geometry["row_spacing_mm"], errors="coerce").gt(0)
        & pd.to_numeric(geometry["col_spacing_mm"], errors="coerce").gt(0)
        & pd.to_numeric(geometry["row_fov_mm"], errors="coerce").gt(0)
        & pd.to_numeric(geometry["col_fov_mm"], errors="coerce").gt(0)
    )
    coverage = float(valid.mean()) if selected_count else 0.0
    if coverage < float(min_geometry_coverage):
        raise ValueError(
            f"B10 physical geometry coverage {coverage:.4f} is below "
            f"the frozen minimum {float(min_geometry_coverage):.4f}"
        )

    planes: dict[str, dict] = {}
    for plane in ("Sagittal", "Coronal", "Axial"):
        part = geometry.loc[geometry["plane"].eq(plane)].copy()
        part_valid = part.loc[
            pd.to_numeric(part["row_spacing_mm"], errors="coerce").gt(0)
            & pd.to_numeric(part["col_spacing_mm"], errors="coerce").gt(0)
            & pd.to_numeric(part["row_fov_mm"], errors="coerce").gt(0)
            & pd.to_numeric(part["col_fov_mm"], errors="coerce").gt(0)
        ]
        if part.empty or part_valid.empty:
            raise ValueError(f"B10 cannot derive physical policy for {plane}")
        row_spacing = part_valid["row_spacing_mm"].astype(float).tolist()
        col_spacing = part_valid["col_spacing_mm"].astype(float).tolist()
        row_fov = part_valid["row_fov_mm"].astype(float).tolist()
        col_fov = part_valid["col_fov_mm"].astype(float).tolist()

        planes[plane] = {
            "selected_series": int(len(part)),
            "valid_geometry_series": int(len(part_valid)),
            "geometry_coverage": float(len(part_valid) / len(part)),
            "target_spacing_mm": [
                float(np.median(row_spacing)),
                float(np.median(col_spacing)),
            ],
            "target_fov_mm": [
                float(np.median(row_fov)),
                float(np.median(col_fov)),
            ],
            "row_spacing_distribution_mm": _quantiles(row_spacing),
            "col_spacing_distribution_mm": _quantiles(col_spacing),
            "row_fov_distribution_mm": _quantiles(row_fov),
            "col_fov_distribution_mm": _quantiles(col_fov),
        }

    policy = {
        "policy_name": B10_PHYSICAL_POLICY,
        "routing_mode": B10_ROUTING_MODE,
        "normalization_scope": "in_plane_only",
        "canonical_statistic": "median over valid selected weak-training series, separately by plane",
        "missing_spacing_action": B10_MISSING_SPACING_ACTION,
        "min_geometry_coverage": float(min_geometry_coverage),
        "uses_gold_labels": False,
        "source_study_count": int(source_study_count),
        "selected_series_count": selected_count,
        "valid_geometry_series": int(valid.sum()),
        "geometry_coverage": coverage,
        "selected_series_signature": str(selected_series_signature_value),
        "planes": planes,
    }
    policy["policy_sha256"] = physical_policy_digest(policy)
    return policy


def physical_policy_digest(policy: dict) -> str:
    payload = dict(policy)
    payload.pop("policy_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_physical_scale_policy(policy: dict) -> None:
    if str(policy.get("policy_name")) != B10_PHYSICAL_POLICY:
        raise ValueError(f"B10 policy_name must be {B10_PHYSICAL_POLICY!r}")
    if str(policy.get("routing_mode")) != B10_ROUTING_MODE:
        raise ValueError("B10 physical policy must be derived from historical dual routing")
    if str(policy.get("normalization_scope")) != "in_plane_only":
        raise ValueError("B10-v1 normalizes in-plane geometry only")
    if str(policy.get("missing_spacing_action")) != B10_MISSING_SPACING_ACTION:
        raise ValueError(
            f"B10 missing-spacing action must remain {B10_MISSING_SPACING_ACTION!r}"
        )
    if bool(policy.get("uses_gold_labels", True)):
        raise ValueError("B10 physical-scale policy must certify zero gold-label use")
    planes = policy.get("planes")
    if not isinstance(planes, dict):
        raise ValueError("B10 physical-scale policy is missing planes")
    for plane in ("Sagittal", "Coronal", "Axial"):
        part = planes.get(plane)
        if not isinstance(part, dict):
            raise ValueError(f"B10 physical policy missing {plane}")
        for field in ("target_spacing_mm", "target_fov_mm"):
            values = np.asarray(part.get(field, []), dtype=float).reshape(-1)
            if len(values) != 2 or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"invalid B10 {plane} {field}: {part.get(field)!r}")
    expected = physical_policy_digest(policy)
    recorded = str(policy.get("policy_sha256", ""))
    if recorded and recorded != expected:
        raise ValueError("B10 physical policy SHA256 does not match its contents")


def load_physical_scale_policy(path: str | Path) -> dict:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError("B10 physical policy must be a JSON object")
    validate_physical_scale_policy(policy)
    if not policy.get("policy_sha256"):
        policy["policy_sha256"] = physical_policy_digest(policy)
    return policy


def audit_selected_series_geometry(
    *,
    data_root: str | Path,
    split: str,
    studies: list[str],
    series_index: dict[str, dict[str, str | None]],
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    missing_dirs = 0
    missing_headers = 0
    for uid in studies:
        mapping = series_index.get(str(uid), {})
        for stream in DUAL_STREAMS:
            series_uid = mapping.get(stream)
            if not series_uid:
                continue
            path = find_series_dir(data_root, split, str(uid), str(series_uid))
            if path is None:
                missing_dirs += 1
                info = {
                    "pixel_spacing_mm": None,
                    "rows": None,
                    "columns": None,
                    "fov_mm": None,
                    "slice_thickness_mm": None,
                    "spacing_between_slices_mm": None,
                    "manufacturer": None,
                    "manufacturer_model": None,
                    "magnetic_field_strength_t": None,
                    "candidate_files": 0,
                    "header_read_failures_before_success": 0,
                }
            else:
                info = inspect_series_geometry(path)
                if info.get("header_file") is None:
                    missing_headers += 1
            spacing = info.get("pixel_spacing_mm")
            fov = info.get("fov_mm")
            rows.append(
                {
                    "StudyInstanceUID": str(uid),
                    "stream": stream,
                    "plane": _plane_from_stream(stream),
                    "SeriesInstanceUID": str(series_uid),
                    "row_spacing_mm": spacing[0] if spacing else np.nan,
                    "col_spacing_mm": spacing[1] if spacing else np.nan,
                    "rows": info.get("rows"),
                    "columns": info.get("columns"),
                    "row_fov_mm": fov[0] if fov else np.nan,
                    "col_fov_mm": fov[1] if fov else np.nan,
                    "slice_thickness_mm": info.get("slice_thickness_mm"),
                    "spacing_between_slices_mm": info.get("spacing_between_slices_mm"),
                    "manufacturer": info.get("manufacturer"),
                    "manufacturer_model": info.get("manufacturer_model"),
                    "magnetic_field_strength_t": info.get("magnetic_field_strength_t"),
                    "candidate_files": info.get("candidate_files"),
                    "header_read_failures_before_success": info.get(
                        "header_read_failures_before_success"
                    ),
                    "series_dir_found": path is not None,
                }
            )

    frame = pd.DataFrame(rows)
    valid = (
        pd.to_numeric(frame.get("row_spacing_mm"), errors="coerce").gt(0)
        & pd.to_numeric(frame.get("col_spacing_mm"), errors="coerce").gt(0)
        & pd.to_numeric(frame.get("row_fov_mm"), errors="coerce").gt(0)
        & pd.to_numeric(frame.get("col_fov_mm"), errors="coerce").gt(0)
    ) if len(frame) else pd.Series([], dtype=bool)
    summary = {
        "studies": int(len(studies)),
        "selected_series": int(len(frame)),
        "valid_geometry_series": int(valid.sum()) if len(frame) else 0,
        "geometry_coverage": float(valid.mean()) if len(frame) else 0.0,
        "missing_series_directories": int(missing_dirs),
        "series_without_readable_header": int(missing_headers),
        "routing_mode": B10_ROUTING_MODE,
        "uses_gold_labels": False,
    }
    return frame, summary


def resample_volume_inplane(
    volume: np.ndarray,
    *,
    source_spacing_mm: tuple[float, float] | list[float] | None,
    plane: str,
    policy: dict,
) -> tuple[np.ndarray, bool]:
    """Resample to canonical mm/pixel, then center crop/pad to canonical physical FOV."""
    validate_physical_scale_policy(policy)
    if source_spacing_mm is None:
        return np.asarray(volume, dtype=np.float32), False
    source = np.asarray(source_spacing_mm, dtype=float).reshape(-1)
    if len(source) != 2 or not np.isfinite(source).all() or np.any(source <= 0):
        return np.asarray(volume, dtype=np.float32), False
    if plane not in policy["planes"]:
        raise ValueError(f"B10 policy does not contain plane {plane!r}")

    target_spacing = np.asarray(policy["planes"][plane]["target_spacing_mm"], dtype=float)
    target_fov = np.asarray(policy["planes"][plane]["target_fov_mm"], dtype=float)

    v = np.asarray(volume, dtype=np.float32)
    if v.ndim != 3 or len(v) == 0:
        raise ValueError(f"expected [S,H,W] volume, got {v.shape}")
    source_h, source_w = int(v.shape[1]), int(v.shape[2])
    new_h = max(1, int(round(source_h * source[0] / target_spacing[0])))
    new_w = max(1, int(round(source_w * source[1] / target_spacing[1])))

    tensor = torch.from_numpy(v).unsqueeze(1)
    if (new_h, new_w) != (source_h, source_w):
        tensor = F.interpolate(
            tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

    target_h = max(1, int(round(target_fov[0] / target_spacing[0])))
    target_w = max(1, int(round(target_fov[1] / target_spacing[1])))
    current_h, current_w = int(tensor.shape[-2]), int(tensor.shape[-1])

    crop_h = min(current_h, target_h)
    crop_w = min(current_w, target_w)
    src_r = max(0, (current_h - crop_h) // 2)
    src_c = max(0, (current_w - crop_w) // 2)
    tensor = tensor[..., src_r : src_r + crop_h, src_c : src_c + crop_w]

    pad_h = target_h - crop_h
    pad_w = target_w - crop_w
    if pad_h > 0 or pad_w > 0:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        tensor = F.pad(tensor, (left, right, top, bottom), value=0.0)

    return tensor.squeeze(1).cpu().numpy().astype(np.float32, copy=False), True
