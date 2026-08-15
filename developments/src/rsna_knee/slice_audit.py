"""Label-free audit of the exact B13 MRI slice-exposure surface.

This audit measures how many actual DICOM frames B13 can expose to the encoder
under its frozen 2.5D sampling policy. It deliberately mirrors the B13 contract:
- exact 3,120 active B6 studies (gold studies excluded by construction);
- exact B12/B13 repaired all-series mapping and frozen SHA-256 signature;
- 16 center positions per real series;
- 2.5D triplets around each center;
- training gap choices [1, 2] with center jitter +/-2;
- evaluation gap 1 with TTA center offsets [-1, 0, 1].

Only DICOM headers are read. No pixels and no gold labels are used.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .b7_weak_supervision import _read_config, load_frozen_b6_export, prepare_b7_supervision
from .b12_training import _load_series_policy
from .b12_variable_series import audit_variable_series_surface
from .b13_training import B13_SERIES_SIGNATURE, _require_b13_contract
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .dicom import DICOM_SUFFIXES, _centers, find_series_dir


def _slice_coordinate(ds) -> float | None:
    """Project ImagePositionPatient onto the acquisition slice normal.

    Using IPP[2] alone is wrong for sagittal/coronal or oblique acquisitions.
    This mirrors the orientation-aware ordering used by ``dicom._sort_key``.
    """
    try:
        ipp = np.asarray(ds.ImagePositionPatient, dtype=float).reshape(-1)
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float).reshape(-1)
        if ipp.size < 3 or iop.size < 6:
            return None
        normal = np.cross(iop[:3], iop[3:6])
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 1e-8:
            return None
        normal = normal / norm
        value = float(np.dot(ipp[:3], normal))
        return value if np.isfinite(value) else None
    except Exception:
        return None


def count_series_slices(series_dir: str | Path) -> dict:
    """Count frames and derive through-plane spacing from DICOM headers."""
    import pydicom

    path = Path(series_dir)
    if not path.is_dir():
        return {
            "n_slices": 0,
            "slice_thickness": float("nan"),
            "spacing": float("nan"),
            "spacing_source": "missing",
            "header_files": 0,
            "header_failures": 0,
        }

    files = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES
    )
    n_slices = 0
    thicknesses: list[float] = []
    tagged_spacings: list[float] = []
    coordinates: list[float] = []
    failures = 0

    for file_path in files:
        try:
            ds = pydicom.dcmread(str(file_path), force=True, stop_before_pixels=True)
        except Exception:
            failures += 1
            continue

        try:
            frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        except Exception:
            frames = 1
        frames = max(frames, 1)
        n_slices += frames

        try:
            value = abs(float(getattr(ds, "SliceThickness")))
            if np.isfinite(value) and value > 0:
                thicknesses.append(value)
        except Exception:
            pass

        try:
            value = abs(float(getattr(ds, "SpacingBetweenSlices")))
            if np.isfinite(value) and value > 0:
                tagged_spacings.append(value)
        except Exception:
            pass

        # One IPP locates one single-frame instance. For enhanced multi-frame
        # objects do not pretend the top-level IPP locates every frame.
        if frames == 1:
            coordinate = _slice_coordinate(ds)
            if coordinate is not None:
                coordinates.append(coordinate)

    spacing = float("nan")
    spacing_source = "unavailable"
    if tagged_spacings:
        spacing = float(np.median(np.asarray(tagged_spacings, dtype=float)))
        spacing_source = "SpacingBetweenSlices"
    elif len(coordinates) > 1:
        values = np.sort(np.asarray(coordinates, dtype=float))
        gaps = np.abs(np.diff(values))
        gaps = gaps[np.isfinite(gaps) & (gaps > 1e-6)]
        if gaps.size:
            spacing = float(np.median(gaps))
            spacing_source = "orientation_projected_IPP"

    thickness = (
        float(np.median(np.asarray(thicknesses, dtype=float)))
        if thicknesses else float("nan")
    )
    return {
        "n_slices": int(n_slices),
        "slice_thickness": thickness,
        "spacing": spacing,
        "spacing_source": spacing_source,
        "header_files": int(len(files)),
        "header_failures": int(failures),
    }


def _triplet_indices(
    n_frames: int,
    n_slices: int,
    gap: int,
    *,
    center_offset: int = 0,
) -> np.ndarray:
    """Return the exact unique frame indices touched by one deterministic view."""
    if n_frames < 1:
        return np.empty(0, dtype=int)
    centers = _centers(
        n_frames,
        n_slices,
        gap,
        center_offset=int(center_offset),
        jitter=0,
    )
    offsets = np.asarray([-gap, 0, gap], dtype=int)
    idx = np.clip(centers[:, None] + offsets[None, :], 0, n_frames - 1)
    return np.unique(idx.reshape(-1))


def _training_possible_union(
    n_frames: int,
    n_slices: int,
    gap_choices: tuple[int, ...],
    center_jitter: int,
) -> np.ndarray:
    """Union of every frame that can be touched by a legal training view."""
    touched: set[int] = set()
    jitter_values = range(-int(center_jitter), int(center_jitter) + 1)
    for gap in gap_choices:
        base = _centers(n_frames, n_slices, int(gap), jitter=0)
        for center in base.tolist():
            for jitter in jitter_values:
                shifted = int(np.clip(center + jitter, 0, n_frames - 1))
                for offset in (-int(gap), 0, int(gap)):
                    touched.add(int(np.clip(shifted + offset, 0, n_frames - 1)))
    return np.asarray(sorted(touched), dtype=int)


def _expected_training_unique_slices(
    n_frames: int,
    n_slices: int,
    gap_choices: tuple[int, ...],
    center_jitter: int,
) -> float:
    """Exact expected unique-frame count for one random training view.

    Gap is uniform over ``gap_choices``. Conditional on a gap, each center's
    integer jitter is independent and uniform over the configured range.
    """
    if n_frames < 1:
        return 0.0
    jitter_values = np.arange(-int(center_jitter), int(center_jitter) + 1, dtype=int)
    if jitter_values.size == 0:
        jitter_values = np.asarray([0], dtype=int)

    expected_by_gap: list[float] = []
    for gap in gap_choices:
        centers = _centers(n_frames, n_slices, int(gap), jitter=0)
        not_touched = np.ones(n_frames, dtype=float)
        for center in centers.tolist():
            hits = np.zeros(n_frames, dtype=float)
            for jitter in jitter_values.tolist():
                shifted = int(np.clip(center + jitter, 0, n_frames - 1))
                local = {
                    int(np.clip(shifted - int(gap), 0, n_frames - 1)),
                    shifted,
                    int(np.clip(shifted + int(gap), 0, n_frames - 1)),
                }
                for index in local:
                    hits[index] += 1.0
            p_hit = hits / float(len(jitter_values))
            not_touched *= (1.0 - p_hit)
        expected_by_gap.append(float(np.sum(1.0 - not_touched)))
    return float(np.mean(expected_by_gap)) if expected_by_gap else 0.0


def _max_unsampled_run(n_frames: int, touched: np.ndarray) -> int:
    if n_frames < 1:
        return 0
    mask = np.ones(n_frames, dtype=bool)
    mask[np.asarray(touched, dtype=int)] = False
    best = run = 0
    for missing in mask.tolist():
        if missing:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def sampling_exposure(
    n_frames: int,
    *,
    n_slices: int = 16,
    eval_gap: int = 1,
    eval_tta_offsets: tuple[int, ...] = (-1, 0, 1),
    train_gap_choices: tuple[int, ...] = (1, 2),
    center_jitter: int = 2,
) -> dict:
    """Describe B13 frame exposure for one series under the frozen contract."""
    if n_frames < 1:
        return {
            "n_slices": int(n_frames),
            "eval_unique_slices": 0,
            "eval_fraction_seen": 0.0,
            "train_expected_unique_slices_per_view": 0.0,
            "train_expected_fraction_per_view": 0.0,
            "train_possible_unique_slices": 0,
            "train_possible_fraction": 0.0,
            "eval_max_unsampled_run_slices": 0,
        }

    eval_views = [
        _triplet_indices(n_frames, n_slices, eval_gap, center_offset=int(offset))
        for offset in eval_tta_offsets
    ]
    eval_union = (
        np.unique(np.concatenate(eval_views))
        if eval_views else np.empty(0, dtype=int)
    )
    train_possible = _training_possible_union(
        n_frames, n_slices, train_gap_choices, center_jitter
    )
    expected_train = _expected_training_unique_slices(
        n_frames, n_slices, train_gap_choices, center_jitter
    )

    center_positions = _centers(n_frames, n_slices, eval_gap, jitter=0)
    center_diffs = np.diff(np.unique(center_positions))
    median_center_stride = float(np.median(center_diffs)) if center_diffs.size else 0.0

    return {
        "n_slices": int(n_frames),
        "center_positions_per_view": int(n_slices),
        "triplet_references_per_view": int(n_slices * 3),
        "eval_gap": int(eval_gap),
        "eval_tta_views": int(len(eval_tta_offsets)),
        "eval_unique_slices": int(len(eval_union)),
        "eval_fraction_seen": float(len(eval_union) / n_frames),
        "eval_max_unsampled_run_slices": _max_unsampled_run(n_frames, eval_union),
        "eval_median_center_stride_slices": median_center_stride,
        "train_gap_choices": [int(x) for x in train_gap_choices],
        "train_center_jitter": int(center_jitter),
        "train_expected_unique_slices_per_view": float(expected_train),
        "train_expected_fraction_per_view": float(expected_train / n_frames),
        "train_possible_unique_slices": int(len(train_possible)),
        "train_possible_fraction": float(len(train_possible) / n_frames),
        "eval_unique_slices_by_tta_offset": {
            str(int(offset)): int(len(indices))
            for offset, indices in zip(eval_tta_offsets, eval_views)
        },
    }


def _audit_one(
    data_root: str,
    split: str,
    study: str,
    series: str,
    plane: str,
    sampling_kwargs: dict,
) -> dict:
    directory = find_series_dir(data_root, split, study, series)
    if directory is None:
        return {
            "StudyInstanceUID": study,
            "SeriesInstanceUID": series,
            "plane": plane,
            "found": False,
            "n_slices": 0,
        }
    info = count_series_slices(directory)
    exposure = sampling_exposure(int(info["n_slices"]), **sampling_kwargs)
    return {
        "StudyInstanceUID": study,
        "SeriesInstanceUID": series,
        "plane": plane,
        "found": True,
        **info,
        **exposure,
    }


def audit_slice_coverage(
    series_df: pd.DataFrame,
    data_root: str | Path,
    *,
    split: str = "train",
    n_slices: int = 16,
    eval_gap: int = 1,
    eval_tta_offsets: tuple[int, ...] = (-1, 0, 1),
    train_gap_choices: tuple[int, ...] = (1, 2),
    center_jitter: int = 2,
    workers: int | None = None,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Audit an already-frozen series surface using the exact B13 sampler."""
    required = {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"}
    missing = sorted(required.difference(series_df.columns))
    if missing:
        raise ValueError(f"series_df missing columns: {missing}")

    rows = series_df[["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"]].astype(str)
    records = list(rows.itertuples(index=False, name=None))
    if limit is not None:
        records = records[: int(limit)]
    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 1)

    sampling_kwargs = {
        "n_slices": int(n_slices),
        "eval_gap": int(eval_gap),
        "eval_tta_offsets": tuple(int(x) for x in eval_tta_offsets),
        "train_gap_choices": tuple(int(x) for x in train_gap_choices),
        "center_jitter": int(center_jitter),
    }

    results: list[dict] = []
    if int(workers) <= 1:
        for study, series, plane in records:
            results.append(
                _audit_one(str(data_root), split, study, series, plane, sampling_kwargs)
            )
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = [
                pool.submit(
                    _audit_one,
                    str(data_root),
                    split,
                    study,
                    series,
                    plane,
                    sampling_kwargs,
                )
                for study, series, plane in records
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if done % 1000 == 0:
                    print(f"  audited {done}/{len(futures)} series")

    frame = pd.DataFrame(results)
    summary = summarise_coverage(frame)
    summary["sampling_contract"] = sampling_kwargs
    summary["limited_run"] = bool(limit is not None)
    return frame, summary


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def summarise_coverage(frame: pd.DataFrame) -> dict:
    """Summarise actual B13 frame exposure, not center-count proxies."""
    if frame.empty:
        return {"n_series_audited": 0, "n_series_readable": 0}
    found = frame.loc[
        frame.get("found", pd.Series(True, index=frame.index)).astype(bool)
        & (pd.to_numeric(frame["n_slices"], errors="coerce").fillna(0) > 0)
    ].copy()
    if found.empty:
        return {"n_series_audited": int(len(frame)), "n_series_readable": 0}

    summary = {
        "n_series_audited": int(len(frame)),
        "n_series_readable": int(len(found)),
        "slices_per_series": _distribution(found["n_slices"].to_numpy(float)),
        "eval_fraction_seen": _distribution(found["eval_fraction_seen"].to_numpy(float)),
        "eval_unique_slices": _distribution(found["eval_unique_slices"].to_numpy(float)),
        "eval_max_unsampled_run_slices": _distribution(
            found["eval_max_unsampled_run_slices"].to_numpy(float)
        ),
        "train_expected_fraction_per_view": _distribution(
            found["train_expected_fraction_per_view"].to_numpy(float)
        ),
        "train_possible_fraction": _distribution(
            found["train_possible_fraction"].to_numpy(float)
        ),
        "series_with_complete_eval_exposure_fraction": float(
            np.mean(found["eval_fraction_seen"].to_numpy(float) >= 1.0 - 1e-12)
        ),
        "series_with_eval_unsampled_run_ge_2_fraction": float(
            np.mean(found["eval_max_unsampled_run_slices"].to_numpy(float) >= 2)
        ),
        "series_with_eval_unsampled_run_ge_3_fraction": float(
            np.mean(found["eval_max_unsampled_run_slices"].to_numpy(float) >= 3)
        ),
    }

    spacing = pd.to_numeric(found.get("spacing"), errors="coerce").to_numpy(float)
    runs = found["eval_max_unsampled_run_slices"].to_numpy(float)
    valid = np.isfinite(spacing) & (spacing > 0)
    if valid.any():
        summary["eval_max_unsampled_run_mm"] = _distribution(runs[valid] * spacing[valid])
        summary["spacing_available_fraction"] = float(valid.mean())
    else:
        summary["spacing_available_fraction"] = 0.0

    if "plane" in found.columns:
        by_plane: dict[str, dict] = {}
        for plane, part in found.groupby("plane", sort=True):
            by_plane[str(plane)] = {
                "n_series": int(len(part)),
                "median_slices": float(np.median(part["n_slices"].to_numpy(float))),
                "median_eval_fraction_seen": float(
                    np.median(part["eval_fraction_seen"].to_numpy(float))
                ),
                "median_eval_max_unsampled_run_slices": float(
                    np.median(part["eval_max_unsampled_run_slices"].to_numpy(float))
                ),
                "median_train_expected_fraction_per_view": float(
                    np.median(part["train_expected_fraction_per_view"].to_numpy(float))
                ),
            }
        summary["by_plane"] = by_plane
    return summary


def format_summary(summary: dict) -> str:
    if not summary.get("n_series_readable"):
        return "no readable series were found; check data_root/split/surface"

    slices = summary["slices_per_series"]
    eval_seen = summary["eval_fraction_seen"]
    eval_run = summary["eval_max_unsampled_run_slices"]
    train_seen = summary["train_expected_fraction_per_view"]

    lines = [
        f"series audited/readable  {summary['n_series_audited']} / {summary['n_series_readable']}",
        f"slices/series median     {slices.get('median', float('nan')):.0f} "
        f"(p95 {slices.get('p95', float('nan')):.0f}, max {slices.get('max', float('nan')):.0f})",
        "",
        "frozen B13 exposure",
        f"  eval unique fraction   median {eval_seen.get('median', float('nan')):.1%} "
        f"(p25 {eval_seen.get('p25', float('nan')):.1%})",
        f"  eval max skipped run   median {eval_run.get('median', float('nan')):.1f} slices "
        f"(p95 {eval_run.get('p95', float('nan')):.1f})",
        f"  training expected/view median {train_seen.get('median', float('nan')):.1%}",
        f"  complete eval exposure {summary['series_with_complete_eval_exposure_fraction']:.1%}",
        f"  eval run >=2 slices    {summary['series_with_eval_unsampled_run_ge_2_fraction']:.1%}",
        f"  eval run >=3 slices    {summary['series_with_eval_unsampled_run_ge_3_fraction']:.1%}",
    ]
    if "eval_max_unsampled_run_mm" in summary:
        mm = summary["eval_max_unsampled_run_mm"]
        lines.append(
            f"  skipped-run length     median {mm.get('median', float('nan')):.1f} mm "
            f"(p95 {mm.get('p95', float('nan')):.1f} mm)"
        )

    if "by_plane" in summary:
        lines += ["", "by plane"]
        for plane, stats in sorted(summary["by_plane"].items()):
            lines.append(
                f"  {plane:<10} n={stats['n_series']:<6} "
                f"eval={stats['median_eval_fraction_seen']:.1%} "
                f"max-run={stats['median_eval_max_unsampled_run_slices']:.1f} "
                f"train/view={stats['median_train_expected_fraction_per_view']:.1%}"
            )

    lines += ["", _interpretation(summary)]
    return "\n".join(lines)


def _interpretation(summary: dict) -> str:
    """Use exposure diagnostics as evidence, not as a target-specific tuning rule."""
    run_p95 = summary["eval_max_unsampled_run_slices"].get("p95", 0.0)
    eval_median = summary["eval_fraction_seen"].get("median", 0.0)
    frac_run3 = summary["series_with_eval_unsampled_run_ge_3_fraction"]

    if eval_median >= 0.95 and run_p95 <= 1:
        return (
            "Interpretation: the frozen B13 evaluation sampler exposes nearly every "
            "frame and leaves no meaningful multi-slice gaps. Slice-count "
            "undersampling is not supported as a primary bottleneck."
        )
    if frac_run3 < 0.10:
        return (
            "Interpretation: some frames are skipped, but long unsampled runs are "
            "uncommon. Treat slice budget as a secondary hypothesis; prioritize "
            "representation/supervision diagnostics before a new slice-count run."
        )
    return (
        "Interpretation: a material fraction of series contains multi-slice gaps "
        "even after the frozen B13 TTA union. This supports slice exposure as a "
        "plausible global bottleneck, but it does not justify target-wise tuning. "
        "Any slice-budget change must be predeclared and evaluated globally."
    )


def build_frozen_b13_surface(
    config: dict,
    *,
    b6_root: str | Path,
    series_policy_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Reconstruct and verify the exact non-gold 17,475-series B13 surface."""
    _require_b13_contract(config)
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, b6_audit = load_frozen_b6_export(b6_root)
    uids, _, _, supervision = prepare_b7_supervision(train, b6_frame)
    if len(uids) != 3120 or int(supervision.get("usable_cells", -1)) != 14123:
        raise ValueError("slice audit requires exact B13 B6 supervision surface")

    frozen_policy = _load_series_policy(series_policy_path)
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, index = audit_variable_series_surface(series, uids)

    signature = str(series_summary.get("series_signature_sha256", ""))
    if signature != B13_SERIES_SIGNATURE:
        raise ValueError(
            f"B13 series signature mismatch: {signature} != {B13_SERIES_SIGNATURE}"
        )
    frozen_signature = str(
        frozen_policy.get("series_summary", {}).get("series_signature_sha256", "")
    )
    if frozen_signature and frozen_signature != signature:
        raise ValueError("reconstructed series surface does not match frozen policy")

    rows = [
        {
            "StudyInstanceUID": str(uid),
            "SeriesInstanceUID": str(record["series_uid"]),
            "Anatomical_Plane": str(record["plane"]),
        }
        for uid in uids
        for record in index[str(uid)]
    ]
    surface = pd.DataFrame(rows)
    if len(surface) != 17475:
        raise ValueError(f"expected 17475 B13 series, reconstructed {len(surface)}")

    metadata = {
        "b6_version": b6_audit.get("b6_version"),
        "active_studies": int(len(uids)),
        "usable_cells": int(supervision["usable_cells"]),
        "series_count": int(len(surface)),
        "series_signature_sha256": signature,
        "metadata_repair": metadata_stats,
        "gold_studies_in_surface": 0,
        "uses_gold_labels": False,
    }
    return surface, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact B13 frame exposure on the frozen all-series surface"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="runs/slice_audit_b13")
    args = parser.parse_args()

    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root

    surface, surface_meta = build_frozen_b13_surface(
        config,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
    )
    frame, summary = audit_slice_coverage(
        surface,
        config["data_root"],
        split="train",
        n_slices=int(config.get("b7_n_slices", 16)),
        eval_gap=int(config.get("b7_triplet_gap", 1)),
        eval_tta_offsets=tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1])),
        train_gap_choices=tuple(int(x) for x in config.get("b7_train_gap_choices", [1, 2])),
        center_jitter=int(config.get("b7_center_jitter", 2)),
        workers=args.workers,
        limit=args.limit,
    )
    summary["surface"] = surface_meta

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "slice_audit.csv", index=False)
    (out / "slice_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(format_summary(summary))
    print()
    print(
        f"verified surface: {surface_meta['active_studies']} studies / "
        f"{surface_meta['series_count']} series / "
        f"{surface_meta['series_signature_sha256']}"
    )
    if args.limit is not None:
        print("NOTE: --limit was used; summary is diagnostic only, not the full surface.")
    print(f"wrote {out/'slice_audit.csv'} and {out/'slice_audit.json'}")


if __name__ == "__main__":
    main()
