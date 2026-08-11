"""Label-free audit of how much of each MRI series training actually sees.

Every experiment from B7 onward samples a fixed `b7_n_slices` (16) positions per
series. Nothing in the repository has ever measured how many slices those series
actually contain, so the sampling fraction is unknown.

This matters because of the per-target pattern in the B13 result. The model does
well on findings that occupy many slices and are visible at low resolution
(Effusion 0.768, Baker's 0.748, Synovitis 0.711) and poorly on focal structural
findings (ACL 0.474, Contusion 0.553, MCL 0.556). An anterior cruciate ligament
tear is visible on a handful of contiguous slices; if a 40-slice sagittal series
is represented by 16 evenly spaced positions, the slices carrying the tear can
simply be skipped.

This module measures that directly. It reads only DICOM geometry — no pixel
decoding, no labels, no gold studies — so it is cheap and cannot influence model
selection. It answers three questions:

1. how many slices does each series actually have;
2. what fraction does 16-position sampling cover;
3. how wide is the gap, in millimetres, between consecutive sampled positions
   relative to slice thickness.

The third is the decisive one. If consecutive sampled positions are several
slices apart, a lesion spanning fewer slices than that gap can fall entirely
between samples, and no amount of architecture change downstream recovers it.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .dicom import find_series_dir


def count_series_slices(series_dir: str | Path) -> dict:
    """Count slices and read spacing for one series without decoding pixels.

    Returns zero counts rather than raising, so one unreadable series cannot
    abort an audit over thousands.
    """
    import pydicom

    path = Path(series_dir)
    if not path.is_dir():
        return {"n_slices": 0, "slice_thickness": float("nan"), "spacing": float("nan")}

    files = [p for p in path.iterdir() if p.is_file()]
    n_slices = 0
    thickness = float("nan")
    spacing = float("nan")
    positions: list[float] = []

    for index, file_path in enumerate(files):
        try:
            # stop_before_pixels keeps this fast: headers only.
            ds = pydicom.dcmread(str(file_path), force=True, stop_before_pixels=True)
        except Exception:
            continue
        frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        n_slices += frames
        if index == 0 or not np.isfinite(thickness):
            try:
                thickness = float(getattr(ds, "SliceThickness", float("nan")))
            except (TypeError, ValueError):
                pass
            try:
                spacing = float(getattr(ds, "SpacingBetweenSlices", float("nan")))
            except (TypeError, ValueError):
                pass
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            try:
                positions.append(float(ipp[2]))
            except (TypeError, ValueError):
                pass

    # Derive spacing from positions when the tag is absent, which is common.
    if not np.isfinite(spacing) and len(positions) > 1:
        gaps = np.abs(np.diff(np.sort(np.asarray(positions))))
        gaps = gaps[gaps > 1e-6]
        if gaps.size:
            spacing = float(np.median(gaps))

    return {"n_slices": int(n_slices), "slice_thickness": thickness, "spacing": spacing}


def _audit_one(data_root: str, split: str, study: str, series: str, plane: str) -> dict:
    directory = find_series_dir(data_root, split, study, series)
    if directory is None:
        return {
            "StudyInstanceUID": study, "SeriesInstanceUID": series, "plane": plane,
            "n_slices": 0, "slice_thickness": float("nan"), "spacing": float("nan"),
            "found": False,
        }
    info = count_series_slices(directory)
    return {
        "StudyInstanceUID": study, "SeriesInstanceUID": series, "plane": plane,
        "found": True, **info,
    }


def audit_slice_coverage(
    series_df: pd.DataFrame,
    data_root: str | Path,
    split: str = "train",
    n_sampled: int = 16,
    workers: int | None = None,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Measure real slice counts against the sampled budget.

    Returns a per-series frame and a summary dictionary. ``limit`` audits only
    the first N series, which is useful for a quick check before committing to
    the full corpus.
    """
    import os

    rows = series_df[["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"]].astype(str)
    records = list(rows.itertuples(index=False, name=None))
    if limit is not None:
        records = records[:limit]
    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 1)

    results: list[dict] = []
    if workers <= 1:
        for study, series, plane in records:
            results.append(_audit_one(str(data_root), split, study, series, plane))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_audit_one, str(data_root), split, study, series, plane)
                for study, series, plane in records
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if done % 1000 == 0:
                    print(f"  audited {done}/{len(futures)} series")

    frame = pd.DataFrame(results)
    return frame, summarise_coverage(frame, n_sampled=n_sampled)


def summarise_coverage(frame: pd.DataFrame, n_sampled: int = 16) -> dict:
    """Turn a per-series audit into the numbers that drive the decision."""
    found = frame[frame.get("found", True) & (frame["n_slices"] > 0)]
    if found.empty:
        return {"n_series_audited": int(len(frame)), "n_series_readable": 0}

    counts = found["n_slices"].to_numpy(dtype=float)
    # With n_sampled evenly spaced positions over n_slices, consecutive samples
    # sit this many slices apart. A value above 1 means slices are skipped.
    stride = np.maximum(counts / max(n_sampled, 1), 1.0)
    covered_fraction = np.minimum(n_sampled / counts, 1.0)

    spacing = found["spacing"].to_numpy(dtype=float)
    finite_spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
    gap_mm = stride[np.isfinite(spacing) & (spacing > 0)] * finite_spacing

    summary = {
        "n_series_audited": int(len(frame)),
        "n_series_readable": int(len(found)),
        "n_sampled_positions": int(n_sampled),
        "slices_per_series": {
            "min": float(np.min(counts)),
            "p25": float(np.percentile(counts, 25)),
            "median": float(np.median(counts)),
            "p75": float(np.percentile(counts, 75)),
            "p95": float(np.percentile(counts, 95)),
            "max": float(np.max(counts)),
            "mean": float(np.mean(counts)),
        },
        "fraction_of_slices_seen": {
            "median": float(np.median(covered_fraction)),
            "mean": float(np.mean(covered_fraction)),
            "p05": float(np.percentile(covered_fraction, 5)),
        },
        "sampling_stride_slices": {
            "median": float(np.median(stride)),
            "p95": float(np.percentile(stride, 95)),
            "max": float(np.max(stride)),
        },
        "series_undersampled_fraction": float(np.mean(counts > n_sampled)),
        "series_fully_covered_fraction": float(np.mean(counts <= n_sampled)),
    }
    if gap_mm.size:
        summary["gap_between_sampled_positions_mm"] = {
            "median": float(np.median(gap_mm)),
            "p95": float(np.percentile(gap_mm, 95)),
        }
    if "plane" in found.columns:
        by_plane = found.groupby("plane")["n_slices"].agg(["count", "median", "max"])
        summary["by_plane"] = {
            str(plane): {
                "n_series": int(row["count"]),
                "median_slices": float(row["median"]),
                "max_slices": float(row["max"]),
            }
            for plane, row in by_plane.iterrows()
        }
    return summary


def format_summary(summary: dict) -> str:
    """Render the audit as the short report a decision can be made from."""
    if not summary.get("n_series_readable"):
        return "no readable series were found; check data_root and split"

    slices = summary["slices_per_series"]
    seen = summary["fraction_of_slices_seen"]
    stride = summary["sampling_stride_slices"]
    n = summary["n_sampled_positions"]

    lines = [
        f"series audited      {summary['n_series_audited']}  "
        f"(readable {summary['n_series_readable']})",
        f"sampled positions   {n}",
        "",
        "slices per series",
        f"  min/median/max    {slices['min']:.0f} / {slices['median']:.0f} / {slices['max']:.0f}",
        f"  p25/p75/p95       {slices['p25']:.0f} / {slices['p75']:.0f} / {slices['p95']:.0f}",
        "",
        f"undersampled series {summary['series_undersampled_fraction']:.1%} "
        f"(more slices than the {n} sampled)",
        f"fraction seen       median {seen['median']:.1%}, worst 5% below {seen['p05']:.1%}",
        f"sampling stride     median {stride['median']:.1f} slices "
        f"(p95 {stride['p95']:.1f}, max {stride['max']:.1f})",
    ]
    if "gap_between_sampled_positions_mm" in summary:
        gap = summary["gap_between_sampled_positions_mm"]
        lines += [
            f"gap between samples median {gap['median']:.1f} mm (p95 {gap['p95']:.1f} mm)",
        ]
    if "by_plane" in summary:
        lines += ["", "by plane"]
        for plane, stats in sorted(summary["by_plane"].items()):
            lines.append(
                f"  {plane:<10} n={stats['n_series']:<6} "
                f"median={stats['median_slices']:.0f} max={stats['max_slices']:.0f}"
            )

    lines += ["", _interpretation(summary)]
    return "\n".join(lines)


def _interpretation(summary: dict) -> str:
    """State plainly what the numbers imply, without overclaiming."""
    stride = summary["sampling_stride_slices"]["median"]
    undersampled = summary["series_undersampled_fraction"]
    if stride <= 1.05:
        return (
            "Interpretation: sampling already covers essentially every slice, so "
            "slice count is NOT limiting focal-lesion detection. Look elsewhere "
            "(in-plane resolution, supervision quality)."
        )
    if stride < 2.0:
        return (
            f"Interpretation: consecutive sampled positions are ~{stride:.1f} slices apart "
            f"and {undersampled:.0%} of series are undersampled. Mild skipping; raising the "
            "slice budget is worth testing but is unlikely to be transformative."
        )
    return (
        f"Interpretation: consecutive sampled positions are ~{stride:.1f} slices apart and "
        f"{undersampled:.0%} of series are undersampled. A lesion spanning fewer than "
        f"{stride:.0f} slices can fall entirely between sampled positions. This is "
        "consistent with the observed failure on focal targets (ACL, MCL, Contusion) "
        "alongside success on diffuse ones (Effusion, Baker's, Synovitis), and makes "
        "the slice budget a strong candidate for the next controlled experiment."
    )


def main() -> None:
    import argparse

    from .data import load_series_csv
    from .b7_weak_supervision import _read_config

    parser = argparse.ArgumentParser(
        description="Measure real MRI slice counts against the sampled slice budget"
    )
    parser.add_argument("--config", default=None, help="config supplying data_root/b7_n_slices")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--series-csv", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-slices", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="audit only the first N series")
    parser.add_argument("--out", default="runs/slice_audit")
    args = parser.parse_args()

    config = _read_config(args.config) if args.config else {}
    data_root = Path(args.data_root or config.get("data_root", "."))
    n_slices = int(args.n_slices or config.get("b7_n_slices", 16))
    series_csv = args.series_csv or (data_root / config.get("train_series_csv", "train_series.csv"))

    series = load_series_csv(series_csv)
    print(f"auditing slice coverage for {len(series)} series (sampled positions = {n_slices}) ...")
    frame, summary = audit_slice_coverage(
        series, data_root, split=args.split, n_sampled=n_slices,
        workers=args.workers, limit=args.limit,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "slice_audit.csv", index=False)
    (out / "slice_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(format_summary(summary))
    print()
    print(f"wrote {out/'slice_audit.csv'} and {out/'slice_audit.json'}")


if __name__ == "__main__":
    main()
