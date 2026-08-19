"""Check whether the sampled slices can see a small structure at all.

The model reads a fixed 16 slice positions per series, spread evenly from one
edge of the scan to the other. That is fine for a finding spread across much of
the knee -- an effusion, arthritis -- and it scores well on those.

It is not obviously fine for a small one. A cruciate ligament occupies only a
few consecutive slices. If a series holds 200 slices and only 16 are read, the
gap between them is over a dozen slices, and a structure four slices thick can
fall entirely between two samples. The model would then be asked about
something it was never shown.

This reports how often that can happen, per plane and per series, using only
the DICOM headers. It reads no pixels and needs no GPU.

It measures opportunity, not error: a series that can see the structure may
still be read wrongly. But a series that cannot see it can never be read right,
and no amount of extra training fixes that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.dataset import read_series
from model._implementation import ensure_developments_source, read_config

ensure_developments_source()

# Roughly how many consecutive slices a small focal structure spans. The exact
# value is not critical -- the report shows a range so the conclusion does not
# rest on one guess.
FOCAL_SPANS = (3, 5, 8)
SAMPLED_SLICES = 16


def _frame_counts(root: Path, series, limit: int | None, *, split: str = "train") -> dict[str, int]:
    """Count frames per series, using the repository's own folder resolution.

    The layout is `<root>/<split>_series/<study>/<series>/`, but that is the
    loader's business rather than this tool's, so ask the loader.
    """
    from rsna_knee.dicom import DICOM_SUFFIXES, find_series_dir

    counts: dict[str, int] = {}
    rows = series if limit is None else series.head(limit)
    total = len(rows)
    for seen, row in enumerate(rows.itertuples(), start=1):
        folder = find_series_dir(root, split, row.StudyInstanceUID, row.SeriesInstanceUID)
        if folder is not None:
            counts[row.SeriesInstanceUID] = sum(
                1 for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES
            )
        if seen % 2000 == 0:
            print(f"   counted {seen}/{total} series", flush=True)
    return counts


def _hit_chance(frames: int, span: int, sampled: int = SAMPLED_SLICES) -> float:
    """Chance that at least one sampled position lands inside the structure."""
    if frames <= sampled:
        return 1.0
    gap = frames / sampled
    return float(min(1.0, span / gap))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check slice coverage of small structures")
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="only inspect this many series (leave off for all)")
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--out", default="runs/slice_coverage.json")
    args = parser.parse_args()

    config = read_config(args.config)
    config["data_root"] = str(Path(args.data_root).resolve())
    root = Path(config["data_root"])

    series, _ = read_series(root, config, split=args.split)
    print(f"series in table: {len(series)}")

    counts = _frame_counts(root, series, args.limit, split=args.split)
    if not counts:
        raise SystemExit(
            f"found no series folders under {root} -- expected "
            f"{root}/{args.split}_series/<study>/<series>/"
        )
    print(f"series measured : {len(counts)}")

    frames = np.array(sorted(counts.values()))
    print("\nslices per series")
    for label, value in [
        ("smallest", frames.min()), ("25%", np.percentile(frames, 25)),
        ("median", np.median(frames)), ("75%", np.percentile(frames, 75)),
        ("95%", np.percentile(frames, 95)), ("largest", frames.max()),
    ]:
        print(f"   {label:9s} {value:8.0f}")

    print(f"\nthe model reads {SAMPLED_SLICES} of them, spread evenly")
    print("gap between the slices it reads:")
    gaps = frames / SAMPLED_SLICES
    print(f"   median {np.median(gaps):.1f} slices    largest {gaps.max():.1f} slices")

    print("\nchance of seeing a small structure at all")
    summary = {}
    for span in FOCAL_SPANS:
        chances = np.array([_hit_chance(int(f), span) for f in frames])
        missed = float((chances < 1.0).mean())
        summary[f"span_{span}"] = {
            "mean_chance": float(chances.mean()),
            "fraction_of_series_at_risk": missed,
            "fraction_below_half": float((chances < 0.5).mean()),
        }
        print(
            f"   a {span}-slice structure: seen {chances.mean() * 100:5.1f}% of the time; "
            f"{missed * 100:5.1f}% of series cannot guarantee seeing it"
        )

    plane_summary = {}
    if "Anatomical_Plane" in series.columns:
        print("\nby plane (a cruciate ligament is read on sagittal, "
              "a collateral ligament on coronal)")
        for plane, group in series.groupby("Anatomical_Plane"):
            found = [counts[s] for s in group.SeriesInstanceUID if s in counts]
            if not found:
                continue
            arr = np.array(found)
            chance = np.array([_hit_chance(int(f), 5) for f in arr]).mean()
            plane_summary[str(plane)] = {
                "series": len(arr),
                "median_slices": float(np.median(arr)),
                "mean_chance_span_5": float(chance),
            }
            print(f"   {str(plane):10s} {len(arr):5d} series   "
                  f"median {np.median(arr):5.0f} slices   sees a 5-slice structure "
                  f"{chance * 100:5.1f}% of the time")

    result = {
        "series_measured": len(counts),
        "sampled_slices": SAMPLED_SLICES,
        "slices_per_series": {
            "min": int(frames.min()), "median": float(np.median(frames)),
            "p95": float(np.percentile(frames, 95)), "max": int(frames.max()),
        },
        "focal_structure_coverage": summary,
        "by_plane": plane_summary,
        "reading": (
            "a low chance means the fixed 16 positions can step over a small "
            "structure entirely; that is a sampling limit, not a training one, "
            "and more epochs cannot fix it"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
