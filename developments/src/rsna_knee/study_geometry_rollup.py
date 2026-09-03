"""The geometry a single study hands the model, rolled up from the series scan.

`slice_geometry_scan` measured the corpus: 24,371 series, triplet depths from
1.59 mm to 16.66 mm. That answered a question about the dataset. This answers
the question about the *prediction*, which is not the same thing, because a
study is one row of the submission and it is built from several series at once.

## Why every eligible series counts

The obvious worry about the corpus figure is that the model might only read a
selected handful per study, making the corpus spread irrelevant. It does not.
Since B12 the dataset policy has been `all_repaired_anatomical_series_v1`
(`b12_variable_series.build_variable_series_index`): every series with a
recognised plane is read, with no fluid or structural winner picked and no cap
on how many a study may contribute. `VariableSeriesKneeDataset` is what B37 and
everything after it trains on, so the corpus *is* the input.

The legacy `select_series` / `build_series_index(mode="dual")` path, which took
at most six per study, is what B7 through B11 used. It is not what the current
lineage reads, and this module reports the dual subset only as a comparison.

## The question this actually opens

If a study contributes five series and their triplets span 1.2 mm, 6.6 mm and
13 mm, the model fuses three very different views of depth inside one
prediction. That is a sharper problem than the corpus spread, and invisible in
a corpus histogram:

```text
within-study spread   how far apart the thinnest and thickest triplet in one
                      study are, in millimetres and as a ratio
mixed studies         how many studies contain both a thin volume and a thick
                      stack, so the fusion has to reconcile them
concentrated loss     the 52,014 unread frames are 6.4% of the corpus, but how
                      much of any single study do they cost
gold against the rest whether the 58 expert studies look like the 4,349 report
                      ones, since every mechanism decision is judged on them
```

## What it costs to run

Nothing. It reads `series_geometry.csv`, which the series scan already wrote.
No DICOM headers are opened again.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import (
    backfill_series_metadata,
    build_series_index,
    load_series_csv,
    load_train_csv,
)

ROLLUP_VERSION = "study_geometry_rollup_v1"

# A study whose thickest triplet is this many times its thinnest is asking the
# model to fuse views that are not comparable.
WIDE_SPREAD_RATIO = 2.0

# The band edges a "mixed" study straddles: a fine volume and a thick stack.
THIN_TRIPLET_MM = 2.0
THICK_TRIPLET_MM = 8.0

REQUIRED_COLUMNS = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "triplet_span_mm",
    "slice_spacing_mm",
    "frames",
    "frames_never_read",
)


def load_series_geometry(path: str | Path) -> pd.DataFrame:
    """Read the per-series table the scan wrote, and check it is the right one."""
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is not a slice_geometry_scan series table; it is missing "
            f"{', '.join(missing)}. Run rsna_knee.slice_geometry_scan --out-csv first."
        )
    frame = frame.copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    frame["SeriesInstanceUID"] = frame["SeriesInstanceUID"].astype(str)
    return frame


def study_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per study: the range of depths it hands the model, and its loss."""
    span = pd.to_numeric(frame["triplet_span_mm"], errors="coerce")
    work = frame.assign(triplet_span_mm=span)

    grouped = work.groupby("StudyInstanceUID", sort=True)
    table = grouped.agg(
        series=("SeriesInstanceUID", "count"),
        frames=("frames", "sum"),
        frames_never_read=("frames_never_read", "sum"),
        thinnest_triplet_mm=("triplet_span_mm", "min"),
        median_triplet_mm=("triplet_span_mm", "median"),
        thickest_triplet_mm=("triplet_span_mm", "max"),
        series_with_a_span=("triplet_span_mm", "count"),
    ).reset_index()

    table["triplet_spread_mm"] = (
        table["thickest_triplet_mm"] - table["thinnest_triplet_mm"]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        table["triplet_spread_ratio"] = (
            table["thickest_triplet_mm"] / table["thinnest_triplet_mm"]
        )
    table["fraction_never_read"] = np.where(
        table["frames"] > 0, table["frames_never_read"] / table["frames"], 0.0
    )
    table["mixes_thin_and_thick"] = (
        table["thinnest_triplet_mm"] < THIN_TRIPLET_MM
    ) & (table["thickest_triplet_mm"] >= THICK_TRIPLET_MM)
    return table


def dual_subset(
    series_csv_frame: pd.DataFrame, data_root: str | Path, *, split: str = "train"
) -> set[str]:
    """The series the legacy `mode="dual"` policy would have selected.

    Kept for comparison only. B12 and everything after it read all of them.

    The metadata is repaired from the headers first, because the selection
    keys on plane, fluid and fat flags and `slice_geometry_scan` measured the
    repaired population. Skipping the repair here would compare two different
    series sets and blame the difference on the policy.
    """
    root = Path(data_root)
    series = load_series_csv(root / "train_series.csv")
    series, _ = backfill_series_metadata(series, root, split=split)
    train = load_train_csv(root / "train.csv")
    index = build_series_index(series, train["StudyInstanceUID"].astype(str), mode="dual")
    chosen = {str(uid) for picks in index.values() for uid in picks.values() if uid}
    return chosen.intersection(set(series_csv_frame["SeriesInstanceUID"]))


def _quantiles(values: pd.Series) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not clean.size:
        return {"n": 0}
    keys = ("p0", "p25", "p50", "p75", "p90", "p95", "p99", "p100")
    points = np.quantile(clean, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {"n": int(clean.size), **{k: float(v) for k, v in zip(keys, points)}}


def _population(table: pd.DataFrame) -> dict:
    """The numbers that describe one group of studies."""
    if not len(table):
        return {"studies": 0}
    losing = table.loc[table["frames_never_read"] > 0]
    ratio = pd.to_numeric(table["triplet_spread_ratio"], errors="coerce")
    return {
        "studies": int(len(table)),
        "series_per_study": _quantiles(table["series"]),
        "thinnest_triplet_mm": _quantiles(table["thinnest_triplet_mm"]),
        "median_triplet_mm": _quantiles(table["median_triplet_mm"]),
        "thickest_triplet_mm": _quantiles(table["thickest_triplet_mm"]),
        "within_study_spread_mm": _quantiles(table["triplet_spread_mm"]),
        "within_study_spread_ratio": _quantiles(ratio),
        "studies_spread_wider_than_ratio": {
            "ratio": float(WIDE_SPREAD_RATIO),
            "studies": int((ratio >= WIDE_SPREAD_RATIO).sum()),
            "fraction": float((ratio >= WIDE_SPREAD_RATIO).mean()),
        },
        "studies_mixing_thin_and_thick": {
            "thin_below_mm": float(THIN_TRIPLET_MM),
            "thick_from_mm": float(THICK_TRIPLET_MM),
            "studies": int(table["mixes_thin_and_thick"].sum()),
            "fraction": float(table["mixes_thin_and_thick"].mean()),
        },
        "loss": {
            "studies_losing_any_frame": int(len(losing)),
            "fraction_of_studies": float(len(losing) / len(table)),
            "frames_never_read": int(table["frames_never_read"].sum()),
            "share_of_own_frames_lost": _quantiles(losing["fraction_never_read"])
            if len(losing)
            else {"n": 0},
        },
    }


def rollup(
    *,
    series_csv: str | Path,
    data_root: str | Path | None = None,
    out_csv: str | Path | None = None,
    out_json: str | Path | None = None,
) -> dict:
    frame = load_series_geometry(series_csv)
    table = study_table(frame)

    result = {
        "version": ROLLUP_VERSION,
        "series_csv": str(Path(series_csv).resolve()),
        "series_rows": int(len(frame)),
        "policy": (
            "all_repaired_anatomical_series_v1 — since B12 every eligible series "
            "is read, so these studies are the model's whole input"
        ),
        "all_studies": _population(table),
    }

    if data_root is not None:
        root = Path(data_root)
        train = load_train_csv(root / "train.csv")
        expert = set(
            train.loc[train[list(TARGETS)].notna().any(axis=1), "StudyInstanceUID"]
            .astype(str)
        )
        is_expert = table["StudyInstanceUID"].isin(expert)
        result["expert_gold"] = _population(table.loc[is_expert])
        result["report_only"] = _population(table.loc[~is_expert])

        chosen = dual_subset(frame, root)
        dual_table = study_table(frame.loc[frame["SeriesInstanceUID"].isin(chosen)])
        result["legacy_dual_subset"] = {
            "note": (
                "What B7-B11 read. B12 onwards reads everything, so this is a "
                "comparison and not the current input."
            ),
            "series": int(len(chosen)),
            **_population(dual_table),
        }

    if out_csv is not None:
        path = Path(out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        result["study_csv"] = str(path)
    if out_json is not None:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _print_population(title: str, block: dict) -> None:
    print()
    print(f"  {title}")
    if not block.get("studies"):
        print("    no studies")
        return
    print(f"    studies                        {block['studies']:>10,}")
    series = block["series_per_study"]
    print(
        f"    series per study               "
        f"p50 {series['p50']:.0f}   p90 {series['p90']:.0f}   max {series['p100']:.0f}"
    )

    for label, key in (
        ("thinnest triplet, mm", "thinnest_triplet_mm"),
        ("thickest triplet, mm", "thickest_triplet_mm"),
        ("spread within a study, mm", "within_study_spread_mm"),
    ):
        block_q = block[key]
        print(
            f"    {label:<31}p25 {block_q['p25']:5.2f}   p50 {block_q['p50']:5.2f}   "
            f"p90 {block_q['p90']:5.2f}   max {block_q['p100']:5.2f}"
        )

    wide = block["studies_spread_wider_than_ratio"]
    mixed = block["studies_mixing_thin_and_thick"]
    loss = block["loss"]
    for label, count, fraction in (
        (
            f"thickest >= {wide['ratio']:.0f}x thinnest",
            wide["studies"],
            wide["fraction"],
        ),
        (
            f"mixes <{mixed['thin_below_mm']:.0f} mm with >={mixed['thick_from_mm']:.0f} mm",
            mixed["studies"],
            mixed["fraction"],
        ),
        (
            "studies losing any frame",
            loss["studies_losing_any_frame"],
            loss["fraction_of_studies"],
        ),
    ):
        print(f"    {label:<31}{count:>10,}   {fraction * 100:5.1f}%")
    if loss["share_of_own_frames_lost"].get("n"):
        share = loss["share_of_own_frames_lost"]
        print(
            f"      of their own frames they lose  "
            f"p50 {share['p50'] * 100:4.1f}%   p90 {share['p90'] * 100:4.1f}%   "
            f"max {share['p100'] * 100:4.1f}%"
        )


def _report(result: dict) -> None:
    print()
    print(f"  series rows read                 {result['series_rows']:>10,}")
    print(f"  policy: {result['policy']}")
    _print_population("every study", result["all_studies"])
    if "expert_gold" in result:
        _print_population("the 58 expert studies", result["expert_gold"])
        _print_population("the report-only studies", result["report_only"])
        _print_population(
            "legacy dual subset, for comparison only", result["legacy_dual_subset"]
        )
    print()
    print(
        "  A wide spread inside one study means the model fuses views whose\n"
        "  physical depth differs, for a single prediction."
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        "Roll the series geometry up to the studies the model actually predicts"
    )
    parser.add_argument(
        "--series-csv",
        required=True,
        help="series_geometry.csv written by rsna_knee.slice_geometry_scan",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="optional: adds the expert/report split and the legacy dual comparison",
    )
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    _report(
        rollup(
            series_csv=args.series_csv,
            data_root=args.data_root,
            out_csv=args.out_csv,
            out_json=args.out_json,
        )
    )


if __name__ == "__main__":
    main()
