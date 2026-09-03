"""What the dataset holds, and how much of it the model never looks at.

Coverage answers whether the files are on disk. They are: 24,371 of 24,371.
This answers a different question, which nobody has asked of the whole corpus —
**of the pixels that are present, how many does the pipeline actually read?**

Three places lose data before the encoder, and only the first is deliberate:

```text
plane           a series whose Anatomical_Plane is not one of the three is
                excluded outright, and the count has never been looked at
slices          32 centres per series, whatever its length. A 320-frame series
                contributes 32 of them
shape           a series reaches the head as a grid sized by its aspect ratio,
                and that grid is pooled onto a fixed 6x6
```

The slice figure is the one to watch. The mean series has 33.6 frames, so a
typical study loses almost nothing, and the average conceals the tail: the
question is not the mean but how much of the corpus sits in series long enough
to be truncated.

## What it costs to run

Slice counts come from counting files in a series folder, which needs no DICOM
parsing at all. The metadata repair does open headers, but only for rows whose
plane, fluid or fat flag is missing from the CSV — the same repair the training
loader performs, so the numbers here are the ones a run would see.

## What it does not do

It reads nothing into a model and writes no labels. It is an inventory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PLANES, backfill_series_metadata, load_series_csv, load_train_csv
from .dicom import DICOM_SUFFIXES, find_series_dir

INVENTORY_VERSION = "data_inventory_v1"

# What the loader takes from every series, however long it is.
SLICE_CENTRES = 32


def _slice_counts(
    root: Path, series: pd.DataFrame, split: str
) -> tuple[np.ndarray, list[str]]:
    """Frames per series, by counting files rather than parsing them."""
    counts: list[int] = []
    missing: list[str] = []
    for row in series.itertuples(index=False):
        folder = find_series_dir(
            root, split, str(row.StudyInstanceUID), str(row.SeriesInstanceUID)
        )
        if folder is None or not folder.is_dir():
            missing.append(f"{row.StudyInstanceUID}/{row.SeriesInstanceUID}")
            counts.append(0)
            continue
        counts.append(
            sum(
                1
                for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES
            )
        )
    return np.asarray(counts, dtype=np.int64), missing


def inventory(
    *,
    data_root: str | Path,
    split: str = "train",
    centres: int = SLICE_CENTRES,
    out_json: str | Path | None = None,
) -> dict:
    root = Path(data_root).resolve()
    train = load_train_csv(root / "train.csv")
    series = load_series_csv(root / "train_series.csv")
    series, repair = backfill_series_metadata(series, root, split=split)

    recognised = series["Anatomical_Plane"].isin(PLANES)
    eligible = series.loc[recognised]

    per_study = eligible.groupby("StudyInstanceUID").size()
    all_studies = train["StudyInstanceUID"].astype(str)
    zero = int(len(set(all_studies).difference(per_study.index.astype(str))))

    counts, missing = _slice_counts(root, series, split)
    series = series.assign(frames=counts)
    usable = series.loc[recognised, "frames"].to_numpy()

    read = np.minimum(usable, centres)
    quantiles = [0, 25, 50, 75, 90, 95, 99, 100]
    result = {
        "version": INVENTORY_VERSION,
        "data_root": str(root),
        "studies": int(len(train)),
        "series_rows": int(len(series)),
        "plane": {
            "recognised": int(recognised.sum()),
            "excluded_unknown_plane": int((~recognised).sum()),
            "distribution": {
                str(k): int(v)
                for k, v in eligible["Anatomical_Plane"].value_counts().items()
            },
        },
        "studies_with_zero_eligible_series": zero,
        "metadata_repair": repair,
        "slices": {
            "centres_taken_per_series": int(centres),
            "frames_total": int(usable.sum()),
            "frames_read": int(read.sum()),
            "frames_never_read": int((usable - read).sum()),
            "fraction_never_read": (
                float((usable - read).sum() / usable.sum()) if usable.sum() else 0.0
            ),
            "series_longer_than_centres": int((usable > centres).sum()),
            "series_shorter_than_centres": int((usable < centres).sum()),
            "frames_per_series": {
                f"p{q}": float(np.percentile(usable, q)) for q in quantiles
            }
            if usable.size
            else {},
        },
        "series_folders_not_found": missing[:5],
        "series_folders_not_found_count": len(missing),
    }
    if out_json is not None:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _report(result: dict) -> None:
    plane, slices = result["plane"], result["slices"]
    print()
    print(f"  studies                          {result['studies']:>10,}")
    print(f"  series rows                      {result['series_rows']:>10,}")
    print()
    print("  plane eligibility")
    print(f"    recognised                     {plane['recognised']:>10,}")
    print(
        f"    excluded, unknown plane        {plane['excluded_unknown_plane']:>10,}"
        f"   {plane['excluded_unknown_plane'] / max(result['series_rows'], 1) * 100:5.1f}%"
    )
    for name, count in plane["distribution"].items():
        print(f"      {name:<28}{count:>10,}")
    print(f"    studies with no eligible series{result['studies_with_zero_eligible_series']:>10,}")

    repair = result["metadata_repair"]
    print()
    print("  metadata repaired from the DICOM headers")
    for key in sorted(repair):
        print(f"    {key:<32}{repair[key]:>10,}")

    print()
    print(f"  slices, taking {slices['centres_taken_per_series']} centres per series")
    print(f"    frames present                 {slices['frames_total']:>10,}")
    print(f"    frames the loader reads        {slices['frames_read']:>10,}")
    print(
        f"    frames never read              {slices['frames_never_read']:>10,}"
        f"   {slices['fraction_never_read'] * 100:5.1f}%"
    )
    print(f"    series longer than {slices['centres_taken_per_series']:<3}         {slices['series_longer_than_centres']:>10,}")
    print(f"    series shorter than {slices['centres_taken_per_series']:<3}        {slices['series_shorter_than_centres']:>10,}")
    print()
    print("    frames per series")
    for key, value in slices["frames_per_series"].items():
        print(f"      {key:<28}{value:>10.0f}")

    if result["series_folders_not_found_count"]:
        print()
        print(
            f"  WARNING {result['series_folders_not_found_count']:,} series folders were "
            f"not found, e.g. {result['series_folders_not_found'][:2]}"
        )
    print()
    print(
        "  A series shorter than the centre count is read completely, with some\n"
        "  centres landing on the same frame. Only the longer ones lose anything."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        "Inventory the dataset, and how much of it the loader never reads"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--centres", type=int, default=SLICE_CENTRES)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    _report(
        inventory(
            data_root=args.data_root,
            split=args.split,
            centres=args.centres,
            out_json=args.out_json,
        )
    )


if __name__ == "__main__":
    main()
