"""Put the 58 expert-labelled studies in one folder, images and tables together.

They are the only studies in this competition with hard labels a radiologist
assigned to all twelve findings, and they are scattered through 4,407 study
folders with nothing marking them apart. Everything else in this project reads
them through `gold_mask` and a data root; nothing has ever collected them in one
place where they can be browsed, copied to another machine, or handed to someone
else to look at.

## What it writes

The competition's own layout, so the result is itself a valid data root and
every existing tool works on it unchanged:

```text
<out>/
  train.csv           the 58 rows, every original column, labels included
  train_series.csv    their series rows, every original column
  train_series/<study>/<series>/*.dcm
  manifest.json       what was copied, from where, and how
```

## Copy, link, or hardlink

```text
copy       real independent files. Portable, zips, survives the source moving.
hardlink   the same bytes under a second name. Free, instant, same filesystem
           only, and editing one edits the other -- which nothing here does.
symlink    a pointer. Free and obvious, but breaks the moment the tree is moved
           or archived without --dereference.
```

`copy` is the default because the request that motivates this is "give me these
studies", and a folder that stops working when something else moves is not that.
The size is reported first, and `--dry-run` reports it without writing.

## What it refuses

Writing into a directory that already holds files, so a second run cannot half
overwrite a first. Writing anywhere inside a study folder of the source. And
finishing with a study or series count that does not match the source tables,
because a silently short export of the project's only expert truth is worse than
no export.

The source is opened read-only throughout. Nothing here can modify the
competition data.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from .data import gold_mask, load_train_csv
from .dicom import DICOM_SUFFIXES, find_series_dir

EXPORT_VERSION = "gold_studies_export_v1"

# The expert surface has been 58 studies since the competition release. A
# different number means the wrong train.csv, not a new discovery.
EXPECTED_GOLD_STUDIES = 58

COPY, HARDLINK, SYMLINK = "copy", "hardlink", "symlink"
MODES = (COPY, HARDLINK, SYMLINK)


def gold_studies(data_root: str | Path, *, train_csv: str = "train.csv") -> pd.DataFrame:
    """The expert-labelled rows of train.csv, with every original column kept."""
    root = Path(data_root)
    train = load_train_csv(root / train_csv)
    gold = train.loc[gold_mask(train)].copy()
    if len(gold) != EXPECTED_GOLD_STUDIES:
        raise ValueError(
            f"{root / train_csv} has {len(gold)} expert-labelled studies, not "
            f"{EXPECTED_GOLD_STUDIES}. This is the wrong train.csv."
        )
    return gold


def gold_series(
    data_root: str | Path, uids: list[str], *, series_csv: str = "train_series.csv"
) -> pd.DataFrame:
    """Their rows of train_series.csv, unparsed so the copy stays byte-faithful.

    Read with `pandas` directly rather than `load_series_csv`: that one coerces
    plane and fluid flags into the project's vocabulary, which is right for
    training and wrong for a copy meant to be the source.
    """
    frame = pd.read_csv(Path(data_root) / series_csv)
    if "StudyInstanceUID" not in frame.columns:
        raise ValueError(f"{series_csv} has no StudyInstanceUID column")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    selected = frame.loc[frame["StudyInstanceUID"].isin(set(uids))].copy()
    missing = sorted(set(uids).difference(selected["StudyInstanceUID"]))
    if missing:
        raise ValueError(
            f"{len(missing)} expert studies have no series row, e.g. {missing[:3]}"
        )
    return selected


def _dicom_files(path: Path) -> list[Path]:
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES
    )


def plan(
    *,
    data_root: str | Path,
    train_csv: str = "train.csv",
    series_csv: str = "train_series.csv",
    split: str = "train",
) -> dict:
    """Locate every file that would be exported, and total its size."""
    root = Path(data_root).resolve()
    studies = gold_studies(root, train_csv=train_csv)
    uids = studies["StudyInstanceUID"].astype(str).tolist()
    series = gold_series(root, uids, series_csv=series_csv)

    items: list[tuple[str, str, Path, list[Path]]] = []
    absent: list[str] = []
    for row in series.itertuples(index=False):
        study, name = str(row.StudyInstanceUID), str(row.SeriesInstanceUID)
        located = find_series_dir(root, split, study, name)
        if located is None or not located.is_dir():
            absent.append(f"{study}/{name}")
            continue
        files = _dicom_files(located)
        if not files:
            absent.append(f"{study}/{name}")
            continue
        items.append((study, name, located, files))

    total = sum(path.stat().st_size for _, _, _, files in items for path in files)
    return {
        "root": root,
        "studies": studies,
        "series": series,
        "items": items,
        "series_missing": absent,
        "file_count": sum(len(files) for _, _, _, files in items),
        "bytes": int(total),
    }


def _place(source: Path, target: Path, mode: str) -> None:
    if mode == COPY:
        shutil.copy2(source, target)
    elif mode == HARDLINK:
        os.link(source, target)
    else:
        target.symlink_to(source.resolve())


def export(
    *,
    data_root: str | Path,
    out_root: str | Path,
    mode: str = COPY,
    train_csv: str = "train.csv",
    series_csv: str = "train_series.csv",
    split: str = "train",
    dry_run: bool = False,
) -> dict:
    """Write the 58 expert studies, their tables and their DICOMs, into one folder."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    prepared = plan(
        data_root=data_root, train_csv=train_csv, series_csv=series_csv, split=split
    )
    root, out = prepared["root"], Path(out_root).resolve()

    # Writing to <root>/gold_58 is the point, so being inside the data root is
    # fine. Being inside a folder we are about to read from is not.
    if out == root:
        raise ValueError(f"refusing to write over the data root itself: {out}")
    for _, _, located, _ in prepared["items"]:
        if out == located or located in out.parents or out in located.parents:
            raise ValueError(
                f"refusing to write into a source series folder: {out} overlaps {located}"
            )
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            f"{out} already holds files. Remove it or choose another path, rather "
            "than half-overwriting an earlier export."
        )

    summary = {
        "version": EXPORT_VERSION,
        "source_root": str(root),
        "out_root": str(out),
        "mode": mode,
        "studies": int(len(prepared["studies"])),
        "series": int(len(prepared["series"])),
        "series_exported": len(prepared["items"]),
        "series_missing": prepared["series_missing"],
        "files": prepared["file_count"],
        "bytes": prepared["bytes"],
        "gibibytes": round(prepared["bytes"] / 1024**3, 3),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return summary

    images = out / f"{split}_series"
    images.mkdir(parents=True, exist_ok=True)
    for study, name, located, files in prepared["items"]:
        destination = images / study / name
        destination.mkdir(parents=True, exist_ok=True)
        for path in files:
            _place(path, destination / path.name, mode)

    prepared["studies"].to_csv(out / train_csv, index=False)
    prepared["series"].to_csv(out / series_csv, index=False)

    written = sum(1 for _ in images.rglob("*") if _.is_file() or _.is_symlink())
    if written != prepared["file_count"]:
        raise RuntimeError(
            f"wrote {written} files, expected {prepared['file_count']}. The export "
            "is incomplete and must not be used."
        )
    summary["files_written"] = written
    (out / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _report(summary: dict) -> None:
    print()
    print(f"  source     {summary['source_root']}")
    print(f"  out        {summary['out_root']}")
    print(f"  mode       {summary['mode']}")
    print()
    print(f"  studies    {summary['studies']:>8}")
    print(f"  series     {summary['series_exported']:>8} of {summary['series']}")
    print(f"  files      {summary['files']:>8,}")
    print(f"  size       {summary['gibibytes']:>8.3f} GiB")
    if summary["series_missing"]:
        print()
        print(
            f"  WARNING {len(summary['series_missing'])} series have no readable "
            f"folder, e.g. {summary['series_missing'][:2]}"
        )
    print()
    if summary["dry_run"]:
        print("  Nothing written. Drop --dry-run to write it.")
    else:
        print(f"  {summary['files_written']:,} files written, manifest.json beside them.")
        print("  This is patient imaging. Local only -- do not commit it.")


def main() -> None:
    parser = argparse.ArgumentParser(
        "Collect the 58 expert-labelled studies, tables and DICOMs, into one folder"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=COPY,
        help=(
            "copy makes real independent files; hardlink shares the bytes on the "
            "same filesystem for nothing; symlink points at the source and breaks "
            "if it moves"
        ),
    )
    parser.add_argument("--train-csv", default="train.csv")
    parser.add_argument("--series-csv", default="train_series.csv")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--dry-run", action="store_true", help="report the size and write nothing"
    )
    args = parser.parse_args()

    _report(
        export(
            data_root=args.data_root,
            out_root=args.out_root,
            mode=args.mode,
            train_csv=args.train_csv,
            series_csv=args.series_csv,
            split=args.split,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
