"""Check that the DICOM folders a run needs are actually on this machine.

A study whose series directories are all missing is not an error anywhere near
where it happens. `find_series_dir` returns `None`, `_load_b42` quietly hands
back a zero placeholder with `present=0.0`, the dataset yields a study in which
nothing is present, and the model finally raises:

```text
RuntimeError: B42 study has no readable MRI series
```

That message names no study, no path and no cause, and it arrives inside the
first epoch -- after the config, the base checkpoint, the label export and the
whole series index have loaded, which on a slow filesystem is twenty minutes.

The cause is almost always one of two things, and they need opposite fixes:

```text
missing images    train.csv and train_series.csv are the full tables, but only
                  some of the DICOM folders were copied to this machine
wrong layout      every folder is here, under a directory name find_series_dir
                  does not look in
```

This module tells them apart before a GPU is spent, and is importable so a
trainer can refuse to start.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .dicom import find_series_dir

# What find_series_dir accepts, in the order it tries them. Repeated here only
# so the diagnostic can say which layout it looked for rather than only that it
# failed; the lookup itself stays in one place.
ACCEPTED_LAYOUTS = ("{split}_series/<study>/<series>", "{split}_images/<study>/<series>", "<study>/<series>")


def accepted_layouts(split: str = "train") -> list[str]:
    """The layouts as a reader would type them, with the real split name.

    Printing a literal `{split}` asks the reader to do the substitution, and the
    whole point of this message is that they are already confused.
    """
    return [layout.format(split=split) for layout in ACCEPTED_LAYOUTS]


def study_coverage(
    data_root: str | Path, series_index: dict, *, split: str = "train"
) -> dict:
    """How much of the requested surface is actually present on disk.

    `series_index` is what `audit_variable_series_surface` returns: one list of
    series records per study UID.
    """
    root = Path(data_root)
    present_studies: list[str] = []
    partial_studies: list[str] = []
    missing_studies: list[str] = []
    series_found = 0
    series_total = 0

    for uid, records in series_index.items():
        found = 0
        for record in records:
            series_total += 1
            if find_series_dir(root, split, str(uid), str(record["series_uid"])) is not None:
                found += 1
        series_found += found

        if not records or found == 0:
            missing_studies.append(str(uid))
        elif found < len(records):
            partial_studies.append(str(uid))
        else:
            present_studies.append(str(uid))

    studies = len(series_index)
    return {
        "data_root": str(root),
        "split": split,
        "studies": studies,
        "studies_complete": len(present_studies),
        "studies_partial": len(partial_studies),
        "studies_missing": len(missing_studies),
        "studies_missing_fraction": (len(missing_studies) / studies) if studies else 0.0,
        "series_total": series_total,
        "series_found": series_found,
        "missing_examples": sorted(missing_studies)[:5],
        "partial_examples": sorted(partial_studies)[:5],
    }


def describe_layout(data_root: str | Path) -> dict:
    """What is actually under the data root, so a wrong layout is visible.

    A run with every file present but one directory renamed looks exactly like a
    run with no files at all, and the two need opposite fixes.
    """
    root = Path(data_root)
    if not root.is_dir():
        return {"data_root": str(root), "exists": False, "entries": []}

    entries = sorted(child.name for child in root.iterdir() if child.is_dir())
    recognised = [name for name in entries if name.endswith(("_series", "_images"))]
    return {
        "data_root": str(root),
        "exists": True,
        "entries": entries[:20],
        "recognised_image_directories": recognised,
        "accepted_layouts": accepted_layouts(),
    }


def format_coverage(report: dict, layout: dict, *, label: str = "surface") -> str:
    """One readable block, saying which of the two causes this is."""
    lines = [
        f"DICOM coverage for the {label}:",
        f"  data root            {report['data_root']}",
        f"  studies requested    {report['studies']}",
        f"  fully present        {report['studies_complete']}",
        f"  partly present       {report['studies_partial']}",
        f"  no series at all     {report['studies_missing']}",
        f"  series found         {report['series_found']} of {report['series_total']}",
    ]
    if report["missing_examples"]:
        lines.append(f"  missing, for example {', '.join(report['missing_examples'])}")

    if report["studies_missing"] == 0:
        lines.append("  -> every study has at least one readable series")
        return "\n".join(lines)

    lines.append("")
    if report["series_found"] == 0:
        lines += [
            "  Nothing at all was found, which usually means the layout rather",
            "  than the files. This is what is under the data root:",
            f"    {', '.join(layout['entries']) or '(nothing)'}",
            "  and these are the layouts the loader accepts:",
            *(f"    {name}" for name in accepted_layouts(report["split"])),
        ]
    else:
        lines += [
            "  Some studies were found and some were not, which means the tables",
            "  describe the full dataset while only part of the images were",
            "  copied to this machine. train.csv is complete -- that is why the",
            "  gate's fingerprint matched -- but the DICOM folders are not.",
        ]
    return "\n".join(lines)


def require_dicom_coverage(
    data_root: str | Path, series_index: dict, *, split: str = "train", label: str = "surface"
) -> dict:
    """Refuse to start a run that would die inside its first epoch."""
    report = study_coverage(data_root, series_index, split=split)
    layout = describe_layout(data_root)
    if report["studies_missing"] == 0:
        return report

    raise RuntimeError(
        format_coverage(report, layout, label=label)
        + "\n\n"
        + f"  {report['studies_missing']} of {report['studies']} studies in the "
        f"{label} have no readable series, so the run would raise "
        '"B42 study has no readable MRI series" partway through its first epoch.'
    )


def _series_index_from_csv(data_root: Path, series_csv: Path) -> dict:
    """Every study in the series table, without needing a gate or a checkpoint.

    Deliberately not `audit_variable_series_surface`: this has to work when
    nothing else does, so it reads the two columns it needs and nothing more.
    """
    import pandas as pd

    frame = pd.read_csv(series_csv)
    for column in ("StudyInstanceUID", "SeriesInstanceUID"):
        if column not in frame.columns:
            raise ValueError(f"{series_csv} has no {column} column")

    index: dict[str, list[dict]] = {}
    for study, series in zip(
        frame["StudyInstanceUID"].astype(str), frame["SeriesInstanceUID"].astype(str)
    ):
        index.setdefault(study, []).append({"series_uid": series})
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        "rsna-knee-dicom-coverage",
        description="Say whether this machine has the DICOM folders a run needs.",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--series-csv", default=None, help="default: <data-root>/train_series.csv")
    parser.add_argument("--split", default="train", choices=("train", "test"))
    args = parser.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    series_csv = Path(args.series_csv) if args.series_csv else root / f"{args.split}_series.csv"
    if not series_csv.is_file():
        raise SystemExit(f"no series table at {series_csv}")

    layout = describe_layout(root)
    index = _series_index_from_csv(root, series_csv)
    report = study_coverage(root, index, split=args.split)
    print(format_coverage(report, layout, label=f"whole {args.split} table"))

    if report["studies_missing"]:
        # A partial study is usable -- the loader masks the series it cannot
        # read. A study with none is what stops a run.
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "study_coverage",
    "describe_layout",
    "format_coverage",
    "require_dicom_coverage",
]
