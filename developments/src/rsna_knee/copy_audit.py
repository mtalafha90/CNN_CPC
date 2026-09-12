"""Find what a machine-to-machine copy of the DICOM tree actually left behind.

`dicom_coverage` answers "can a run start?" -- it asks `find_series_dir` whether
each series directory exists, and refuses a run whose studies have nothing
readable. That is the right check before training, and this module does not
repeat it.

It does not answer the two questions a half-finished copy raises:

```text
which ones            coverage prints five examples. Re-copying needs the
                      whole list, in a form rsync will read.

did they copy whole   a directory that exists is not a directory that is
                      complete. An interrupted transfer leaves a series with
                      40 of its 160 slices, and every existence check on this
                      machine passes.
```

The second is the dangerous one. A truncated series trains: `_load_b42` reads
what is there, the study has `present=1.0`, nothing raises, and the model
quietly learns from a third of a knee. Nothing downstream can tell that from a
scanner that genuinely acquired fewer slices.

So the only honest way to find truncation is to compare against the machine the
files came from. Hence two modes:

```text
scan      walk a data root, record every series with its file count and bytes.
          Run it on the SOURCE machine and copy the small manifest across.

compare   hold the destination against the source manifest, and list what is
          missing, what is short, and what is byte-for-byte the same size.
```

`check` is the one-machine fallback for when the source is already gone: it can
still find series the table names and the disk does not have, but it cannot see
a truncated one, and it says so rather than implying a clean bill of health.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dicom import DICOM_SUFFIXES, find_series_dir

COPY_AUDIT_VERSION = "copy_audit_v1"

#: A series directory with fewer slices than this is reported even when no
#: source manifest exists. It is a smell, not a rule: a localiser series really
#: can be three images, which is why `check` reports these separately from the
#: missing ones rather than failing on them.
SUSPICIOUSLY_FEW_SLICES = 3


def series_files(directory: Path) -> list[Path]:
    """The DICOM files in one series directory, by the loader's own rule."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in DICOM_SUFFIXES
    )


def scan_manifest(data_root: str | Path, *, split: str = "train", series_csv=None) -> dict:
    """Every series the table names, with what is on this disk for it.

    Driven by the series table rather than by walking the tree, so a directory
    the table does not mention cannot pad the count and make a short copy look
    complete.
    """
    root = Path(data_root).expanduser().resolve()
    table = Path(series_csv) if series_csv else root / f"{split}_series.csv"
    entries: dict[str, dict] = {}

    for study, series in _table_pairs(table):
        directory = find_series_dir(root, split, study, series)
        key = f"{study}/{series}"
        if directory is None:
            entries[key] = {"files": 0, "bytes": 0, "present": False, "path": None}
            continue
        files = series_files(directory)
        entries[key] = {
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
            "present": True,
            # The path as found, not as assumed. `find_series_dir` accepts
            # three layouts; the first version of this module recorded none of
            # them and `rsync_list` guessed `train_images`, which produced a
            # list of paths that did not exist on a `train_series` machine --
            # and cp reported "failed to get attributes" for every line.
            "path": str(directory.relative_to(root)),
        }

    return {
        "version": COPY_AUDIT_VERSION,
        "data_root": str(root),
        "split": split,
        "series": entries,
        "totals": _totals(entries),
    }


def _table_pairs(series_csv: Path):
    import pandas as pd

    frame = pd.read_csv(series_csv)
    for column in ("StudyInstanceUID", "SeriesInstanceUID"):
        if column not in frame.columns:
            raise ValueError(f"{series_csv} has no {column} column")
    return list(
        zip(
            frame["StudyInstanceUID"].astype(str),
            frame["SeriesInstanceUID"].astype(str),
        )
    )


def _totals(entries: dict) -> dict:
    present = [entry for entry in entries.values() if entry["present"]]
    return {
        "series_in_table": len(entries),
        "series_present": len(present),
        "series_absent": len(entries) - len(present),
        "files": sum(entry["files"] for entry in present),
        "bytes": sum(entry["bytes"] for entry in present),
    }


def compare_manifests(source: dict, destination: dict) -> dict:
    """What the destination is missing or holding only part of.

    Byte totals rather than file counts decide truncation, because a transfer
    interrupted inside one file leaves the right number of files and the wrong
    number of bytes -- which a count-only check reports as complete.
    """
    absent: list[str] = []
    short: list[dict] = []
    complete = 0

    for key, expected in sorted(source["series"].items()):
        if not expected["present"]:
            # Absent at the source too. Not this copy's doing, and re-copying
            # cannot fix it; it belongs in the source machine's own report.
            continue
        actual = destination["series"].get(key)
        if actual is None or not actual["present"] or actual["files"] == 0:
            absent.append(key)
        elif actual["files"] < expected["files"] or actual["bytes"] < expected["bytes"]:
            short.append(
                {
                    "series": key,
                    "files": [expected["files"], actual["files"]],
                    "bytes": [expected["bytes"], actual["bytes"]],
                }
            )
        else:
            complete += 1

    expected_total = sum(1 for entry in source["series"].values() if entry["present"])
    return {
        "version": COPY_AUDIT_VERSION,
        "source_root": source["data_root"],
        "destination_root": destination["data_root"],
        "expected_series": expected_total,
        "complete": complete,
        "absent": absent,
        "short": short,
        "paths": _paths_for(source, absent + [row["series"] for row in short]),
        "bytes_missing": _bytes_missing(source, absent, short),
    }


def dominant_layout(manifest: dict) -> str:
    """The directory the present series actually live under, or "" for none.

    An absent series has no path to read, so its layout has to be inferred --
    and the only honest source is the series that did arrive. If twenty-four
    thousand of them sit under `train_series`, the missing ones belong there
    too. Guessing from a template is what broke this.
    """
    counts: dict[str, int] = {}
    for entry in manifest.get("series", {}).values():
        path = entry.get("path")
        if not entry.get("present") or not path:
            continue
        parts = Path(path).parts
        prefix = parts[0] if len(parts) > 2 else ""
        counts[prefix] = counts.get(prefix, 0) + 1
    if not counts:
        return ""
    return max(sorted(counts), key=lambda name: counts[name])


def _paths_for(source: dict, keys) -> dict:
    """Each series' real relative path, falling back to the common layout."""
    prefix = dominant_layout(source)
    resolved = {}
    for key in keys:
        recorded = source["series"].get(key, {}).get("path")
        resolved[key] = recorded or (f"{prefix}/{key}" if prefix else key)
    return resolved


def _bytes_missing(source: dict, absent: list[str], short: list[dict]) -> int:
    total = sum(source["series"][key]["bytes"] for key in absent)
    total += sum(row["bytes"][0] - row["bytes"][1] for row in short)
    return int(total)


def check_against_table(data_root: str | Path, *, split: str = "train", series_csv=None) -> dict:
    """One machine, no source manifest: what the table names and the disk lacks."""
    manifest = scan_manifest(data_root, split=split, series_csv=series_csv)
    absent = sorted(key for key, entry in manifest["series"].items() if not entry["present"])
    thin = sorted(
        key
        for key, entry in manifest["series"].items()
        if entry["present"] and entry["files"] < SUSPICIOUSLY_FEW_SLICES
    )
    return {
        "version": COPY_AUDIT_VERSION,
        "data_root": manifest["data_root"],
        "expected_series": manifest["totals"]["series_in_table"],
        "absent": absent,
        "thin": thin,
        "paths": _paths_for(manifest, absent),
        "totals": manifest["totals"],
    }


def rsync_list(result: dict, *, split: str = "train", layout: str | None = None) -> list[str]:
    """Paths to re-copy, relative to the data root, for `rsync --files-from`.

    Read from the paths the scan actually found, not from a template. Pass
    `layout` only to override them -- it exists for a manifest written before
    paths were recorded, and for nothing else.

    Every entry names a directory and ends with a slash, so a copy takes the
    series whole rather than the one file that happened to be listed.
    """
    keys = sorted(
        set(result.get("absent", [])) | {row["series"] for row in result.get("short", [])}
    )
    if layout is not None:
        prefix = layout.format(split=split)
        return [f"{prefix}/{key}/" if prefix else f"{key}/" for key in keys]

    paths = result.get("paths") or {}
    return [f"{paths.get(key, key)}/" for key in keys]


def format_comparison(result: dict) -> str:
    lines = [
        "Copy audit:",
        f"  source            {result['source_root']}",
        f"  destination       {result['destination_root']}",
        f"  series expected   {result['expected_series']}",
        f"  complete          {result['complete']}",
        f"  absent            {len(result['absent'])}",
        f"  short             {len(result['short'])}",
        f"  bytes missing     {result['bytes_missing'] / 2**30:.2f} GiB",
    ]
    for row in result["short"][:5]:
        lines.append(
            f"    {row['series']}  files {row['files'][1]}/{row['files'][0]}  "
            f"bytes {row['bytes'][1]}/{row['bytes'][0]}"
        )
    if not result["absent"] and not result["short"]:
        lines.append("  -> every series the source holds arrived whole")
    return "\n".join(lines)


def format_check(result: dict) -> str:
    lines = [
        "Copy check (no source manifest):",
        f"  data root         {result['data_root']}",
        f"  series in table   {result['expected_series']}",
        f"  present           {result['totals']['series_present']}",
        f"  absent            {len(result['absent'])}",
        f"  suspiciously thin {len(result['thin'])}",
        "",
        "  This mode cannot detect a truncated series. A directory that exists",
        "  counts as present however few slices it holds, so a clean result",
        "  here is not proof the copy finished. Run `scan` on the source",
        "  machine and `compare` for that.",
    ]
    return "\n".join(lines)


def _write(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        "rsna-knee-copy-audit",
        description="Find series a machine-to-machine copy missed or truncated.",
    )
    parser.add_argument("stage", choices=("scan", "compare", "check"))
    parser.add_argument("--data-root", help="required for scan and check")
    parser.add_argument("--series-csv", default=None)
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--out", default=None, help="where to write the manifest or report")
    parser.add_argument("--source-manifest", help="compare: the scan from the source machine")
    parser.add_argument("--destination-manifest", help="compare: default is to scan --data-root now")
    parser.add_argument(
        "--rsync-list",
        default=None,
        help="write the paths to re-copy, one per line, for rsync --files-from",
    )
    parser.add_argument(
        "--layout",
        default=None,
        help=(
            "override the directory holding the studies, e.g. '{split}_series'. "
            "Normally unnecessary: the list uses the paths the scan found."
        ),
    )
    args = parser.parse_args()

    if args.stage == "scan":
        if not args.data_root:
            parser.error("scan requires --data-root")
        manifest = scan_manifest(args.data_root, split=args.split, series_csv=args.series_csv)
        out = args.out or f"copy_manifest_{args.split}.json"
        _write(out, manifest)
        totals = manifest["totals"]
        print(
            f"[copy-audit] {totals['series_present']}/{totals['series_in_table']} series, "
            f"{totals['files']:,} files, {totals['bytes'] / 2**30:.1f} GiB -> {out}",
            flush=True,
        )
        return

    if args.stage == "check":
        if not args.data_root:
            parser.error("check requires --data-root")
        result = check_against_table(args.data_root, split=args.split, series_csv=args.series_csv)
        print(format_check(result))
        if args.out:
            _write(args.out, result)
        if args.rsync_list:
            _write_lines(args.rsync_list, rsync_list(result, split=args.split, layout=args.layout))
        raise SystemExit(1 if result["absent"] else 0)

    if not args.source_manifest:
        parser.error("compare requires --source-manifest")
    source = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
    if args.destination_manifest:
        destination = json.loads(Path(args.destination_manifest).read_text(encoding="utf-8"))
    else:
        if not args.data_root:
            parser.error("compare requires --data-root or --destination-manifest")
        destination = scan_manifest(args.data_root, split=args.split, series_csv=args.series_csv)

    result = compare_manifests(source, destination)
    print(format_comparison(result))
    if args.out:
        _write(args.out, result)
    if args.rsync_list:
        paths = rsync_list(result, split=args.split, layout=args.layout)
        _write_lines(args.rsync_list, paths)
        print(f"[copy-audit] {len(paths)} series to re-copy -> {args.rsync_list}", flush=True)
    raise SystemExit(1 if result["absent"] or result["short"] else 0)


def _write_lines(path, lines) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


if __name__ == "__main__":
    main()


__all__ = [
    "COPY_AUDIT_VERSION",
    "SUSPICIOUSLY_FEW_SLICES",
    "check_against_table",
    "compare_manifests",
    "dominant_layout",
    "format_check",
    "format_comparison",
    "rsync_list",
    "scan_manifest",
    "series_files",
]
