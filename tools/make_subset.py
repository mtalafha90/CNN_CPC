"""Build a small, self-contained copy of the dataset for Colab.

The full training release is 24,371 series across 819,078 DICOM files. Colab
will not hold that, and uploading it would take longer than training on it. A
subset works instead -- but only if it is *self-contained*: the same folder
layout, and CSVs trimmed to exactly the studies present, so the notebook needs
no special case for it.

Two choices here are deliberate.

**Expert-labelled studies are always included.** There are only 58 in the whole
release, and they are the notebook's scoring cell. A subset without them trains
fine and can be measured by nothing at all.

**Size is reported before anything is copied.** MRI series vary by more than a
factor of ten in file count, so "200 studies" is not a size. The default run
measures and stops; copying needs `--copy`.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from model._implementation import ensure_developments_source, read_config

ensure_developments_source()

# How the competition lays out its images, in the order the loaders try.
SERIES_LAYOUTS = ("{split}_series/{study}/{series}", "{split}_images/{study}/{series}", "{study}/{series}")


def _series_directory(root: Path, split: str, study: str, series: str) -> Path | None:
    for layout in SERIES_LAYOUTS:
        candidate = root / layout.format(split=split, study=study, series=series)
        if candidate.is_dir():
            return candidate
    return None


def _directory_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def choose_studies(
    studies: pd.DataFrame,
    targets: tuple[str, ...],
    *,
    count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Pick the studies to copy: every expert-labelled one, then a random rest.

    The random part is seeded, so the same command twice gives the same subset
    and a result stays reproducible. Expert studies come first because they are
    the only honest measurement available and there are only 58 of them.
    """
    expert_mask = studies[list(targets)].notna().any(axis=1)
    expert = studies.loc[expert_mask, "StudyInstanceUID"].astype(str).tolist()
    rest = studies.loc[~expert_mask, "StudyInstanceUID"].astype(str)

    remaining = max(0, count - len(expert))
    sampled = rest.sample(n=min(remaining, len(rest)), random_state=seed).tolist()
    return expert, sampled


def measure(
    root: Path,
    split: str,
    series: pd.DataFrame,
    study_uids: list[str],
) -> tuple[list[tuple[str, str, Path, int]], int]:
    """Find every series folder for these studies and add up their size."""
    wanted = series[series["StudyInstanceUID"].astype(str).isin(set(study_uids))]
    found: list[tuple[str, str, Path, int]] = []
    total = 0
    missing = 0
    for _, row in wanted.iterrows():
        study = str(row["StudyInstanceUID"])
        uid = str(row["SeriesInstanceUID"])
        directory = _series_directory(root, split, study, uid)
        if directory is None:
            missing += 1
            continue
        size = _directory_bytes(directory)
        found.append((study, uid, directory, size))
        total += size
    if missing:
        print(f"note: {missing} series listed in the CSV have no folder on disk")
    return found, total


def fit_within_budget(
    found: list[tuple[str, str, Path, int]],
    expert_uids: set[str],
    *,
    max_bytes: int,
) -> tuple[list[tuple[str, str, Path, int]], list[str]]:
    """Keep whole studies until the budget runs out; never a partial study.

    Studies are added entire or not at all. Copying half a study's series would
    silently change what the model sees -- it attends across a study's series,
    so a study missing two of its five is a different training example, not a
    smaller one.

    Expert-labelled studies are placed first and are never dropped: they are the
    only scoring surface, and a subset that cannot be measured is not worth
    uploading. Everything after them is taken smallest-first, which fits the
    most studies into the space rather than the largest ones.
    """
    by_study: dict[str, list[tuple[str, str, Path, int]]] = {}
    for entry in found:
        by_study.setdefault(entry[0], []).append(entry)
    sizes = {uid: sum(e[3] for e in entries) for uid, entries in by_study.items()}

    experts = [uid for uid in by_study if uid in expert_uids]
    others = sorted((uid for uid in by_study if uid not in expert_uids),
                    key=lambda uid: sizes[uid])

    kept: list[tuple[str, str, Path, int]] = []
    kept_uids: list[str] = []
    used = 0
    for uid in experts:                       # unconditional
        kept.extend(by_study[uid])
        kept_uids.append(uid)
        used += sizes[uid]
    if used > max_bytes:
        print(f"warning: the expert studies alone are {used / 1e9:.2f} GB, "
              f"over the {max_bytes / 1e9:.2f} GB budget -- keeping them anyway")

    for uid in others:
        if used + sizes[uid] > max_bytes:
            continue                          # try the next, smaller one
        kept.extend(by_study[uid])
        kept_uids.append(uid)
        used += sizes[uid]
    return kept, kept_uids


def copy_subset(
    out_root: Path,
    split: str,
    found: list[tuple[str, str, Path, int]],
) -> None:
    """Copy each series into the layout the notebook looks for first."""
    for index, (study, uid, source, _) in enumerate(found, start=1):
        destination = out_root / f"{split}_series" / study / uid
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        if index % 50 == 0 or index == len(found):
            print(f"  copied {index}/{len(found)} series")


def write_tables(
    out_root: Path,
    split: str,
    studies: pd.DataFrame,
    series: pd.DataFrame,
    kept_studies: list[str],
    kept_series: set[tuple[str, str]],
    labels_path: Path | None,
) -> None:
    """Trim every table to exactly what was copied.

    A CSV listing studies whose images are absent is worse than a smaller CSV:
    the loaders would read them, find nothing, and quietly return zeros with the
    presence flag off -- a study that trains on blank images rather than an error.
    """
    keep = set(kept_studies)
    study_rows = studies[studies["StudyInstanceUID"].astype(str).isin(keep)]
    study_rows.to_csv(out_root / f"{split}.csv", index=False)

    pairs = series.apply(
        lambda r: (str(r["StudyInstanceUID"]), str(r["SeriesInstanceUID"])), axis=1
    )
    series_rows = series[pairs.isin(kept_series)]
    series_rows.to_csv(out_root / f"{split}_series.csv", index=False)

    print(f"  {len(study_rows)} rows -> {split}.csv")
    print(f"  {len(series_rows)} rows -> {split}_series.csv")

    if labels_path is not None:
        labels = pd.read_csv(labels_path)
        labels["StudyInstanceUID"] = labels["StudyInstanceUID"].astype(str)
        subset = labels[labels["StudyInstanceUID"].isin(keep)]
        subset.to_csv(out_root / "training_targets.csv", index=False)
        print(f"  {len(subset)} rows -> training_targets.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a small self-contained copy of the dataset for Colab"
    )
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument(
        "--studies",
        type=int,
        default=4407,
        help="an upper bound on studies to consider; the size budget usually binds first",
    )
    parser.add_argument(
        "--max-gigabytes",
        type=float,
        default=8.0,
        help=(
            "the real limit. Studies are added whole, smallest first, until this "
            "is reached. A free Google Drive holds 15 GB in total, so leave room"
        ),
    )
    parser.add_argument(
        "--labels",
        help="the parsed report states to trim alongside; omit for a test split",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="actually copy. Without this the size is measured and nothing is written",
    )
    args = parser.parse_args()

    from rsna_knee.constants import TARGETS
    from rsna_knee.data import load_series_csv, load_train_csv

    root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    config = read_config(args.config)

    if args.split == "train":
        studies = load_train_csv(root / config.get("train_csv", "train.csv"))
    else:
        studies = pd.read_csv(root / "test.csv")
        studies["StudyInstanceUID"] = studies["StudyInstanceUID"].astype(str)
    series = load_series_csv(root / f"{args.split}_series.csv")

    if args.split == "train":
        expert, sampled = choose_studies(
            studies, tuple(TARGETS), count=args.studies, seed=args.seed
        )
        kept = expert + sampled
        print(f"{len(expert)} expert-labelled + {len(sampled)} report-only = {len(kept)} studies")
    else:
        kept = studies["StudyInstanceUID"].astype(str).tolist()[: args.studies]
        print(f"{len(kept)} test studies")

    print(f"measuring {len(kept)} candidate studies (this reads directory sizes)...")
    found, total = measure(root, args.split, series, kept)
    if not found:
        raise SystemExit("no series found on disk -- is --data-root right?")

    expert_uids = set(expert) if args.split == "train" else set()
    budget = int(args.max_gigabytes * 1e9)
    found, kept = fit_within_budget(found, expert_uids, max_bytes=budget)

    total = sum(entry[3] for entry in found)
    gigabytes = total / 1e9
    per_study = total / max(len(kept), 1) / 1e6

    print(f"\n{len(kept)} studies, {len(found)} series, {gigabytes:.2f} GB "
          f"({per_study:.0f} MB per study)")
    print(f"budget {args.max_gigabytes:.1f} GB, {(budget - total) / 1e9:.2f} GB spare")
    if not args.copy:
        print(
            "\nNothing copied. Re-run with --copy when this looks right.\n"
            "Raise --max-gigabytes for more studies; an upload runs at roughly a "
            "gigabyte per few minutes."
        )
        return

    print(f"\ncopying into {out_root} ...")
    out_root.mkdir(parents=True, exist_ok=True)
    copy_subset(out_root, args.split, found)

    kept_series = {(study, uid) for study, uid, _, _ in found}
    kept_studies = sorted({study for study, _, _, _ in found})
    write_tables(
        out_root,
        args.split,
        studies,
        series,
        kept_studies,
        kept_series,
        Path(args.labels) if args.labels else None,
    )

    print(f"\ndone: {out_root}")
    print(f"zip it with:  cd {out_root.parent} && zip -r {out_root.name}.zip {out_root.name}")


if __name__ == "__main__":
    main()
