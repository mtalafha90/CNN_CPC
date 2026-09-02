"""Look at a study the way the model looks at it.

Every measurement in this project is taken at arm's length -- an AUC, an error
rate, a coverage fraction. None of it shows a knee. When a report says one thing
and the expert label says another, the only way to judge which is right is to
look at the pictures, and there has been no way to do that.

This renders one study to PNG. It deliberately reuses the *same* reader and the
*same* normalisation the training pipeline uses, so what appears is what the
encoder sees, not a prettier version of it:

```text
read_dicom_series      the frame ordering and rescale the loader applies
_normalise_volume      the 1st-99th percentile window, per series
```

A separate `--raw` switch shows the untouched pixels instead, which is the thing
to reach for when an image looks wrong and the question is whether the
normalisation did it.

## Choosing what to look at

With no `--series`, every series in the study is listed with its plane and
weighting, and the one most likely to be diagnostic is rendered -- a sagittal
fluid-sensitive series where one exists, since that is where menisci and cruciate
ligaments are read. That is a convenience for browsing, not a claim about which
series matters; pass `--series` to override it.

## This writes pictures of patient data

The PNGs are competition images. They are local-only, like the report text. Do
not commit them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .dicom import _normalise_volume, find_series_dir, read_dicom_series
from .dicom_meta import read_series_metadata

VIEWER_VERSION = "show_series_v1"

# Enough slices to see the joint through, few enough to fit on a screen.
DEFAULT_SLICES = 12
DEFAULT_COLUMNS = 4


def list_series(data_root: str | Path, study: str, *, split: str = "train") -> list[dict]:
    """Every series folder of one study, with what the DICOM says it is."""
    root = Path(data_root)
    for parent in (root / f"{split}_series" / study, root / f"{split}_images" / study, root / study):
        if parent.is_dir():
            break
    else:
        raise FileNotFoundError(
            f"no folder for study {study} under {root}. Checked "
            f"{split}_series/, {split}_images/ and the root itself"
        )

    found = []
    for series_dir in sorted(p for p in parent.iterdir() if p.is_dir()):
        metadata = read_series_metadata(series_dir)
        found.append(
            {
                "series": series_dir.name,
                "path": series_dir,
                "slices": sum(1 for p in series_dir.iterdir() if p.is_file()),
                "plane": metadata.get("Anatomical_Plane"),
                "weighting": metadata.get("weighting"),
                "fluid_sensitive": metadata.get("Fluid_Sensitive"),
            }
        )
    if not found:
        raise FileNotFoundError(f"study {study} has no series folders under {parent}")
    return found


def choose_series(series: list[dict]) -> dict:
    """Pick the series most likely to be worth looking at first.

    Sagittal fluid-sensitive, else sagittal, else whichever has the most slices.
    A browsing convenience, not a statement about which series carries signal.
    """
    def sagittal(item: dict) -> bool:
        return str(item.get("plane") or "").lower().startswith("sag")

    for test in (
        lambda item: sagittal(item) and bool(item.get("fluid_sensitive")),
        sagittal,
    ):
        candidates = [item for item in series if test(item)]
        if candidates:
            return max(candidates, key=lambda item: item["slices"])
    return max(series, key=lambda item: item["slices"])


def _slice_indices(depth: int, count: int) -> list[int]:
    """Evenly spaced slices through the volume, always including the middle."""
    if depth <= count:
        return list(range(depth))
    return [int(round(value)) for value in np.linspace(0, depth - 1, count)]


def montage(volume: np.ndarray, *, columns: int = DEFAULT_COLUMNS) -> np.ndarray:
    """Tile slices into one image, padding the last row if it is short."""
    if volume.ndim != 3 or not len(volume):
        raise ValueError(f"expected a non-empty (slices, height, width) volume, got {volume.shape}")
    depth, height, width = volume.shape
    columns = max(1, min(int(columns), depth))
    rows = int(np.ceil(depth / columns))

    canvas = np.zeros((rows * height, columns * width), dtype=volume.dtype)
    for index in range(depth):
        row, column = divmod(index, columns)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = volume[index]
    return canvas


def to_png_bytes(image: np.ndarray) -> np.ndarray:
    """Scale any finite array onto 0-255 for saving."""
    finite = image[np.isfinite(image)]
    if not finite.size:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (np.nan_to_num(image, nan=low) - low) / (high - low)
    return (scaled * 255.0).round().astype(np.uint8)


def render(
    *,
    data_root: str | Path,
    study: str,
    series: str | None = None,
    split: str = "train",
    slices: int = DEFAULT_SLICES,
    columns: int = DEFAULT_COLUMNS,
    raw: bool = False,
    out: str | Path,
) -> dict:
    """Render one series of one study to a PNG, and say what was rendered."""
    available = list_series(data_root, study, split=split)
    if series is None:
        chosen = choose_series(available)
    else:
        matching = [item for item in available if item["series"] == str(series)]
        if not matching:
            names = ", ".join(item["series"] for item in available[:6])
            raise ValueError(f"study {study} has no series {series}. It has: {names}")
        chosen = matching[0]

    located = find_series_dir(data_root, split, study, chosen["series"]) or chosen["path"]
    try:
        volume = read_dicom_series(located)
    except ValueError as error:
        # np.stack on an empty list, i.e. every frame failed to decode.
        raise RuntimeError(
            f"no frame in {located} could be decoded. If this says 'no decoder', "
            "the codec plugins are installed but were imported from a stale cache: "
            "call importlib.invalidate_caches() before the first pydicom import"
        ) from error
    if volume is None or not len(volume):
        raise RuntimeError(f"no readable frames in {located}")
    volume = np.asarray(volume, dtype=np.float32)

    indices = _slice_indices(len(volume), int(slices))
    selected = volume[indices]
    # Normalise the whole series, then select -- the window is a property of the
    # series, so taking it from a subset would show a different picture than the
    # model sees.
    shown = selected if raw else _normalise_volume(volume)[indices]

    picture = to_png_bytes(montage(shown, columns=columns))
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(path, picture)

    return {
        "version": VIEWER_VERSION,
        "study": str(study),
        "series": chosen["series"],
        "series_path": str(located),
        "plane": chosen["plane"],
        "weighting": chosen["weighting"],
        "fluid_sensitive": chosen["fluid_sensitive"],
        "frames_in_series": int(len(volume)),
        "frames_shown": len(indices),
        "slice_indices": indices,
        "normalised": not raw,
        "series_available": len(available),
        "png": str(path),
    }


def _write_png(path: Path, image: np.ndarray) -> None:
    """Save a greyscale array, without requiring matplotlib."""
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "saving a PNG needs Pillow: pip install pillow"
        ) from error
    Image.fromarray(image, mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser("Render one study's series to a PNG")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--study", required=True, help="StudyInstanceUID")
    parser.add_argument("--series", default=None, help="SeriesInstanceUID; omit to choose one")
    parser.add_argument("--split", default="train")
    parser.add_argument("--slices", type=int, default=DEFAULT_SLICES)
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="show untouched pixels instead of the 1-99 percentile window the model sees",
    )
    parser.add_argument("--out", default=None, help="PNG path; defaults beside the study id")
    parser.add_argument(
        "--list", action="store_true", help="list the study's series and stop"
    )
    args = parser.parse_args()

    if args.list:
        for item in list_series(args.data_root, args.study, split=args.split):
            print(
                f"  {item['series']:<48}{item['slices']:>5} slices   "
                f"{str(item['plane'] or '?'):<10}{str(item['weighting'] or '?')}"
            )
        return

    out = args.out or f"{args.study}.png"
    result = render(
        data_root=args.data_root,
        study=args.study,
        series=args.series,
        split=args.split,
        slices=args.slices,
        columns=args.columns,
        raw=args.raw,
        out=out,
    )
    print()
    print(f"  study        {result['study']}")
    print(f"  series       {result['series']}   ({result['series_available']} in this study)")
    print(f"  plane        {result['plane'] or '?'}    weighting {result['weighting'] or '?'}")
    print(
        f"  slices       {result['frames_shown']} of {result['frames_in_series']}"
        f"   {'normalised as the model sees it' if result['normalised'] else 'raw pixels'}"
    )
    print(f"  written      {result['png']}")
    print()
    print("  This is patient imaging. Local only -- do not commit it.")


if __name__ == "__main__":
    main()
