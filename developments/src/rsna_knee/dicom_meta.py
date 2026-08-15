"""Derive series metadata from DICOM headers.

`train_series.csv` supplies `Anatomical_Plane`, `Fluid_Sensitive` and
`Fat_Suppression`, and those columns should be preferred wherever they are
populated. This module covers the gaps: series that the CSV omits, or rows
where a field is blank. Without a fallback, `select_series` finds nothing for
the affected plane and the study silently trains on a stream of zeros.

Everything here is derived from acquisition parameters rather than free text.
`SeriesDescription` is unreliable in this dataset — the studies come from
sixteen sites and the reports appear in twelve languages, so a sagittal series
may be labelled "SAG", "SAGITAL", "T2 SAG FS" or nothing at all. Geometry and
timings mean the same thing everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Capitalised to match the convention used by `load_series_csv`.
PLANES = ("Sagittal", "Coronal", "Axial")


def plane_from_orientation(orientation: Sequence[float] | None) -> str | None:
    """Classify the imaging plane from `ImageOrientationPatient`.

    The tag holds the row and column direction cosines. Their cross product is
    the slice normal, and whichever patient axis it aligns with most strongly
    names the plane: x is sagittal, y coronal, z axial.

    Returns ``None`` when the tag is missing or degenerate, so callers can tell
    "unknown" apart from a genuine answer.
    """
    if orientation is None or len(orientation) < 6:
        return None
    row = np.asarray(orientation[:3], dtype=np.float64)
    column = np.asarray(orientation[3:6], dtype=np.float64)
    normal = np.cross(row, column)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-6:
        return None
    return PLANES[int(np.argmax(np.abs(normal / norm)))]


def weighting_from_parameters(
    echo_time: float | None,
    repetition_time: float | None,
    inversion_time: float | None = None,
    scan_options: str = "",
    image_type: Sequence[str] | None = None,
) -> tuple[str, bool]:
    """Infer contrast weighting and fat suppression from acquisition timings.

    Returns a ``(weighting, fat_suppressed)`` pair, where weighting is one of
    ``t1``, ``pd``, ``t2``, ``stir`` or ``unknown``. The thresholds are the
    conventional musculoskeletal ones: short TE with short TR is T1, short TE
    with long TR is proton density, long TE with long TR is T2.
    """
    haystack = " ".join(
        [str(scan_options or "")] + [str(value) for value in (image_type or [])]
    ).upper()
    fat_suppressed = any(
        token in haystack
        for token in ("FS", "FAT_SAT", "FATSAT", "SPAIR", "SPIR", "DIXON", "STIR")
    )

    # A short inversion time is STIR, which is both fluid sensitive and fat
    # suppressed regardless of what the other timings suggest.
    if inversion_time is not None and 0 < float(inversion_time) < 200:
        return "stir", True

    if echo_time is None or repetition_time is None:
        return "unknown", fat_suppressed

    te = float(echo_time)
    tr = float(repetition_time)
    if te < 35.0:
        return ("t1" if tr < 900.0 else "pd"), fat_suppressed
    if tr >= 900.0:
        return "t2", fat_suppressed
    return "unknown", fat_suppressed


def is_fluid_sensitive(weighting: str) -> bool:
    """Whether a weighting shows fluid brightly.

    T2, proton density and STIR are fluid sensitive; T1 is structural. This
    mirrors the meaning of the `Fluid_Sensitive` column in `train_series.csv`.
    """
    return weighting in {"t2", "pd", "stir"}


def read_series_metadata(series_dir: str | Path) -> dict[str, Any]:
    """Read one DICOM from a series directory and describe the series.

    Only a single instance is opened — every slice of a series shares these
    attributes, so reading more would cost time for nothing.

    Returns a dict with ``Anatomical_Plane``, ``Fluid_Sensitive``,
    ``Fat_Suppression`` and ``weighting``. Values are ``None`` when they cannot
    be determined, so a caller can leave the CSV value in place.
    """
    import pydicom

    unknown: dict[str, Any] = {
        "Anatomical_Plane": None,
        "Fluid_Sensitive": None,
        "Fat_Suppression": None,
        "weighting": None,
    }

    path = Path(series_dir)
    if not path.is_dir():
        return unknown

    dataset = None
    for candidate in sorted(path.iterdir()):
        if not candidate.is_file():
            continue
        try:
            dataset = pydicom.dcmread(str(candidate), force=True, stop_before_pixels=True)
            break
        except Exception:
            continue
    if dataset is None:
        return unknown

    orientation = getattr(dataset, "ImageOrientationPatient", None)
    if orientation is None:
        orientation = _multiframe_orientation(dataset)

    weighting, fat_suppressed = weighting_from_parameters(
        getattr(dataset, "EchoTime", None),
        getattr(dataset, "RepetitionTime", None),
        getattr(dataset, "InversionTime", None),
        str(getattr(dataset, "ScanOptions", "") or ""),
        getattr(dataset, "ImageType", []) or [],
    )

    return {
        "Anatomical_Plane": plane_from_orientation(orientation),
        # An unknown weighting must not masquerade as a confident False.
        "Fluid_Sensitive": None if weighting == "unknown" else is_fluid_sensitive(weighting),
        "Fat_Suppression": bool(fat_suppressed),
        "weighting": None if weighting == "unknown" else weighting,
    }


def _multiframe_orientation(dataset: Any) -> Sequence[float] | None:
    """Recover orientation from an enhanced multi-frame instance."""
    try:
        shared = dataset.SharedFunctionalGroupsSequence[0]
        return shared.PlaneOrientationSequence[0].ImageOrientationPatient
    except Exception:
        return None
