"""Reading knee MRI DICOM series into normalised volumes.

Two design choices matter here.

**Plane detection is geometric, not textual.** The challenge data comes from
sixteen sites and the reports are in twelve languages, so ``SeriesDescription``
is unreliable: a sagittal series may be labelled "SAG", "SAGITAL", "СAГ" or
nothing at all. The slice normal derived from ``ImageOrientationPatient`` gives
the same answer everywhere, so that is what we use.

**Contrast weighting comes from acquisition parameters.** ``EchoTime`` and
``RepetitionTime`` separate T1, PD and T2 without reading any free text, and
``ScanOptions`` / ``ImageType`` flag fat saturation. This keeps series routing
robust across sites.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .utils import get_logger

LOGGER = get_logger()

PLANES = ("sagittal", "coronal", "axial")
WEIGHTINGS = ("t1", "pd", "t2", "stir", "unknown")


def _get(dataset: Any, name: str, default: Any = None) -> Any:
    """Read a DICOM attribute without raising when it is absent."""
    value = getattr(dataset, name, default)
    if value is None:
        return default
    return value


def plane_from_orientation(orientation: Sequence[float] | None) -> str:
    """Classify a series as sagittal, coronal or axial from its direction cosines.

    ``ImageOrientationPatient`` holds the row and column direction cosines. The
    slice normal is their cross product; whichever patient axis it aligns with
    most strongly names the plane (x -> sagittal, y -> coronal, z -> axial).
    """
    if orientation is None or len(orientation) < 6:
        return "unknown"
    row = np.asarray(orientation[:3], dtype=np.float64)
    column = np.asarray(orientation[3:6], dtype=np.float64)
    normal = np.cross(row, column)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return "unknown"
    normal /= norm
    return PLANES[int(np.argmax(np.abs(normal)))]


def weighting_from_parameters(
    echo_time: float | None,
    repetition_time: float | None,
    inversion_time: float | None = None,
    scan_options: str = "",
    image_type: Sequence[str] | None = None,
) -> tuple[str, bool]:
    """Infer contrast weighting and fat saturation from acquisition parameters.

    Returns a ``(weighting, fat_saturated)`` pair. The thresholds are the
    conventional musculoskeletal MRI ones: short TE with short TR is T1,
    short TE with long TR is proton density, and long TE with long TR is T2.
    """
    haystack = " ".join(
        [str(scan_options or "")] + [str(v) for v in (image_type or [])]
    ).upper()
    fat_saturated = any(
        token in haystack for token in ("FS", "FAT_SAT", "FATSAT", "SPAIR", "SPIR", "DIXON")
    )

    if inversion_time is not None and 0 < float(inversion_time) < 200:
        return "stir", True

    if echo_time is None or repetition_time is None:
        return "unknown", fat_saturated

    te = float(echo_time)
    tr = float(repetition_time)
    if te < 35.0:
        if tr < 900.0:
            return "t1", fat_saturated
        return "pd", fat_saturated
    if tr >= 900.0:
        return "t2", fat_saturated
    return "unknown", fat_saturated


@dataclass
class SeriesInfo:
    """Metadata describing one MRI series."""

    exam_id: str
    series_id: str
    plane: str = "unknown"
    weighting: str = "unknown"
    fat_saturated: bool = False
    num_slices: int = 0
    rows: int = 0
    columns: int = 0
    pixel_spacing: tuple[float, float] = (1.0, 1.0)
    slice_spacing: float = 1.0
    paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = self.__dict__.copy()
        payload["pixel_spacing"] = list(self.pixel_spacing)
        payload.pop("paths", None)
        return payload

    @property
    def key(self) -> str:
        """A short, language-independent description such as ``sag_pd_fs``."""
        suffix = "_fs" if self.fat_saturated else ""
        return f"{self.plane[:3]}_{self.weighting}{suffix}"


def _slice_sort_key(dataset: Any, normal: np.ndarray) -> float:
    """Project a slice position onto the slice normal so ordering is anatomical."""
    position = _get(dataset, "ImagePositionPatient")
    if position is not None and len(position) >= 3:
        return float(np.dot(np.asarray(position[:3], dtype=np.float64), normal))
    instance = _get(dataset, "InstanceNumber", 0)
    return float(instance or 0)


def _normal_vector(orientation: Sequence[float] | None) -> np.ndarray:
    if orientation is None or len(orientation) < 6:
        return np.array([0.0, 0.0, 1.0])
    row = np.asarray(orientation[:3], dtype=np.float64)
    column = np.asarray(orientation[3:6], dtype=np.float64)
    normal = np.cross(row, column)
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])


def normalise_volume(volume: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Scale a volume to uint8 using robust percentiles.

    MRI intensities carry no absolute meaning and vary wildly between vendors,
    so we stretch each volume between its own percentiles rather than using a
    fixed window. Percentiles (not min/max) keep a single bright artefact from
    flattening the whole volume.
    """
    volume = volume.astype(np.float32)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros(volume.shape, dtype=np.uint8)
    lower, upper = np.percentile(finite, [low, high])
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        lower, upper = float(finite.min()), float(finite.max())
    if upper <= lower:
        return np.zeros(volume.shape, dtype=np.uint8)
    volume = np.clip((volume - lower) / (upper - lower), 0.0, 1.0)
    return (volume * 255.0).round().astype(np.uint8)


def resize_volume(volume: np.ndarray, size: int) -> np.ndarray:
    """Resize each slice to ``size`` x ``size``.

    OpenCV is used when available because it is by far the fastest option; the
    NumPy fallback keeps the pipeline importable in minimal environments.
    """
    if volume.shape[-2:] == (size, size):
        return volume
    try:
        import cv2

        return np.stack(
            [cv2.resize(slice_, (size, size), interpolation=cv2.INTER_AREA) for slice_ in volume]
        )
    except ImportError:
        rows = np.linspace(0, volume.shape[1] - 1, size).round().astype(int)
        columns = np.linspace(0, volume.shape[2] - 1, size).round().astype(int)
        return volume[:, rows][:, :, columns]


def read_series(paths: Sequence[str | Path]) -> tuple[np.ndarray, SeriesInfo]:
    """Read a list of DICOM files belonging to one series into a volume.

    Returns the volume as ``[slices, rows, columns]`` uint8 together with the
    series metadata. Multi-frame instances (one file holding the whole series)
    are handled as well as the classic one-file-per-slice layout.
    """
    import pydicom

    datasets = []
    for path in paths:
        try:
            datasets.append(pydicom.dcmread(str(path), force=True))
        except Exception as error:  # pragma: no cover - corrupt files are rare
            LOGGER.warning("Skipping unreadable DICOM %s: %s", path, error)
    if not datasets:
        raise ValueError("No readable DICOM files in series")

    reference = datasets[0]
    orientation = _get(reference, "ImageOrientationPatient")
    if orientation is None:
        # Multi-frame files hide geometry inside the shared functional groups.
        orientation = _multiframe_orientation(reference)
    normal = _normal_vector(orientation)

    if len(datasets) == 1 and int(_get(reference, "NumberOfFrames", 1) or 1) > 1:
        volume = _apply_rescale(reference.pixel_array.astype(np.float32), reference)
    else:
        datasets.sort(key=lambda ds: _slice_sort_key(ds, normal))
        slices = []
        for dataset in datasets:
            try:
                slices.append(_apply_rescale(dataset.pixel_array.astype(np.float32), dataset))
            except Exception as error:  # pragma: no cover
                LOGGER.warning("Skipping slice with unreadable pixels: %s", error)
        if not slices:
            raise ValueError("No decodable pixel data in series")
        shapes = {s.shape for s in slices}
        if len(shapes) > 1:
            target = max(shapes, key=lambda s: s[0] * s[1])
            slices = [_pad_or_crop(s, target) for s in slices]
        volume = np.stack(slices)

    pixel_spacing = _get(reference, "PixelSpacing", [1.0, 1.0])
    inversion_time = _get(reference, "InversionTime")
    weighting, fat_saturated = weighting_from_parameters(
        _get(reference, "EchoTime"),
        _get(reference, "RepetitionTime"),
        inversion_time,
        str(_get(reference, "ScanOptions", "")),
        _get(reference, "ImageType", []),
    )

    info = SeriesInfo(
        exam_id=str(_get(reference, "StudyInstanceUID", "")),
        series_id=str(_get(reference, "SeriesInstanceUID", "")),
        plane=plane_from_orientation(orientation),
        weighting=weighting,
        fat_saturated=bool(fat_saturated),
        num_slices=int(volume.shape[0]),
        rows=int(volume.shape[1]),
        columns=int(volume.shape[2]),
        pixel_spacing=(float(pixel_spacing[0]), float(pixel_spacing[1])),
        slice_spacing=float(_get(reference, "SpacingBetweenSlices", _get(reference, "SliceThickness", 1.0)) or 1.0),
        paths=[str(p) for p in paths],
    )
    return normalise_volume(volume), info


def _multiframe_orientation(dataset: Any) -> Sequence[float] | None:
    """Dig the orientation out of an enhanced multi-frame DICOM."""
    try:
        shared = dataset.SharedFunctionalGroupsSequence[0]
        return shared.PlaneOrientationSequence[0].ImageOrientationPatient
    except Exception:
        return None


def _apply_rescale(pixels: np.ndarray, dataset: Any) -> np.ndarray:
    slope = float(_get(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(_get(dataset, "RescaleIntercept", 0.0) or 0.0)
    return pixels * slope + intercept


def _pad_or_crop(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Make a slice match ``target`` by centre cropping then zero padding."""
    output = np.zeros(target, dtype=image.dtype)
    rows = min(image.shape[0], target[0])
    columns = min(image.shape[1], target[1])
    source_row = (image.shape[0] - rows) // 2
    source_column = (image.shape[1] - columns) // 2
    target_row = (target[0] - rows) // 2
    target_column = (target[1] - columns) // 2
    output[target_row : target_row + rows, target_column : target_column + columns] = image[
        source_row : source_row + rows, source_column : source_column + columns
    ]
    return output
