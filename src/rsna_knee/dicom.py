from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _sort_key(ds) -> float:
    try:
        iop = np.asarray(ds.ImageOrientationPatient, float)
        ipp = np.asarray(ds.ImagePositionPatient, float)
        return float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
    except Exception:
        return float(getattr(ds, "InstanceNumber", 0))


def find_series_dir(root: str | Path, split: str, study: str, series: str) -> Path | None:
    root = Path(root)
    for p in [
        root / f"{split}_series" / study / series,
        root / f"{split}_images" / study / series,
        root / study / series,
    ]:
        if p.is_dir():
            return p
    return None


def _iter_dicom_files(path: Path) -> list[Path]:
    """List candidate DICOM files in a series directory.

    Some sites export instances without a `.dcm` suffix. Files that are not
    actually DICOM are ignored by :func:`read_dicom_series`.
    """
    suffixed = sorted(path.glob("*.dcm"))
    if suffixed:
        return suffixed
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in {"", ".dicom", ".ima"}
    )


def read_dicom_series(path: str | Path, *, return_stats: bool = False):
    """Decode a DICOM series in physical slice order.

    Parameters
    ----------
    path:
        Directory containing one MRI series.
    return_stats:
        When true, return ``(volume, stats)`` so preflight can distinguish
        discovered files from successfully decoded frames.

    Raises
    ------
    RuntimeError
        If no readable DICOM pixels are found.
    """
    import pydicom

    path = Path(path)
    candidates = _iter_dicom_files(path)
    items: list[tuple[float, np.ndarray]] = []
    failed = 0

    for p in candidates:
        try:
            ds = pydicom.dcmread(str(p), force=True)
            arr = ds.pixel_array.astype(np.float32)
            arr = (
                arr * float(getattr(ds, "RescaleSlope", 1.0))
                + float(getattr(ds, "RescaleIntercept", 0.0))
            )
            if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
                arr = arr.max() - arr
            if arr.ndim == 3:
                items.extend((float(i), frame) for i, frame in enumerate(arr))
            else:
                items.append((_sort_key(ds), arr))
        except Exception:
            failed += 1

    if not items:
        raise RuntimeError(
            f"No readable DICOM pixels in {path} "
            f"({len(candidates)} candidate files, {failed} decode failures)"
        )

    items.sort(key=lambda x: x[0])
    frames = [x[1] for x in items]
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        target = max(shapes, key=lambda s: s[0] * s[1])
        frames = [_pad_or_crop(f, target) for f in frames]

    volume = np.stack(frames)
    if return_stats:
        return volume, {
            "candidate_files": len(candidates),
            "decode_failures": failed,
            "decoded_frames": len(frames),
        }
    return volume


def _pad_or_crop(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Centre crop, then zero pad, so an image matches ``target``."""
    out = np.zeros(target, dtype=image.dtype)
    rows = min(image.shape[0], target[0])
    cols = min(image.shape[1], target[1])
    sr, sc = (image.shape[0] - rows) // 2, (image.shape[1] - cols) // 2
    tr, tc = (target[0] - rows) // 2, (target[1] - cols) // 2
    out[tr:tr + rows, tc:tc + cols] = image[sr:sr + rows, sc:sc + cols]
    return out


def _normalise_volume(v: np.ndarray) -> np.ndarray:
    """Robustly map one MRI series to [0, 1]."""
    v = np.asarray(v, dtype=np.float32)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        raise RuntimeError("DICOM volume contains no finite pixels")
    lo, hi = np.percentile(finite, [1, 99])
    v = np.nan_to_num(v, nan=float(lo), posinf=float(hi), neginf=float(lo))
    v = np.clip(v, lo, hi)
    return ((v - lo) / max(float(hi - lo), 1e-6)).astype(np.float32)


def preprocess_volume(
    v: np.ndarray,
    n_slices: int = 16,
    image_size: int = 224,
) -> torch.Tensor:
    """Uniformly sample a normalized MRI series as ``[N,H,W]``."""
    v = _normalise_volume(v)
    idx = np.round(np.linspace(0, len(v) - 1, n_slices)).astype(int)
    t = torch.from_numpy(v[idx]).unsqueeze(1)
    return F.interpolate(
        t, (image_size, image_size), mode="bilinear", align_corners=False
    ).squeeze(1)


def preprocess_triplets(
    v: np.ndarray,
    n_slices: int = 16,
    image_size: int = 224,
    gap: int = 1,
) -> torch.Tensor:
    """Build 2.5D ``[z-gap, z, z+gap]`` triplets from the original series.

    Centers are distributed uniformly over the original slice axis, so local
    channels remain genuinely neighboring slices even for long series.
    Returns ``[N,3,H,W]``.
    """
    if gap < 1:
        raise ValueError("2.5D triplet gap must be >= 1")
    v = _normalise_volume(v)
    centers = np.round(np.linspace(0, len(v) - 1, n_slices)).astype(int)
    offsets = np.asarray([-gap, 0, gap], dtype=int)
    triplet_idx = np.clip(centers[:, None] + offsets[None, :], 0, len(v) - 1)
    t = torch.from_numpy(v[triplet_idx].astype(np.float32))
    return F.interpolate(
        t, (image_size, image_size), mode="bilinear", align_corners=False
    )
