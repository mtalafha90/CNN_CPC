from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


DICOM_SUFFIXES = {"", ".dcm", ".dicom", ".ima"}


def _sort_key(ds) -> float:
    try:
        iop = np.asarray(ds.ImageOrientationPatient, float)
        ipp = np.asarray(ds.ImagePositionPatient, float)
        return float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
    except Exception:
        return float(getattr(ds, "InstanceNumber", 0))


def find_series_dir(root: str | Path, split: str, study: str, series: str) -> Path | None:
    root = Path(root)
    study, series = str(study), str(series)
    for p in (
        root / f"{split}_series" / study / series,
        root / f"{split}_images" / study / series,
        root / study / series,
    ):
        if p.is_dir():
            return p
    return None


def _iter_dicom_files(path: Path) -> list[Path]:
    """Return every candidate DICOM instance, including mixed suffix layouts."""
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES
    )


def read_dicom_series(path: str | Path, *, return_stats: bool = False):
    """Decode one MRI series and sort frames along the physical slice axis."""
    import pydicom

    path = Path(path)
    candidates = _iter_dicom_files(path)
    items: list[tuple[float, np.ndarray]] = []
    failed = 0

    for file_index, p in enumerate(candidates):
        try:
            ds = pydicom.dcmread(str(p), force=True)
            arr = np.asarray(ds.pixel_array, dtype=np.float32)
            arr = (
                arr * float(getattr(ds, "RescaleSlope", 1.0))
                + float(getattr(ds, "RescaleIntercept", 0.0))
            )
            if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
                arr = arr.max() - arr

            base = _sort_key(ds)
            if arr.ndim == 2:
                items.append((base, arr))
            elif arr.ndim == 3:
                # Enhanced MR is commonly one multi-frame instance. Preserve
                # frame order while anchoring it to the file's physical key so
                # multiple multi-frame files cannot all restart at zero.
                epsilon = 1e-4
                items.extend((base + i * epsilon, frame) for i, frame in enumerate(arr))
            else:
                raise RuntimeError(f"unsupported pixel array shape {arr.shape}")
        except Exception:
            failed += 1

    if not items:
        raise RuntimeError(
            f"No readable DICOM pixels in {path} "
            f"({len(candidates)} candidate files, {failed} decode failures)"
        )

    items.sort(key=lambda x: x[0])
    frames = [frame for _, frame in items]
    shapes = {frame.shape for frame in frames}
    if len(shapes) > 1:
        target = max(shapes, key=lambda shape: shape[0] * shape[1])
        frames = [_pad_or_crop(frame, target) for frame in frames]

    volume = np.stack(frames).astype(np.float32, copy=False)
    if return_stats:
        return volume, {
            "candidate_files": len(candidates),
            "decode_failures": failed,
            "decoded_frames": len(frames),
        }
    return volume


def _pad_or_crop(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    out = np.zeros(target, dtype=image.dtype)
    rows = min(image.shape[0], target[0])
    cols = min(image.shape[1], target[1])
    sr, sc = (image.shape[0] - rows) // 2, (image.shape[1] - cols) // 2
    tr, tc = (target[0] - rows) // 2, (target[1] - cols) // 2
    out[tr:tr + rows, tc:tc + cols] = image[sr:sr + rows, sc:sc + cols]
    return out


def _normalise_volume(v: np.ndarray) -> np.ndarray:
    """Robustly map one MRI series to [0, 1] using global 1st/99th percentiles."""
    v = np.asarray(v, dtype=np.float32)
    if v.ndim != 3 or len(v) == 0:
        raise RuntimeError(f"expected non-empty [S,H,W] volume, got {v.shape}")
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        raise RuntimeError("DICOM volume contains no finite pixels")
    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise RuntimeError("invalid DICOM intensity percentiles")
    v = np.nan_to_num(v, nan=float(lo), posinf=float(hi), neginf=float(lo))
    v = np.clip(v, lo, hi)
    scale = max(float(hi - lo), 1e-6)
    return ((v - lo) / scale).astype(np.float32, copy=False)


def preprocess_volume(v: np.ndarray, n_slices: int = 16, image_size: int = 224) -> torch.Tensor:
    if n_slices < 1 or image_size < 1:
        raise ValueError("n_slices and image_size must be positive")
    v = _normalise_volume(v)
    idx = np.round(np.linspace(0, len(v) - 1, n_slices)).astype(int)
    tensor = torch.from_numpy(v[idx]).unsqueeze(1)
    return F.interpolate(
        tensor, (image_size, image_size), mode="bilinear", align_corners=False
    ).squeeze(1)


def preprocess_triplets(
    v: np.ndarray,
    n_slices: int = 16,
    image_size: int = 224,
    gap: int = 1,
) -> torch.Tensor:
    """Build uniformly sampled 2.5D [z-gap,z,z+gap] tensors: [N,3,H,W]."""
    if gap < 1:
        raise ValueError("2.5D triplet gap must be >= 1")
    if n_slices < 1 or image_size < 1:
        raise ValueError("n_slices and image_size must be positive")
    v = _normalise_volume(v)
    centers = np.round(np.linspace(0, len(v) - 1, n_slices)).astype(int)
    offsets = np.asarray([-gap, 0, gap], dtype=int)
    triplet_idx = np.clip(centers[:, None] + offsets[None, :], 0, len(v) - 1)
    tensor = torch.from_numpy(v[triplet_idx].astype(np.float32, copy=False))
    return F.interpolate(
        tensor, (image_size, image_size), mode="bilinear", align_corners=False
    )
