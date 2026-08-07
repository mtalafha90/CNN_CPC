from __future__ import annotations

from pathlib import Path
import numpy as np
import pydicom
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
    for p in [root / f"{split}_series" / study / series, root / f"{split}_images" / study / series, root / study / series]:
        if p.is_dir():
            return p
    return None


def _iter_dicom_files(path: Path) -> list[Path]:
    """List candidate DICOM files in a series directory.

    Some sites export instances without a `.dcm` suffix, in which case a plain
    ``glob("*.dcm")`` returns nothing and the study silently trains on zeros.
    Other suffixes are therefore accepted as a fallback; files that are not
    really DICOM fail to parse and are skipped by the caller.
    """
    suffixed = sorted(path.glob("*.dcm"))
    if suffixed:
        return suffixed
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {"", ".dicom", ".ima"}
    )


def read_dicom_series(path: str | Path) -> np.ndarray:
    items = []
    for p in _iter_dicom_files(Path(path)):
        try:
            ds = pydicom.dcmread(str(p), force=True)
            arr = ds.pixel_array.astype(np.float32)
            arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
            if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
                arr = arr.max() - arr
            if arr.ndim == 3:
                # An enhanced multi-frame instance stores the whole series in a
                # single file, with its frames already in acquisition order.
                items.extend((float(i), frame) for i, frame in enumerate(arr))
            else:
                items.append((_sort_key(ds), arr))
        except Exception:
            pass
    if not items:
        raise RuntimeError(f"No readable DICOMs in {path}")
    items.sort(key=lambda x: x[0])
    frames = [x[1] for x in items]
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        # Mixed in-plane sizes occur when a localiser is stored alongside the
        # series. np.stack would raise, so normalise to the dominant shape.
        target = max(shapes, key=lambda s: s[0] * s[1])
        frames = [_pad_or_crop(f, target) for f in frames]
    return np.stack(frames)


def _pad_or_crop(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Centre crop, then zero pad, so an image matches `target`."""
    out = np.zeros(target, dtype=image.dtype)
    rows = min(image.shape[0], target[0])
    cols = min(image.shape[1], target[1])
    sr, sc = (image.shape[0] - rows) // 2, (image.shape[1] - cols) // 2
    tr, tc = (target[0] - rows) // 2, (target[1] - cols) // 2
    out[tr:tr + rows, tc:tc + cols] = image[sr:sr + rows, sc:sc + cols]
    return out


def preprocess_volume(v: np.ndarray, n_slices: int = 16, image_size: int = 224) -> torch.Tensor:
    lo, hi = np.percentile(v[np.isfinite(v)], [1, 99])
    v = np.clip(v, lo, hi)
    v = (v - lo) / max(float(hi - lo), 1e-6)
    idx = np.round(np.linspace(0, len(v) - 1, n_slices)).astype(int)
    t = torch.from_numpy(v[idx].astype(np.float32)).unsqueeze(1)
    return F.interpolate(t, (image_size, image_size), mode="bilinear", align_corners=False).squeeze(1)
