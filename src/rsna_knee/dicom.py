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


def read_dicom_series(path: str | Path) -> np.ndarray:
    items = []
    for p in Path(path).glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(p), force=True)
            arr = ds.pixel_array.astype(np.float32)
            arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
            if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
                arr = arr.max() - arr
            items.append((_sort_key(ds), arr))
        except Exception:
            pass
    if not items:
        raise RuntimeError(f"No readable DICOMs in {path}")
    items.sort(key=lambda x: x[0])
    return np.stack([x[1] for x in items])


def preprocess_volume(v: np.ndarray, n_slices: int = 16, image_size: int = 224) -> torch.Tensor:
    lo, hi = np.percentile(v[np.isfinite(v)], [1, 99])
    v = np.clip(v, lo, hi)
    v = (v - lo) / max(float(hi - lo), 1e-6)
    idx = np.round(np.linspace(0, len(v) - 1, n_slices)).astype(int)
    t = torch.from_numpy(v[idx].astype(np.float32)).unsqueeze(1)
    return F.interpolate(t, (image_size, image_size), mode="bilinear", align_corners=False).squeeze(1)
