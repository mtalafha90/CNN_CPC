"""B55's dataset: a fixed physical crop, one canonical side, and less resolution.

Three changes to B42's geometry, and nothing else. Everything above the dataset
-- the encoder, the sparse head, the study hierarchy, the supervision, the
split, the schedule -- is B52's, imported rather than restated.

```text
crop            90% of native            ->  130 mm, measured
laterality      whichever side was scanned -> one canonical orientation
resolution      448^2 reference area     ->  336^2
```

## Why these three together

They are one competition endpoint, not three experiments, and the run cannot
attribute its result to any one of them. That is a deliberate trade and it is
the same one B52 made: this project is 0.18 behind the field's public
notebooks, which is not a gap that a single careful ablation closes.

The crop and the laterality are a pair by nature -- both are about showing the
model the same anatomy in the same orientation, and competitors measured them
together at `+0.030`. The resolution belongs with them because it is the same
question asked a third way: 448 was chosen when the crop was fractional, and a
supervised ImageNet ConvNeXt measurably degrades above ~256 while the
competition's own controlled arm put 384 against 224 at `-0.004` for 3.5x the
cost.

## The header read

Pixel spacing, laterality and slice positions come from three DICOM headers per
series, cached per directory. That is trivial beside decoding every frame,
which this dataset does anyway, and it means the physical crop needs no
precomputed table -- so training and submission take the same path, which is
the mistake `resolve_spacing` was written to avoid in B54.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .b35_target_spatial_residual import b35_centers
from .b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectDataset,
    resize_triplets_constant_area,
)
from .dicom import _normalise_volume, find_series_dir
from .physical_crop import (
    CROP_MM,
    crop_to_millimetres,
    normalise_laterality,
    resolve_laterality,
)

B55_VERSION = "b55_physical_geometry_v1"

#: 336 rather than 448. See the module docstring.
B55_REFERENCE_SIDE = 336
B55_REFERENCE_AREA = B55_REFERENCE_SIDE * B55_REFERENCE_SIDE

#: How many headers to read for the geometry. The same three the geometry scan
#: uses, for the same reason: spacing and laterality do not vary within a series.
GEOMETRY_SAMPLES = 3


@lru_cache(maxsize=32768)
def read_series_geometry(series_dir: str) -> dict:
    """Pixel spacing, laterality and plane positions, from three headers.

    Cached on the directory, so a study read three times for three TTA offsets
    pays for this once. Returns a dict with `None` entries rather than raising:
    a series whose headers cannot be read still has pixels, and the crop falls
    back rather than the study being lost.
    """
    import pydicom

    path = Path(series_dir)
    files = sorted(p for p in path.iterdir() if p.is_file())
    if not files:
        return {"pixel_spacing_mm": float("nan"), "laterality": None, "source": None}

    step = max(1, len(files) // GEOMETRY_SAMPLES)
    spacings: list[float] = []
    positions: list[list[float]] = []
    tag_value = None

    for chosen in files[::step][:GEOMETRY_SAMPLES]:
        try:
            ds = pydicom.dcmread(str(chosen), stop_before_pixels=True, force=True)
        except Exception:
            continue
        spacing = getattr(ds, "PixelSpacing", None)
        if spacing is not None:
            try:
                values = np.asarray(spacing, dtype=float).reshape(-1)
                if values.size and np.isfinite(values[0]) and values[0] > 0:
                    spacings.append(float(values[0]))
            except Exception:
                pass
        position = getattr(ds, "ImagePositionPatient", None)
        if position is not None:
            try:
                values = np.asarray(position, dtype=float).reshape(-1)
                if values.size == 3 and np.all(np.isfinite(values)):
                    positions.append([float(v) for v in values])
            except Exception:
                pass
        if tag_value is None:
            tag_value = getattr(ds, "Laterality", None) or getattr(
                ds, "ImageLaterality", None
            )

    resolved = resolve_laterality(tag_value, positions or None)
    return {
        "pixel_spacing_mm": float(np.median(spacings)) if spacings else float("nan"),
        **resolved,
    }


def preprocess_dense_triplets_b55(
    raw: np.ndarray,
    *,
    gap: int = 1,
    center_offset: int = 0,
    pixel_spacing_mm: float = float("nan"),
    laterality: str | None = None,
    plane: str | None = None,
    crop_mm: float = CROP_MM,
    fallback_fraction: float = 0.90,
    reference_area: int = B55_REFERENCE_AREA,
) -> tuple[torch.Tensor, np.ndarray, dict]:
    """B42's triplet build, with a physical crop and a canonical orientation.

    The order matters and is deliberate:

    ```text
    1  sample centres      unchanged -- slice choice is the frozen B35 contract
    2  crop in millimetres before any resize, so the crop is measured against
                           the pixels the scanner actually produced
    3  normalise the side  after cropping, so a mirrored series is mirrored
                           about the crop's centre rather than the image's
    4  resize once         to the reference area, exactly as B42 does
    ```
    """
    if int(gap) < 1:
        raise ValueError("B55 2.5D gap must be positive")
    normalized = _normalise_volume(raw)
    centers, position = b35_centers(
        len(normalized), gap=int(gap), center_offset=int(center_offset)
    )
    offsets = np.asarray([-int(gap), 0, int(gap)], dtype=np.int64)
    index = np.clip(centers[:, None] + offsets[None, :], 0, len(normalized) - 1)
    triplets = normalized[index].astype(np.float32, copy=False)

    cropped, crop_audit = crop_to_millimetres(
        triplets,
        pixel_spacing_mm,
        crop_mm=float(crop_mm),
        fallback_fraction=float(fallback_fraction),
    )
    oriented, side_audit = normalise_laterality(cropped, laterality, plane)

    # The slice order can be reversed by the laterality step, and the position
    # basis describes where each slice sits, so it has to follow.
    if side_audit.get("applied") == "slice_order_reversed":
        position = np.ascontiguousarray(position[::-1])

    images = resize_triplets_constant_area(oriented, reference_area=int(reference_area))
    return images, position, {**crop_audit, **side_audit}


class B55PhysicalGeometryDataset(B42ConstantAreaAspectDataset):
    """B42's dataset with B55's geometry, and every per-series decision recorded.

    `geometry_audit` accumulates one row per series read, so a run can report
    how often the crop was physical rather than clipped or fallback, and how
    often the laterality tag agreed with the derivation. Without that a run
    could change nothing at all and look identical -- which is precisely how
    B52 trained on unaugmented pixels for 27 hours.
    """

    reference_area: int = B55_REFERENCE_AREA
    crop_mm: float = CROP_MM

    def __init__(self, *args, **kwargs):
        self.reference_area = int(kwargs.pop("reference_area", B55_REFERENCE_AREA))
        self.crop_mm = float(kwargs.pop("crop_mm", CROP_MM))
        super().__init__(*args, **kwargs)
        self.geometry_audit: list[dict] = []

    def _load_b42(self, uid: str, series_uid: str, plane: str):
        path = find_series_dir(
            self.config.data_root, self.config.split, uid, str(series_uid)
        )
        if path is None:
            if self.config.strict_dicom:
                raise FileNotFoundError(f"missing series {uid}/{series_uid}")
            image, position = self._zero_b42()
            return image, position, 0.0
        try:
            geometry = read_series_geometry(str(path))
            raw = self._read_volume(path, plane.lower())
            images, positions, audit = [], [], None
            for offset in self.center_offsets:
                image, position, audit = preprocess_dense_triplets_b55(
                    raw,
                    gap=int(self.config.triplet_gap),
                    center_offset=int(offset),
                    pixel_spacing_mm=geometry["pixel_spacing_mm"],
                    laterality=geometry["laterality"],
                    plane=plane,
                    crop_mm=self.crop_mm,
                    fallback_fraction=float(
                        self.crop_focus_policy["crop_fraction"]
                    ),
                    reference_area=self.reference_area,
                )
                images.append(image)
                positions.append(torch.from_numpy(position))
            if audit is not None:
                self.geometry_audit.append(
                    {**audit, "agree": geometry.get("agree"),
                     "laterality_source": geometry.get("source")}
                )
            return torch.stack(images), torch.stack(positions), 1.0
        except Exception:
            if self.config.strict_dicom:
                raise
            image, position = self._zero_b42()
            return image, position, 0.0


__all__ = [
    "B55PhysicalGeometryDataset",
    "B55_REFERENCE_AREA",
    "B55_REFERENCE_SIDE",
    "B55_VERSION",
    "preprocess_dense_triplets_b55",
    "read_series_geometry",
]
