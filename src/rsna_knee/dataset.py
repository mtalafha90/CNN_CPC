from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from .constants import DUAL_STREAMS
from .dicom import find_series_dir, preprocess_triplets, read_dicom_series
from .physical_scale import resample_volume_inplane, validate_physical_scale_policy


@dataclass
class DatasetConfig:
    data_root: str
    split: str = "train"
    n_slices: int = 16
    image_size: int = 224
    noise_std: float = 0.02
    slice_dropout: float = 0.08
    triplet_gap: int = 1
    strict_dicom: bool = False
    train_gap_choices: tuple[int, ...] = (1, 2)
    center_jitter: int = 2
    center_offset: int = 0
    tta_center_offsets: tuple[int, ...] = ()
    rotation_deg: float = 5.0
    translate_frac: float = 0.03
    scale_jitter: float = 0.05
    gamma_jitter: float = 0.12
    bias_field_strength: float = 0.08
    series_cache_mb: int = 256
    physical_scale_policy: dict | None = None

    def __post_init__(self) -> None:
        if self.n_slices < 1 or self.image_size < 1 or self.triplet_gap < 1:
            raise ValueError("n_slices/image_size/triplet_gap must be positive")
        if any(g < 1 for g in self.train_gap_choices):
            raise ValueError("train_gap_choices must be >=1")
        if self.center_jitter < 0 or self.noise_std < 0 or self.series_cache_mb < 0:
            raise ValueError("jitter/noise/cache size must be non-negative")
        if not 0 <= self.slice_dropout < 1:
            raise ValueError("slice_dropout must be in [0,1)")
        if self.physical_scale_policy is not None:
            validate_physical_scale_policy(self.physical_scale_policy)


class _VolumeLRU:
    """Bounded per-process cache of decoded/preprocessed DICOM volumes."""

    def __init__(self, max_mb: int) -> None:
        self.max_bytes = int(max_mb) * 1024 * 1024
        self.items: OrderedDict[str, np.ndarray] = OrderedDict()
        self.bytes = 0

    def get(self, key: str) -> np.ndarray | None:
        value = self.items.pop(key, None)
        if value is not None:
            self.items[key] = value
        return value

    def put(self, key: str, value: np.ndarray) -> None:
        if self.max_bytes <= 0 or value.nbytes > self.max_bytes:
            return
        old = self.items.pop(key, None)
        if old is not None:
            self.bytes -= old.nbytes
        self.items[key] = value
        self.bytes += value.nbytes
        while self.items and self.bytes > self.max_bytes:
            _, evicted = self.items.popitem(last=False)
            self.bytes -= evicted.nbytes


class KneeStudyDataset(Dataset):
    def __init__(
        self,
        study_uids,
        series_index,
        config: DatasetConfig,
        targets=None,
        weights=None,
        train: bool = False,
    ):
        self.study_uids = [str(x) for x in study_uids]
        self.series_index = series_index
        self.config = config
        self.targets = targets
        self.weights = weights
        self.train = bool(train)
        self.stream_names = list(DUAL_STREAMS)
        self._cache = _VolumeLRU(config.series_cache_mb)
        n = len(self.study_uids)
        if targets is not None and len(targets) != n:
            raise ValueError("targets length mismatch")
        if weights is not None and len(weights) != n:
            raise ValueError("weights length mismatch")
        if self.train and self.config.tta_center_offsets:
            raise ValueError("TTA offsets are inference-only")

    @property
    def in_channels(self):
        return 3

    @property
    def n_views(self) -> int:
        return len(self.config.tta_center_offsets) if self.config.tta_center_offsets else 1

    def __len__(self):
        return len(self.study_uids)

    def _zero(self):
        base = torch.zeros(
            self.config.n_slices,
            3,
            self.config.image_size,
            self.config.image_size,
            dtype=torch.float32,
        )
        if self.config.tta_center_offsets:
            return base.unsqueeze(0).repeat(self.n_views, 1, 1, 1, 1)
        return base

    def _augment_mri(self, volume: torch.Tensor) -> torch.Tensor:
        """Mild acquisition-like augmentation shared across one series."""
        angle = float(
            torch.empty(1).uniform_(-self.config.rotation_deg, self.config.rotation_deg)
        )
        max_shift = int(round(self.config.translate_frac * self.config.image_size))
        translate = [
            int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
            if max_shift
            else 0,
            int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
            if max_shift
            else 0,
        ]
        scale = float(
            torch.empty(1).uniform_(
                1 - self.config.scale_jitter, 1 + self.config.scale_jitter
            )
        )
        volume = TVF.affine(
            volume,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
        )
        if self.config.gamma_jitter > 0:
            gamma = float(
                torch.empty(1).uniform_(
                    1 - self.config.gamma_jitter, 1 + self.config.gamma_jitter
                )
            )
            volume = volume.clamp(0, 1).pow(gamma)
        if self.config.bias_field_strength > 0:
            h, w = volume.shape[-2:]
            yy = torch.linspace(-1, 1, h, device=volume.device).view(1, 1, h, 1)
            xx = torch.linspace(-1, 1, w, device=volume.device).view(1, 1, 1, w)
            ax = float(
                torch.empty(1).uniform_(
                    -self.config.bias_field_strength, self.config.bias_field_strength
                )
            )
            ay = float(
                torch.empty(1).uniform_(
                    -self.config.bias_field_strength, self.config.bias_field_strength
                )
            )
            field = (1 + ax * xx + ay * yy).clamp(0.8, 1.2)
            volume = (volume * field).clamp(0, 1)
        if self.config.noise_std > 0:
            volume = (volume + torch.randn_like(volume) * self.config.noise_std).clamp(
                0, 1
            )
        if self.config.slice_dropout > 0:
            drop = torch.rand(volume.shape[0]) < self.config.slice_dropout
            volume[drop] = 0
        return volume

    def _read_volume(self, path, stream_name: str) -> np.ndarray:
        policy = self.config.physical_scale_policy
        policy_tag = policy.get("policy_sha256", "") if policy is not None else "legacy"
        key = f"{path}|{stream_name}|{policy_tag}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if policy is None:
            raw = read_dicom_series(path)
        else:
            raw, stats = read_dicom_series(path, return_stats=True)
            plane = {
                "sagittal": "Sagittal",
                "coronal": "Coronal",
                "axial": "Axial",
            }[stream_name.split("_", 1)[0]]
            raw, _ = resample_volume_inplane(
                raw,
                source_spacing_mm=stats.get("pixel_spacing_mm"),
                plane=plane,
                policy=policy,
            )
        self._cache.put(key, raw)
        return raw

    def _training_view(self, raw: np.ndarray) -> torch.Tensor:
        choices = self.config.train_gap_choices
        gap = int(choices[int(torch.randint(len(choices), (1,)).item())])
        jitter_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = np.random.default_rng(jitter_seed)
        volume = preprocess_triplets(
            raw,
            n_slices=self.config.n_slices,
            image_size=self.config.image_size,
            gap=gap,
            center_offset=0,
            jitter=self.config.center_jitter,
            rng=rng,
        )
        return self._augment_mri(volume)

    def _evaluation_view(self, raw: np.ndarray) -> torch.Tensor:
        if self.config.tta_center_offsets:
            return torch.stack(
                [
                    preprocess_triplets(
                        raw,
                        n_slices=self.config.n_slices,
                        image_size=self.config.image_size,
                        gap=self.config.triplet_gap,
                        center_offset=int(offset),
                        jitter=0,
                    )
                    for offset in self.config.tta_center_offsets
                ],
                dim=0,
            )
        return preprocess_triplets(
            raw,
            n_slices=self.config.n_slices,
            image_size=self.config.image_size,
            gap=self.config.triplet_gap,
            center_offset=self.config.center_offset,
            jitter=0,
        )

    def _load(self, uid, series_uid, stream_name: str):
        if not series_uid:
            return self._zero(), 0.0
        path = find_series_dir(
            self.config.data_root,
            self.config.split,
            uid,
            str(series_uid),
        )
        if path is None:
            if self.config.strict_dicom:
                raise FileNotFoundError(f"missing series {uid}/{series_uid}")
            return self._zero(), 0.0
        try:
            raw = self._read_volume(path, stream_name)
            volume = self._training_view(raw) if self.train else self._evaluation_view(raw)
        except Exception:
            if self.config.strict_dicom:
                raise
            return self._zero(), 0.0
        return volume, 1.0

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        mapping = self.series_index.get(uid, {})
        volumes, present = [], []
        for name in self.stream_names:
            volume, flag = self._load(uid, mapping.get(name), name)
            volumes.append(volume)
            present.append(flag)
        stacked = torch.stack(volumes)
        if self.config.tta_center_offsets:
            stacked = stacked.permute(1, 0, 2, 3, 4, 5).contiguous()
        item = {
            "study_uid": uid,
            "volumes": stacked,
            "present": torch.tensor(present, dtype=torch.float32),
        }
        if self.targets is not None:
            item["target"] = torch.from_numpy(
                np.asarray(self.targets[idx], dtype=np.float32)
            )
        if self.weights is not None:
            item["weight"] = torch.from_numpy(
                np.asarray(self.weights[idx], dtype=np.float32)
            )
        return item
