from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import DUAL_STREAMS
from .dicom import find_series_dir, preprocess_triplets, read_dicom_series


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

    def __post_init__(self) -> None:
        if self.n_slices < 1 or self.image_size < 1 or self.triplet_gap < 1:
            raise ValueError("n_slices, image_size and triplet_gap must be positive")
        if self.noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        if not 0 <= self.slice_dropout < 1:
            raise ValueError("slice_dropout must be in [0,1)")


class KneeStudyDataset(Dataset):
    """Study-level dataset with a fixed six-stream, 2.5D MRI contract."""

    def __init__(
        self,
        study_uids,
        series_index,
        config: DatasetConfig,
        targets=None,
        weights=None,
        train: bool = False,
    ) -> None:
        self.study_uids = [str(x) for x in study_uids]
        self.series_index = series_index
        self.config = config
        self.targets = targets
        self.weights = weights
        self.train = bool(train)
        self.stream_names = list(DUAL_STREAMS)

        n = len(self.study_uids)
        if targets is not None and len(targets) != n:
            raise ValueError("targets length must match study_uids")
        if weights is not None and len(weights) != n:
            raise ValueError("weights length must match study_uids")

    @property
    def in_channels(self) -> int:
        return 3

    def __len__(self) -> int:
        return len(self.study_uids)

    def _zero(self) -> torch.Tensor:
        return torch.zeros(
            self.config.n_slices,
            3,
            self.config.image_size,
            self.config.image_size,
            dtype=torch.float32,
        )

    def _load(self, uid: str, series_uid: str | None) -> tuple[torch.Tensor, float]:
        if not series_uid:
            return self._zero(), 0.0

        path = find_series_dir(self.config.data_root, self.config.split, uid, str(series_uid))
        if path is None:
            if self.config.strict_dicom:
                raise FileNotFoundError(
                    f"series directory not found: split={self.config.split} "
                    f"study={uid} series={series_uid}"
                )
            return self._zero(), 0.0

        try:
            volume = preprocess_triplets(
                read_dicom_series(path),
                n_slices=self.config.n_slices,
                image_size=self.config.image_size,
                gap=self.config.triplet_gap,
            )
        except Exception:
            if self.config.strict_dicom:
                raise
            return self._zero(), 0.0

        if self.train:
            if self.config.noise_std > 0:
                volume = (
                    volume + torch.randn_like(volume) * self.config.noise_std
                ).clamp_(0, 1)
            if self.config.slice_dropout > 0:
                drop = torch.rand(volume.shape[0]) < self.config.slice_dropout
                volume[drop] = 0
        return volume, 1.0

    def __getitem__(self, idx: int) -> dict:
        uid = self.study_uids[idx]
        mapping = self.series_index.get(uid, {})
        volumes, present = [], []
        for stream_name in self.stream_names:
            volume, flag = self._load(uid, mapping.get(stream_name))
            volumes.append(volume)
            present.append(flag)

        item = {
            "study_uid": uid,
            "volumes": torch.stack(volumes),
            "present": torch.tensor(present, dtype=torch.float32),
        }
        if self.targets is not None:
            item["target"] = torch.from_numpy(np.asarray(self.targets[idx], dtype=np.float32))
        if self.weights is not None:
            item["weight"] = torch.from_numpy(np.asarray(self.weights[idx], dtype=np.float32))
        return item
