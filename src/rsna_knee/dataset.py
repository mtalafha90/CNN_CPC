from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

from .dicom import find_series_dir, preprocess_volume, read_dicom_series


@dataclass
class DatasetConfig:
    data_root: str
    split: str = "train"
    n_slices: int = 16
    image_size: int = 224
    noise_std: float = 0.02
    slice_dropout: float = 0.08


class KneeStudyDataset(Dataset):
    def __init__(self, study_uids, series_index, config, targets=None, weights=None, train=False):
        self.study_uids = [str(x) for x in study_uids]
        self.series_index = series_index
        self.config = config
        self.targets = targets
        self.weights = weights
        self.train = train
        self.stream_names = sorted(next(iter(series_index.values())).keys()) if series_index else ["sagittal", "coronal", "axial"]

    def __len__(self):
        return len(self.study_uids)

    def _load(self, uid, series_uid):
        zero = torch.zeros(self.config.n_slices, self.config.image_size, self.config.image_size)
        if not series_uid:
            return zero, 0.0
        p = find_series_dir(self.config.data_root, self.config.split, uid, str(series_uid))
        if p is None:
            return zero, 0.0
        try:
            v = preprocess_volume(read_dicom_series(p), self.config.n_slices, self.config.image_size)
        except Exception:
            return zero, 0.0
        if self.train:
            if self.config.noise_std:
                v = (v + torch.randn_like(v) * self.config.noise_std).clamp(0, 1)
            if self.config.slice_dropout:
                v[torch.rand(v.shape[0]) < self.config.slice_dropout] = 0
        return v, 1.0

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        volumes, present = [], []
        mapping = self.series_index.get(uid, {})
        for name in self.stream_names:
            v, flag = self._load(uid, mapping.get(name))
            volumes.append(v)
            present.append(flag)
        item = {"study_uid": uid, "volumes": torch.stack(volumes), "present": torch.tensor(present, dtype=torch.float32)}
        if self.targets is not None:
            item["target"] = torch.from_numpy(np.asarray(self.targets[idx], np.float32))
        if self.weights is not None:
            item["weight"] = torch.from_numpy(np.asarray(self.weights[idx], np.float32))
        return item
