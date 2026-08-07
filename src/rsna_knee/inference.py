from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .constants import SUBMISSION_COLUMNS, TARGETS
from .data import build_series_index, load_series_csv, load_test_csv
from .dataset import DatasetConfig, KneeStudyDataset
from .model import MultiSeriesKneeNet
from .report_labels import label_dataframe
from .training import predict


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MultiSeriesKneeNet(len(ckpt["stream_names"]), pretrained=False, dropout=float(ckpt["config"].get("dropout", 0.25)))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


def infer_checkpoints(data_root: str | Path, checkpoint_paths, config: dict, fusion_alpha: float = 0.7) -> pd.DataFrame:
    root = Path(data_root)
    test = load_test_csv(root / config.get("test_csv", "test.csv"))
    series = load_series_csv(root / config.get("test_series_csv", "test_series.csv"))
    index = build_series_index(series, test["StudyInstanceUID"].astype(str), config.get("stream_mode", "best"))
    dcfg = DatasetConfig(str(root), "test", int(config.get("n_slices", 16)), int(config.get("image_size", 224)), 0, 0)
    ds = KneeStudyDataset(test["StudyInstanceUID"].astype(str).tolist(), index, dcfg, train=False)
    loader = DataLoader(ds, batch_size=max(1, int(config.get("batch_size", 2))), shuffle=False, num_workers=int(config.get("num_workers", 2)), pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_p, uids = [], None
    for path in checkpoint_paths:
        model, _ = load_checkpoint(path, device)
        fold_uids, p, _ = predict(model, loader, device)
        if uids is None:
            uids = fold_uids
        elif uids != fold_uids:
            raise ValueError("checkpoint inference order mismatch")
        all_p.append(p)
    image_p = np.mean(np.stack(all_p), axis=0) if all_p else np.full((len(test), len(TARGETS)), 0.5, np.float32)
    report_p, _ = label_dataframe(test)
    alpha = float(np.clip(fusion_alpha, 0, 1))
    final = alpha * image_p + (1 - alpha) * report_p
    sub = pd.DataFrame(final, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", uids or test["StudyInstanceUID"].astype(str).tolist())
    return sub[SUBMISSION_COLUMNS]


def validate_submission(df: pd.DataFrame) -> None:
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"submission columns must be exactly {SUBMISSION_COLUMNS}")
    p = df[TARGETS].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("submission probabilities must be finite and in [0,1]")
