from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .constants import SUBMISSION_COLUMNS, TARGETS
from .data import backfill_series_metadata, build_series_index, load_series_csv, load_test_csv
from .dataset import DatasetConfig, KneeStudyDataset
from .model import MultiSeriesKneeNet
from .runtime import resolve_runtime
from .training import predict


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = MultiSeriesKneeNet(
        len(ckpt["stream_names"]),
        pretrained=False,
        dropout=float(cfg.get("dropout", 0.25)),
        in_channels=int(ckpt.get("in_channels", 3 if cfg.get("input_mode") == "2p5d" else 1)),
        backbone=str(cfg.get("backbone", "resnet18")),
        target_attention=bool(cfg.get("target_attention", False)),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


def infer_checkpoints(
    data_root: str | Path,
    checkpoint_paths,
    config: dict,
    fusion_alpha: float = 1.0,
) -> pd.DataFrame:
    """Run image-only fold-ensemble inference.

    ``fusion_alpha`` is retained for backward API compatibility, but values
    below 1 are rejected by default because the hidden test workflow must not
    depend on report text. Reports remain a training teacher, not an inference
    requirement.
    """
    if float(fusion_alpha) != 1.0 and not bool(config.get("allow_test_report_fusion", False)):
        raise ValueError(
            "test report fusion is disabled; use fusion_alpha=1.0 or explicitly set "
            "allow_test_report_fusion=true after verifying reports exist in the official test data"
        )

    root = Path(data_root)
    test = load_test_csv(root / config.get("test_csv", "test.csv"))
    series = load_series_csv(root / config.get("test_series_csv", "test_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="test")
    print(f"[test metadata] {metadata_stats}")

    stream_mode = str(config.get("stream_mode", "best"))
    index = build_series_index(series, test["StudyInstanceUID"].astype(str), stream_mode)
    dcfg = DatasetConfig(
        data_root=str(root),
        split="test",
        n_slices=int(config.get("n_slices", 16)),
        image_size=int(config.get("image_size", 224)),
        noise_std=0.0,
        slice_dropout=0.0,
        input_mode=str(config.get("input_mode", "2d")),
        triplet_gap=int(config.get("triplet_gap", 1)),
        strict_dicom=bool(config.get("strict_dicom_inference", True)),
    )
    ds = KneeStudyDataset(test["StudyInstanceUID"].astype(str).tolist(), index, dcfg, train=False)
    runtime = resolve_runtime(config)
    loader = DataLoader(
        ds,
        batch_size=max(1, int(config.get("batch_size", 2))),
        shuffle=False,
        **runtime.loader_kwargs(),
    )

    device = runtime.device
    all_p, uids = [], None
    for path in checkpoint_paths:
        model, ckpt = load_checkpoint(path, device)
        ckpt_cfg = ckpt.get("config", {})
        if str(ckpt_cfg.get("input_mode", "2d")) != dcfg.input_mode:
            raise ValueError(f"checkpoint {path} input_mode does not match inference config")
        fold_uids, p, _ = predict(model, loader, device, runtime)
        if uids is None:
            uids = fold_uids
        elif uids != fold_uids:
            raise ValueError("checkpoint inference order mismatch")
        all_p.append(p)

    image_p = (
        np.mean(np.stack(all_p), axis=0)
        if all_p
        else np.full((len(test), len(TARGETS)), 0.5, np.float32)
    )
    final = image_p

    if float(fusion_alpha) < 1.0:
        # Explicit opt-in only; this branch exists for diagnostic experiments
        # where reports are actually present in the test-like data.
        from .report_labels import label_dataframe
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
