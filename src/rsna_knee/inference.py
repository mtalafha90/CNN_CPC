from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .constants import SUBMISSION_COLUMNS, TARGETS
from .data import backfill_series_metadata, build_series_index, load_series_csv, load_test_csv
from .dataset import DatasetConfig, KneeStudyDataset
from .model import KneeMILNet
from .runtime import resolve_runtime
from .training import predict


def _load_checkpoint_payload(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "model_spec", "stream_names"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"checkpoint {path} missing keys: {missing}")
    return payload


def _same_model_spec(a: dict, b: dict) -> bool:
    keys = {
        "n_streams", "n_slices", "in_channels", "image_size", "triplet_gap",
        "stream_mode", "dropout", "normalize_input",
    }
    return all(a.get(k) == b.get(k) for k in keys)


def load_checkpoint(path: str | Path, device: torch.device):
    payload = _load_checkpoint_payload(path)
    spec = payload["model_spec"]
    model = KneeMILNet(
        int(spec["n_streams"]),
        int(spec["n_slices"]),
        in_channels=int(spec.get("in_channels", 3)),
        pretrained_weights=False,
        normalize_input=bool(spec.get("normalize_input", True)),
        dropout=float(spec.get("dropout", 0.25)),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), payload


def infer_checkpoints(data_root: str | Path, checkpoint_paths, config: dict) -> pd.DataFrame:
    """Average aligned fold checkpoints using MRI images only.

    Model/preprocessing settings are recovered from the checkpoint. The runtime
    YAML controls only data locations and hardware settings, so a stale config
    cannot silently change slice count, triplet construction or normalization.
    """
    paths = [Path(p) for p in checkpoint_paths]
    if not paths:
        raise ValueError("at least one checkpoint is required for inference")

    payloads = [_load_checkpoint_payload(p) for p in paths]
    reference_spec = payloads[0]["model_spec"]
    reference_stream_names = list(payloads[0]["stream_names"])
    for path, payload in zip(paths[1:], payloads[1:]):
        if not _same_model_spec(reference_spec, payload["model_spec"]):
            raise ValueError(f"checkpoint model_spec mismatch: {path}")
        if list(payload["stream_names"]) != reference_stream_names:
            raise ValueError(f"checkpoint stream ordering mismatch: {path}")

    root = Path(data_root)
    test = load_test_csv(root / config.get("test_csv", "test.csv"))
    if test["StudyInstanceUID"].astype(str).duplicated().any():
        raise ValueError("test.csv contains duplicate StudyInstanceUID values")

    series = load_series_csv(root / config.get("test_series_csv", "test_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="test")
    print(f"[test metadata] {metadata_stats}")

    stream_mode = str(reference_spec.get("stream_mode", "dual"))
    index = build_series_index(series, test["StudyInstanceUID"].astype(str), stream_mode)
    dataset_config = DatasetConfig(
        data_root=str(root),
        split="test",
        n_slices=int(reference_spec["n_slices"]),
        image_size=int(reference_spec["image_size"]),
        noise_std=0.0,
        slice_dropout=0.0,
        input_mode="2p5d",
        triplet_gap=int(reference_spec.get("triplet_gap", 1)),
        strict_dicom=bool(config.get("strict_dicom_inference", True)),
    )
    dataset = KneeStudyDataset(test["StudyInstanceUID"].astype(str).tolist(), index, dataset_config, train=False)
    if dataset.stream_names != reference_stream_names:
        raise ValueError(
            f"test stream order {dataset.stream_names} does not match checkpoints {reference_stream_names}"
        )

    runtime = resolve_runtime(config)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(config.get("batch_size", 4))),
        shuffle=False,
        **runtime.loader_kwargs(),
    )

    all_predictions: list[np.ndarray] = []
    reference_uids: list[str] | None = None
    for path in paths:
        model, _ = load_checkpoint(path, runtime.device)
        uids, probabilities, _ = predict(model, loader, runtime.device, runtime)
        if reference_uids is None:
            reference_uids = uids
        elif uids != reference_uids:
            raise ValueError(f"checkpoint inference order mismatch: {path}")
        all_predictions.append(probabilities)

    probabilities = np.mean(np.stack(all_predictions, axis=0), axis=0)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("non-finite probabilities produced during inference")

    submission = pd.DataFrame(probabilities, columns=TARGETS)
    submission.insert(0, "StudyInstanceUID", reference_uids)
    validate_submission(submission)
    return submission[SUBMISSION_COLUMNS]


def validate_submission(df: pd.DataFrame) -> None:
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"submission columns must be exactly {SUBMISSION_COLUMNS}")
    if df["StudyInstanceUID"].astype(str).duplicated().any():
        raise ValueError("submission contains duplicate StudyInstanceUID values")
    values = df[TARGETS].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("submission probabilities must be finite and in [0,1]")
