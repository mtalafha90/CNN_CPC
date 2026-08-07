from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .constants import DUAL_STREAMS, SUBMISSION_COLUMNS, TARGETS
from .data import backfill_series_metadata, build_series_index, load_series_csv, load_test_csv
from .dataset import DatasetConfig, KneeStudyDataset
from .model import KneeMILNet
from .runtime import resolve_runtime
from .training import predict

ARCHITECTURE = "cross_sequence_pathology_queries_v1"
MODEL_SPEC_KEYS = {
    "architecture",
    "n_streams",
    "n_slices",
    "in_channels",
    "image_size",
    "triplet_gap",
    "stream_mode",
    "dropout",
    "normalize_input",
    "encoder_batch_size",
    "gradient_checkpointing",
    "transformer_layers",
    "transformer_heads",
    "transformer_ff_mult",
    "pathology_layers",
}


def _load_checkpoint_payload(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "model_spec", "stream_names"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"checkpoint {path} missing keys: {missing}")
    missing_spec = sorted(MODEL_SPEC_KEYS.difference(payload["model_spec"]))
    if missing_spec:
        raise ValueError(f"checkpoint {path} is not a current production checkpoint; missing model_spec: {missing_spec}")
    return payload


def _same_model_spec(a: dict, b: dict) -> bool:
    return all(a.get(key) == b.get(key) for key in MODEL_SPEC_KEYS)


def load_checkpoint(path: str | Path, device: torch.device):
    payload = _load_checkpoint_payload(path)
    spec = payload["model_spec"]
    if spec["architecture"] != ARCHITECTURE:
        raise ValueError(f"unsupported checkpoint architecture: {spec['architecture']}")
    if int(spec["n_streams"]) != len(DUAL_STREAMS) or int(spec["in_channels"]) != 3:
        raise ValueError("checkpoint violates production stream/channel contract")
    if str(spec["stream_mode"]) != "dual":
        raise ValueError("checkpoint is not a dual-sequence production model")

    model = KneeMILNet(
        int(spec["n_streams"]),
        int(spec["n_slices"]),
        in_channels=3,
        pretrained_weights=False,
        normalize_input=bool(spec["normalize_input"]),
        dropout=float(spec["dropout"]),
        encoder_batch_size=int(spec["encoder_batch_size"]),
        gradient_checkpointing=bool(spec["gradient_checkpointing"]),
        transformer_layers=int(spec["transformer_layers"]),
        transformer_heads=int(spec["transformer_heads"]),
        transformer_ff_mult=float(spec["transformer_ff_mult"]),
        pathology_layers=int(spec["pathology_layers"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), payload


def _dataset(root, test, index, spec, config, offset: int):
    return KneeStudyDataset(
        test["StudyInstanceUID"].tolist(),
        index,
        DatasetConfig(
            data_root=str(root),
            split="test",
            n_slices=int(spec["n_slices"]),
            image_size=int(spec["image_size"]),
            noise_std=0.0,
            slice_dropout=0.0,
            triplet_gap=int(spec["triplet_gap"]),
            strict_dicom=bool(config.get("strict_dicom_inference", True)),
            center_offset=int(offset),
            center_jitter=0,
            rotation_deg=0.0,
            translate_frac=0.0,
            scale_jitter=0.0,
            gamma_jitter=0.0,
            bias_field_strength=0.0,
        ),
        train=False,
    )


def infer_checkpoints(data_root: str | Path, checkpoint_paths, config: dict) -> pd.DataFrame:
    paths = [Path(path) for path in checkpoint_paths]
    if not paths:
        raise ValueError("at least one checkpoint is required")
    payloads = [_load_checkpoint_payload(path) for path in paths]
    spec = payloads[0]["model_spec"]
    if list(payloads[0]["stream_names"]) != DUAL_STREAMS:
        raise ValueError("checkpoint stream order mismatch")
    for path, payload in zip(paths[1:], payloads[1:]):
        if not _same_model_spec(spec, payload["model_spec"]):
            raise ValueError(f"checkpoint model_spec mismatch: {path}")
        if list(payload["stream_names"]) != DUAL_STREAMS:
            raise ValueError(f"checkpoint stream order mismatch: {path}")

    root = Path(data_root)
    test = load_test_csv(root / config.get("test_csv", "test.csv"))
    series = load_series_csv(root / config.get("test_series_csv", "test_series.csv"))
    series, stats = backfill_series_metadata(series, root, split="test")
    print(f"[test metadata] {stats}")
    index = build_series_index(series, test["StudyInstanceUID"], mode="dual")
    runtime = resolve_runtime(config)
    if runtime.distributed:
        raise RuntimeError("submission inference should be launched as one process; DDP inference is not supported")
    offsets = [int(x) for x in config.get("tta_center_offsets", [-1, 0, 1])] or [0]

    all_predictions = []
    reference_uids = None
    for offset in offsets:
        dataset = _dataset(root, test, index, spec, config, offset)
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(config.get("batch_size", 2))),
            shuffle=False,
            **runtime.loader_kwargs(),
        )
        for path in paths:
            model, _ = load_checkpoint(path, runtime.device)
            uids, probability, _ = predict(model, loader, runtime.device, runtime)
            if reference_uids is None:
                reference_uids = uids
            elif uids != reference_uids:
                raise ValueError("inference order mismatch")
            all_predictions.append(probability)

    probabilities = np.mean(np.stack(all_predictions), axis=0)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("non-finite probabilities")
    submission = pd.DataFrame(probabilities, columns=TARGETS)
    submission.insert(0, "StudyInstanceUID", reference_uids)
    validate_submission(submission)
    return submission[SUBMISSION_COLUMNS]


def validate_submission(df: pd.DataFrame) -> None:
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"submission columns must be exactly {SUBMISSION_COLUMNS}")
    if df["StudyInstanceUID"].astype(str).duplicated().any():
        raise ValueError("duplicate StudyInstanceUID")
    values = df[TARGETS].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("submission probabilities must be finite and in [0,1]")
