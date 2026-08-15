from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .budget import RuntimeBudget
from .constants import DUAL_STREAMS, SUBMISSION_COLUMNS, TARGETS
from .data import backfill_series_metadata, build_series_index, load_series_csv, load_test_csv
from .dataset import DatasetConfig, KneeStudyDataset
from .model import KneeMILNet
from .policy import validate_competition_config
from .runtime import autocast, resolve_runtime

ARCHITECTURE = "cross_sequence_pathology_queries_v1"
MODEL_SPEC_KEYS = {
    "architecture", "n_streams", "n_slices", "in_channels", "image_size",
    "triplet_gap", "stream_mode", "dropout", "normalize_input",
    "encoder_batch_size", "gradient_checkpointing", "transformer_layers",
    "transformer_heads", "transformer_ff_mult", "pathology_layers",
}
CHECKPOINT_REQUIRED_KEYS = {
    "model", "model_spec", "stream_names", "config", "fold", "stage",
    "validation_tta_offsets",
}


def _load_checkpoint_payload(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = sorted(CHECKPOINT_REQUIRED_KEYS.difference(payload))
    if missing:
        raise ValueError(f"checkpoint {path} missing keys: {missing}")
    if not isinstance(payload["config"], dict):
        raise ValueError(f"checkpoint {path} has invalid training config")
    validate_competition_config(payload["config"], purpose="train")
    if str(payload["stage"]) not in {"stage1", "stage2"}:
        raise ValueError(f"checkpoint {path} has invalid stage={payload['stage']!r}")
    if not isinstance(payload["validation_tta_offsets"], (list, tuple)):
        raise ValueError(f"checkpoint {path} is missing its validation TTA contract")
    missing_spec = sorted(MODEL_SPEC_KEYS.difference(payload["model_spec"]))
    if missing_spec:
        raise ValueError(
            f"checkpoint {path} is not current production architecture; missing {missing_spec}"
        )
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
        int(spec["n_streams"]), int(spec["n_slices"]), in_channels=3,
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
    return model.to(device).eval(), payload


def _dataset(root, test, index, spec, config, offsets):
    return KneeStudyDataset(
        test["StudyInstanceUID"].tolist(),
        index,
        DatasetConfig(
            data_root=str(root), split="test",
            n_slices=int(spec["n_slices"]), image_size=int(spec["image_size"]),
            noise_std=0.0, slice_dropout=0.0,
            triplet_gap=int(spec["triplet_gap"]),
            strict_dicom=bool(config.get("strict_dicom_inference", True)),
            tta_center_offsets=tuple(int(x) for x in offsets),
            center_jitter=0, rotation_deg=0.0, translate_frac=0.0,
            scale_jitter=0.0, gamma_jitter=0.0, bias_field_strength=0.0,
            series_cache_mb=int(config.get("series_cache_mb_per_worker", 256)),
        ),
        train=False,
    )


def _central_view_index(offsets: list[int]) -> int:
    return offsets.index(0) if 0 in offsets else min(range(len(offsets)), key=lambda i: abs(offsets[i]))


def _validate_ensemble_contract(paths: list[Path], payloads: list[dict], config: dict) -> tuple[list[Path], list[dict]]:
    n_folds = int(config.get("n_folds", 3))
    if len(paths) != n_folds:
        raise ValueError(f"expected exactly {n_folds} fold checkpoints, received {len(paths)}")
    folds = [int(payload["fold"]) for payload in payloads]
    expected_folds = list(range(n_folds))
    if sorted(folds) != expected_folds:
        raise ValueError(f"checkpoint folds must be exactly {expected_folds}; received {sorted(folds)}")

    stages = {str(payload["stage"]) for payload in payloads}
    if len(stages) != 1:
        raise ValueError(f"cannot mix checkpoint stages in one ensemble: {sorted(stages)}")
    stage = next(iter(stages))
    expected_stage = config.get("expected_checkpoint_stage")
    if expected_stage and str(expected_stage) != stage:
        raise ValueError(f"expected {expected_stage} checkpoints but received stage={stage}")

    requested_offsets = tuple(int(x) for x in (config.get("tta_center_offsets", [-1, 0, 1]) or [0]))
    for path, payload in zip(paths, payloads):
        trained_offsets = tuple(int(x) for x in payload["validation_tta_offsets"])
        if trained_offsets != requested_offsets:
            raise ValueError(
                f"checkpoint {path} validated TTA offsets {trained_offsets}, but submission requests "
                f"{requested_offsets}; retrain/re-evaluate instead of changing TTA after OOF"
            )

    ordered = sorted(zip(paths, payloads), key=lambda item: int(item[1]["fold"]))
    return [item[0] for item in ordered], [item[1] for item in ordered]


@torch.no_grad()
def infer_checkpoints(data_root: str | Path, checkpoint_paths, config: dict) -> pd.DataFrame:
    """One-pass image-only inference under a strict wall-clock budget."""
    validate_competition_config(config, purpose="infer")
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )
    paths = [Path(path) for path in checkpoint_paths]
    if not paths:
        raise ValueError("at least one checkpoint is required")
    payloads = [_load_checkpoint_payload(path) for path in paths]
    paths, payloads = _validate_ensemble_contract(paths, payloads, config)

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
    offsets = [int(x) for x in config.get("tta_center_offsets", [-1, 0, 1])] or [0]
    dataset = _dataset(root, test, index, spec, config, offsets)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(config.get("inference_batch_size", config.get("batch_size", 2)))),
        shuffle=False,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 900_000),
    )

    models = [load_checkpoint(path, runtime.device)[0] for path in paths]
    view_indices = list(range(len(offsets)))
    central = _central_view_index(offsets)
    probability_rows = []
    uid_rows = []
    steady_batch_times: list[float] = []
    auto_tta = bool(config.get("auto_tta_budget", True))
    allow_fallback = bool(config.get("allow_tta_fallback", True))
    tta_fallback = False
    initial_guard = float(config.get("prediction_initial_batch_guard_seconds", 180.0))

    for batch_index, batch in enumerate(loader):
        guard = initial_guard if not steady_batch_times else max(
            5.0, float(np.mean(steady_batch_times[-5:])) * 1.35
        )
        budget.require(guard, label=f"submission inference batch {batch_index + 1}")

        batch_start = time.monotonic()
        volumes = batch["volumes"]
        present = batch["present"].to(runtime.device, non_blocking=True)
        if volumes.ndim != 7:
            raise RuntimeError("inference dataset must return [B,V,K,S,C,H,W]")

        per_view_model = []
        for view in view_indices:
            model_probs = []
            x = volumes[:, view].to(runtime.device, non_blocking=True)
            for model in models:
                with autocast(runtime):
                    logits = model(x, present)
                model_probs.append(torch.sigmoid(logits.float()))
            per_view_model.append(torch.stack(model_probs).mean(dim=0))

        elapsed = time.monotonic() - batch_start
        fallback_this_batch = False
        if batch_index == 0 and auto_tta and len(view_indices) > 1:
            projected = max(elapsed, 1e-3) * len(loader) * 1.35
            if projected + budget.reserve_seconds > budget.remaining_seconds:
                if not allow_fallback:
                    raise RuntimeError(
                        f"projected TTA inference {projected/3600:.2f} h exceeds the safe runtime budget"
                    )
                print(
                    f"[budget] projected multi-view inference {projected/3600:.2f} h; "
                    "falling back to the center view to guarantee completion"
                )
                probability = per_view_model[central]
                view_indices = [central]
                tta_fallback = True
                fallback_this_batch = True
            else:
                probability = torch.stack(per_view_model).mean(dim=0)
        elif len(view_indices) == 1:
            probability = per_view_model[0]
        else:
            probability = torch.stack(per_view_model).mean(dim=0)

        probability_rows.append(probability.cpu().numpy())
        uid_rows.extend(list(batch["study_uid"]))

        if not fallback_this_batch:
            steady_batch_times.append(elapsed)
        remaining_batches = len(loader) - batch_index - 1
        if remaining_batches and steady_batch_times:
            mean_batch = float(np.mean(steady_batch_times[-5:]))
            if not budget.can_start(mean_batch * remaining_batches * 1.35):
                raise RuntimeError(
                    "inference cannot finish safely inside the configured runtime budget; "
                    "reduce preprocessing cost or inference batch overhead before submission"
                )

    probabilities = np.concatenate(probability_rows, axis=0) if probability_rows else np.empty((0, len(TARGETS)))
    if not np.isfinite(probabilities).all():
        raise RuntimeError("non-finite probabilities")
    submission = pd.DataFrame(probabilities, columns=TARGETS)
    submission.insert(0, "StudyInstanceUID", uid_rows)
    validate_submission(submission)
    print(
        f"[runtime] inference elapsed={budget.elapsed_seconds/3600:.2f} h "
        f"budget={budget.max_hours:.2f} h tta_fallback={tta_fallback} "
        f"stage={payloads[0]['stage']} folds={[int(p['fold']) for p in payloads]}"
    )
    return submission[SUBMISSION_COLUMNS]


def validate_submission(df: pd.DataFrame) -> None:
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"submission columns must be exactly {SUBMISSION_COLUMNS}")
    if df["StudyInstanceUID"].astype(str).duplicated().any():
        raise ValueError("duplicate StudyInstanceUID")
    values = df[TARGETS].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("submission probabilities must be finite and in [0,1]")
