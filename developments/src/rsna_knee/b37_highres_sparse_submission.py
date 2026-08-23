"""Hidden-test inference for the completed frozen B37 sparse-MIL endpoint.

This module is deliberately separate from ``b37_highres_sparse_eval``.  That
module evaluates the reused 58-study expert surface, while this one reads only
``test.csv``/``test_series.csv`` and writes a Kaggle-compatible probability
submission.  It does not use labels, thresholds, blending, or post-hoc tuning.

The B37 checkpoint is not standalone: it reconstructs the exact B34 base from
the supplied Phase-9 checkpoint and verifies that checkpoint's SHA-256 digest
before loading B37's fine-tuned encoder and sparse-MIL head.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config
from .b12_variable_series import build_variable_series_index
from .b17_submission import _validate_sample_submission, _validate_submission
from .b35_training import (
    B35_EXPECTED_CELLS,
    B35_EXPECTED_SERIES,
    sha256_file,
)
from .b37_highres_sparse_eval import load_b37_checkpoint
from .b37_highres_sparse_mil import (
    B37_IMAGE_SIZE,
    B37_TOP_K,
    B37_VERSION,
    B37HighResSparseDataset,
    collate_b35,
    require_b37_sparse_contract,
)
from .b37_highres_sparse_training import (
    B37_EPOCHS,
    B37_EXPERIMENT,
)
from .budget import RuntimeBudget
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_test_csv
from .dataset import DatasetConfig
from .phase9_supervision import REPORT_ONLY_STUDIES
from .policy import validate_competition_config
from .runtime import autocast, resolve_runtime

B37_SUBMISSION_EXPERIMENT = "B37_frozen_highres448_sparse_mil_hidden_test_inference"
B37_SUBMISSION_TTA_OFFSETS = (-1, 0, 1)
B37_SUBMISSION_BATCH_SIZE = 1
B37_SUBMISSION_LOADER_SEED_OFFSET = 47_500_000
B37_SUBMISSION_MAX_HOURS = 8.25
B37_SUBMISSION_MIN_RESERVE_MINUTES = 30.0


def _b37_test_dataset_config(config: dict, root: Path) -> DatasetConfig:
    """Build the deterministic test-time dataset configuration for B37."""
    return DatasetConfig(
        data_root=str(root),
        split="test",
        n_slices=int(config.get("b7_n_slices", 16)),
        image_size=B37_IMAGE_SIZE,
        noise_std=0.0,
        slice_dropout=0.0,
        triplet_gap=int(config.get("b7_triplet_gap", 1)),
        strict_dicom=True,
        train_gap_choices=tuple(
            int(value) for value in config.get("b7_train_gap_choices", [1, 2])
        ),
        center_jitter=0,
        tta_center_offsets=(),
        rotation_deg=0.0,
        translate_frac=0.0,
        scale_jitter=0.0,
        gamma_jitter=0.0,
        bias_field_strength=0.0,
        series_cache_mb=0,
        physical_scale_policy=None,
    )


def _require_sparse_mil_submission_contract(
    config: dict,
    *,
    expected_offsets: tuple[int, ...],
    endpoint_name: str,
) -> dict:
    """Validate a fixed B37-family inference recipe without changing B37 itself."""
    policy = require_b37_sparse_contract(config)
    offsets = tuple(
        int(value)
        for value in config.get("b7_eval_tta_offsets", expected_offsets)
    )
    expected = tuple(int(value) for value in expected_offsets)
    if offsets != expected:
        raise ValueError(
            f"{endpoint_name} submission freezes "
            f"b7_eval_tta_offsets={list(expected)}"
        )
    for key in ("tta_center_offsets", "validation_tta_offsets"):
        values = config.get(key)
        if values is not None and tuple(int(value) for value in values) != expected:
            raise ValueError(
                f"{endpoint_name} submission freezes {key}={list(expected)}"
            )
    if (
        int(config.get("b7_eval_batch_size", B37_SUBMISSION_BATCH_SIZE))
        != B37_SUBMISSION_BATCH_SIZE
    ):
        raise ValueError(f"{endpoint_name} submission freezes b7_eval_batch_size=1")
    if int(config.get("num_workers", 0)) != 0:
        raise ValueError(
            f"{endpoint_name} submission requires num_workers=0 "
            "to avoid host-memory multiplication"
        )
    if bool(config.get("pin_memory", False)):
        raise ValueError(f"{endpoint_name} submission requires pin_memory=false")
    if int(config.get("series_cache_mb_per_worker", 0)) != 0:
        raise ValueError(
            f"{endpoint_name} submission requires series_cache_mb_per_worker=0"
        )
    if bool(config.get("strict_dicom_inference", True)) is not True:
        raise ValueError(f"{endpoint_name} submission requires strict_dicom_inference=true")
    validate_competition_config(config, purpose="infer")
    return policy


def require_b37_submission_contract(config: dict) -> dict:
    """Reject inference settings that would alter the frozen B37 endpoint."""
    return _require_sparse_mil_submission_contract(
        config,
        expected_offsets=B37_SUBMISSION_TTA_OFFSETS,
        endpoint_name="B37",
    )


def _require_b37_checkpoint_contract(payload: dict) -> None:
    """Check checkpoint facts that are specific to the completed B37 run."""
    if payload.get("experiment") != B37_EXPERIMENT:
        raise ValueError("checkpoint is not the completed B37 experiment")
    if payload.get("version") != B37_VERSION:
        raise ValueError("checkpoint is not a B37 high-resolution sparse-MIL checkpoint")
    if payload.get("fixed_endpoint") is not True or int(
        payload.get("completed_epochs", -1)
    ) != B37_EPOCHS:
        raise ValueError("B37 submission requires the completed fixed-E2 checkpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0 or bool(
        payload.get("gold_labels_used", True)
    ):
        raise ValueError("B37 checkpoint unexpectedly used expert labels")
    if int(payload.get("training_studies", -1)) != REPORT_ONLY_STUDIES:
        raise ValueError("B37 checkpoint has the wrong report-only training population")
    if int(payload.get("training_series", -1)) != B35_EXPECTED_SERIES:
        raise ValueError("B37 checkpoint has the wrong MRI series training surface")
    if int(payload.get("training_supervision_cells", -1)) != B35_EXPECTED_CELLS:
        raise ValueError("B37 checkpoint has the wrong supervision surface")
    initial = str(payload.get("encoder_sha256_initial", ""))
    final = str(payload.get("encoder_sha256_final", ""))
    if not initial or not final or initial == final:
        raise ValueError("B37 checkpoint lacks the required encoder-tail adaptation fingerprint")

    sparse = payload.get("sparse_mil")
    if not isinstance(sparse, dict):
        raise ValueError("B37 checkpoint is missing sparse-MIL metadata")
    if int(sparse.get("grid_size", -1)) != 6:
        raise ValueError("B37 checkpoint does not use the frozen 6x6 sparse-MIL grid")
    if int(sparse.get("top_k", -1)) != B37_TOP_K:
        raise ValueError("B37 checkpoint does not use frozen top-k=8 sparse MIL")
    if int(sparse.get("dense_slices", -1)) != 32:
        raise ValueError("B37 checkpoint does not use 32 dense slice centres")
    preprocessing = payload.get("preprocessing")
    if not isinstance(preprocessing, dict) or int(
        preprocessing.get("image_size", -1)
    ) != B37_IMAGE_SIZE:
        raise ValueError("B37 checkpoint does not certify the frozen 448x448 preprocessing")


def _release_memory() -> None:
    """Drop completed high-resolution batches before constructing the next one."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _submission_budget(
    config: dict,
    *,
    max_hours: float = B37_SUBMISSION_MAX_HOURS,
    min_reserve_minutes: float = B37_SUBMISSION_MIN_RESERVE_MINUTES,
) -> RuntimeBudget:
    """Stay within a fixed endpoint's wall-clock budget and leave time for output."""
    return RuntimeBudget(
        max_hours=min(
            float(config.get("runtime_budget_hours", max_hours)),
            float(max_hours),
        ),
        reserve_minutes=max(
            float(config.get("runtime_reserve_minutes", min_reserve_minutes)),
            float(min_reserve_minutes),
        ),
    )


def projected_remaining_seconds(
    completed_study_seconds: list[float],
    *,
    remaining_studies: int,
    safety_factor: float = 1.35,
) -> float:
    """Conservative remaining-time estimate from complete study wall-clock times.

    Each recorded duration includes DataLoader materialization, all TTA views,
    device transfer, model inference, and post-study memory release.  This is
    deliberately broader than GPU-only timing so a Kaggle run cannot silently
    spend its reserve on DICOM decoding.
    """
    if int(remaining_studies) < 0:
        raise ValueError("remaining_studies must be non-negative")
    if not completed_study_seconds:
        return 180.0
    if float(safety_factor) < 1.0:
        raise ValueError("safety_factor must be at least one")
    window = np.asarray(completed_study_seconds[-5:], dtype=np.float64)
    if not np.isfinite(window).all() or np.any(window <= 0):
        raise ValueError("completed study timings must be finite and positive")
    return float(window.mean() * int(remaining_studies) * float(safety_factor))


def _largest_submission_index(dataset) -> int:
    """Choose the test study with the most eligible MRI series for a memory probe."""
    if len(dataset) < 1:
        raise ValueError("submission dataset is empty")
    return max(
        range(len(dataset)),
        key=lambda index: len(dataset.series_records[dataset.study_uids[index]]),
    )


@torch.no_grad()
def preflight_sparse_mil_submission(
    model,
    dataset,
    runtime,
    *,
    tta_offsets: tuple[int, ...],
    endpoint_name: str,
) -> None:
    """Run the highest-series hidden-test study once before a long endpoint."""
    if runtime.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(runtime.device)
    index = _largest_submission_index(dataset)
    item = dataset[index]
    batch = collate_b35([item])
    volumes = batch["volumes"].to(runtime.device, non_blocking=True)
    position = batch["slice_position"].to(runtime.device, non_blocking=True)
    present = batch["present"].to(runtime.device, non_blocking=True)
    series_meta = batch["series_meta"].to(runtime.device, non_blocking=True)
    expected_views = len(tta_offsets)
    if volumes.ndim != 7 or int(volumes.shape[1]) != expected_views:
        raise RuntimeError(
            f"{endpoint_name} preflight expected [B,{expected_views},K,32,3,448,448], "
            f"got {tuple(volumes.shape)}"
        )
    if position.ndim != 4 or int(position.shape[1]) != expected_views:
        raise RuntimeError(f"{endpoint_name} preflight slice-position TTA shape changed")

    views = []
    for view in range(expected_views):
        with autocast(runtime):
            output = model(
                volumes[:, view],
                present,
                series_meta,
                position[:, view],
            )
        views.append(torch.sigmoid(output.logits.float()))
    probability = torch.stack(views, dim=0).mean(dim=0)
    if not torch.isfinite(probability).all():
        raise RuntimeError(f"{endpoint_name} preflight produced non-finite probabilities")
    peak_gib = 0.0
    if runtime.device.type == "cuda" and torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_allocated(runtime.device) / (1024**3)
    print(
        f"[{endpoint_name} preflight] study={batch['study_uid'][0]} "
        f"series={int(present.sum().item())} views={expected_views} "
        f"cuda_peak={peak_gib:.2f}GiB PASS",
        flush=True,
    )
    del batch, volumes, position, present, series_meta, output, views, probability
    _release_memory()


@torch.no_grad()
def generate_b37_sparse_mil_submission(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_path: str | Path = "submission.csv",
    tta_offsets: tuple[int, ...] = B37_SUBMISSION_TTA_OFFSETS,
    submission_experiment: str = B37_SUBMISSION_EXPERIMENT,
    submission_version: str = B37_VERSION,
    endpoint_name: str = "B37",
    min_reserve_minutes: float = B37_SUBMISSION_MIN_RESERVE_MINUTES,
    preflight_only: bool = False,
    governance: str | None = None,
) -> Path | None:
    """Run a fixed B37-family sparse-MIL inference recipe on the hidden test set."""
    config = dict(config)
    root = Path(data_root).resolve()
    config["data_root"] = str(root)
    offsets = tuple(int(value) for value in tta_offsets)
    if not offsets:
        raise ValueError("a sparse-MIL submission requires at least one TTA offset")
    crop_policy = _require_sparse_mil_submission_contract(
        config,
        expected_offsets=offsets,
        endpoint_name=endpoint_name,
    )
    runtime = resolve_runtime(config)
    if runtime.num_workers != 0 or runtime.pin_memory:
        raise RuntimeError(
            f"{endpoint_name} submission runtime must use workers=0 and pin_memory=False"
        )
    print(runtime.describe(), flush=True)

    checkpoint_path = Path(checkpoint).resolve()
    base_path = Path(base_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{endpoint_name} checkpoint is missing {checkpoint_path}")
    if not base_path.is_file():
        raise FileNotFoundError(
            f"{endpoint_name} base checkpoint is missing {base_path}"
        )
    model, payload = load_b37_checkpoint(
        checkpoint_path,
        base_checkpoint=base_path,
        device=runtime.device,
    )
    _require_b37_checkpoint_contract(payload)
    model.eval()

    test = load_test_csv(root / config.get("test_csv", "test.csv"))
    uids = test["StudyInstanceUID"].astype(str).tolist()
    if not uids:
        raise ValueError("test.csv contains no studies")
    series = load_series_csv(root / config.get("test_series_csv", "test_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="test")
    variable_index = build_variable_series_index(series, uids)
    counts = [len(variable_index.get(uid, [])) for uid in uids]
    missing = [uid for uid, count in zip(uids, counts) if count == 0]
    if missing:
        raise ValueError(
            f"{endpoint_name} submission found {len(missing)} test study/studies "
            "with zero eligible MRI series"
        )

    dataset = B37HighResSparseDataset(
        uids,
        variable_index,
        _b37_test_dataset_config(config, root),
        crop_focus_policy=crop_policy,
        center_offsets=offsets,
    )
    if preflight_only:
        preflight_sparse_mil_submission(
            model,
            dataset,
            runtime,
            tta_offsets=offsets,
            endpoint_name=endpoint_name,
        )
        return None

    loader = DataLoader(
        dataset,
        batch_size=B37_SUBMISSION_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_b35,
        **runtime.loader_kwargs(
            seed=int(config.get("seed", 2026)) + B37_SUBMISSION_LOADER_SEED_OFFSET
        ),
    )

    budget = _submission_budget(
        config,
        max_hours=B37_SUBMISSION_MAX_HOURS,
        min_reserve_minutes=min_reserve_minutes,
    )
    probability_rows: list[np.ndarray] = []
    uid_rows: list[str] = []
    study_times: list[float] = []
    iterator = iter(loader)
    for batch_index in range(len(loader)):
        remaining_before = len(loader) - batch_index
        projected_seconds = projected_remaining_seconds(
            study_times,
            remaining_studies=remaining_before,
        )
        budget.require(
            projected_seconds,
            label=f"remaining {endpoint_name} submission inference",
        )

        started = time.monotonic()
        batch = next(iterator)
        volumes = batch["volumes"].to(runtime.device, non_blocking=True)
        position = batch["slice_position"].to(runtime.device, non_blocking=True)
        present = batch["present"].to(runtime.device, non_blocking=True)
        series_meta = batch["series_meta"].to(runtime.device, non_blocking=True)
        if volumes.ndim != 7:
            raise RuntimeError(
                f"{endpoint_name} submission expects [B,V,K,32,3,448,448], "
                f"got {tuple(volumes.shape)}"
            )
        if int(volumes.shape[1]) != len(offsets):
            raise RuntimeError(f"{endpoint_name} submission TTA view count changed")
        if position.ndim != 4 or int(position.shape[1]) != len(offsets):
            raise RuntimeError(
                f"{endpoint_name} submission slice-position TTA shape changed"
            )

        view_probabilities = []
        for view in range(volumes.shape[1]):
            with autocast(runtime):
                output = model(
                    volumes[:, view],
                    present,
                    series_meta,
                    position[:, view],
                )
            view_probabilities.append(torch.sigmoid(output.logits.float()))
        probability = torch.stack(view_probabilities, dim=0).mean(dim=0)
        probability_rows.append(probability.cpu().numpy())
        uid_rows.extend(str(uid) for uid in batch["study_uid"])

        del (
            batch,
            volumes,
            position,
            present,
            series_meta,
            output,
            view_probabilities,
            probability,
        )
        _release_memory()
        study_times.append(time.monotonic() - started)
        completed = batch_index + 1
        if completed % 10 == 0 or completed == len(loader):
            elapsed = budget.elapsed_seconds / 60.0
            remaining = projected_remaining_seconds(
                study_times,
                remaining_studies=len(loader) - completed,
            )
            print(
                f"[{endpoint_name} submit] {completed}/{len(loader)} "
                f"elapsed={elapsed:.1f} min "
                f"estimated_remaining={remaining / 60.0:.1f} min "
                f"reserve={budget.remaining_seconds / 60.0:.1f} min",
                flush=True,
            )

    if uid_rows != uids:
        raise RuntimeError(f"{endpoint_name} submission changed StudyInstanceUID order")
    probabilities = np.concatenate(probability_rows, axis=0)
    frame = pd.DataFrame(probabilities, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    _validate_submission(frame, uids)
    sample_validation = _validate_sample_submission(root, frame)

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    checkpoint_sha256 = sha256_file(checkpoint_path)
    base_checkpoint_sha256 = sha256_file(base_path)
    manifest = {
        "experiment": submission_experiment,
        "version": submission_version,
        "parent_checkpoint_experiment": payload.get("experiment"),
        "parent_checkpoint_version": payload.get("version"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "checkpoint_base_sha256_verified": base_checkpoint_sha256
        == str(payload.get("base_checkpoint_sha256", "")),
        "fixed_endpoint": bool(payload.get("fixed_endpoint")),
        "completed_epochs": int(payload.get("completed_epochs", -1)),
        "training_studies": int(payload.get("training_studies", -1)),
        "training_series": int(payload.get("training_series", -1)),
        "training_supervision_cells": int(
            payload.get("training_supervision_cells", -1)
        ),
        "gold_studies_used_in_gradient": int(
            payload.get("gold_studies_used_in_gradient", -1)
        ),
        "prediction": "frozen B37 combined sparse-MIL logits; raw sigmoid probability",
        "thresholding_used": False,
        "blending_used": False,
        "input_image_size": B37_IMAGE_SIZE,
        "crop_policy": crop_policy,
        "sparse_mil": payload.get("sparse_mil"),
        "encoder_sha256_initial": payload.get("encoder_sha256_initial"),
        "encoder_sha256_final": payload.get("encoder_sha256_final"),
        "test_rows": int(len(frame)),
        "test_series_total": int(sum(counts)),
        "test_series_min": int(min(counts)),
        "test_series_median": float(np.median(counts)),
        "test_series_max": int(max(counts)),
        "tta_center_offsets": list(offsets),
        "eval_batch_size": B37_SUBMISSION_BATCH_SIZE,
        "workers": int(runtime.num_workers),
        "pin_memory": bool(runtime.pin_memory),
        "strict_dicom": True,
        "metadata_repair": metadata_stats,
        "runtime_elapsed_hours": float(budget.elapsed_seconds / 3600.0),
        "runtime_budget_hours": float(budget.max_hours),
        "runtime_reserve_minutes": float(budget.reserve_minutes),
        "timing_scope": (
            "per-study full wall time: DICOM/DataLoader materialization, device "
            "transfer, all TTA views, inference, and memory release"
        ),
        "submission_sha256": sha256_file(output_path),
        "governance": governance
        or (
            "Exact frozen B37 fixed-E2 combined sparse-MIL endpoint. Do not tune "
            "thresholds, blend weights, TTA, resolution, crop, grid, top-k, or "
            "target-specific behavior after observing hidden competition evidence."
        ),
        **sample_validation,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if output_path.name != "submission.csv":
        print(
            f"[{endpoint_name} submit] local smoke output written. "
            "The Kaggle final output must be named submission.csv.",
            flush=True,
        )
    print(output_path, flush=True)
    print(manifest_path, flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    return output_path


def generate_b37_submission(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_path: str | Path = "submission.csv",
) -> Path:
    """Run the exact completed B37 endpoint on the hidden test surface."""
    output = generate_b37_sparse_mil_submission(
        config,
        data_root=data_root,
        checkpoint=checkpoint,
        base_checkpoint=base_checkpoint,
        out_path=out_path,
        tta_offsets=B37_SUBMISSION_TTA_OFFSETS,
        submission_experiment=B37_SUBMISSION_EXPERIMENT,
        submission_version=B37_VERSION,
        endpoint_name="B37",
        min_reserve_minutes=B37_SUBMISSION_MIN_RESERVE_MINUTES,
    )
    if output is None:
        raise RuntimeError("B37 submission unexpectedly returned from preflight")
    return output


def main() -> None:
    parser = argparse.ArgumentParser("Generate the frozen B37 Kaggle submission")
    parser.add_argument("--config", default="config/b37_highres_sparse_448.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()
    config = dict(_read_config(args.config))
    generate_b37_submission(
        config,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
