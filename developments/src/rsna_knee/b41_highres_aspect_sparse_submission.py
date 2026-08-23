"""Hidden-test inference for the frozen B41 native-aspect sparse-MIL endpoint.

B41 must be scored with the same geometry used during training: full-native
normalization, a native 90% centre crop, one aspect-preserving resize-to-fit,
and symmetric zero padding to 448x448.  Reusing B37's submission dataset would
silently square-stretch rectangular test series and would therefore evaluate a
different model/input contract.
"""
from __future__ import annotations

import argparse
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
from .b35_training import B35_EXPECTED_CELLS, B35_EXPECTED_SERIES, sha256_file
from .b37_highres_sparse_eval import load_b37_checkpoint
from .b37_highres_sparse_mil import collate_b35
from .b37_highres_sparse_submission import (
    B37_SUBMISSION_BATCH_SIZE,
    B37_SUBMISSION_MAX_HOURS,
    B37_SUBMISSION_MIN_RESERVE_MINUTES,
    B37_SUBMISSION_TTA_OFFSETS,
    _b37_test_dataset_config,
    _release_memory,
    _require_sparse_mil_submission_contract,
    _submission_budget,
    preflight_sparse_mil_submission,
    projected_remaining_seconds,
)
from .b41_highres_aspect_sparse_mil import (
    B41_EXPERIMENT,
    B41_IMAGE_SIZE,
    B41_RESIZE_POLICY,
    B41_TOP_K,
    B41_VERSION,
    B41HighResAspectSparseDataset,
    b41_preprocessing_state,
    require_b41_aspect_contract,
)
from .b41_highres_aspect_sparse_training import B41_EPOCHS
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_test_csv
from .phase9_supervision import REPORT_ONLY_STUDIES
from .runtime import autocast, resolve_runtime

B41_SUBMISSION_EXPERIMENT = "B41_frozen_native_aspect_sparse_mil_hidden_test_inference"
B41_SUBMISSION_TTA_OFFSETS = B37_SUBMISSION_TTA_OFFSETS
B41_SUBMISSION_LOADER_SEED_OFFSET = 51_500_000
B41_SUBMISSION_MIN_RESERVE_MINUTES = B37_SUBMISSION_MIN_RESERVE_MINUTES


def require_b41_submission_contract(config: dict) -> dict:
    """Reject inference settings that differ from the frozen B41 endpoint."""
    # First require B41's geometry policy, not merely B37's shared model controls.
    policy = require_b41_aspect_contract(config)
    # Then reuse the audited B37-family hidden-test safety/TTA contract.
    _require_sparse_mil_submission_contract(
        config,
        expected_offsets=B41_SUBMISSION_TTA_OFFSETS,
        endpoint_name="B41",
    )
    return policy


def _require_b41_checkpoint_contract(payload: dict) -> None:
    """Verify that the supplied checkpoint is the completed fixed B41 E2 model."""
    if payload.get("experiment") != B41_EXPERIMENT:
        raise ValueError("checkpoint is not the completed B41 experiment")
    if payload.get("version") != B41_VERSION:
        raise ValueError("checkpoint is not a B41 native-aspect sparse-MIL checkpoint")
    if payload.get("fixed_endpoint") is not True or int(payload.get("completed_epochs", -1)) != B41_EPOCHS:
        raise ValueError("B41 submission requires the completed fixed-E2 checkpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0 or bool(payload.get("gold_labels_used", True)):
        raise ValueError("B41 checkpoint unexpectedly used expert labels")
    if int(payload.get("training_studies", -1)) != REPORT_ONLY_STUDIES:
        raise ValueError("B41 checkpoint has the wrong report-only training population")
    if int(payload.get("training_series", -1)) != B35_EXPECTED_SERIES:
        raise ValueError("B41 checkpoint has the wrong MRI series training surface")
    if int(payload.get("training_supervision_cells", -1)) != B35_EXPECTED_CELLS:
        raise ValueError("B41 checkpoint has the wrong supervision surface")

    sparse = payload.get("sparse_mil")
    if not isinstance(sparse, dict):
        raise ValueError("B41 checkpoint is missing sparse-MIL metadata")
    if int(sparse.get("grid_size", -1)) != 6 or int(sparse.get("top_k", -1)) != B41_TOP_K:
        raise ValueError("B41 checkpoint sparse-MIL grid/top-k contract changed")
    if int(sparse.get("dense_slices", -1)) != 32:
        raise ValueError("B41 checkpoint does not use 32 dense slice centres")

    preprocessing = payload.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("B41 checkpoint is missing preprocessing metadata")
    if preprocessing.get("resize_policy") != B41_RESIZE_POLICY:
        raise ValueError("B41 checkpoint does not certify aspect-preserving resize/pad")
    if preprocessing.get("preserves_in_plane_aspect_ratio") is not True:
        raise ValueError("B41 checkpoint does not certify preserved in-plane aspect ratio")
    if int(preprocessing.get("image_size", -1)) != B41_IMAGE_SIZE:
        raise ValueError("B41 checkpoint does not certify the frozen 448 canvas")
    padding = preprocessing.get("padding")
    if not isinstance(padding, dict) or float(padding.get("value", float("nan"))) != 0.0:
        raise ValueError("B41 checkpoint does not certify zero padding")


@torch.no_grad()
def generate_b41_submission(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_path: str | Path = "submission.csv",
    preflight_only: bool = False,
) -> Path | None:
    """Run the exact frozen B41 endpoint on the competition test surface."""
    settings = dict(config)
    root = Path(data_root).resolve()
    settings["data_root"] = str(root)
    crop_policy = require_b41_submission_contract(settings)

    runtime = resolve_runtime(settings)
    if runtime.num_workers != 0 or runtime.pin_memory:
        raise RuntimeError("B41 submission requires workers=0 and pin_memory=False")
    print(runtime.describe(), flush=True)

    checkpoint_path = Path(checkpoint).resolve()
    base_path = Path(base_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"B41 checkpoint is missing: {checkpoint_path}")
    if not base_path.is_file():
        raise FileNotFoundError(f"B41 base checkpoint is missing: {base_path}")

    model, payload = load_b37_checkpoint(
        checkpoint_path,
        base_checkpoint=base_path,
        device=runtime.device,
        expected_version=B41_VERSION,
        expected_experiment=B41_EXPERIMENT,
        checkpoint_label="B41",
    )
    _require_b41_checkpoint_contract(payload)
    model.eval()

    test = load_test_csv(root / settings.get("test_csv", "test.csv"))
    uids = test["StudyInstanceUID"].astype(str).tolist()
    if not uids:
        raise ValueError("test.csv contains no studies")
    series = load_series_csv(root / settings.get("test_series_csv", "test_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="test")
    variable_index = build_variable_series_index(series, uids)
    counts = [len(variable_index.get(uid, [])) for uid in uids]
    missing = [uid for uid, count in zip(uids, counts) if count == 0]
    if missing:
        raise ValueError(f"B41 submission found {len(missing)} test study/studies with zero eligible MRI series")

    dataset = B41HighResAspectSparseDataset(
        uids,
        variable_index,
        _b37_test_dataset_config(settings, root),
        crop_focus_policy=crop_policy,
        center_offsets=B41_SUBMISSION_TTA_OFFSETS,
    )
    if preflight_only:
        preflight_sparse_mil_submission(
            model,
            dataset,
            runtime,
            tta_offsets=B41_SUBMISSION_TTA_OFFSETS,
            endpoint_name="B41",
        )
        return None

    loader = DataLoader(
        dataset,
        batch_size=B37_SUBMISSION_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_b35,
        **runtime.loader_kwargs(seed=int(settings.get("seed", 2026)) + B41_SUBMISSION_LOADER_SEED_OFFSET),
    )
    budget = _submission_budget(
        settings,
        max_hours=B37_SUBMISSION_MAX_HOURS,
        min_reserve_minutes=B41_SUBMISSION_MIN_RESERVE_MINUTES,
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
            safety_factor=1.35,
        )
        budget.require(projected_seconds, label="remaining B41 submission inference")

        started = time.monotonic()
        batch = next(iterator)
        volumes = batch["volumes"].to(runtime.device, non_blocking=True)
        position = batch["slice_position"].to(runtime.device, non_blocking=True)
        present = batch["present"].to(runtime.device, non_blocking=True)
        series_meta = batch["series_meta"].to(runtime.device, non_blocking=True)
        if volumes.ndim != 7 or int(volumes.shape[1]) != len(B41_SUBMISSION_TTA_OFFSETS):
            raise RuntimeError(f"B41 submission TTA volume shape changed: {tuple(volumes.shape)}")
        if position.ndim != 4 or int(position.shape[1]) != len(B41_SUBMISSION_TTA_OFFSETS):
            raise RuntimeError("B41 submission slice-position TTA shape changed")

        view_probabilities = []
        for view in range(volumes.shape[1]):
            with autocast(runtime):
                output = model(volumes[:, view], present, series_meta, position[:, view])
            view_probabilities.append(torch.sigmoid(output.logits.float()))
        probability = torch.stack(view_probabilities, dim=0).mean(dim=0)
        if not torch.isfinite(probability).all():
            raise RuntimeError("B41 submission produced non-finite probabilities")
        probability_rows.append(probability.cpu().numpy())
        uid_rows.extend(str(uid) for uid in batch["study_uid"])

        del batch, volumes, position, present, series_meta, output, view_probabilities, probability
        _release_memory()
        study_times.append(time.monotonic() - started)
        completed = batch_index + 1
        if completed % 10 == 0 or completed == len(loader):
            remaining = projected_remaining_seconds(
                study_times,
                remaining_studies=len(loader) - completed,
                safety_factor=1.35,
            )
            print(
                f"[B41 submit] {completed}/{len(loader)} elapsed={budget.elapsed_seconds / 60.0:.1f} min "
                f"estimated_remaining={remaining / 60.0:.1f} min "
                f"work_remaining={budget.remaining_work_seconds / 60.0:.1f} min",
                flush=True,
            )

    if uid_rows != uids:
        raise RuntimeError("B41 submission changed StudyInstanceUID order")
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
        "experiment": B41_SUBMISSION_EXPERIMENT,
        "version": B41_VERSION,
        "parent_checkpoint_experiment": payload.get("experiment"),
        "parent_checkpoint_version": payload.get("version"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "checkpoint_base_sha256_verified": base_checkpoint_sha256 == str(payload.get("base_checkpoint_sha256", "")),
        "fixed_endpoint": bool(payload.get("fixed_endpoint")),
        "completed_epochs": int(payload.get("completed_epochs", -1)),
        "training_studies": int(payload.get("training_studies", -1)),
        "training_series": int(payload.get("training_series", -1)),
        "training_supervision_cells": int(payload.get("training_supervision_cells", -1)),
        "gold_studies_used_in_gradient": int(payload.get("gold_studies_used_in_gradient", -1)),
        "prediction": "frozen B41 combined sparse-MIL logits; raw sigmoid probability",
        "thresholding_used": False,
        "blending_used": False,
        "preprocessing": b41_preprocessing_state(),
        "crop_policy": crop_policy,
        "sparse_mil": payload.get("sparse_mil"),
        "encoder_sha256_initial": payload.get("encoder_sha256_initial"),
        "encoder_sha256_final": payload.get("encoder_sha256_final"),
        "test_rows": int(len(frame)),
        "test_series_total": int(sum(counts)),
        "test_series_min": int(min(counts)),
        "test_series_median": float(np.median(counts)),
        "test_series_max": int(max(counts)),
        "tta_center_offsets": list(B41_SUBMISSION_TTA_OFFSETS),
        "eval_batch_size": B37_SUBMISSION_BATCH_SIZE,
        "workers": int(runtime.num_workers),
        "pin_memory": bool(runtime.pin_memory),
        "strict_dicom": True,
        "metadata_repair": metadata_stats,
        "runtime_elapsed_hours": float(budget.elapsed_seconds / 3600.0),
        "runtime_budget_hours": float(budget.max_hours),
        "runtime_reserve_minutes": float(budget.reserve_minutes),
        "submission_sha256": sha256_file(output_path),
        "governance": (
            "Exact frozen B41 fixed-E2 native-aspect endpoint. Do not tune thresholds, blend weights, "
            "TTA, crop, resize/padding policy, grid, top-k, or target behavior after hidden evidence."
        ),
        **sample_validation,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if output_path.name != "submission.csv":
        print("[B41 submit] local smoke output written; Kaggle final output must be named submission.csv", flush=True)
    print(output_path, flush=True)
    print(manifest_path, flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser("Generate the frozen B41 Kaggle submission")
    parser.add_argument("--config", default="config/b41_highres_aspect_sparse_448.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = dict(_read_config(args.config))
    generate_b41_submission(
        config,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        out_path=args.out,
        preflight_only=bool(args.preflight_only),
    )


if __name__ == "__main__":
    main()
