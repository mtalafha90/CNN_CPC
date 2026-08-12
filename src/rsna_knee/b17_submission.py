"""Kaggle test inference for the frozen B17 champion checkpoint."""
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
from .b12_variable_series import (
    VariableSeriesKneeDataset,
    build_variable_series_index,
    collate_variable_series,
)
from .b17_training import B17_EXPERIMENT, B17_VARIANT, load_b17_checkpoint, require_b17_contract
from .budget import RuntimeBudget
from .constants import SUBMISSION_COLUMNS, TARGETS
from .data import backfill_series_metadata, load_series_csv, load_test_csv
from .dataset import DatasetConfig
from .runtime import autocast, resolve_runtime


def _test_dataset_config(config: dict, root: Path, offsets: tuple[int, ...]) -> DatasetConfig:
    return DatasetConfig(
        data_root=str(root),
        split="test",
        n_slices=int(config.get("b7_n_slices", 16)),
        image_size=int(config.get("b7_image_size", 224)),
        noise_std=0.0,
        slice_dropout=0.0,
        triplet_gap=int(config.get("b7_triplet_gap", 1)),
        strict_dicom=bool(config.get("strict_dicom_inference", True)),
        train_gap_choices=tuple(int(x) for x in config.get("b7_train_gap_choices", [1, 2])),
        center_jitter=0,
        tta_center_offsets=offsets,
        rotation_deg=0.0,
        translate_frac=0.0,
        scale_jitter=0.0,
        gamma_jitter=0.0,
        bias_field_strength=0.0,
        series_cache_mb=int(config.get("series_cache_mb_per_worker", 256)),
        physical_scale_policy=None,
    )


def _validate_submission(frame: pd.DataFrame, test_uids: list[str]) -> None:
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"submission columns must be exactly {SUBMISSION_COLUMNS}")
    observed = frame["StudyInstanceUID"].astype(str).tolist()
    if observed != [str(uid) for uid in test_uids]:
        raise ValueError("submission StudyInstanceUID order does not match test.csv")
    if frame["StudyInstanceUID"].astype(str).duplicated().any():
        raise ValueError("submission contains duplicate StudyInstanceUID values")
    values = frame[TARGETS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("submission contains non-finite probabilities")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("submission probabilities must lie in [0,1]")


def _validate_sample_submission(root: Path, frame: pd.DataFrame) -> dict:
    path = root / "sample_submission.csv"
    if not path.is_file():
        return {"sample_submission_present": False}
    sample = pd.read_csv(path)
    if list(sample.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            "sample_submission.csv columns differ from the frozen submission contract: "
            f"{list(sample.columns)}"
        )
    sample_uids = sample["StudyInstanceUID"].astype(str).tolist()
    pred_uids = frame["StudyInstanceUID"].astype(str).tolist()
    if sample_uids != pred_uids:
        raise ValueError("sample_submission.csv StudyInstanceUID order differs from test.csv")
    return {
        "sample_submission_present": True,
        "sample_submission_rows": int(len(sample)),
        "sample_submission_columns_match": True,
        "sample_submission_uid_order_match": True,
    }


@torch.no_grad()
def generate_b17_submission(
    config: dict,
    *,
    checkpoint: str | Path,
    out_path: str | Path = "submission.csv",
) -> Path:
    require_b17_contract(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, payload = load_b17_checkpoint(checkpoint, device=runtime.device)
    if payload.get("variant") != B17_VARIANT or payload.get("experiment") != B17_EXPERIMENT:
        raise ValueError("checkpoint is not the completed frozen B17 experiment")

    root = Path(config["data_root"])
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
        raise ValueError(f"B17 test inference found {len(missing)} study/studies with zero eligible series")

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B17 submission freezes TTA at [-1,0,1]")

    ds = VariableSeriesKneeDataset(
        uids,
        variable_index,
        _test_dataset_config(config, root, offsets),
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=max(1, int(config.get("b7_eval_batch_size", 2))),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 22_100_000),
    )

    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )
    model.eval()
    probability_rows: list[np.ndarray] = []
    uid_rows: list[str] = []
    batch_times: list[float] = []

    for batch_index, batch in enumerate(loader):
        if batch_times:
            projected = float(np.mean(batch_times[-5:])) * max(1, len(loader) - batch_index) * 1.35
            budget.require(projected, label="remaining frozen B17 submission inference")
        else:
            budget.require(180.0, label="first frozen B17 submission batch")

        start = time.monotonic()
        present = batch["present"].to(runtime.device, non_blocking=True)
        series_meta = batch["series_meta"].to(runtime.device, non_blocking=True)
        volumes = batch["volumes"]
        if volumes.ndim != 7:
            raise RuntimeError(
                f"B17 submission expects [B,V,K,S,C,H,W], got {tuple(volumes.shape)}"
            )
        if int(volumes.shape[1]) != len(offsets):
            raise RuntimeError("B17 submission TTA view count changed")

        view_probabilities = []
        for view in range(volumes.shape[1]):
            with autocast(runtime):
                logits = model(
                    volumes[:, view].to(runtime.device, non_blocking=True),
                    present,
                    series_meta,
                )
            view_probabilities.append(torch.sigmoid(logits.float()))
        probability = torch.stack(view_probabilities, dim=0).mean(dim=0)
        probability_rows.append(probability.cpu().numpy())
        uid_rows.extend([str(uid) for uid in batch["study_uid"]])
        batch_times.append(time.monotonic() - start)

    if uid_rows != uids:
        raise RuntimeError("B17 test inference changed StudyInstanceUID order")
    probabilities = np.concatenate(probability_rows, axis=0)
    frame = pd.DataFrame(probabilities, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    _validate_submission(frame, uids)
    sample_validation = _validate_sample_submission(root, frame)

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    manifest = {
        "experiment": "B17_hidden_test_submission_inference",
        "variant": B17_VARIANT,
        "checkpoint": str(Path(checkpoint).resolve()),
        "completed_epochs": int(payload.get("completed_epochs", -1)),
        "encoder_frozen": bool(payload.get("encoder_frozen")),
        "encoder_sha256": payload.get("encoder_sha256_final"),
        "gold_labels_used_in_gradient": False,
        "gold_labels_used_for_early_stopping": False,
        "gold_checkpoint_selection": False,
        "test_rows": int(len(frame)),
        "test_series_total": int(sum(counts)),
        "test_series_min": int(min(counts)),
        "test_series_median": float(np.median(counts)),
        "test_series_max": int(max(counts)),
        "tta_center_offsets": list(offsets),
        "metadata_repair": metadata_stats,
        "runtime_elapsed_hours": float(budget.elapsed_seconds / 3600.0),
        "runtime_budget_hours": float(budget.max_hours),
        **sample_validation,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)
    print(manifest_path)
    print(json.dumps(manifest, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b17-submit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    generate_b17_submission(config, checkpoint=args.checkpoint, out_path=args.out)


if __name__ == "__main__":
    main()
