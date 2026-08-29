"""Candidate-only exploratory Kaggle inference for the completed B49 run.

This module deliberately does not select, blend, calibrate, or otherwise alter
the B49 result.  It reconstructs the fixed post-cross-attention candidate,
verifies its report-only training provenance, then writes raw sigmoid
probabilities in the competition submission format.  Any score obtained after
upload is exploratory hidden-test evidence, not a B49 promotion decision.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import build_variable_series_index
from .b17_submission import _validate_sample_submission, _validate_submission
from .b35_training import sha256_file
from .b37_highres_sparse_eval import B37_EVAL_OFFSETS
from .b48_global_conditioned_sparse_training import (
    _indices_for_split,
    _report_only_surface,
    b48_fill_artifacts,
    load_b48_domain_split,
)
from .b49_native_tiled_multiscale_eval import load_b49_checkpoint
from .b49_native_tiled_multiscale_mil import (
    B49_EXPERIMENT,
    B49_POST_CROSS_ATTENTION_CANDIDATE,
    B49_VERSION,
    B49NativeTiledFullFOVDataset,
    collate_b49,
    require_b49_contract,
)
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_test_csv
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import autocast, resolve_runtime


B49_SUBMISSION_EXPERIMENT = "B49_candidate_only_exploratory_hidden_test_inference"
B49_SUBMISSION_VERSION = "b49_candidate_only_raw_probability_submission_v1"
B49_SUBMISSION_TTA_OFFSETS = B37_EVAL_OFFSETS
B49_SUBMISSION_LOADER_SEED_OFFSET = 59_100_000


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def require_b49_candidate_submission_contract(config: dict) -> tuple[int, ...]:
    """Reject changes to B49 candidate-only inference semantics."""
    require_b49_contract(config, arm=B49_POST_CROSS_ATTENTION_CANDIDATE)
    offsets = tuple(
        int(value)
        for value in config.get("b7_eval_tta_offsets", B49_SUBMISSION_TTA_OFFSETS)
    )
    if offsets != B49_SUBMISSION_TTA_OFFSETS:
        raise ValueError(
            "B49 candidate submission freezes b7_eval_tta_offsets="
            f"{list(B49_SUBMISSION_TTA_OFFSETS)}"
        )
    if int(config.get("b7_eval_batch_size", 1)) != 1:
        raise ValueError("B49 candidate submission requires b7_eval_batch_size=1")
    if int(config.get("num_workers", 0)) != 0:
        raise ValueError("B49 candidate submission requires num_workers=0")
    if bool(config.get("pin_memory", False)):
        raise ValueError("B49 candidate submission requires pin_memory=false")
    if int(config.get("series_cache_mb_per_worker", 0)) != 0:
        raise ValueError("B49 candidate submission requires series_cache_mb_per_worker=0")
    if bool(config.get("strict_dicom_inference", True)) is not True:
        raise ValueError("B49 candidate submission requires strict_dicom_inference=true")
    return offsets


def _verified_candidate(
    *,
    settings: dict,
    root: Path,
    labels_root: str | Path,
    base_checkpoint: str | Path,
    domain_split: str | Path,
    candidate_checkpoint: str | Path,
    device,
):
    """Reconstruct the candidate with the same provenance checks as B49 eval."""
    domain_payload, domain_rows, domain_meta = load_b48_domain_split(domain_split)
    train_path = root / settings.get("train_csv", "train.csv")
    expected_train_sha = str(domain_payload.get("source_train_csv_sha256", ""))
    if not expected_train_sha or sha256_file(train_path) != expected_train_sha:
        raise ValueError("B49 candidate submission train.csv fingerprint mismatch")

    base_path = Path(base_checkpoint).resolve()
    base_probe, base_payload = load_phase9_checkpoint(
        base_path, expected_arm="llm_fill", device="cpu"
    )
    del base_probe
    (
        _train,
        all_uids,
        all_targets,
        all_weights,
        _lookup,
        _confidence,
        _fill_policy,
        _fill_audit,
        _supervision,
    ) = _report_only_surface(
        data_root=root,
        labels_root=labels_root,
        config=settings,
        domain_rows=domain_rows,
        base_payload=base_payload,
    )
    del all_targets, all_weights
    train_indices = _indices_for_split(all_uids, domain_rows, "train")
    train_uids = [all_uids[index] for index in train_indices]
    model, payload = load_b49_checkpoint(
        candidate_checkpoint,
        config=settings,
        base_checkpoint=base_path,
        arm=B49_POST_CROSS_ATTENTION_CANDIDATE,
        domain_sha256=domain_meta["sha256"],
        domain_rows_sha256=domain_meta["rows_sha256"],
        training_uids=train_uids,
        fill_artifacts=b48_fill_artifacts(labels_root),
        device=device,
    )
    if payload.get("experiment") != B49_EXPERIMENT or payload.get("version") != B49_VERSION:
        raise ValueError("B49 candidate submission checkpoint identity changed")
    return model, payload, domain_meta


@torch.no_grad()
def generate_b49_candidate_submission(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    base_checkpoint: str | Path,
    domain_split: str | Path,
    candidate_checkpoint: str | Path,
    out_path: str | Path = "submission_b49_candidate.csv",
    manifest_path: str | Path | None = None,
) -> Path:
    """Write raw candidate probabilities in Kaggle's required row order."""
    settings = dict(config)
    root = Path(data_root).resolve()
    settings["data_root"] = str(root)
    offsets = require_b49_candidate_submission_contract(settings)
    runtime = resolve_runtime(settings)
    print(runtime.describe(), flush=True)

    model, payload, domain_meta = _verified_candidate(
        settings=settings,
        root=root,
        labels_root=labels_root,
        base_checkpoint=base_checkpoint,
        domain_split=domain_split,
        candidate_checkpoint=candidate_checkpoint,
        device=runtime.device,
    )
    model.eval()

    test = load_test_csv(root / settings.get("test_csv", "test.csv"))
    uids = test["StudyInstanceUID"].astype(str).tolist()
    if not uids:
        raise ValueError("B49 candidate submission test.csv has no studies")
    series = load_series_csv(root / settings.get("test_series_csv", "test_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="test")
    test_index = build_variable_series_index(series, uids)
    missing = [uid for uid in uids if not test_index.get(uid)]
    if missing:
        raise ValueError(
            f"B49 candidate submission found {len(missing)} test study/studies with no eligible MRI series"
        )

    dataset_config = make_b7_dataset_config(settings, root, train=False)
    dataset_config.split = "test"
    dataset_config.tta_center_offsets = ()
    dataset = B49NativeTiledFullFOVDataset(
        uids,
        test_index,
        dataset_config,
        center_offsets=offsets,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_b49,
        **runtime.loader_kwargs(seed=int(payload["seed"]) + B49_SUBMISSION_LOADER_SEED_OFFSET),
    )

    probabilities: list[np.ndarray] = []
    observed_uids: list[str] = []
    for batch_index, items in enumerate(loader, start=1):
        if len(items) != 1:
            raise RuntimeError("B49 candidate submission requires one ragged study per batch")
        item = items[0]
        present = item["present"].to(runtime.device, non_blocking=True)
        meta = item["series_meta"].to(runtime.device, non_blocking=True)
        if len(item["views"]) != len(offsets):
            raise RuntimeError("B49 candidate submission TTA view count changed")
        views: list[torch.Tensor] = []
        started = time.monotonic()
        for view in item["views"]:
            context = [volume.to(runtime.device, non_blocking=True) for volume in view["context_volumes"]]
            position = view["slice_position"].to(runtime.device, non_blocking=True)
            with autocast(runtime):
                output = model(context, view["local_sources"], present, meta, position)
            views.append(torch.sigmoid(output.logits.float()).cpu())
            del context, position, output
        probability = torch.stack(views, dim=0).mean(dim=0)
        if tuple(probability.shape) != (1, len(TARGETS)) or not torch.isfinite(probability).all():
            raise RuntimeError("B49 candidate submission produced invalid probabilities")
        observed_uids.append(str(item["study_uid"]))
        probabilities.append(probability.numpy()[0])
        del item, items, present, meta, views, probability
        _release()
        if batch_index % 10 == 0 or batch_index == len(loader):
            print(
                f"[B49 candidate submission] {batch_index}/{len(loader)} "
                f"last_study={time.monotonic() - started:.1f}s",
                flush=True,
            )

    if observed_uids != uids:
        raise RuntimeError("B49 candidate submission changed test.csv StudyInstanceUID order")
    frame = pd.DataFrame(np.asarray(probabilities, dtype=np.float64), columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", observed_uids)
    _validate_submission(frame, uids)
    sample_validation = _validate_sample_submission(root, frame)

    output = Path(out_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    manifest = {
        "submission_role": "exploratory_candidate_only_posthoc_hidden_test_inference_not_promotion_evidence",
        "experiment": B49_SUBMISSION_EXPERIMENT,
        "version": B49_SUBMISSION_VERSION,
        "candidate_arm": B49_POST_CROSS_ATTENTION_CANDIDATE,
        "candidate_checkpoint": str(Path(candidate_checkpoint).resolve()),
        "candidate_checkpoint_sha256": sha256_file(candidate_checkpoint),
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "base_checkpoint_sha256": str(payload["base_checkpoint_sha256"]),
        "domain_split_sha256": domain_meta["sha256"],
        "tta_offsets": list(offsets),
        "prediction_policy": "raw_sigmoid_probabilities; no_thresholding; no_calibration; no_blending",
        "test_studies": len(uids),
        "metadata_repair": metadata_stats,
        "sample_submission_validation": sample_validation,
        "submission_path": str(output),
        "submission_sha256": sha256_file(output),
    }
    manifest_output = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else output.with_suffix(".manifest.json")
    )
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output, flush=True)
    print(manifest_output, flush=True)
    print(f"submission_sha256 {manifest['submission_sha256']}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser("Write an exploratory B49 candidate-only Kaggle submission")
    parser.add_argument("--config", default="config/b49_native_tiled_multiscale.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--domain-split", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--out", default="submission_b49_candidate.csv")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    generate_b49_candidate_submission(
        dict(_read_config(args.config)),
        data_root=args.data_root,
        labels_root=args.labels_root,
        base_checkpoint=args.base_checkpoint,
        domain_split=args.domain_split,
        candidate_checkpoint=args.candidate_checkpoint,
        out_path=args.out,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B49_SUBMISSION_EXPERIMENT",
    "B49_SUBMISSION_TTA_OFFSETS",
    "generate_b49_candidate_submission",
    "require_b49_candidate_submission_contract",
]
