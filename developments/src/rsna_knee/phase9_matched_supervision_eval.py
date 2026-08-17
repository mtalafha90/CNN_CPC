"""Paired diagnostic evaluation for the frozen Phase-9 supervision experiment.

The repeatedly reused 58-study expert surface is diagnostic only.  It is not a
promotion gate and is not independent validation.  The control and candidate are
both fixed-E2 B34 models differing only in their training supervision table.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import build_variable_series_index, collate_variable_series
from .b12_1_gold_eval import predict_b12_1
from .b18_fisher_selection import B18_EXPECTED_GOLD_SERIES, B18_EXPECTED_GOLD_STUDIES
from .b20_crop_focus import CropFocusedVariableSeriesKneeDataset, require_b20_contract
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import bootstrap_macro_auc, compare_runs
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .phase9_supervision import PHASE9_VERSION
from .prospective_weak_v1_eval import (
    PV1_EVAL_BATCH_SIZE,
    PV1_EVAL_NUM_WORKERS,
    PV1_EVAL_PERSISTENT_WORKERS,
    PV1_EVAL_PREFETCH_FACTOR,
    PV1_EVAL_SERIES_CACHE_MB,
    low_memory_eval_config,
)
from .runtime import resolve_runtime

PHASE9_EVAL_VERSION = "1.0.0"


def _release_unused_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _write_predictions(out: Path, name: str, uids: list[str], prediction: np.ndarray, encoder_sha: str) -> None:
    frame = pd.DataFrame(prediction, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    frame.to_csv(out / f"{name}_gold_predictions.csv", index=False)
    (out / f"{name}_prediction_meta.json").write_text(
        json.dumps(
            {
                "evaluation_version": PHASE9_EVAL_VERSION,
                "phase9_version": PHASE9_VERSION,
                "arm": name,
                "encoder_sha256": encoder_sha,
                "gold_studies": len(uids),
                "prediction_shape": list(prediction.shape),
                "independent_validation": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def evaluate_phase9(
    config: dict,
    *,
    control_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    out_root: str | Path = "runs/phase9_matched_supervision/eval",
    n_bootstrap: int = 5000,
) -> dict:
    crop_policy = require_b20_contract(config)
    eval_config = low_memory_eval_config(config)
    runtime = resolve_runtime(eval_config)
    print(runtime.describe())

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != B18_EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("Phase 9 requires the complete reused 58-study expert diagnostic surface")
    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(np.float64)

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_variable_series_index(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts):
        raise RuntimeError("Phase-9 gold diagnostic contains a study with no eligible MRI series")
    if int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise RuntimeError(
            f"Phase 9 expected {B18_EXPECTED_GOLD_SERIES} gold MRI series, got {sum(counts)}"
        )

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("Phase 9 freezes TTA at [-1,0,1]")
    ds = CropFocusedVariableSeriesKneeDataset(
        uids,
        index,
        make_b7_dataset_config(eval_config, root, train=False, tta_offsets=offsets),
        targets=gold[TARGETS].to_numpy(np.float32),
        train=False,
        crop_focus_policy=crop_policy,
    )
    loader = DataLoader(
        ds,
        batch_size=PV1_EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 40_300_000),
    )

    checkpoints = {"control": control_checkpoint, "candidate": candidate_checkpoint}
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    encoder_shas: set[str] = set()
    model_specs: list[dict] = []
    matched_contract: list[dict] = []

    for arm in ("control", "candidate"):
        print(f"[Phase9 eval] loading {arm}: {checkpoints[arm]}")
        model, payload = load_phase9_checkpoint(checkpoints[arm], expected_arm=arm, device=runtime.device)
        if payload.get("crop_focus_policy") != crop_policy:
            raise ValueError(f"Phase-9 {arm} crop policy mismatch")
        encoder_sha = str(payload.get("encoder_sha256_initial", ""))
        encoder_shas.add(encoder_sha)
        if len(encoder_shas) > 1:
            raise RuntimeError("Phase-9 arms do not share the exact same frozen B16 encoder")
        model_specs.append(payload.get("model_spec"))
        matched_contract.append(
            {
                "arm": arm,
                "construction_seed": payload.get("construction_seed"),
                "loader_seed": payload.get("loader_seed"),
                "post_seed": payload.get("post_construction_training_seed"),
                "training_series": payload.get("training_series"),
                "report_only_studies_exposed": payload.get("report_only_studies_exposed"),
                "completed_epochs": payload.get("completed_epochs"),
            }
        )

        model.eval()
        state = model.b34_state()
        if state.get("training_context_active") is not False:
            raise RuntimeError("Phase-9 B34 diagnostic did not deactivate the training scaffold")
        if state.get("eval_context_exact_bypass") is not True:
            raise RuntimeError("Phase-9 B34 exact evaluation bypass contract missing")
        if int(state.get("inference_context_parameters_used", -1)) != 0:
            raise RuntimeError("Phase-9 B34 unexpectedly uses scaffold parameters at inference")

        pred_uids, pred = predict_b12_1(model, loader, runtime)
        if [str(x) for x in pred_uids] != uids:
            raise RuntimeError(f"Phase-9 {arm} gold prediction order changed")
        pred = np.asarray(pred, dtype=np.float32)
        if pred.shape != truth.shape or not np.isfinite(pred).all():
            raise RuntimeError(f"Phase-9 {arm} predictions invalid")
        predictions[arm] = pred
        metrics[arm] = bootstrap_macro_auc(
            truth,
            pred,
            n_bootstrap=n_bootstrap,
            seed=int(config.get("seed", 2026)) + (40_301 if arm == "control" else 40_302),
        ).to_dict()
        _write_predictions(out, arm, uids, pred, encoder_sha)
        print(f"[Phase9 eval] {arm} macro AUC={metrics[arm]['macro_auc']:.10f}")
        del model, payload, pred_uids
        _release_unused_memory()

    if model_specs[0] != model_specs[1]:
        raise RuntimeError("Phase-9 arms do not share the exact same B34 model specification")
    left, right = matched_contract
    for key in (
        "construction_seed",
        "loader_seed",
        "post_seed",
        "training_series",
        "report_only_studies_exposed",
        "completed_epochs",
    ):
        if left[key] != right[key]:
            raise RuntimeError(f"Phase-9 matched contract differs between arms for {key}")

    paired = compare_runs(
        truth,
        predictions["control"],
        predictions["candidate"],
        n_bootstrap=n_bootstrap,
        seed=int(config.get("seed", 2026)) + 40_303,
    )
    data = {"StudyInstanceUID": uids}
    for j, target in enumerate(TARGETS):
        data[f"{target}__truth"] = truth[:, j]
        data[f"{target}__control"] = predictions["control"][:, j]
        data[f"{target}__candidate"] = predictions["candidate"][:, j]
    pd.DataFrame(data).to_csv(out / "paired_gold_predictions.csv", index=False)

    result = {
        "evaluation_version": PHASE9_EVAL_VERSION,
        "phase9_version": PHASE9_VERSION,
        "surface": "reused 58-study expert development diagnostic",
        "independent_validation": False,
        "promotion_gate": False,
        "gold_studies": len(uids),
        "gold_series": int(sum(counts)),
        "gold_labels_used_in_training": False,
        "tta_offsets": list(offsets),
        "primary_diagnostic_metric": "macro ROC AUC across 12 expert-labelled targets",
        "paired_difference_definition": "candidate macro AUC minus control macro AUC; positive favors Phase-8 supervision",
        "metrics": metrics,
        "paired_macro_auc_bootstrap": paired,
        "matched_training_contract": matched_contract,
        "encoder_sha256": next(iter(encoder_shas)),
        "model_spec_identical": True,
        "metadata_repair": metadata_stats,
        "memory_policy": {
            "sequential_model_loading": True,
            "models_resident_simultaneously": 1,
            "eval_batch_size": PV1_EVAL_BATCH_SIZE,
            "num_workers": PV1_EVAL_NUM_WORKERS,
            "prefetch_factor": PV1_EVAL_PREFETCH_FACTOR,
            "persistent_workers": PV1_EVAL_PERSISTENT_WORKERS,
            "series_cache_mb_per_worker": PV1_EVAL_SERIES_CACHE_MB,
        },
        "governance": (
            "This 58-study surface has been repeatedly reused and is diagnostic only. "
            "Do not filter Phase-8 targets/scripts, retune B34, or promote a model from this result. "
            "Independent hidden competition or new external expert-labelled evidence remains required."
        ),
    }
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "control_macro_auc": metrics["control"]["macro_auc"],
        "candidate_macro_auc": metrics["candidate"]["macro_auc"],
        "paired_candidate_minus_control": paired,
    }, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate frozen Phase-9 matched supervision arms")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--control-checkpoint", required=True)
    ap.add_argument("--candidate-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/phase9_matched_supervision/eval")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    evaluate_phase9(
        config,
        control_checkpoint=args.control_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
