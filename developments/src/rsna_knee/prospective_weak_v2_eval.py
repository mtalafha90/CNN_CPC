"""Evaluate the predeclared PV2 B34 training-scaffold hypothesis.

Primary mechanistic test:
    B34 - B29
Both have the same inference-time complementary-query form; only B34 had B31's
local-context scaffold during training.  Negative weighted-BCE difference favors
B34 and supports a training-path benefit.

Inference simplification replication:
    B34 - B31
The training paths are matched; B34 bypasses local context exactly at eval while
B31 retains it.  A predeclared absolute equivalence margin of 0.001 macro weighted
BCE is used.  The entire paired 95% interval must lie in [-0.001,+0.001] to claim
metric-resolution equivalence.

Reference mechanism comparison:
    B31 - B29
This checks whether the B31 training-context advantage itself replicates on PV2.

PV2 remains weak-label internal evidence with historical downstream exposure.
The original PV1 validation set is never used here.
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

from .b7_weak_supervision import _read_config, load_frozen_b6_export, make_b7_dataset_config, prepare_b7_supervision
from .b12_variable_series import audit_variable_series_surface, build_variable_series_index, collate_variable_series
from .b12_1_gold_eval import predict_b12_1
from .b13_training import B13_SERIES_SIGNATURE
from .b20_crop_focus import CropFocusedVariableSeriesKneeDataset, require_b20_contract
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .prospective_weak_v1 import validate_prospective_weak_v1_manifest
from .prospective_weak_v1_eval import (
    PV1_EVAL_BATCH_SIZE, PV1_EVAL_NUM_WORKERS, PV1_EVAL_PERSISTENT_WORKERS,
    PV1_EVAL_PREFETCH_FACTOR, PV1_EVAL_SERIES_CACHE_MB, low_memory_eval_config,
    macro_weighted_soft_bce, paired_bootstrap_loss_difference, weak_state_auc,
)
from .prospective_weak_v2 import (
    PV2_PARENT_PV1_SPLIT_SHA256, PV2_VALIDATION_STUDIES, PV2_VERSION,
    validate_prospective_weak_v2_manifest,
)
from .prospective_weak_v2_b29_training import load_pv2_b29_checkpoint
from .prospective_weak_v2_training import load_prospective_weak_v2_checkpoint
from .runtime import resolve_runtime

PV2_EVAL_VERSION = "1.0.0"
PV2_B34_EQUIVALENCE_MARGIN = 0.001
PV2_PREDECLARED_COMPARISONS = (
    "b34_minus_b29",
    "b34_minus_b31",
    "b31_minus_b29",
)


def _release_unused_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _subset(full_uids, targets, weights, subset_uids):
    row = {str(uid): i for i, uid in enumerate(full_uids)}
    idx = np.asarray([row[str(uid)] for uid in subset_uids], dtype=np.int64)
    return targets[idx], weights[idx]


def _write_predictions(out: Path, name: str, uids: list[str], pred: np.ndarray, split_sha: str, encoder_sha: str):
    frame = pd.DataFrame(pred, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    frame.to_csv(out / f"{name}_predictions.csv", index=False)
    (out / f"{name}_prediction_meta.json").write_text(json.dumps({
        "evaluation_version": PV2_EVAL_VERSION,
        "validation_framework": PV2_VERSION,
        "model_name": name,
        "split_sha256": split_sha,
        "encoder_sha256": encoder_sha,
        "validation_studies": len(uids),
        "prediction_shape": list(pred.shape),
    }, indent=2), encoding="utf-8")


def evaluate_prospective_weak_v2(
    config: dict,
    *,
    split_manifest_path: str | Path,
    parent_pv1_manifest_path: str | Path,
    b6_root: str | Path,
    b29_checkpoint: str | Path,
    b31_checkpoint: str | Path,
    b34_checkpoint: str | Path,
    out_root: str | Path = "runs/prospective_weak_v2/eval",
    n_bootstrap: int = 5000,
) -> dict:
    crop_policy = require_b20_contract(config)
    eval_config = low_memory_eval_config(config)
    runtime = resolve_runtime(eval_config)
    print(runtime.describe())
    print(
        "[PV2 eval] sequential low-memory evaluation | "
        f"batch={PV1_EVAL_BATCH_SIZE} workers={PV1_EVAL_NUM_WORKERS} "
        f"prefetch={PV1_EVAL_PREFETCH_FACTOR} persistent={PV1_EVAL_PERSISTENT_WORKERS} "
        f"cache_mb={PV1_EVAL_SERIES_CACHE_MB}"
    )

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, _ = load_frozen_b6_export(b6_root)
    full_uids, full_targets, full_weights, _ = prepare_b7_supervision(train, b6_frame)
    full_uids = [str(x) for x in full_uids]
    parent = json.loads(Path(parent_pv1_manifest_path).read_text(encoding="utf-8"))
    validate_prospective_weak_v1_manifest(parent, full_uids)
    if str(parent.get("split_sha256", "")) != PV2_PARENT_PV1_SPLIT_SHA256:
        raise ValueError("PV2 eval requires exact frozen parent PV1 split")
    split = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    validate_prospective_weak_v2_manifest(split, parent, full_uids)
    val_uids = [str(x) for x in split["validation_uids"]]
    if len(val_uids) != PV2_VALIDATION_STUDIES:
        raise RuntimeError("PV2 validation count changed")
    val_targets, val_weights = _subset(full_uids, full_targets, full_weights, val_uids)
    expected_audit = split["post_assignment_supervision_audit"]["validation"]
    if int((val_weights > 0).sum()) != int(expected_audit["usable_cells"]):
        raise RuntimeError("PV2 validation supervision changed")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    full_summary, _ = audit_variable_series_surface(series, full_uids)
    if full_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("PV2 full series surface no longer matches frozen B13 signature")
    val_index = build_variable_series_index(series, val_uids)
    val_series = int(sum(len(val_index.get(uid, [])) for uid in val_uids))
    if val_series <= 0 or any(len(val_index.get(uid, [])) == 0 for uid in val_uids):
        raise RuntimeError("PV2 validation contains study with no eligible MRI series")

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("PV2 freezes TTA at [-1,0,1]")
    ds = CropFocusedVariableSeriesKneeDataset(
        val_uids, val_index, make_b7_dataset_config(eval_config, root, train=False, tta_offsets=offsets),
        targets=val_targets, weights=val_weights, train=False, crop_focus_policy=crop_policy,
    )
    loader = DataLoader(
        ds, batch_size=PV1_EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 35_100_000),
    )

    checkpoints = {"b29": b29_checkpoint, "b31": b31_checkpoint, "b34": b34_checkpoint}
    split_sha = str(split["split_sha256"])
    out = Path(out_root); out.mkdir(parents=True, exist_ok=True)
    predictions = {}
    metrics = {}
    encoder_shas = set()
    b34_scaffold_state = None

    for name in ("b29", "b31", "b34"):
        print(f"[PV2 eval] loading {name}: {checkpoints[name]}")
        if name == "b29":
            model, payload = load_pv2_b29_checkpoint(
                checkpoints[name], expected_split_sha256=split_sha, device=runtime.device
            )
        else:
            model, payload = load_prospective_weak_v2_checkpoint(
                checkpoints[name], expected_split_sha256=split_sha, device=runtime.device
            )
        if str(payload.get("model_name", "")) != name:
            raise ValueError(f"PV2 checkpoint/model mismatch for {name}")
        if payload.get("crop_focus_policy") != crop_policy:
            raise ValueError(f"PV2 crop policy mismatch for {name}")
        encoder_sha = str(payload.get("encoder_sha256_initial", ""))
        if not encoder_sha:
            raise RuntimeError(f"PV2 {name} missing encoder fingerprint")
        encoder_shas.add(encoder_sha)
        if len(encoder_shas) > 1:
            raise RuntimeError("PV2 controls do not share one frozen encoder")

        model.eval()
        if name == "b34":
            state = model.b34_state()
            if state.get("training_context_active") is not False:
                raise RuntimeError("B34 evaluation did not deactivate training scaffold")
            if state.get("eval_context_exact_bypass") is not True:
                raise RuntimeError("B34 exact eval bypass contract missing")
            if int(state.get("inference_context_parameters_used", -1)) != 0:
                raise RuntimeError("B34 unexpectedly uses context parameters at inference")
            b34_scaffold_state = state

        print(f"[PV2 eval] predicting {name}")
        pred_uids, pred = predict_b12_1(model, loader, runtime)
        if [str(x) for x in pred_uids] != val_uids:
            raise RuntimeError(f"PV2 {name} prediction order changed")
        pred = np.asarray(pred, dtype=np.float32)
        if pred.shape != (len(val_uids), len(TARGETS)) or not np.isfinite(pred).all():
            raise RuntimeError(f"PV2 {name} predictions invalid")
        predictions[name] = pred
        metrics[name] = {
            "primary": macro_weighted_soft_bce(val_targets, val_weights, pred),
            "secondary": weak_state_auc(val_targets, val_weights, pred),
        }
        _write_predictions(out, name, val_uids, pred, split_sha, encoder_sha)
        print(f"[PV2 eval] {name} BCE={metrics[name]['primary']['macro_weighted_soft_bce']:.10f}")
        del pred_uids, model, payload
        _release_unused_memory()

    seed = int(config.get("seed", 2026))
    paired = {
        "b34_minus_b29": paired_bootstrap_loss_difference(
            val_targets, val_weights, predictions["b29"], predictions["b34"],
            n_bootstrap=n_bootstrap, seed=seed + 35_201,
        ),
        "b34_minus_b31": paired_bootstrap_loss_difference(
            val_targets, val_weights, predictions["b31"], predictions["b34"],
            n_bootstrap=n_bootstrap, seed=seed + 35_202,
        ),
        "b31_minus_b29": paired_bootstrap_loss_difference(
            val_targets, val_weights, predictions["b29"], predictions["b31"],
            n_bootstrap=n_bootstrap, seed=seed + 35_203,
        ),
    }

    scaffold_cmp = paired["b34_minus_b29"]
    bypass_cmp = paired["b34_minus_b31"]
    if scaffold_cmp["ci_upper"] < 0.0:
        scaffold_interpretation = "training_scaffold_benefit_supported"
    elif scaffold_cmp["ci_lower"] > 0.0:
        scaffold_interpretation = "training_scaffold_hypothesis_rejected"
    else:
        scaffold_interpretation = "training_scaffold_effect_unresolved"

    margin = PV2_B34_EQUIVALENCE_MARGIN
    bypass_equivalent = bool(
        bypass_cmp["ci_lower"] >= -margin and bypass_cmp["ci_upper"] <= margin
    )
    mechanism_success = bool(scaffold_cmp["ci_upper"] < 0.0 and bypass_equivalent)

    data = {"StudyInstanceUID": val_uids}
    for j, target in enumerate(TARGETS):
        data[f"{target}__target"] = val_targets[:, j]
        data[f"{target}__weight"] = val_weights[:, j]
        for name in ("b29", "b31", "b34"):
            data[f"{target}__{name}"] = predictions[name][:, j]
    pd.DataFrame(data).to_csv(out / "paired_predictions.csv", index=False)

    result = {
        "evaluation_version": PV2_EVAL_VERSION,
        "validation_framework": PV2_VERSION,
        "surface": "nested PV2 499-study weak-label metric surface from the old PV1 training partition",
        "split_sha256": split_sha,
        "parent_pv1_split_sha256": split["parent_pv1_split_sha256"],
        "training_studies_per_control": int(split["training_studies"]),
        "validation_studies": len(val_uids),
        "validation_series": val_series,
        "locked_parent_pv1_validation_studies": int(split["locked_parent_pv1_validation_studies"]),
        "locked_parent_pv1_validation_used": False,
        "independent_expert_validation": False,
        "historically_downstream_unseen": False,
        "weak_label_validation": True,
        "expert_labels_read": False,
        "exposure_note": split["exposure_note"],
        "primary_metric": "macro of per-target B6-weighted soft-label BCE; lower is better",
        "secondary_metric": "macro AUC over B6 positive/negated states where both classes are present",
        "predeclared_global_comparisons": list(PV2_PREDECLARED_COMPARISONS),
        "primary_b34_mechanism_test": "b34_minus_b29",
        "inference_bypass_replication_test": "b34_minus_b31",
        "b34_b31_equivalence_margin_macro_bce": margin,
        "training_scaffold_interpretation": scaffold_interpretation,
        "b34_b31_equivalent_within_predeclared_margin": bypass_equivalent,
        "b34_mechanism_success": mechanism_success,
        "tta_offsets": list(offsets),
        "memory_policy": {
            "sequential_model_loading": True,
            "models_resident_simultaneously": 1,
            "eval_batch_size": PV1_EVAL_BATCH_SIZE,
            "num_workers": PV1_EVAL_NUM_WORKERS,
            "prefetch_factor": PV1_EVAL_PREFETCH_FACTOR,
            "persistent_workers": PV1_EVAL_PERSISTENT_WORKERS,
            "series_cache_mb_per_worker": PV1_EVAL_SERIES_CACHE_MB,
        },
        "metrics": metrics,
        "paired_primary_loss_bootstrap": paired,
        "b34_scaffold_state_at_eval": b34_scaffold_state,
        "encoder_sha256": next(iter(encoder_shas)),
        "metadata_repair": metadata_stats,
        "governance": (
            "PV2 is an internal post-PV1 mechanism test with historical downstream exposure. "
            "Do not use per-target outcomes for switching, blending, scaffold masks, kernel retuning, or B34.1. "
            "The original PV1 result remains the stronger prospective architecture-selection record, and "
            "independent hidden/external evidence remains required for active-model promotion."
        ),
    }
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("[PV2 eval] complete")
    print(json.dumps({
        "B29_BCE": metrics["b29"]["primary"]["macro_weighted_soft_bce"],
        "B31_BCE": metrics["b31"]["primary"]["macro_weighted_soft_bce"],
        "B34_BCE": metrics["b34"]["primary"]["macro_weighted_soft_bce"],
        "scaffold_interpretation": scaffold_interpretation,
        "b34_b31_equivalent": bypass_equivalent,
        "mechanism_success": mechanism_success,
    }, indent=2))
    print(out / "comparison.json")
    return result


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate matched PV2 B29/B31/B34 controls")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--parent-pv1-manifest", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--b29-checkpoint", required=True)
    ap.add_argument("--b31-checkpoint", required=True)
    ap.add_argument("--b34-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/prospective_weak_v2/eval")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()
    config = dict(_read_config(args.config)); config["data_root"] = str(Path(args.data_root).resolve())
    evaluate_prospective_weak_v2(
        config, split_manifest_path=args.split_manifest,
        parent_pv1_manifest_path=args.parent_pv1_manifest, b6_root=args.b6_root,
        b29_checkpoint=args.b29_checkpoint, b31_checkpoint=args.b31_checkpoint,
        b34_checkpoint=args.b34_checkpoint, out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
