"""Reused Expert-58 diagnostic comparing B40’s fixed E3 endpoint to B37 E2.

This module never chooses an epoch or changes either endpoint. It reports the
historical 224 base replay, frozen B37 E2, and B40 E3 through the same fixed
three-offset 448 sparse-MIL evaluation path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config
from .b12_1_gold_eval import predict_b12_1
from .b12_variable_series import build_variable_series_index
from .b17_training import encoder_state_sha256
from .b18_fisher_selection import B18_EXPECTED_GOLD_SERIES, B18_EXPECTED_GOLD_STUDIES
from .b35_training import sha256_file
from .b37_highres_sparse_eval import (
    B37_BASE_TOLERANCE,
    B37_EVAL_OFFSETS,
    B37_EXPECTED_BASE_MACRO,
    FOCAL_SIX,
    _base_loader,
    _candidate_loader,
    _release,
    load_b37_checkpoint,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import compare_runs, macro_auc_from_arrays
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import autocast
from .b40_b37_e2_continuation import (
    B40_EXPERIMENT,
    B40_RUN_ROOT,
    B40_VERSION,
    load_b40_checkpoint,
    require_b40_continuation_contract,
)


def _target_rows(truth, base, b37, b40) -> tuple[float, float, float, dict]:
    base_macro, base_auc = macro_auc_from_arrays(truth, base)
    b37_macro, b37_auc = macro_auc_from_arrays(truth, b37)
    b40_macro, b40_auc = macro_auc_from_arrays(truth, b40)
    rows = {
        target: {
            "base_224_auc": float(base_auc[index]),
            "b37_e2_auc": float(b37_auc[index]),
            "b40_e3_auc": float(b40_auc[index]),
            "b40_minus_b37": float(b40_auc[index] - b37_auc[index]),
            "b40_minus_base": float(b40_auc[index] - base_auc[index]),
        }
        for index, target in enumerate(TARGETS)
    }
    return float(base_macro), float(b37_macro), float(b40_macro), rows


def _global_macro_auc(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Return only the scalar macro AUC for a global-logit prediction matrix."""
    macro_auc, _ = macro_auc_from_arrays(truth, prediction)
    return float(macro_auc)


@torch.no_grad()
def evaluate_b40(
    config: dict,
    *,
    data_root: str | Path,
    parent_checkpoint: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path = f"{B40_RUN_ROOT}/expert58",
    n_bootstrap: int = 5000,
) -> dict:
    """Compare fixed B40 E3 to its immutable B37 E2 parent on Expert-58."""
    config = dict(config)
    root = Path(data_root).resolve()
    parent_path = Path(parent_checkpoint).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    base_path = Path(base_checkpoint).resolve()
    config["data_root"] = str(root)
    require_b40_continuation_contract(config)

    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != B18_EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("B40 requires the complete reused 58-study expert surface")
    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(np.float64)

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_variable_series_index(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts) or int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise ValueError("B40 expert MRI series surface changed")

    base_loader, base_runtime = _base_loader(config, root, uids, index, truth)
    print(base_runtime.describe(), flush=True)
    base_model, _ = load_phase9_checkpoint(base_path, expected_arm="llm_fill", device=base_runtime.device)
    base_model.eval()
    base_uids, base_prediction = predict_b12_1(base_model, base_loader, base_runtime)
    if [str(uid) for uid in base_uids] != uids:
        raise RuntimeError("B40 historical base study order changed")
    base_prediction = np.asarray(base_prediction, dtype=np.float32)
    del base_model, base_loader
    _release()

    candidate_loader, runtime = _candidate_loader(config, root, uids, index, truth)
    print(runtime.describe(), flush=True)
    b37_model, b37_payload = load_b37_checkpoint(
        parent_path,
        base_checkpoint=base_path,
        device=runtime.device,
    )
    b40_model, b40_payload = load_b40_checkpoint(
        checkpoint_path,
        base_checkpoint=base_path,
        device=runtime.device,
    )
    if sha256_file(parent_path) != str(b40_payload.get("parent_b37_checkpoint_sha256", "")):
        raise ValueError("B40 checkpoint’s B37 parent fingerprint does not match the supplied parent")
    if b40_payload.get("experiment") != B40_EXPERIMENT or b40_payload.get("version") != B40_VERSION:
        raise ValueError("B40 payload identity changed during load")
    b37_model.eval()
    b40_model.eval()

    b37_global_blocks: list[np.ndarray] = []
    b37_combined_blocks: list[np.ndarray] = []
    b40_global_blocks: list[np.ndarray] = []
    b40_combined_blocks: list[np.ndarray] = []
    scored_uids: list[str] = []
    for batch in candidate_loader:
        volumes = batch["volumes"].to(runtime.device, non_blocking=True)
        position = batch["slice_position"].to(runtime.device, non_blocking=True)
        present = batch["present"].to(runtime.device, non_blocking=True)
        meta = batch["series_meta"].to(runtime.device, non_blocking=True)
        if volumes.ndim != 7 or int(volumes.shape[1]) != len(B37_EVAL_OFFSETS):
            raise RuntimeError("B40 evaluation TTA view shape changed")
        b37_global_views, b37_combined_views = [], []
        b40_global_views, b40_combined_views = [], []
        for view in range(volumes.shape[1]):
            with autocast(runtime):
                b37_output = b37_model(volumes[:, view], present, meta, position[:, view])
                b40_output = b40_model(volumes[:, view], present, meta, position[:, view])
            b37_global_views.append(torch.sigmoid(b37_output.base_logits.float()))
            b37_combined_views.append(torch.sigmoid(b37_output.logits.float()))
            b40_global_views.append(torch.sigmoid(b40_output.base_logits.float()))
            b40_combined_views.append(torch.sigmoid(b40_output.logits.float()))
        b37_global_blocks.append(torch.stack(b37_global_views).mean(dim=0).cpu().numpy())
        b37_combined_blocks.append(torch.stack(b37_combined_views).mean(dim=0).cpu().numpy())
        b40_global_blocks.append(torch.stack(b40_global_views).mean(dim=0).cpu().numpy())
        b40_combined_blocks.append(torch.stack(b40_combined_views).mean(dim=0).cpu().numpy())
        scored_uids.extend(str(uid) for uid in batch["study_uid"])
        del (
            batch,
            volumes,
            position,
            present,
            meta,
            b37_output,
            b40_output,
            b37_global_views,
            b37_combined_views,
            b40_global_views,
            b40_combined_views,
        )
        _release()

    if scored_uids != uids:
        raise RuntimeError("B40 candidate study order changed")
    b37_global_prediction = np.concatenate(b37_global_blocks, axis=0)
    b37_combined_prediction = np.concatenate(b37_combined_blocks, axis=0)
    b40_global_prediction = np.concatenate(b40_global_blocks, axis=0)
    b40_combined_prediction = np.concatenate(b40_combined_blocks, axis=0)
    arrays = (
        base_prediction,
        b37_global_prediction,
        b37_combined_prediction,
        b40_global_prediction,
        b40_combined_prediction,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("B40 evaluation produced non-finite predictions")

    base_macro, b37_macro, b40_macro, per_target = _target_rows(
        truth,
        base_prediction,
        b37_combined_prediction,
        b40_combined_prediction,
    )
    if abs(base_macro - B37_EXPECTED_BASE_MACRO) > B37_BASE_TOLERANCE:
        raise RuntimeError(
            f"historical base replay changed: expected ~{B37_EXPECTED_BASE_MACRO:.10f}, got {base_macro:.10f}"
        )
    b37_global_macro = _global_macro_auc(truth, b37_global_prediction)
    b40_global_macro = _global_macro_auc(truth, b40_global_prediction)
    focal_b37 = float(np.mean([per_target[target]["b37_e2_auc"] for target in FOCAL_SIX]))
    focal_b40 = float(np.mean([per_target[target]["b40_e3_auc"] for target in FOCAL_SIX]))
    paired = compare_runs(
        truth,
        b37_combined_prediction,
        b40_combined_prediction,
        n_bootstrap=int(n_bootstrap),
        seed=int(config.get("seed", 2026)) + 47_700_000,
    )
    result = {
        "evaluation_role": "reused post-B39 Expert-58 development diagnostic; not independent test evidence and not a B40 tuning or promotion criterion",
        "parent_b37_checkpoint": str(parent_path),
        "parent_b37_checkpoint_sha256": sha256_file(parent_path),
        "b40_checkpoint": str(checkpoint_path),
        "b40_checkpoint_sha256": sha256_file(checkpoint_path),
        "base_checkpoint": str(base_path),
        "n_studies": len(uids),
        "n_series": int(sum(counts)),
        "tta_offsets": list(B37_EVAL_OFFSETS),
        "base_224_macro_auc": base_macro,
        "b37_e2_global_448_macro_auc": float(b37_global_macro),
        "b37_e2_combined_448_macro_auc": b37_macro,
        "b40_e3_global_448_macro_auc": float(b40_global_macro),
        "b40_e3_combined_448_macro_auc": b40_macro,
        "b40_minus_b37_macro": float(b40_macro - b37_macro),
        "focal_six": list(FOCAL_SIX),
        "b37_e2_focal_six_auc": focal_b37,
        "b40_e3_focal_six_auc": focal_b40,
        "b40_minus_b37_focal_six": float(focal_b40 - focal_b37),
        "per_target": per_target,
        "paired_macro_auc_bootstrap": paired,
        "parent_encoder_sha256_final": b37_payload.get("encoder_sha256_final"),
        "b40_encoder_sha256_final": b40_payload.get("encoder_sha256_final"),
        "metadata_repair": metadata_stats,
        "continuation": b40_payload.get("continuation"),
        "governance": (
            "B40 is a separately fixed optimizer-reset continuation. Do not use this reused Expert-58 result "
            "to change B40’s one additional epoch, optimizer, learning rate, TTA, or model structure. "
            "B37 and B39 remain immutable; hidden competition evidence is required for promotion."
        ),
    }
    output_root = Path(out_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "b40_vs_b37_expert58.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output, flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate B40 optimizer-reset continuation against B37 E2")
    parser.add_argument("--config", default="config/b40_b37_e2_continuation.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--out-root", default=f"{B40_RUN_ROOT}/expert58")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    config = dict(_read_config(args.config))
    evaluate_b40(
        config,
        data_root=args.data_root,
        parent_checkpoint=args.parent_checkpoint,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        out_root=args.out_root,
        n_bootstrap=int(args.n_bootstrap),
    )


if __name__ == "__main__":
    main()
