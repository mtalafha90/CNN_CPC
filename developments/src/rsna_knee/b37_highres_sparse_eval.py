"""Expert-58 diagnostic for B37 high-resolution sparse MIL.

Three predictions are reported without post-hoc tuning:

1. historical base: original full-fill B34 checkpoint at historical 224/B20 input;
2. B37 global: the same frozen B34 hierarchy fed by the B37-adapted 448 encoder;
3. B37 combined: B37 global plus the learned B36 sparse-MIL residual.

The primary predeclared endpoint is B37 combined versus the historical base.
The global-only line is mechanistic decomposition, not a second selection gate.
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
from .b20_crop_focus import CropFocusedVariableSeriesKneeDataset, b20_crop_focus_policy
from .b35_training import sha256_file
from .b37_highres_sparse_mil import (
    B37_EXPERT58_ROOT,
    B37_VERSION,
    B37HighResSparseDataset,
    B37HighResSparseMILResidual,
    collate_b35,
    require_b37_sparse_contract,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import compare_runs, macro_auc_from_arrays
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import autocast, resolve_runtime

B37_EVAL_OFFSETS = (-1, 0, 1)
B37_EXPECTED_BASE_MACRO = 0.6686507522833671
B37_BASE_TOLERANCE = 5e-4
FOCAL_SIX = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Contusion",
    "Fracture",
)


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_b37_checkpoint(
    path: str | Path,
    *,
    base_checkpoint: str | Path,
    device,
    expected_version: str = B37_VERSION,
    expected_experiment: str | None = None,
    checkpoint_label: str = "B37",
):
    """Load a fixed sparse-MIL checkpoint using B37's shared architecture.

    B37 supplies the defaults.  A later isolated preprocessing ablation can
    request its own immutable checkpoint identity while reusing the exact same
    sparse-MIL model reconstruction, avoiding a second checkpoint loader that
    might drift from the B37/B40 compatibility contract.
    """
    checkpoint = Path(path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("version") != str(expected_version):
        raise ValueError(f"not a {checkpoint_label} high-resolution sparse-MIL checkpoint")
    if expected_experiment is not None and payload.get("experiment") != str(expected_experiment):
        raise ValueError(f"{checkpoint_label} checkpoint experiment identity changed")
    if bool(payload.get("fixed_endpoint")) is not True or int(payload.get("completed_epochs", -1)) != 2:
        raise ValueError(f"{checkpoint_label} evaluation requires the complete fixed-E2 checkpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError(f"{checkpoint_label} checkpoint unexpectedly used expert labels")

    base_path = Path(base_checkpoint).resolve()
    if sha256_file(base_path) != str(payload.get("base_checkpoint_sha256", "")):
        raise ValueError(f"{checkpoint_label} base checkpoint fingerprint mismatch")
    base, _ = load_phase9_checkpoint(
        base_path,
        expected_arm="llm_fill",
        device="cpu",
    )
    sparse = payload.get("sparse_mil", {})
    finetune = payload.get("encoder_finetune", {})
    model_state = payload.get("model_state", {})
    model = B37HighResSparseMILResidual(
        base,
        grid_size=int(sparse.get("grid_size", 6)),
        top_k=int(sparse.get("top_k", 8)),
        temperature=float(sparse.get("temperature", 1.0)),
        encoder_trainable_stages=int(finetune.get("encoder_trainable_stages", 1)),
        encoder_chunk_size=int(model_state.get("encoder_chunk_size", 4)),
    )
    model.base.load_state_dict(payload["base_state"], strict=True)
    model.head.load_state_dict(payload["head_state"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload


def _base_loader(config: dict, root: Path, uids: list[str], index: dict, truth: np.ndarray):
    cfg = dict(config)
    cfg["b7_image_size"] = 224
    cfg["b7_eval_batch_size"] = 1
    runtime = resolve_runtime(cfg)
    policy = b20_crop_focus_policy(cfg)
    ds = CropFocusedVariableSeriesKneeDataset(
        uids,
        index,
        make_b7_dataset_config(
            cfg,
            root,
            train=False,
            tta_offsets=B37_EVAL_OFFSETS,
        ),
        targets=truth.astype(np.float32),
        train=False,
        crop_focus_policy=policy,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 47_200_000),
    )
    return loader, runtime


def _candidate_loader(
    config: dict,
    root: Path,
    uids: list[str],
    index: dict,
    truth: np.ndarray,
    *,
    dataset_class=B37HighResSparseDataset,
    contract_validator=require_b37_sparse_contract,
    loader_seed_offset: int = 47_300_000,
):
    """Build a fixed sparse-MIL Expert-58 loader, defaulting exactly to B37."""
    cfg = dict(config)
    cfg["b7_eval_batch_size"] = 1
    runtime = resolve_runtime(cfg)
    policy = contract_validator(cfg)
    dcfg = make_b7_dataset_config(cfg, root, train=False)
    dcfg.tta_center_offsets = ()
    ds = dataset_class(
        uids,
        index,
        dcfg,
        crop_focus_policy=policy,
        center_offsets=B37_EVAL_OFFSETS,
        targets=truth.astype(np.float32),
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_b35,
        **runtime.loader_kwargs(
            seed=int(config.get("seed", 2026)) + int(loader_seed_offset)
        ),
    )
    return loader, runtime


def _per_target(truth, base, global448, combined):
    base_macro, base_auc = macro_auc_from_arrays(truth, base)
    global_macro, global_auc = macro_auc_from_arrays(truth, global448)
    combined_macro, combined_auc = macro_auc_from_arrays(truth, combined)
    rows = {}
    for target, a, b, c in zip(TARGETS, base_auc, global_auc, combined_auc):
        rows[target] = {
            "base_224_auc": float(a),
            "b37_global_448_auc": float(b),
            "b37_combined_auc": float(c),
            "global_minus_base": float(b - a),
            "combined_minus_base": float(c - a),
            "sparse_residual_increment": float(c - b),
        }
    return float(base_macro), float(global_macro), float(combined_macro), rows


@torch.no_grad()
def evaluate_b37(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path = B37_EXPERT58_ROOT,
    n_bootstrap: int = 5000,
) -> dict:
    config = dict(config)
    config["data_root"] = str(Path(data_root).resolve())
    require_b37_sparse_contract(config)
    root = Path(config["data_root"])

    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != B18_EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("B37 requires the complete reused 58-study expert surface")
    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(np.float64)

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_variable_series_index(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts) or int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise ValueError("B37 expert MRI series surface changed")

    # Historical 224 base.
    base_loader, base_runtime = _base_loader(config, root, uids, index, truth)
    print(base_runtime.describe(), flush=True)
    base_model, _ = load_phase9_checkpoint(
        Path(base_checkpoint).resolve(),
        expected_arm="llm_fill",
        device=base_runtime.device,
    )
    base_model.eval()
    base_uids, base_pred = predict_b12_1(base_model, base_loader, base_runtime)
    if [str(x) for x in base_uids] != uids:
        raise RuntimeError("B37 historical base study order changed")
    base_pred = np.asarray(base_pred, dtype=np.float32)
    del base_model, base_loader
    _release()

    # Candidate 448 global branch and sparse residual, both with identical TTA.
    candidate_loader, runtime = _candidate_loader(config, root, uids, index, truth)
    print(runtime.describe(), flush=True)
    model, payload = load_b37_checkpoint(
        checkpoint,
        base_checkpoint=base_checkpoint,
        device=runtime.device,
    )

    global_blocks, combined_blocks = [], []
    scored_uids: list[str] = []
    top1_sum = np.zeros(len(TARGETS), dtype=np.float64)
    topk_sum = np.zeros(len(TARGETS), dtype=np.float64)
    gap_sum = np.zeros(len(TARGETS), dtype=np.float64)
    selected_count = 0
    unique_sum = unique_n = 0

    for batch in candidate_loader:
        volumes = batch["volumes"].to(runtime.device, non_blocking=True)
        position = batch["slice_position"].to(runtime.device, non_blocking=True)
        present = batch["present"].to(runtime.device, non_blocking=True)
        meta = batch["series_meta"].to(runtime.device, non_blocking=True)
        if volumes.ndim != 7:
            raise RuntimeError("B37 evaluation expects [B,V,K,S,C,H,W]")
        global_views, combined_views = [], []
        for view in range(volumes.shape[1]):
            with autocast(runtime):
                out = model(
                    volumes[:, view],
                    present,
                    meta,
                    position[:, view],
                )
            global_views.append(torch.sigmoid(out.base_logits.float()))
            combined_views.append(torch.sigmoid(out.logits.float()))
            values = out.top_values.float()
            top1_sum += values[:, :, 0].sum(dim=0).cpu().numpy()
            topk_sum += values.mean(dim=-1).sum(dim=0).cpu().numpy()
            gap_sum += (values[:, :, 0] - values[:, :, -1]).sum(dim=0).cpu().numpy()
            selected_count += int(values.shape[0])
            indices = out.top_indices.detach().cpu()
            for bi in range(indices.shape[0]):
                unique_sum += int(torch.unique(indices[bi]).numel())
                unique_n += 1
        global_blocks.append(torch.stack(global_views).mean(dim=0).cpu().numpy())
        combined_blocks.append(torch.stack(combined_views).mean(dim=0).cpu().numpy())
        scored_uids.extend(str(x) for x in batch["study_uid"])

    if scored_uids != uids:
        raise RuntimeError("B37 candidate study order changed")
    global_pred = np.concatenate(global_blocks, axis=0)
    combined_pred = np.concatenate(combined_blocks, axis=0)
    if not np.isfinite(global_pred).all() or not np.isfinite(combined_pred).all():
        raise RuntimeError("B37 produced non-finite predictions")

    base_macro, global_macro, combined_macro, per_target = _per_target(
        truth,
        base_pred,
        global_pred,
        combined_pred,
    )
    if abs(base_macro - B37_EXPECTED_BASE_MACRO) > B37_BASE_TOLERANCE:
        raise RuntimeError(
            f"historical base replay changed: expected ~{B37_EXPECTED_BASE_MACRO:.10f}, got {base_macro:.10f}"
        )

    focal_base = float(np.mean([per_target[t]["base_224_auc"] for t in FOCAL_SIX]))
    focal_global = float(np.mean([per_target[t]["b37_global_448_auc"] for t in FOCAL_SIX]))
    focal_combined = float(np.mean([per_target[t]["b37_combined_auc"] for t in FOCAL_SIX]))
    macro_delta = float(combined_macro - base_macro)
    focal_delta = float(focal_combined - focal_base)

    paired = compare_runs(
        truth,
        base_pred,
        combined_pred,
        n_bootstrap=int(n_bootstrap),
        seed=int(config.get("seed", 2026)) + 47_400_000,
    )
    mil = {
        "top_k": int(model.head.top_k),
        "grid_size": int(model.head.grid_size),
        "regions_per_slice": int(model.head.n_regions),
        "mean_unique_selected_locations_across_12_targets": float(unique_sum / max(unique_n, 1)),
        "max_possible_selected_locations_across_12_targets": int(len(TARGETS) * model.head.top_k),
        "per_target": {
            target: {
                "mean_top1_evidence": float(top1_sum[i] / max(selected_count, 1)),
                "mean_topk_evidence": float(topk_sum[i] / max(selected_count, 1)),
                "mean_top1_minus_kth": float(gap_sum[i] / max(selected_count, 1)),
            }
            for i, target in enumerate(TARGETS)
        },
    }

    result = {
        "evaluation_role": "reused expert development diagnostic; not independent test evidence",
        "checkpoint": str(Path(checkpoint).resolve()),
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "n_studies": len(uids),
        "n_series": int(sum(counts)),
        "tta_offsets": list(B37_EVAL_OFFSETS),
        "base_224_macro_auc": base_macro,
        "b37_global_448_macro_auc": global_macro,
        "b37_combined_macro_auc": combined_macro,
        "global_448_minus_base_224": float(global_macro - base_macro),
        "sparse_residual_macro_increment": float(combined_macro - global_macro),
        "macro_delta_primary": macro_delta,
        "focal_six": list(FOCAL_SIX),
        "focal_six_base_224": focal_base,
        "focal_six_global_448": focal_global,
        "focal_six_combined": focal_combined,
        "focal_six_delta_primary": focal_delta,
        "per_target": per_target,
        "mil": mil,
        "head": model.head.state(),
        "paired_macro_auc_bootstrap": paired,
        "predeclared_decision": {
            "primary_endpoint": "B37 combined 448 sparse-MIL versus historical 224 full-fill B34",
            "strong_go_rule": "macro_delta >= +0.02 OR focal_six_delta >= +0.03 with macro_delta >= 0",
            "kill_joint_mechanism_rule": "macro_delta <= +0.005 AND focal_six_delta < +0.01",
            "strong_go": bool(
                macro_delta >= 0.02
                or (focal_delta >= 0.03 and macro_delta >= 0.0)
            ),
            "kill_joint_mechanism": bool(
                macro_delta <= 0.005 and focal_delta < 0.01
            ),
            "global_448_line_is_mechanistic_decomposition_not_selection_gate": True,
        },
        "training_contract": {
            "completed_epochs": int(payload.get("completed_epochs", -1)),
            "training_studies": int(payload.get("training_studies", -1)),
            "training_series": int(payload.get("training_series", -1)),
            "training_supervision_cells": int(payload.get("training_supervision_cells", -1)),
            "encoder_sha256_initial": payload.get("encoder_sha256_initial"),
            "encoder_sha256_final": payload.get("encoder_sha256_final"),
        },
        "metadata_repair": metadata_stats,
        "governance": (
            "Do not tune 448, 6x6, top-k=8, crop fraction, endpoint or target subsets from this reused expert58 result. "
            "Independent hidden competition evidence is required for promotion."
        ),
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    for name, pred in (
        ("base_224", base_pred),
        ("b37_global_448", global_pred),
        ("b37_combined", combined_pred),
    ):
        frame = pd.DataFrame(pred, columns=TARGETS)
        frame.insert(0, "StudyInstanceUID", uids)
        frame.to_csv(out / f"{name}_predictions.csv", index=False)
    (out / "expert58.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate B37 high-resolution sparse MIL")
    ap.add_argument("--config", default="config/b37_highres_sparse_448.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--out-root", default=B37_EXPERT58_ROOT)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    evaluate_b37(
        config,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
