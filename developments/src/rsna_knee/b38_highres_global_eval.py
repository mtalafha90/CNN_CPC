"""Expert-58 diagnostic for the fixed B38 global-only 448 ablation.

B38 compares its 448 native-crop, 16-centre global path against the historical
224 full-fill B34 replay.  Expert-58 is a reused development diagnostic only:
this module records the result but contains no tuning or promotion rule.
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
from .b38_highres_global import (
    B38_EXPERT58_ROOT,
    B38_VERSION,
    B38HighResGlobalDataset,
    B38HighResGlobalTail,
    collate_b35,
    require_b38_global_contract,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import compare_runs, macro_auc_from_arrays
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import autocast, resolve_runtime

B38_EVAL_OFFSETS = (-1, 0, 1)
B38_EXPECTED_BASE_MACRO = 0.6686507522833671
B38_BASE_TOLERANCE = 5e-4


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_b38_checkpoint(
    path: str | Path,
    *,
    base_checkpoint: str | Path,
    device,
):
    """Rebuild a completed B38 endpoint and validate its frozen lineage."""
    checkpoint = Path(path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("version") != B38_VERSION:
        raise ValueError("not a B38 high-resolution global-tail checkpoint")
    if (
        bool(payload.get("fixed_endpoint")) is not True
        or int(payload.get("completed_epochs", -1)) != 2
    ):
        raise ValueError("B38 evaluation requires the complete fixed-E2 checkpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B38 checkpoint unexpectedly used expert labels")

    global_model = payload.get("global_model", {})
    if (
        int(global_model.get("n_slices", -1)) != 16
        or global_model.get("aggregation") != "frozen B34 global hierarchy only"
        or global_model.get("sparse_mil") is not False
        or global_model.get("local_auxiliary_loss") is not False
    ):
        raise ValueError("B38 global-only checkpoint contract mismatch")

    base_path = Path(base_checkpoint).resolve()
    if sha256_file(base_path) != str(payload.get("base_checkpoint_sha256", "")):
        raise ValueError("B38 base checkpoint fingerprint mismatch")
    base, _ = load_phase9_checkpoint(
        base_path,
        expected_arm="llm_fill",
        device="cpu",
    )
    finetune = payload.get("encoder_finetune", {})
    model_state = payload.get("model_state", {})
    model = B38HighResGlobalTail(
        base,
        encoder_trainable_stages=int(
            finetune.get("encoder_trainable_stages", 1)
        ),
        encoder_chunk_size=int(model_state.get("encoder_chunk_size", 4)),
    )
    model.base.load_state_dict(payload["base_state"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload


def _base_loader(
    config: dict,
    root: Path,
    uids: list[str],
    index: dict,
    truth: np.ndarray,
):
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
            tta_offsets=B38_EVAL_OFFSETS,
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
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 48_200_000),
    )
    return loader, runtime


def _candidate_loader(
    config: dict,
    root: Path,
    uids: list[str],
    index: dict,
    truth: np.ndarray,
):
    cfg = dict(config)
    cfg["b7_eval_batch_size"] = 1
    runtime = resolve_runtime(cfg)
    policy = require_b38_global_contract(cfg)
    dcfg = make_b7_dataset_config(cfg, root, train=False)
    dcfg.tta_center_offsets = ()
    ds = B38HighResGlobalDataset(
        uids,
        index,
        dcfg,
        crop_focus_policy=policy,
        center_offsets=B38_EVAL_OFFSETS,
        targets=truth.astype(np.float32),
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_b35,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 48_300_000),
    )
    return loader, runtime


def _per_target(truth: np.ndarray, base: np.ndarray, b38: np.ndarray):
    base_macro, base_auc = macro_auc_from_arrays(truth, base)
    b38_macro, b38_auc = macro_auc_from_arrays(truth, b38)
    rows = {}
    for target, base_value, b38_value in zip(TARGETS, base_auc, b38_auc):
        rows[target] = {
            "base_224_auc": float(base_value),
            "b38_global_448_auc": float(b38_value),
            "b38_minus_base": float(b38_value - base_value),
        }
    return float(base_macro), float(b38_macro), rows


@torch.no_grad()
def evaluate_b38(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path = B38_EXPERT58_ROOT,
    n_bootstrap: int = 5000,
) -> dict:
    """Score the completed B38 endpoint on the reused Expert-58 diagnostic."""
    config = dict(config)
    config["data_root"] = str(Path(data_root).resolve())
    require_b38_global_contract(config)
    root = Path(config["data_root"])

    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if (
        len(gold) != B18_EXPECTED_GOLD_STUDIES
        or gold[TARGETS].isna().any().any()
    ):
        raise ValueError("B38 requires the complete reused 58-study expert surface")
    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(np.float64)

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_variable_series_index(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if (
        any(count == 0 for count in counts)
        or int(sum(counts)) != B18_EXPECTED_GOLD_SERIES
    ):
        raise ValueError("B38 expert MRI series surface changed")

    # Historical 224 base replay with its original three-centre TTA.
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
        raise RuntimeError("B38 historical base study order changed")
    base_pred = np.asarray(base_pred, dtype=np.float32)
    del base_model, base_loader
    _release()

    # B38 448 global path using the same three evaluation offsets and only 16
    # centres per offset.
    candidate_loader, runtime = _candidate_loader(config, root, uids, index, truth)
    print(runtime.describe(), flush=True)
    model, payload = load_b38_checkpoint(
        checkpoint,
        base_checkpoint=base_checkpoint,
        device=runtime.device,
    )
    blocks: list[np.ndarray] = []
    scored_uids: list[str] = []
    for batch in candidate_loader:
        volumes = batch["volumes"].to(runtime.device, non_blocking=True)
        present = batch["present"].to(runtime.device, non_blocking=True)
        meta = batch["series_meta"].to(runtime.device, non_blocking=True)
        if volumes.ndim != 7:
            raise RuntimeError("B38 evaluation expects [B,V,K,S,C,H,W]")
        views = []
        for view in range(volumes.shape[1]):
            with autocast(runtime):
                out = model(volumes[:, view], present, meta)
            views.append(torch.sigmoid(out.logits.float()))
        blocks.append(torch.stack(views).mean(dim=0).cpu().numpy())
        scored_uids.extend(str(x) for x in batch["study_uid"])

    if scored_uids != uids:
        raise RuntimeError("B38 candidate study order changed")
    b38_pred = np.concatenate(blocks, axis=0)
    if not np.isfinite(base_pred).all() or not np.isfinite(b38_pred).all():
        raise RuntimeError("B38 produced non-finite predictions")

    base_macro, b38_macro, per_target = _per_target(truth, base_pred, b38_pred)
    if abs(base_macro - B38_EXPECTED_BASE_MACRO) > B38_BASE_TOLERANCE:
        raise RuntimeError(
            "historical base replay changed: "
            f"expected ~{B38_EXPECTED_BASE_MACRO:.10f}, got {base_macro:.10f}"
        )
    paired = compare_runs(
        truth,
        base_pred,
        b38_pred,
        n_bootstrap=int(n_bootstrap),
        seed=int(config.get("seed", 2026)) + 48_400_000,
    )

    result = {
        "evaluation_role": (
            "reused post-B37 Expert-58 development diagnostic; not independent "
            "test evidence and not a B38 tuning or promotion criterion"
        ),
        "checkpoint": str(Path(checkpoint).resolve()),
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "n_studies": len(uids),
        "n_series": int(sum(counts)),
        "tta_offsets": list(B38_EVAL_OFFSETS),
        "base_224_macro_auc": base_macro,
        "b38_global_448_macro_auc": b38_macro,
        "macro_delta_vs_base_224": float(b38_macro - base_macro),
        "per_target": per_target,
        "paired_macro_auc_bootstrap": paired,
        "training_contract": {
            "completed_epochs": int(payload.get("completed_epochs", -1)),
            "training_studies": int(payload.get("training_studies", -1)),
            "training_series": int(payload.get("training_series", -1)),
            "training_supervision_cells": int(
                payload.get("training_supervision_cells", -1)
            ),
            "encoder_sha256_initial": payload.get("encoder_sha256_initial"),
            "encoder_sha256_final": payload.get("encoder_sha256_final"),
            "global_only": True,
            "sparse_mil": False,
            "local_auxiliary_loss": False,
        },
        "metadata_repair": metadata_stats,
        "governance": (
            "Do not use this reused Expert-58 score to change B38.  The model is "
            "already fixed at 448, native 90% crop, 16 centres, one encoder tail "
            "stage, LR 5e-6, and two epochs.  Hidden competition evidence is "
            "required for promotion."
        ),
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    for name, prediction in (
        ("base_224", base_pred),
        ("b38_global_448", b38_pred),
    ):
        frame = pd.DataFrame(prediction, columns=TARGETS)
        frame.insert(0, "StudyInstanceUID", uids)
        frame.to_csv(out / f"{name}_predictions.csv", index=False)
    (out / "expert58.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate B38 high-resolution global-tail ablation")
    ap.add_argument("--config", default="config/b38_highres_global_448.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--out-root", default=B38_EXPERT58_ROOT)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    evaluate_b38(
        config,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
