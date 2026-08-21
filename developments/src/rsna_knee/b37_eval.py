"""Expert-58 diagnostic for B37 high-resolution preprocessing.

The reused expert surface is diagnostic only.  B37 is compared against the
existing full-LLM-fill B34 fixed-E2 checkpoint.  Each model is scored through
its own declared preprocessing path:

base: full normalization -> resize 224 -> 90% crop -> resize 224
B37: full normalization -> native 90% crop -> one resize 288

Both use the same [-1,0,+1] slice-center TTA and the same 336 expert MRI series.
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
from .b37_highres import (
    B37_VERSION,
    B37_IMAGE_SIZE,
    B37HighResVariableSeriesKneeDataset,
    b37_preprocessing_state,
    require_b37_preprocessing_contract,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import compare_runs, macro_auc_from_arrays
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import resolve_runtime

B37_BASE_IMAGE_SIZE = 224
B37_EXPECTED_BASE_MACRO = 0.6686507522833671
B37_BASE_MACRO_TOLERANCE = 5e-4
B37_EVAL_OFFSETS = (-1, 0, 1)
FOCAL_SIX = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Contusion",
    "Fracture",
)


def _low_memory_eval_config(config: dict, *, image_size: int) -> dict:
    out = dict(config)
    out["b7_image_size"] = int(image_size)
    out["num_workers"] = 2
    out["persistent_workers"] = False
    out["prefetch_factor"] = 1
    out["series_cache_mb_per_worker"] = 64
    out["b7_eval_batch_size"] = 1
    return out


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _loader(
    config: dict,
    *,
    root: Path,
    uids: list[str],
    index: dict,
    truth: np.ndarray,
    candidate: bool,
):
    image_size = B37_IMAGE_SIZE if candidate else B37_BASE_IMAGE_SIZE
    cfg = _low_memory_eval_config(config, image_size=image_size)
    runtime = resolve_runtime(cfg)
    policy = b20_crop_focus_policy({**config, "b7_image_size": 224})
    dataset_cls = (
        B37HighResVariableSeriesKneeDataset
        if candidate
        else CropFocusedVariableSeriesKneeDataset
    )
    ds = dataset_cls(
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
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + (47_100_000 if candidate else 47_000_000)),
    )
    return loader, runtime


def _per_target(truth: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> tuple[dict, float, float]:
    _, base_auc = macro_auc_from_arrays(truth, base)
    _, candidate_auc = macro_auc_from_arrays(truth, candidate)
    per_target = {}
    for target, left, right in zip(TARGETS, base_auc, candidate_auc):
        per_target[target] = {
            "base_auc": float(left),
            "b37_auc": float(right),
            "delta": float(right - left),
        }
    focal_base = float(np.mean([per_target[t]["base_auc"] for t in FOCAL_SIX]))
    focal_candidate = float(np.mean([per_target[t]["b37_auc"] for t in FOCAL_SIX]))
    return per_target, focal_base, focal_candidate


def evaluate_b37(
    config: dict,
    *,
    data_root: str | Path,
    base_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    out_root: str | Path = "runs/b37_highres_288/expert58",
    n_bootstrap: int = 5000,
) -> dict:
    config = dict(config)
    config["data_root"] = str(Path(data_root).resolve())
    require_b37_preprocessing_contract(config)
    root = Path(config["data_root"])

    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != B18_EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("B37 requires the complete 58-study / 696-label expert surface")
    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(np.float64)

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_variable_series_index(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts) or int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise ValueError("B37 expert series surface changed")

    predictions: dict[str, np.ndarray] = {}
    payloads: dict[str, dict] = {}
    for name, checkpoint, candidate in (
        ("base", base_checkpoint, False),
        ("b37", candidate_checkpoint, True),
    ):
        loader, runtime = _loader(
            config,
            root=root,
            uids=uids,
            index=index,
            truth=truth,
            candidate=candidate,
        )
        print(runtime.describe())
        model, payload = load_phase9_checkpoint(
            checkpoint,
            expected_arm="llm_fill",
            device=runtime.device,
        )
        if candidate:
            if payload.get("b37_version") != B37_VERSION:
                raise ValueError("candidate checkpoint is not the declared B37 variant")
            if payload.get("b37_preprocessing") != b37_preprocessing_state():
                raise ValueError("B37 checkpoint preprocessing contract mismatch")
        else:
            if payload.get("b37_version") is not None:
                raise ValueError("base checkpoint unexpectedly contains B37 preprocessing")

        model.eval()
        state = model.b34_state()
        if state.get("eval_context_exact_bypass") is not True:
            raise RuntimeError(f"{name} B34 evaluation bypass contract missing")
        pred_uids, pred = predict_b12_1(model, loader, runtime)
        if [str(x) for x in pred_uids] != uids:
            raise RuntimeError(f"B37 {name} study order changed")
        pred = np.asarray(pred, dtype=np.float32)
        if pred.shape != truth.shape or not np.isfinite(pred).all():
            raise RuntimeError(f"B37 {name} predictions are invalid")
        predictions[name] = pred
        payloads[name] = payload
        del model, loader
        _release()

    base_macro, _ = macro_auc_from_arrays(truth, predictions["base"])
    candidate_macro, _ = macro_auc_from_arrays(truth, predictions["b37"])
    if abs(float(base_macro) - B37_EXPECTED_BASE_MACRO) > B37_BASE_MACRO_TOLERANCE:
        raise RuntimeError(
            "B37 historical-base replay changed materially: "
            f"expected about {B37_EXPECTED_BASE_MACRO:.10f}, got {float(base_macro):.10f}"
        )

    per_target, focal_base, focal_b37 = _per_target(
        truth,
        predictions["base"],
        predictions["b37"],
    )
    paired = compare_runs(
        truth,
        predictions["base"],
        predictions["b37"],
        n_bootstrap=int(n_bootstrap),
        seed=int(config.get("seed", 2026)) + 47_200_000,
    )

    macro_delta = float(candidate_macro - base_macro)
    focal_delta = float(focal_b37 - focal_base)
    strong_go = bool(
        macro_delta >= 0.02
        or (focal_delta >= 0.03 and macro_delta >= 0.0)
    )
    kill_main_resolution = bool(macro_delta <= 0.005 and focal_delta < 0.01)

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("base", "b37"):
        frame = pd.DataFrame(predictions[name], columns=TARGETS)
        frame.insert(0, "StudyInstanceUID", uids)
        frame.to_csv(out / f"{name}_predictions.csv", index=False)

    result = {
        "evaluation_role": "reused expert development diagnostic; not independent test evidence",
        "n_studies": len(uids),
        "n_series": int(sum(counts)),
        "tta_offsets": list(B37_EVAL_OFFSETS),
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "b37_checkpoint": str(Path(candidate_checkpoint).resolve()),
        "base_image_size": B37_BASE_IMAGE_SIZE,
        "b37_preprocessing": b37_preprocessing_state(),
        "base_macro_auc": float(base_macro),
        "b37_macro_auc": float(candidate_macro),
        "macro_delta": macro_delta,
        "focal_six": list(FOCAL_SIX),
        "focal_six_base_mean_auc": focal_base,
        "focal_six_b37_mean_auc": focal_b37,
        "focal_six_delta": focal_delta,
        "per_target": per_target,
        "paired_macro_auc_bootstrap": paired,
        "predeclared_decision": {
            "strong_go_rule": "macro_delta >= +0.02 OR focal_six_delta >= +0.03 with macro_delta >= 0",
            "kill_main_resolution_rule": "macro_delta <= +0.005 AND focal_six_delta < +0.01",
            "strong_go": strong_go,
            "kill_main_resolution": kill_main_resolution,
        },
        "training_contract": {
            "base_encoder_trainable_stages": int(payloads["base"].get("encoder_trainable_stages", -1)),
            "b37_encoder_trainable_stages": int(payloads["b37"].get("encoder_trainable_stages", -1)),
            "base_encoder_lr_scale": float(payloads["base"].get("encoder_lr_scale", -1.0)),
            "b37_encoder_lr_scale": float(payloads["b37"].get("encoder_lr_scale", -1.0)),
            "base_completed_epochs": int(payloads["base"].get("completed_epochs", -1)),
            "b37_completed_epochs": int(payloads["b37"].get("completed_epochs", -1)),
            "base_training_series": int(payloads["base"].get("training_series", -1)),
            "b37_training_series": int(payloads["b37"].get("training_series", -1)),
            "base_report_only_studies": int(payloads["base"].get("report_only_studies_exposed", -1)),
            "b37_report_only_studies": int(payloads["b37"].get("report_only_studies_exposed", -1)),
        },
        "metadata_repair": metadata_stats,
        "governance": (
            "Do not tune 288, crop fraction, endpoint or target subsets from this reused expert-58 result. "
            "A promoted B37 still requires hidden competition evidence."
        ),
    }
    (out / "expert58.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate B37 high-resolution candidate")
    ap.add_argument("--config", default="config/b37_highres_288.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/b37_highres_288/expert58")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    evaluate_b37(
        config,
        data_root=args.data_root,
        base_checkpoint=args.base_checkpoint,
        candidate_checkpoint=args.checkpoint,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
