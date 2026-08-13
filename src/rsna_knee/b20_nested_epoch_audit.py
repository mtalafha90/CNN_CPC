"""Post-hoc nested epoch-selection audit for the completed B20 experiment.

This command does not train a new model. It reuses B20's five saved candidate
checkpoints, predicts the 58-study expert surface once per epoch, and asks how
much of the all-58 selected score can be attributed to epoch selection.

Two estimates are reported:

1. ``strict_manifest``: for each outer fold, use only the manifest's single
   ``inner_selection`` fold to choose the epoch, then score the untouched outer
   fold. This mirrors :func:`data.build_validation_manifest` exactly.
2. ``crossfit_two_fold_selection``: for each outer fold, use the other two folds
   together to choose the epoch, then score the held-out outer fold. Because B20
   never trains on gold labels, this uses the available gold data more efficiently
   while still keeping each scored study out of its own epoch-selection set.

The resulting outer/OOF scores estimate checkpoint-selection optimism only. The
58 gold studies have already been reused throughout model development, so these
numbers are not pristine independent validation evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_1_gold_eval import predict_b12_1
from .b12_1_hierarchical import build_b12_1_model
from .b12_variable_series import build_variable_series_index, collate_variable_series
from .b17_training import encoder_state_sha256, freeze_encoder
from .b18_fisher_selection import (
    B18_CANDIDATE_EPOCHS,
    B18_EXPECTED_GOLD_SERIES,
    B18_EXPECTED_GOLD_STUDIES,
)
from .b20_crop_focus import (
    B20_EXPERIMENT,
    B20_VARIANT,
    CropFocusedVariableSeriesKneeDataset,
    require_b20_contract,
)
from .constants import TARGETS
from .data import (
    backfill_series_metadata,
    build_validation_manifest,
    gold_mask,
    load_series_csv,
    load_train_csv,
    make_balanced_gold_folds,
)
from .evaluation import macro_auc_from_arrays
from .runtime import resolve_runtime

AUDIT_VARIANT = "b20_nested_epoch_selection_audit_v1"
N_SPLITS = 3


def _score_rows(truth: np.ndarray, prediction: np.ndarray) -> dict:
    macro, per_target = macro_auc_from_arrays(truth, prediction)
    return {
        "macro_auc": float(macro),
        "defined_targets": int(np.isfinite(per_target).sum()),
        "all_12_targets_defined": bool(np.isfinite(per_target).all()),
        "per_target_auc": {
            target: (float(value) if np.isfinite(value) else None)
            for target, value in zip(TARGETS, per_target)
        },
    }


def _epoch_scores(
    truth: np.ndarray,
    predictions: dict[int, np.ndarray],
    mask: np.ndarray,
) -> list[dict]:
    rows = []
    for epoch in sorted(predictions):
        score = _score_rows(truth[mask], predictions[epoch][mask])
        rows.append({"epoch": int(epoch), **score})
    return rows


def _select_epoch_exact_12(rows: list[dict]) -> tuple[int | None, str | None]:
    if not rows:
        return None, "no epoch scores"
    if not all(bool(row["all_12_targets_defined"]) for row in rows):
        defined = sorted({int(row["defined_targets"]) for row in rows})
        return None, f"selection subset does not define all 12 target AUCs (defined={defined})"
    best = max(float(row["macro_auc"]) for row in rows)
    selected = min(
        int(row["epoch"])
        for row in rows
        if np.isclose(float(row["macro_auc"]), best, atol=1e-12, rtol=0)
    )
    return selected, None


def nested_epoch_audit_from_predictions(
    *,
    truth: np.ndarray,
    predictions: dict[int, np.ndarray],
    folds: np.ndarray,
) -> dict:
    """Pure-array B20 nested audit, separated for deterministic unit testing."""
    truth = np.asarray(truth, dtype=np.float64)
    folds = np.asarray(folds, dtype=int)
    if truth.ndim != 2 or truth.shape[1] != len(TARGETS):
        raise ValueError("truth must have shape [N,12]")
    if folds.shape != (truth.shape[0],):
        raise ValueError("fold vector length does not match truth")
    if set(np.unique(folds).tolist()) != set(range(N_SPLITS)):
        raise ValueError("B20 nested audit requires exactly folds 0,1,2")
    if set(predictions) != set(range(1, B18_CANDIDATE_EPOCHS + 1)):
        raise ValueError("B20 nested audit requires predictions for epochs 1..5")
    if any(np.asarray(pred).shape != truth.shape for pred in predictions.values()):
        raise ValueError("candidate prediction shape mismatch")

    full_epoch_rows = _epoch_scores(
        truth, predictions, np.ones(truth.shape[0], dtype=bool)
    )
    global_epoch, global_error = _select_epoch_exact_12(full_epoch_rows)
    if global_epoch is None:
        raise RuntimeError(f"full 58-study epoch selection is undefined: {global_error}")
    global_score = float(
        next(row["macro_auc"] for row in full_epoch_rows if row["epoch"] == global_epoch)
    )

    crossfit_oof = np.full_like(truth, np.nan, dtype=np.float64)
    crossfit_rows: list[dict] = []
    for outer_fold in range(N_SPLITS):
        outer = folds == outer_fold
        selection = ~outer
        scores = _epoch_scores(truth, predictions, selection)
        selected_epoch, error = _select_epoch_exact_12(scores)
        if selected_epoch is None:
            raise RuntimeError(
                f"cross-fitted selection is undefined for outer fold {outer_fold}: {error}"
            )
        crossfit_oof[outer] = predictions[selected_epoch][outer]
        outer_score = _score_rows(truth[outer], predictions[selected_epoch][outer])
        selected_inner = next(row for row in scores if row["epoch"] == selected_epoch)
        crossfit_rows.append(
            {
                "outer_fold": outer_fold,
                "selection_folds": [f for f in range(N_SPLITS) if f != outer_fold],
                "n_selection": int(selection.sum()),
                "n_outer": int(outer.sum()),
                "selected_epoch": int(selected_epoch),
                "selection_macro_auc": float(selected_inner["macro_auc"]),
                "outer_macro_auc_defined_targets": float(outer_score["macro_auc"]),
                "outer_defined_targets": int(outer_score["defined_targets"]),
                "epoch_selection_scores": {
                    str(row["epoch"]): float(row["macro_auc"]) for row in scores
                },
            }
        )
    if not np.isfinite(crossfit_oof).all():
        raise RuntimeError("cross-fitted OOF prediction matrix is incomplete")
    crossfit_oof_score = _score_rows(truth, crossfit_oof)
    if not crossfit_oof_score["all_12_targets_defined"]:
        raise RuntimeError("cross-fitted OOF surface does not define all 12 targets")

    strict_oof = np.full_like(truth, np.nan, dtype=np.float64)
    strict_rows: list[dict] = []
    strict_complete = True
    for outer_fold in range(N_SPLITS):
        inner_fold = (outer_fold + 1) % N_SPLITS
        inner = folds == inner_fold
        outer = folds == outer_fold
        scores = _epoch_scores(truth, predictions, inner)
        selected_epoch, error = _select_epoch_exact_12(scores)
        if selected_epoch is None:
            strict_complete = False
            strict_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "n_inner": int(inner.sum()),
                    "n_outer": int(outer.sum()),
                    "status": "unavailable",
                    "reason": error,
                    "epoch_selection_scores": {
                        str(row["epoch"]): float(row["macro_auc"]) for row in scores
                    },
                    "defined_targets": int(scores[0]["defined_targets"]),
                }
            )
            continue
        strict_oof[outer] = predictions[selected_epoch][outer]
        outer_score = _score_rows(truth[outer], predictions[selected_epoch][outer])
        selected_inner = next(row for row in scores if row["epoch"] == selected_epoch)
        strict_rows.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "n_inner": int(inner.sum()),
                "n_outer": int(outer.sum()),
                "status": "ok",
                "selected_epoch": int(selected_epoch),
                "selection_macro_auc": float(selected_inner["macro_auc"]),
                "outer_macro_auc_defined_targets": float(outer_score["macro_auc"]),
                "outer_defined_targets": int(outer_score["defined_targets"]),
                "epoch_selection_scores": {
                    str(row["epoch"]): float(row["macro_auc"]) for row in scores
                },
            }
        )

    strict_oof_score = None
    if strict_complete and np.isfinite(strict_oof).all():
        score = _score_rows(truth, strict_oof)
        if score["all_12_targets_defined"]:
            strict_oof_score = score
        else:
            strict_complete = False

    fixed_epoch5_score = float(
        next(row["macro_auc"] for row in full_epoch_rows if row["epoch"] == 5)
    )
    return {
        "full_epoch_scores": full_epoch_rows,
        "global_selected_epoch": int(global_epoch),
        "global_selected_macro_auc": global_score,
        "fixed_epoch5_macro_auc": fixed_epoch5_score,
        "global_selection_uplift_vs_epoch5": float(global_score - fixed_epoch5_score),
        "crossfit_rows": crossfit_rows,
        "crossfit_oof_predictions": crossfit_oof,
        "crossfit_oof_score": crossfit_oof_score,
        "crossfit_selection_optimism": float(
            global_score - float(crossfit_oof_score["macro_auc"])
        ),
        "strict_rows": strict_rows,
        "strict_complete": bool(strict_complete),
        "strict_oof_predictions": strict_oof if strict_complete else None,
        "strict_oof_score": strict_oof_score,
        "strict_selection_optimism": (
            float(global_score - float(strict_oof_score["macro_auc"]))
            if strict_oof_score is not None
            else None
        ),
    }


def _load_candidate(path: Path, expected_epoch: int, device) -> tuple[torch.nn.Module, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("variant") != B20_VARIANT or payload.get("experiment") != B20_EXPERIMENT:
        raise ValueError(f"{path} is not a B20 candidate checkpoint")
    if int(payload.get("model_epoch", -1)) != int(expected_epoch):
        raise ValueError(f"{path} model_epoch does not match epoch {expected_epoch}")
    if payload.get("crop_focus_enabled") is not True or bool(payload.get("b19_cosine_mask_used", True)):
        raise ValueError(f"{path} does not certify B20 crop-only preprocessing")
    initial_sha = str(payload.get("encoder_sha256_initial", ""))
    final_sha = str(payload.get("encoder_sha256_final", ""))
    if not initial_sha or initial_sha != final_sha:
        raise ValueError(f"{path} encoder fingerprint changed")
    spec = payload.get("model_spec")
    state = payload.get("model_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError(f"{path} is missing model specification/state")
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.load_state_dict(state, strict=True)
    freeze_encoder(model)
    if encoder_state_sha256(model.encoder) != initial_sha:
        raise ValueError(f"{path} reconstructed encoder fingerprint mismatch")
    model = model.to(device)
    model.eval()
    return model, payload


def _prepare_gold(config: dict, root: Path):
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != B18_EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("B20 nested audit requires the complete 58-study expert surface")

    gold_uids = gold["StudyInstanceUID"].tolist()
    uid_set = set(gold_uids)
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series = series.loc[series["StudyInstanceUID"].astype(str).isin(uid_set)].copy()
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    variable_index = build_variable_series_index(series, gold_uids)
    counts = [len(variable_index.get(uid, [])) for uid in gold_uids]
    if any(count == 0 for count in counts):
        raise ValueError("B20 nested audit found an expert study with zero eligible series")
    if int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise ValueError(
            f"B20 nested audit expected {B18_EXPECTED_GOLD_SERIES} expert series, got {sum(counts)}"
        )
    return train, gold, variable_index, metadata_stats


def _predict_candidate(model, config, root, gold, variable_index, runtime, crop_policy):
    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B20 nested audit freezes TTA at [-1,0,1]")
    uids = gold["StudyInstanceUID"].tolist()
    ds = CropFocusedVariableSeriesKneeDataset(
        uids,
        variable_index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        targets=gold[TARGETS].to_numpy(np.float32),
        train=False,
        crop_focus_policy=crop_policy,
    )
    loader = DataLoader(
        ds,
        batch_size=max(1, int(config.get("b7_eval_batch_size", 2))),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 29_100_000),
    )
    pred_uids, prediction = predict_b12_1(model, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("B20 nested audit prediction order changed")
    return prediction


def run_audit(
    config: dict,
    *,
    candidates_root: str | Path = "runs/b20_crop_focus/candidates",
    out_root: str | Path = "runs/b20_crop_focus/nested_epoch_audit",
) -> Path:
    crop_policy = require_b20_contract(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    root = Path(config["data_root"])
    candidates_root = Path(candidates_root)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    train, gold, variable_index, metadata_stats = _prepare_gold(config, root)
    truth = gold[TARGETS].to_numpy(np.float64)
    gold_uids = gold["StudyInstanceUID"].astype(str).tolist()

    folds_all = make_balanced_gold_folds(
        train, n_splits=N_SPLITS, seed=int(config.get("seed", 2026))
    )
    folds = folds_all.loc[gold.index].to_numpy(dtype=int)
    if set(np.unique(folds).tolist()) != {0, 1, 2}:
        raise ValueError("B20 nested audit failed to create all three gold folds")

    manifests = []
    for outer_fold in range(N_SPLITS):
        manifest = build_validation_manifest(
            train,
            outer_fold=outer_fold,
            n_splits=N_SPLITS,
            seed=int(config.get("seed", 2026)),
            inner_fold=(outer_fold + 1) % N_SPLITS,
        )
        manifests.append(manifest)
    pd.concat(manifests, ignore_index=True).to_csv(
        out / "validation_manifests.csv", index=False
    )

    predictions: dict[int, np.ndarray] = {}
    payload_epochs = {}
    for epoch in range(1, B18_CANDIDATE_EPOCHS + 1):
        path = candidates_root / f"epoch_{epoch}.pt"
        print(f"[B20 nested audit] predicting epoch {epoch}: {path}", flush=True)
        model, payload = _load_candidate(path, epoch, runtime.device)
        if payload.get("crop_focus_policy") != crop_policy:
            raise ValueError(f"epoch {epoch} crop-focus policy differs from config")
        predictions[epoch] = _predict_candidate(
            model, config, root, gold, variable_index, runtime, crop_policy
        )
        payload_epochs[epoch] = int(payload.get("model_epoch", -1))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    audit = nested_epoch_audit_from_predictions(
        truth=truth,
        predictions=predictions,
        folds=folds,
    )

    full_scores = pd.DataFrame(
        [
            {
                "epoch": row["epoch"],
                "macro_auc": row["macro_auc"],
                "defined_targets": row["defined_targets"],
                "all_12_targets_defined": row["all_12_targets_defined"],
            }
            for row in audit["full_epoch_scores"]
        ]
    )
    full_scores.to_csv(out / "full_epoch_scores.csv", index=False)
    pd.DataFrame(audit["crossfit_rows"]).to_csv(out / "crossfit_folds.csv", index=False)
    pd.DataFrame(audit["strict_rows"]).to_csv(out / "strict_manifest_folds.csv", index=False)

    fold_frame = pd.DataFrame({"StudyInstanceUID": gold_uids, "fold": folds})
    fold_frame.to_csv(out / "gold_folds.csv", index=False)

    crossfit_oof = pd.DataFrame(audit["crossfit_oof_predictions"], columns=TARGETS)
    crossfit_oof.insert(0, "fold", folds)
    crossfit_oof.insert(0, "StudyInstanceUID", gold_uids)
    crossfit_oof.to_csv(out / "crossfit_oof_predictions.csv", index=False)

    if audit["strict_oof_predictions"] is not None:
        strict_oof = pd.DataFrame(audit["strict_oof_predictions"], columns=TARGETS)
        strict_oof.insert(0, "fold", folds)
        strict_oof.insert(0, "StudyInstanceUID", gold_uids)
        strict_oof.to_csv(out / "strict_manifest_oof_predictions.csv", index=False)

    stacked = []
    for epoch, prediction in predictions.items():
        frame = pd.DataFrame(prediction, columns=TARGETS)
        frame.insert(0, "epoch", epoch)
        frame.insert(0, "StudyInstanceUID", gold_uids)
        stacked.append(frame)
    pd.concat(stacked, ignore_index=True).to_csv(
        out / "all_epoch_gold_predictions.csv", index=False
    )

    summary = {
        "variant": AUDIT_VARIANT,
        "model": "B20_crop_only_joint_focus",
        "candidate_root": str(candidates_root),
        "candidate_epochs": payload_epochs,
        "n_gold_studies": int(len(gold)),
        "n_splits": N_SPLITS,
        "fold_seed": int(config.get("seed", 2026)),
        "crop_focus_policy": crop_policy,
        "metadata_repair": metadata_stats,
        "global_all58_selected_epoch": audit["global_selected_epoch"],
        "global_all58_selected_macro_auc": audit["global_selected_macro_auc"],
        "fixed_epoch5_macro_auc": audit["fixed_epoch5_macro_auc"],
        "global_selection_uplift_vs_epoch5": audit["global_selection_uplift_vs_epoch5"],
        "crossfit_two_fold_selection": {
            "oof_macro_auc": audit["crossfit_oof_score"]["macro_auc"],
            "all_12_targets_defined": audit["crossfit_oof_score"]["all_12_targets_defined"],
            "selected_epochs_by_outer_fold": [
                row["selected_epoch"] for row in audit["crossfit_rows"]
            ],
            "estimated_selection_optimism": audit["crossfit_selection_optimism"],
        },
        "strict_manifest": {
            "complete": audit["strict_complete"],
            "oof_macro_auc": (
                audit["strict_oof_score"]["macro_auc"]
                if audit["strict_oof_score"] is not None
                else None
            ),
            "estimated_selection_optimism": audit["strict_selection_optimism"],
            "selected_epochs_by_outer_fold": [
                row.get("selected_epoch") for row in audit["strict_rows"]
            ],
        },
        "interpretation": (
            "Post-hoc estimate of B20 epoch-selection optimism. Each cross-fitted outer study is "
            "scored using an epoch chosen without that study's fold. Because the same 58 gold studies "
            "have influenced earlier model development, this is not pristine independent validation."
        ),
        "training_performed": False,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(summary_path)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b20-nested-audit")
    parser.add_argument("--config", default="configs/b20_crop_focus.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--candidates-root", default="runs/b20_crop_focus/candidates")
    parser.add_argument("--out-root", default="runs/b20_crop_focus/nested_epoch_audit")
    args = parser.parse_args()

    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    run_audit(
        config,
        candidates_root=args.candidates_root,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
