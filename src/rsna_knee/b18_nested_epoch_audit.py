"""Post-hoc nested epoch-selection audit for completed B18.

This is the direct B18 counterpart of the B20 nested audit. It performs no
training: the five saved B18 candidate checkpoints are rescored on the same
three gold folds, using the same global 12-target macro-AUC selection rule and
earliest-epoch tie break.

The primary estimate uses the two non-outer folds together for epoch selection;
the strict historical manifest uses only one inner fold. Both estimate
checkpoint-selection optimism only. The 58 expert studies are repeatedly reused
development data and are not pristine independent validation.
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
from .b12_variable_series import (
    VariableSeriesKneeDataset,
    build_variable_series_index,
    collate_variable_series,
)
from .b17_training import encoder_state_sha256, freeze_encoder
from .b18_fisher_selection import (
    B18_CANDIDATE_EPOCHS,
    B18_EXPECTED_GOLD_SERIES,
    B18_EXPECTED_GOLD_STUDIES,
    B18_EXPERIMENT,
    B18_VARIANT,
    require_b18_contract,
)
from .b20_nested_epoch_audit import N_SPLITS, nested_epoch_audit_from_predictions
from .constants import TARGETS
from .data import (
    backfill_series_metadata,
    build_validation_manifest,
    gold_mask,
    load_series_csv,
    load_train_csv,
    make_balanced_gold_folds,
)
from .runtime import resolve_runtime

AUDIT_VARIANT = "b18_nested_epoch_selection_audit_v1"


def _load_candidate(path: Path, expected_epoch: int, device) -> tuple[torch.nn.Module, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("variant") != B18_VARIANT or payload.get("experiment") != B18_EXPERIMENT:
        raise ValueError(f"{path} is not a B18 candidate checkpoint")
    if int(payload.get("model_epoch", -1)) != int(expected_epoch):
        raise ValueError(f"{path} model_epoch does not match epoch {expected_epoch}")
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
        raise ValueError("B18 nested audit requires the complete 58-study expert surface")

    gold_uids = gold["StudyInstanceUID"].tolist()
    uid_set = set(gold_uids)
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series = series.loc[series["StudyInstanceUID"].astype(str).isin(uid_set)].copy()
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    variable_index = build_variable_series_index(series, gold_uids)
    counts = [len(variable_index.get(uid, [])) for uid in gold_uids]
    if any(count == 0 for count in counts):
        raise ValueError("B18 nested audit found an expert study with zero eligible series")
    if int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise ValueError(
            f"B18 nested audit expected {B18_EXPECTED_GOLD_SERIES} expert series, got {sum(counts)}"
        )
    return train, gold, variable_index, metadata_stats


def _predict_candidate(model, config, root, gold, variable_index, runtime):
    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B18 nested audit freezes TTA at [-1,0,1]")
    uids = gold["StudyInstanceUID"].tolist()
    ds = VariableSeriesKneeDataset(
        uids,
        variable_index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        targets=gold[TARGETS].to_numpy(np.float32),
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=max(1, int(config.get("b7_eval_batch_size", 2))),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 29_000_000),
    )
    pred_uids, prediction = predict_b12_1(model, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("B18 nested audit prediction order changed")
    return prediction


def run_audit(
    config: dict,
    *,
    candidates_root: str | Path = "runs/b18_fisher_selection/candidates",
    out_root: str | Path = "runs/b18_fisher_selection/nested_epoch_audit",
) -> Path:
    require_b18_contract(config)
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
        raise ValueError("B18 nested audit failed to create all three gold folds")

    manifests = []
    for outer_fold in range(N_SPLITS):
        manifests.append(
            build_validation_manifest(
                train,
                outer_fold=outer_fold,
                n_splits=N_SPLITS,
                seed=int(config.get("seed", 2026)),
                inner_fold=(outer_fold + 1) % N_SPLITS,
            )
        )
    pd.concat(manifests, ignore_index=True).to_csv(
        out / "validation_manifests.csv", index=False
    )

    predictions: dict[int, np.ndarray] = {}
    payload_epochs = {}
    for epoch in range(1, B18_CANDIDATE_EPOCHS + 1):
        path = candidates_root / f"epoch_{epoch}.pt"
        print(f"[B18 nested audit] predicting epoch {epoch}: {path}", flush=True)
        model, payload = _load_candidate(path, epoch, runtime.device)
        predictions[epoch] = _predict_candidate(
            model, config, root, gold, variable_index, runtime
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

    pd.DataFrame(
        [
            {
                "epoch": row["epoch"],
                "macro_auc": row["macro_auc"],
                "defined_targets": row["defined_targets"],
                "all_12_targets_defined": row["all_12_targets_defined"],
            }
            for row in audit["full_epoch_scores"]
        ]
    ).to_csv(out / "full_epoch_scores.csv", index=False)
    pd.DataFrame(audit["crossfit_rows"]).to_csv(out / "crossfit_folds.csv", index=False)
    pd.DataFrame(audit["strict_rows"]).to_csv(out / "strict_manifest_folds.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": gold_uids, "fold": folds}).to_csv(
        out / "gold_folds.csv", index=False
    )

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
        "model": "B18_fisher_expert_guided_epoch_selection",
        "candidate_root": str(candidates_root),
        "candidate_epochs": payload_epochs,
        "n_gold_studies": int(len(gold)),
        "n_splits": N_SPLITS,
        "fold_seed": int(config.get("seed", 2026)),
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
            "Post-hoc estimate of B18 epoch-selection optimism using the same folds and selection "
            "rule as the B20 audit. This isolates the Fisher-style checkpoint-selection step, but "
            "the repeatedly reused 58-study gold surface is not pristine independent validation."
        ),
        "training_performed": False,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(summary_path)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b18-nested-audit")
    parser.add_argument("--config", default="configs/b18_fisher_selection.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--candidates-root", default="runs/b18_fisher_selection/candidates")
    parser.add_argument("--out-root", default="runs/b18_fisher_selection/nested_epoch_audit")
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
