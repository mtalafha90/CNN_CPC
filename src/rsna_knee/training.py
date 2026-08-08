from __future__ import annotations

import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .budget import RuntimeBudget
from .calibration import fit_calibration
from .constants import DUAL_STREAMS, TARGETS
from .cotrain import assign_crossfit_folds, consensus_arrays, load_fold_image_teacher
from .data import (
    add_report_groups,
    backfill_series_metadata,
    build_series_index,
    gold_mask,
    load_series_csv,
    load_train_csv,
    make_balanced_gold_folds,
)
from .dataset import DatasetConfig, KneeStudyDataset
from .evaluation import bootstrap_macro_auc, macro_auc_from_arrays
from .model import KneeMILNet
from .policy import validate_competition_config
from .preflight import run_preflight
from .report_labels import (
    STATE_UNCERTAIN,
    STATE_UNMENTIONED,
    combine_gold_and_pseudo,
    label_dataframe,
    state_dataframe,
)
from .runtime import autocast, make_scaler, resolve_runtime
from .sampling import TwoPoolBatchSampler, trusted_study_mask


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_weighted_bce(
    logits,
    target,
    weight,
    *,
    target_denominator=None,
    batch_scale: float = 1.0,
):
    """Macro-balanced BCE.

    With ``target_denominator=None`` this is the historical per-batch objective.
    During production training an epoch-level denominator is supplied from the
    deterministic sampler plan. Summing batch contributions then gives every
    pathology equal total supervision mass over the planned epoch, even when
    sparse report labels make some targets participate in fewer batches.
    """
    cell = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    numerator = (cell * weight).sum(dim=0)
    if target_denominator is None:
        denominator = weight.sum(dim=0)
        valid = denominator > 0
        return (
            (numerator[valid] / denominator[valid].clamp_min(1e-8)).mean()
            if valid.any()
            else logits.sum() * 0.0
        )

    denominator = torch.as_tensor(target_denominator, dtype=logits.dtype, device=logits.device)
    if denominator.ndim != 1 or denominator.numel() != logits.shape[1]:
        raise ValueError("target_denominator must contain one value per target")
    valid = denominator > 0
    if not valid.any():
        return logits.sum() * 0.0
    return (numerator[valid] / denominator[valid].clamp_min(1e-8)).mean() * float(batch_scale)


def confidence_gated_ranking_loss(
    logits,
    target,
    weight,
    *,
    pairs_per_target=32,
    min_confidence=0.35,
    positive_threshold=0.75,
    negative_threshold=0.25,
    return_counts: bool = False,
):
    losses = []
    counts = torch.zeros(logits.shape[1], dtype=torch.long, device=logits.device)
    for j in range(logits.shape[1]):
        trusted = weight[:, j] >= float(min_confidence)
        positives = torch.nonzero(
            trusted & (target[:, j] >= positive_threshold), as_tuple=False
        ).flatten()
        negatives = torch.nonzero(
            trusted & (target[:, j] <= negative_threshold), as_tuple=False
        ).flatten()
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        n = min(int(pairs_per_target), int(positives.numel() * negatives.numel()))
        pi = positives[torch.randint(positives.numel(), (n,), device=logits.device)]
        ni = negatives[torch.randint(negatives.numel(), (n,), device=logits.device)]
        pair_weight = torch.minimum(weight[pi, j], weight[ni, j]).clamp(max=1.0)
        losses.append(
            (nn.functional.softplus(-(logits[pi, j] - logits[ni, j])) * pair_weight).mean()
        )
        counts[j] = n
    loss = torch.stack(losses).mean() if losses else logits.new_zeros(())
    return (loss, counts) if return_counts else loss


def _central_view_index(offsets: tuple[int, ...]) -> int:
    if 0 in offsets:
        return offsets.index(0)
    return min(range(len(offsets)), key=lambda i: abs(offsets[i]))


@torch.no_grad()
def predict(
    model,
    loader,
    device,
    runtime=None,
    *,
    budget: RuntimeBudget | None = None,
    label: str = "prediction",
    central_view_index: int = 0,
    initial_batch_guard_seconds: float = 180.0,
    safety_factor: float = 1.35,
    return_details: bool = False,
):
    """Budget-aware prediction with optional multi-view average and center output."""
    model.eval()
    started = time.monotonic()
    uids, probs, central_probs, targets = [], [], [], []
    batch_times: list[float] = []
    for batch_index, batch in enumerate(loader):
        if budget is not None:
            if batch_times:
                guard = max(5.0, float(np.median(batch_times[-5:])) * float(safety_factor))
            else:
                guard = float(initial_batch_guard_seconds)
            budget.require(guard, label=f"{label} batch {batch_index + 1}")

        batch_started = time.monotonic()
        volumes = batch["volumes"]
        present = batch["present"].to(device, non_blocking=True)
        if volumes.ndim != 7:
            with (autocast(runtime) if runtime is not None else nullcontext()):
                logits = model(volumes.to(device, non_blocking=True), present)
            probability_tensor = torch.sigmoid(logits.float())
            central_tensor = probability_tensor
        else:
            view_probs = []
            for view in range(volumes.shape[1]):
                with (autocast(runtime) if runtime is not None else nullcontext()):
                    logits = model(volumes[:, view].to(device, non_blocking=True), present)
                view_probs.append(torch.sigmoid(logits.float()))
            if not 0 <= int(central_view_index) < len(view_probs):
                raise ValueError("central_view_index is outside the available TTA views")
            central_tensor = view_probs[int(central_view_index)]
            probability_tensor = torch.stack(view_probs).mean(dim=0)

        probs.append(probability_tensor.cpu().numpy())
        central_probs.append(central_tensor.cpu().numpy())
        uids.extend(list(batch["study_uid"]))
        if "target" in batch:
            targets.append(batch["target"].numpy())
        batch_times.append(time.monotonic() - batch_started)

    elapsed = time.monotonic() - started
    probability = np.concatenate(probs) if probs else np.empty((0, len(TARGETS)), np.float32)
    central_probability = (
        np.concatenate(central_probs) if central_probs else np.empty((0, len(TARGETS)), np.float32)
    )
    truth = np.concatenate(targets) if targets else None
    result = (uids, probability, truth)
    if not return_details:
        return result
    details = {
        "elapsed_seconds": float(elapsed),
        "studies": int(len(uids)),
        "batches": int(len(batch_times)),
        "seconds_per_study": float(elapsed / max(len(uids), 1)),
        "median_batch_seconds": float(np.median(batch_times)) if batch_times else 0.0,
        "central_probability": central_probability,
    }
    return (*result, details)


def _offsets(config: dict, key: str, fallback) -> tuple[int, ...]:
    values = config.get(key, fallback)
    values = [0] if values is None or len(values) == 0 else values
    return tuple(int(value) for value in values)


def _dataset_config(config, root, *, train, split="train", center_offset=0, tta_offsets=()):
    return DatasetConfig(
        data_root=str(root),
        split=split,
        n_slices=int(config.get("n_slices", 16)),
        image_size=int(config.get("image_size", 224)),
        noise_std=float(config.get("noise_std", 0.02)) if train else 0.0,
        slice_dropout=float(config.get("slice_dropout", 0.08)) if train else 0.0,
        triplet_gap=int(config.get("triplet_gap", 1)),
        strict_dicom=bool(config.get("strict_dicom", False)),
        center_offset=int(center_offset),
        tta_center_offsets=tuple(int(x) for x in tta_offsets),
        train_gap_choices=tuple(int(x) for x in config.get("train_gap_choices", [1, 2])),
        center_jitter=int(config.get("center_jitter", 2)) if train else 0,
        rotation_deg=float(config.get("rotation_deg", 5.0)) if train else 0.0,
        translate_frac=float(config.get("translate_frac", 0.03)) if train else 0.0,
        scale_jitter=float(config.get("scale_jitter", 0.05)) if train else 0.0,
        gamma_jitter=float(config.get("gamma_jitter", 0.12)) if train else 0.0,
        bias_field_strength=float(config.get("bias_field_strength", 0.08)) if train else 0.0,
        series_cache_mb=int(config.get("series_cache_mb_per_worker", 256)),
    )


def _model_spec(config):
    return {
        "architecture": "cross_sequence_pathology_queries_v1",
        "n_streams": len(DUAL_STREAMS),
        "n_slices": int(config.get("n_slices", 16)),
        "in_channels": 3,
        "image_size": int(config.get("image_size", 224)),
        "triplet_gap": int(config.get("triplet_gap", 1)),
        "stream_mode": "dual",
        "dropout": float(config.get("dropout", 0.25)),
        "normalize_input": bool(config.get("normalize_input", False)),
        "encoder_batch_size": int(config.get("encoder_batch_size", 24)),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "transformer_layers": int(config.get("transformer_layers", 2)),
        "transformer_heads": int(config.get("transformer_heads", 8)),
        "transformer_ff_mult": float(config.get("transformer_ff_mult", 2.0)),
        "pathology_layers": int(config.get("pathology_layers", 1)),
    }


def _build_model(spec, config, device):
    model = KneeMILNet(
        spec["n_streams"], spec["n_slices"], in_channels=3,
        pretrained_weights=bool(config.get("pretrained", False)),
        normalize_input=spec["normalize_input"],
        dropout=spec["dropout"],
        encoder_batch_size=spec["encoder_batch_size"],
        gradient_checkpointing=spec["gradient_checkpointing"],
        transformer_layers=spec["transformer_layers"],
        transformer_heads=spec["transformer_heads"],
        transformer_ff_mult=spec["transformer_ff_mult"],
        pathology_layers=spec["pathology_layers"],
    )
    ssl_path = config.get("ssl_encoder_checkpoint")
    if ssl_path:
        payload = torch.load(Path(ssl_path), map_location="cpu", weights_only=False)
        model.encoder.load_state_dict(payload.get("encoder", payload), strict=True)
    return model.to(device)


def _stage1_source(config: dict):
    candidates = config.get("cotrain_stage1_candidates")
    if candidates:
        return candidates
    return config.get("cotrain_stage1_root")


def _teacher_arrays(
    df,
    states,
    gold,
    allowed_mask,
    config,
    fold: int,
    *,
    use_image_teacher: bool,
):
    calibration = None
    calibration_mask = gold.to_numpy() & allowed_mask.to_numpy()
    if bool(config.get("calibrate_teacher", True)) and calibration_mask.sum() >= int(
        config.get("min_calibration_studies", 8)
    ):
        calibration = fit_calibration(
            states[calibration_mask],
            df.loc[calibration_mask, TARGETS].to_numpy(np.float64),
            alpha=float(config.get("calibration_alpha", 5.0)),
        )
        pseudo = calibration.apply(states)
        confidence = calibration.confidence(
            states,
            unmentioned_weight=float(config.get("unmentioned_weight", 0.0)),
            uncertain_weight_cap=float(config.get("uncertain_weight_cap", 0.10)),
        )
    else:
        pseudo, confidence = label_dataframe(df)
        confidence[states == STATE_UNMENTIONED] = float(config.get("unmentioned_weight", 0.0))
        confidence[states == STATE_UNCERTAIN] = np.minimum(
            confidence[states == STATE_UNCERTAIN],
            float(config.get("uncertain_weight_cap", 0.10)),
        )

    source_meta = None
    stage1_source = _stage1_source(config)
    if stage1_source and use_image_teacher:
        image, source_meta = load_fold_image_teacher(
            stage1_source, fold, df, gold.to_numpy(), return_source=True
        )
        pseudo, confidence = consensus_arrays(
            pseudo,
            confidence,
            image,
            positive_threshold=float(config.get("cotrain_positive_threshold", 0.80)),
            negative_threshold=float(config.get("cotrain_negative_threshold", 0.20)),
            agreement_weight=float(config.get("cotrain_agreement_weight", 0.90)),
            disagreement_weight=float(config.get("cotrain_disagreement_weight", 0.05)),
            blend=float(config.get("cotrain_blend", 0.50)),
            report_low_confidence=float(config.get("cotrain_report_low_confidence", 0.10)),
            image_only_positive_threshold=float(
                config.get("cotrain_image_only_positive_threshold", 0.95)
            ),
            image_only_negative_threshold=float(
                config.get("cotrain_image_only_negative_threshold", 0.05)
            ),
            image_only_weight=float(config.get("cotrain_image_only_weight", 0.20)),
            image_only_blend=float(config.get("cotrain_image_only_blend", 0.75)),
        )
    return pseudo, confidence, calibration, source_meta


def _save_predictions(path: Path, uids, probabilities):
    frame = pd.DataFrame(probabilities, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    frame.to_csv(path, index=False)


def _supervision_summary(targets: np.ndarray, weights: np.ndarray) -> dict:
    summary = {}
    for j, target in enumerate(TARGETS):
        w = np.asarray(weights[:, j], dtype=float)
        y = np.asarray(targets[:, j], dtype=float)
        active = w > 0
        summary[target] = {
            "cells": int(len(w)),
            "nonzero_weight_cells": int(active.sum()),
            "weight_sum": float(w.sum()),
            "high_confidence_cells": int((w >= 0.60).sum()),
            "positive_mass": float((w * y).sum()),
            "negative_mass": float((w * (1.0 - y)).sum()),
        }
    return summary


def _stage2_delta_summary(before_p, before_w, after_p, after_w, row_mask) -> dict:
    rows = np.asarray(row_mask, dtype=bool)
    payload = {"rows": int(rows.sum()), "targets": {}}
    for j, target in enumerate(TARGETS):
        bp, bw = before_p[rows, j], before_w[rows, j]
        ap, aw = after_p[rows, j], after_w[rows, j]
        payload["targets"][target] = {
            "cells": int(len(ap)),
            "report_nonzero_weight": int((bw > 0).sum()),
            "stage2_nonzero_weight": int((aw > 0).sum()),
            "zero_to_nonzero_weight": int(((bw == 0) & (aw > 0)).sum()),
            "stage2_high_confidence": int((aw >= 0.60).sum()),
            "probability_changed_gt_0.05": int((np.abs(ap - bp) > 0.05).sum()),
            "mean_abs_probability_change": float(np.mean(np.abs(ap - bp))) if len(ap) else 0.0,
        }
    return payload


def _balanced_train_loader(dataset, gold_rows, weights, config, runtime, seed, budget):
    batch_size = int(config.get("batch_size", 2))
    trusted = trusted_study_mask(
        np.asarray(gold_rows, dtype=bool),
        weights,
        float(config.get("trusted_pseudo_threshold", 0.60)),
    )
    sampler = TwoPoolBatchSampler(
        trusted,
        batch_size,
        trusted_fraction=float(config.get("trusted_fraction", 0.30)),
        seed=seed,
        drop_last=True,
        max_batches=int(config.get("max_train_batches_per_epoch", 300)),
        deadline_monotonic=budget.work_deadline_monotonic,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, **runtime.loader_kwargs(seed=seed))
    return loader, sampler, trusted


def _optimizer_bundle(model, config, epochs, runtime):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs)),
        eta_min=float(config.get("min_lr", 1e-6)),
    )
    return optimizer, scheduler, make_scaler(runtime)


def _planned_epoch_denominator(sampler, weights: np.ndarray) -> tuple[np.ndarray, int]:
    planned = list(iter(sampler))
    if not planned:
        raise RuntimeError("runtime deadline reached before a training epoch could start")
    flat = np.asarray([index for batch in planned for index in batch], dtype=int)
    denominator = np.asarray(weights, dtype=np.float64)[flat].sum(axis=0)
    return denominator, len(planned)


def _train_epoch(model, loader, sampler, optimizer, scaler, runtime, config, epoch):
    sampler.set_epoch(epoch)
    target_denominator, planned_batches = _planned_epoch_denominator(sampler, loader.dataset.weights)
    model.train()
    total, seen = 0.0, 0
    rank_weight = float(config.get("rank_loss_weight", 0.10))
    pair_counts = np.zeros(len(TARGETS), dtype=np.int64)
    weight_sum = np.zeros(len(TARGETS), dtype=np.float64)
    nonzero_cells = np.zeros(len(TARGETS), dtype=np.int64)
    participating_batches = np.zeros(len(TARGETS), dtype=np.int64)
    actual_batches = 0
    for batch in loader:
        batch_weights_cpu = batch["weight"].numpy().astype(np.float64, copy=False)
        weight_sum += batch_weights_cpu.sum(axis=0)
        nonzero_cells += (batch_weights_cpu > 0).sum(axis=0)
        participating_batches += (batch_weights_cpu.sum(axis=0) > 0).astype(np.int64)

        optimizer.zero_grad(set_to_none=True)
        volumes = batch["volumes"].to(runtime.device, non_blocking=True)
        present = batch["present"].to(runtime.device, non_blocking=True)
        target = batch["target"].to(runtime.device, non_blocking=True)
        weight = batch["weight"].to(runtime.device, non_blocking=True)
        with autocast(runtime):
            logits = model(volumes, present)
            bce = macro_weighted_bce(
                logits,
                target,
                weight,
                target_denominator=target_denominator,
                batch_scale=float(planned_batches),
            )
            if rank_weight > 0:
                rank, counts = confidence_gated_ranking_loss(
                    logits,
                    target,
                    weight,
                    pairs_per_target=int(config.get("rank_pairs_per_target", 32)),
                    min_confidence=float(config.get("rank_min_confidence", 0.35)),
                    positive_threshold=float(config.get("rank_positive_threshold", 0.75)),
                    negative_threshold=float(config.get("rank_negative_threshold", 0.25)),
                    return_counts=True,
                )
                pair_counts += counts.detach().cpu().numpy()
            else:
                rank = logits.new_zeros(())
            loss = bce + rank_weight * rank
        scaler.scale(loss).backward()
        clip = float(config.get("grad_clip", 1.0))
        if clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item()) * len(target)
        seen += len(target)
        actual_batches += 1

    return {
        "loss": total / max(seen, 1),
        "planned_batches": int(planned_batches),
        "actual_batches": int(actual_batches),
        "rank_pairs": {target: int(pair_counts[j]) for j, target in enumerate(TARGETS)},
        "effective_supervision": {
            target: {
                "weight_sum": float(weight_sum[j]),
                "nonzero_cells": int(nonzero_cells[j]),
                "participating_batches": int(participating_batches[j]),
                "planned_epoch_weight": float(target_denominator[j]),
            }
            for j, target in enumerate(TARGETS)
        },
    }


def _add_training_diagnostics(accumulator: dict, metrics: dict) -> None:
    for target in TARGETS:
        accumulator["rank_pairs"][target] += int(metrics["rank_pairs"][target])
        for key, value in metrics["effective_supervision"][target].items():
            accumulator["effective_supervision"][target][key] += value
    accumulator["planned_batches"] += int(metrics["planned_batches"])
    accumulator["actual_batches"] += int(metrics["actual_batches"])


def _empty_training_diagnostics() -> dict:
    return {
        "rank_pairs": {target: 0 for target in TARGETS},
        "effective_supervision": {
            target: {
                "weight_sum": 0.0,
                "nonzero_cells": 0,
                "participating_batches": 0,
                "planned_epoch_weight": 0.0,
            }
            for target in TARGETS
        },
        "planned_batches": 0,
        "actual_batches": 0,
    }


def _finish_components(
    config: dict,
    seconds_per_study: float,
    outer_studies: int,
    weak_studies: int,
    validation_views: int,
    weak_views: int,
) -> dict[str, float]:
    base = max(float(seconds_per_study), float(config.get("finish_seconds_per_study_floor", 0.25)))
    safety = float(config.get("finish_inference_safety_factor", 1.75))
    if safety < 1.0:
        raise ValueError("finish_inference_safety_factor must be >=1")
    # Inner timing already includes validation_views. Never scale weak prediction
    # below that measured per-study time: DICOM decode/loader overhead remains.
    weak_view_scale = max(1.0, float(weak_views) / max(float(validation_views), 1.0))
    return {
        "outer_oof_inference": base * int(outer_studies) * safety,
        "weak_oof_inference": base * int(weak_studies) * safety * weak_view_scale,
        "bootstrap": float(config.get("finish_bootstrap_reserve_seconds", 120.0)),
        "serialization": float(config.get("finish_serialization_reserve_seconds", 180.0)),
        "loader_startup": float(config.get("finish_loader_startup_reserve_seconds", 120.0)),
    }


def train_fold(config: dict, fold: int) -> Path:
    """One-GPU nested training plus leakage-safe weak cross-fitting."""
    validate_competition_config(config, purpose="train")
    start = time.time()
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )
    runtime = resolve_runtime(config)
    seed = int(config.get("seed", 2026))
    n_folds = int(config.get("n_folds", 3))
    if n_folds < 3:
        raise ValueError("nested selection requires at least 3 folds")
    if not 0 <= fold < n_folds:
        raise ValueError("invalid fold")
    inner_fold = int(config.get("inner_selection_fold", (fold + 1) % n_folds))
    if inner_fold == fold:
        raise ValueError("inner fold must differ from outer fold")

    validation_offsets = _offsets(
        config, "validation_tta_offsets", config.get("tta_center_offsets", [-1, 0, 1])
    )
    weak_offsets = _offsets(config, "weak_oof_tta_offsets", [0])
    validation_center_index = _central_view_index(validation_offsets)
    weak_center_index = _central_view_index(weak_offsets)

    seed_everything(seed + fold)
    root = Path(config["data_root"])
    df = load_train_csv(root / config.get("train_csv", "train.csv"))
    series_path = root / config.get("train_series_csv", "train_series.csv")

    preflight_payload = None
    if bool(config.get("preflight_before_train", True)):
        result = run_preflight(
            root,
            split="train",
            series_csv=series_path,
            study_uids=df["StudyInstanceUID"].tolist(),
            sample_size=int(config.get("preflight_sample_size", 24)),
            stream_mode="dual",
            seed=seed,
            max_decode_failure_rate=float(config.get("preflight_max_decode_failure_rate", 0.05)),
            max_file_decode_failure_rate=float(config.get("preflight_max_file_decode_failure_rate", 0.05)),
            strict=True,
        )
        preflight_payload = result.to_dict()
        print(result.summary())

    series = load_series_csv(series_path)
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    df = add_report_groups(df)
    df["fold"] = make_balanced_gold_folds(df, n_folds, seed)
    df["crossfit_fold"] = assign_crossfit_folds(df, n_folds)
    gold = gold_mask(df)
    outer_mask = gold & df["fold"].eq(fold)
    inner_mask = gold & df["fold"].eq(inner_fold)
    image_teacher_mask = (~gold) & df["crossfit_fold"].eq(fold)
    if not outer_mask.any() or not inner_mask.any():
        raise ValueError("outer and inner folds must both contain gold studies")

    stage2 = bool(_stage1_source(config))
    selection_holdout_mask = outer_mask | inner_mask
    final_holdout_mask = outer_mask.copy()
    if not stage2:
        selection_holdout_mask = selection_holdout_mask | image_teacher_mask
        final_holdout_mask = final_holdout_mask | image_teacher_mask

    selection_heldout = set(df.loc[selection_holdout_mask, "report_group"].astype(str))
    final_heldout = set(df.loc[final_holdout_mask, "report_group"].astype(str))
    selection_train_mask = ~df["report_group"].astype(str).isin(selection_heldout)
    final_train_mask = ~df["report_group"].astype(str).isin(final_heldout)

    states = state_dataframe(df)
    selection_pseudo, selection_conf, selection_cal, _ = _teacher_arrays(
        df, states, gold, selection_train_mask, config, fold, use_image_teacher=False
    )
    selection_targets, selection_weights = combine_gold_and_pseudo(
        df, selection_pseudo, selection_conf, float(config.get("gold_weight", 8.0))
    )

    final_report_pseudo, final_report_conf, final_cal, _ = _teacher_arrays(
        df, states, gold, final_train_mask, config, fold, use_image_teacher=False
    )
    if stage2:
        final_pseudo, final_conf, final_cal, stage1_source_meta = _teacher_arrays(
            df, states, gold, final_train_mask, config, fold, use_image_teacher=True
        )
    else:
        final_pseudo, final_conf = final_report_pseudo, final_report_conf
        stage1_source_meta = None
    final_targets, final_weights = combine_gold_and_pseudo(
        df, final_pseudo, final_conf, float(config.get("gold_weight", 8.0))
    )

    index = build_series_index(series, df["StudyInstanceUID"], mode="dual")
    si = np.flatnonzero(selection_train_mask.to_numpy())
    fi = np.flatnonzero(final_train_mask.to_numpy())
    ii = np.flatnonzero(inner_mask.to_numpy())
    oi = np.flatnonzero(outer_mask.to_numpy())
    wi = np.flatnonzero(image_teacher_mask.to_numpy())

    selection_ds = KneeStudyDataset(
        df.iloc[si]["StudyInstanceUID"].tolist(), index, _dataset_config(config, root, train=True),
        selection_targets[si], selection_weights[si], True
    )
    final_ds = KneeStudyDataset(
        df.iloc[fi]["StudyInstanceUID"].tolist(), index, _dataset_config(config, root, train=True),
        final_targets[fi], final_weights[fi], True
    )
    validation_cfg = _dataset_config(config, root, train=False, tta_offsets=validation_offsets)
    inner_ds = KneeStudyDataset(
        df.iloc[ii]["StudyInstanceUID"].tolist(), index, validation_cfg,
        df.iloc[ii][TARGETS].to_numpy(np.float32), None, False
    )
    outer_ds = KneeStudyDataset(
        df.iloc[oi]["StudyInstanceUID"].tolist(), index, validation_cfg,
        df.iloc[oi][TARGETS].to_numpy(np.float32), None, False
    )
    weak_ds = None
    if not stage2:
        weak_ds = KneeStudyDataset(
            df.iloc[wi]["StudyInstanceUID"].tolist(), index,
            _dataset_config(config, root, train=False, tta_offsets=weak_offsets), train=False
        )

    selection_loader, selection_sampler, selection_trusted = _balanced_train_loader(
        selection_ds, gold.iloc[si].to_numpy(), selection_weights[si], config, runtime, seed + fold, budget
    )
    eval_batch_size = max(1, int(config.get("oof_batch_size", config.get("inference_batch_size", 2))))
    inner_loader = DataLoader(
        inner_ds, batch_size=eval_batch_size, shuffle=False,
        **runtime.loader_kwargs(seed=seed + 10_000 + fold)
    )
    outer_loader = DataLoader(
        outer_ds, batch_size=eval_batch_size, shuffle=False,
        **runtime.loader_kwargs(seed=seed + 20_000 + fold)
    )
    weak_loader = None
    if weak_ds is not None:
        weak_loader = DataLoader(
            weak_ds, batch_size=max(1, int(config.get("weak_oof_batch_size", eval_batch_size))), shuffle=False,
            **runtime.loader_kwargs(seed=seed + 30_000 + fold)
        )

    spec = _model_spec(config)
    outdir = Path(config.get("output_dir", "runs/model")) / f"fold{fold}"
    outdir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = outdir / "best.pt"

    assignments = df[["StudyInstanceUID", "report_group", "fold", "crossfit_fold"]].copy()
    assignments["role"] = "weak_train"
    assignments.loc[selection_train_mask & gold, "role"] = "gold_train_selection"
    assignments.loc[inner_mask, "role"] = "inner_selection"
    assignments.loc[outer_mask, "role"] = "outer_oof"
    assignments.loc[image_teacher_mask, "role"] = "image_teacher_train" if stage2 else "weak_oof"
    assignments.to_csv(outdir / "fold_assignments.csv", index=False)
    (outdir / "metadata_repair.json").write_text(json.dumps(metadata_stats, indent=2))
    if preflight_payload is not None:
        (outdir / "preflight.json").write_text(json.dumps(preflight_payload, indent=2))
    if selection_cal is not None:
        (outdir / "calibration_selection.json").write_text(json.dumps(selection_cal.to_dict(), indent=2))
    if final_cal is not None:
        (outdir / "calibration.json").write_text(json.dumps(final_cal.to_dict(), indent=2))

    supervision_plan = {
        "selection": _supervision_summary(selection_targets[si], selection_weights[si]),
        "final": _supervision_summary(final_targets[fi], final_weights[fi]),
    }
    (outdir / "supervision_plan.json").write_text(json.dumps(supervision_plan, indent=2))
    if stage2:
        stage2_delta = _stage2_delta_summary(
            final_report_pseudo,
            final_report_conf,
            final_pseudo,
            final_conf,
            image_teacher_mask.to_numpy(),
        )
        stage2_delta["source"] = stage1_source_meta
        (outdir / "stage2_supervision.json").write_text(json.dumps(stage2_delta, indent=2))

    model = _build_model(spec, config, runtime.device)
    max_epochs = int(config.get("epochs", 8))
    optimizer, scheduler, scaler = _optimizer_bundle(model, config, max_epochs, runtime)
    best_score, best_epoch, bad_epochs = -np.inf, 0, 0
    history = []
    diagnostics = {"selection": _empty_training_diagnostics(), "retrain": _empty_training_diagnostics()}
    epoch_times: list[float] = []
    prediction_seconds_per_study: list[float] = []
    budget_limited = False

    finish_components = _finish_components(
        config,
        float(config.get("finish_seconds_per_study_fallback", 5.0)),
        len(oi),
        len(wi) if not stage2 else 0,
        len(validation_offsets),
        len(weak_offsets),
    )

    for epoch in range(max_epochs):
        epoch_start = time.monotonic()
        metrics = _train_epoch(
            model, selection_loader, selection_sampler, optimizer, scaler, runtime, config, epoch
        )
        scheduler.step()
        _, inner_probability, inner_truth, inner_details = predict(
            model,
            inner_loader,
            runtime.device,
            runtime,
            budget=budget,
            label="inner TTA evaluation",
            central_view_index=validation_center_index,
            initial_batch_guard_seconds=float(config.get("prediction_initial_batch_guard_seconds", 180.0)),
            return_details=True,
        )
        inner_score = macro_auc_from_arrays(inner_truth, inner_probability)[0]
        inner_center_score = macro_auc_from_arrays(
            inner_truth, inner_details["central_probability"]
        )[0]
        prediction_seconds_per_study.append(float(inner_details["seconds_per_study"]))
        finish_components = _finish_components(
            config,
            float(np.median(prediction_seconds_per_study)),
            len(oi),
            len(wi) if not stage2 else 0,
            len(validation_offsets),
            len(weak_offsets),
        )

        epoch_seconds = time.monotonic() - epoch_start
        epoch_times.append(epoch_seconds)
        _add_training_diagnostics(diagnostics["selection"], metrics)
        row = {
            "phase": "selection",
            "epoch": epoch + 1,
            "train_loss": metrics["loss"],
            "inner_macro_auc": inner_score,
            "inner_center_macro_auc": inner_center_score,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
            "train_batches": metrics["actual_batches"],
        }
        history.append(row)
        print(row)
        if np.isfinite(inner_score) and inner_score > best_score:
            best_score, best_epoch, bad_epochs = float(inner_score), epoch + 1, 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(config.get("patience", 2)):
            break

        train_estimate = float(np.median(epoch_times))
        possible_selected_epoch = epoch + 2
        future = {
            "possible_phase_b": 1.15 * possible_selected_epoch * train_estimate,
            **finish_components,
        }
        if not budget.can_start(sum(future.values())):
            budget_limited = True
            print("[budget] stopping selection early to reserve retraining + all finish work")
            break

    if best_epoch == 0:
        raise RuntimeError("no finite inner-selection score")

    del model
    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()
    seed_everything(seed + 100_000 + fold)
    final_loader, final_sampler, final_trusted = _balanced_train_loader(
        final_ds, gold.iloc[fi].to_numpy(), final_weights[fi], config, runtime, seed + 50_000 + fold, budget
    )
    train_estimate = float(np.median(epoch_times)) if epoch_times else 60.0
    budget.require_components(
        {"phase_b_retrain": 1.15 * best_epoch * train_estimate, **finish_components},
        label="Phase-B retraining plus outer/weak OOF/bootstrap/serialization",
    )

    model = _build_model(spec, config, runtime.device)
    optimizer, scheduler, scaler = _optimizer_bundle(model, config, best_epoch, runtime)
    for epoch in range(best_epoch):
        remaining_epochs = best_epoch - epoch
        budget.require_components(
            {"remaining_retrain": 1.15 * remaining_epochs * train_estimate, **finish_components},
            label=f"retrain epoch {epoch + 1} and finish work",
        )
        epoch_start = time.monotonic()
        metrics = _train_epoch(model, final_loader, final_sampler, optimizer, scaler, runtime, config, epoch)
        scheduler.step()
        epoch_seconds = time.monotonic() - epoch_start
        _add_training_diagnostics(diagnostics["retrain"], metrics)
        history.append(
            {
                "phase": "retrain",
                "epoch": epoch + 1,
                "train_loss": metrics["loss"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": epoch_seconds,
                "train_batches": metrics["actual_batches"],
            }
        )

    stage_name = "stage2" if stage2 else "stage1"
    budget.require_components(finish_components, label="checkpoint and final prediction work")
    torch.save(
        {
            "model": model.state_dict(),
            "model_spec": spec,
            "config": config,
            "stream_names": list(DUAL_STREAMS),
            "fold": fold,
            "inner_fold": inner_fold,
            "stage": stage_name,
            "selected_epoch": best_epoch,
            "inner_score": best_score,
            "validation_tta_offsets": list(validation_offsets),
            "stage1_teacher_source": stage1_source_meta,
        },
        final_checkpoint,
    )

    outer_uids, outer_probability, outer_truth, outer_details = predict(
        model,
        outer_loader,
        runtime.device,
        runtime,
        budget=budget,
        label="outer OOF TTA inference",
        central_view_index=validation_center_index,
        initial_batch_guard_seconds=float(config.get("prediction_initial_batch_guard_seconds", 180.0)),
        return_details=True,
    )
    _save_predictions(outdir / "oof.csv", outer_uids, outer_probability)
    _save_predictions(outdir / "oof_center.csv", outer_uids, outer_details["central_probability"])

    weak_details = None
    weak_oof_path = outdir / "weak_oof.csv"
    if weak_loader is not None:
        weak_uids, weak_probability, _, weak_details = predict(
            model,
            weak_loader,
            runtime.device,
            runtime,
            budget=budget,
            label="weak OOF teacher inference",
            central_view_index=weak_center_index,
            initial_batch_guard_seconds=float(config.get("prediction_initial_batch_guard_seconds", 180.0)),
            return_details=True,
        )
        _save_predictions(weak_oof_path, weak_uids, weak_probability)
    else:
        weak_oof_path.unlink(missing_ok=True)

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
    (outdir / "training_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    outer_score = macro_auc_from_arrays(outer_truth, outer_probability)[0]
    outer_center_score = macro_auc_from_arrays(outer_truth, outer_details["central_probability"])[0]
    (outdir / "selection.json").write_text(
        json.dumps(
            {
                "outer_fold": fold,
                "inner_fold": inner_fold,
                "selected_epoch": best_epoch,
                "inner_macro_auc": best_score,
                "outer_macro_auc": float(outer_score),
                "outer_center_macro_auc": float(outer_center_score),
                "validation_tta_offsets": list(validation_offsets),
                "selection_gold_train": int((selection_train_mask & gold).sum()),
                "final_gold_train": int((final_train_mask & gold).sum()),
                "budget_limited_selection": bool(budget_limited),
                "stage": stage_name,
                "selection_image_teacher": False,
                "final_image_teacher": bool(stage2),
                "image_teacher_training_rows": int(image_teacher_mask.sum()) if stage2 else 0,
                "stage1_teacher_source": stage1_source_meta,
            },
            indent=2,
        )
    )
    (outdir / "sampling.json").write_text(
        json.dumps(
            {
                "selection_trusted": int(selection_trusted.sum()),
                "selection_general": int((~selection_trusted).sum()),
                "final_trusted": int(final_trusted.sum()),
                "final_general": int((~final_trusted).sum()),
                "stage1_weak_oof_rows": int(image_teacher_mask.sum()) if not stage2 else 0,
                "stage2_image_teacher_rows": int(image_teacher_mask.sum()) if stage2 else 0,
            },
            indent=2,
        )
    )

    bootstrap_started = time.monotonic()
    budget.require(float(config.get("finish_bootstrap_reserve_seconds", 120.0)), label="outer bootstrap")
    bootstrap = bootstrap_macro_auc(
        outer_truth,
        outer_probability,
        n_bootstrap=int(config.get("n_bootstrap", 2000)),
        seed=seed + fold,
    )
    (outdir / "bootstrap.json").write_text(json.dumps(bootstrap.to_dict(), indent=2))
    bootstrap_seconds = time.monotonic() - bootstrap_started
    (outdir / "config.json").write_text(json.dumps(config, indent=2))

    runtime_payload = {
        "elapsed_seconds": float(time.time() - start),
        "device": runtime.device_name,
        "runtime": runtime.describe(),
        "num_workers": runtime.num_workers,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(runtime.device)) if runtime.device.type == "cuda" else 0,
        "git_sha_env": os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT"),
        "budget": budget.to_dict(),
        "finish_budget_components_seconds": finish_components,
        "outer_prediction": {key: value for key, value in outer_details.items() if key != "central_probability"},
        "weak_prediction": (
            {key: value for key, value in weak_details.items() if key != "central_probability"}
            if weak_details is not None else None
        ),
        "bootstrap_seconds": float(bootstrap_seconds),
        "validation_tta_offsets": list(validation_offsets),
        "weak_oof_tta_offsets": list(weak_offsets),
    }
    (outdir / "runtime.json").write_text(json.dumps(runtime_payload, indent=2))
    return final_checkpoint
