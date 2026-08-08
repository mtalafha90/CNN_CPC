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
from .cotrain import assign_crossfit_folds, consensus_arrays, load_image_predictions
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


def macro_weighted_bce(logits, target, weight):
    cell = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    numerator = (cell * weight).sum(dim=0)
    denominator = weight.sum(dim=0)
    valid = denominator > 0
    return (
        (numerator[valid] / denominator[valid].clamp_min(1e-8)).mean()
        if valid.any()
        else logits.sum() * 0.0
    )


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


@torch.no_grad()
def predict(model, loader, device, runtime=None):
    model.eval()
    uids, probs, targets = [], [], []
    for batch in loader:
        volumes = batch["volumes"]
        if volumes.ndim != 7:
            with (autocast(runtime) if runtime is not None else nullcontext()):
                logits = model(
                    volumes.to(device, non_blocking=True),
                    batch["present"].to(device, non_blocking=True),
                )
            probability = torch.sigmoid(logits.float()).cpu().numpy()
        else:
            # Evaluation TTA: [B,V,K,S,C,H,W]. Decode happened once in Dataset.
            present = batch["present"].to(device, non_blocking=True)
            view_probs = []
            for view in range(volumes.shape[1]):
                with (autocast(runtime) if runtime is not None else nullcontext()):
                    logits = model(volumes[:, view].to(device, non_blocking=True), present)
                view_probs.append(torch.sigmoid(logits.float()))
            probability = torch.stack(view_probs).mean(dim=0).cpu().numpy()
        probs.append(probability)
        uids.extend(list(batch["study_uid"]))
        if "target" in batch:
            targets.append(batch["target"].numpy())
    return (
        uids,
        np.concatenate(probs) if probs else np.empty((0, len(TARGETS)), np.float32),
        np.concatenate(targets) if targets else None,
    )


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
        "normalize_input": bool(config.get("normalize_input", True)),
        "encoder_batch_size": int(config.get("encoder_batch_size", 24)),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "transformer_layers": int(config.get("transformer_layers", 2)),
        "transformer_heads": int(config.get("transformer_heads", 8)),
        "transformer_ff_mult": float(config.get("transformer_ff_mult", 2.0)),
        "pathology_layers": int(config.get("pathology_layers", 1)),
    }


def _build_model(spec, config, device):
    model = KneeMILNet(
        spec["n_streams"],
        spec["n_slices"],
        in_channels=3,
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


def _teacher_arrays(df, states, gold, allowed_mask, config, fold: int):
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

    # Leakage-safe Stage 2: outer fold k may use only weak predictions from the
    # Stage-1 model k. That model never saw outer-gold fold k nor those weak rows.
    stage1_root = config.get("cotrain_stage1_root")
    if stage1_root:
        image_path = Path(stage1_root) / f"fold{fold}" / "weak_oof.csv"
        if not image_path.is_file():
            raise FileNotFoundError(f"missing leakage-safe Stage-1 teacher: {image_path}")
        image = load_image_predictions([str(image_path)], df["StudyInstanceUID"])
        allowed_image_rows = (~gold.to_numpy()) & df["crossfit_fold"].eq(fold).to_numpy()
        available_rows = np.isfinite(image).any(axis=1)
        if np.any(available_rows & ~allowed_image_rows):
            raise ValueError(
                f"Stage-1 weak teacher for outer fold {fold} contains predictions outside its held-out weak fold"
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
        )
    return pseudo, confidence, calibration


def _save_predictions(path: Path, uids, probabilities):
    frame = pd.DataFrame(probabilities, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    frame.to_csv(path, index=False)


def _balanced_train_loader(dataset, gold_rows, weights, config, runtime, seed):
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
        rank=0,
        world_size=1,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        **runtime.loader_kwargs(seed=seed),
    )
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


def _train_epoch(model, loader, sampler, optimizer, scaler, runtime, config, epoch):
    sampler.set_epoch(epoch)
    model.train()
    total, seen = 0.0, 0
    rank_weight = float(config.get("rank_loss_weight", 0.10))
    pair_counts = np.zeros(len(TARGETS), dtype=np.int64)
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        volumes = batch["volumes"].to(runtime.device, non_blocking=True)
        present = batch["present"].to(runtime.device, non_blocking=True)
        target = batch["target"].to(runtime.device, non_blocking=True)
        weight = batch["weight"].to(runtime.device, non_blocking=True)
        with autocast(runtime):
            logits = model(volumes, present)
            bce = macro_weighted_bce(logits, target, weight)
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
    return {
        "loss": total / max(seen, 1),
        "rank_pairs": {target: int(pair_counts[j]) for j, target in enumerate(TARGETS)},
    }


def train_fold(config: dict, fold: int) -> Path:
    """One-GPU nested training with leakage-safe weak cross-fitting."""
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
    weak_oof_mask = (~gold) & df["crossfit_fold"].eq(fold)
    if not outer_mask.any() or not inner_mask.any():
        raise ValueError("outer and inner folds must both contain gold studies")

    selection_heldout = set(
        df.loc[outer_mask | inner_mask | weak_oof_mask, "report_group"].astype(str)
    )
    selection_train_mask = ~df["report_group"].astype(str).isin(selection_heldout)
    final_heldout = set(df.loc[outer_mask | weak_oof_mask, "report_group"].astype(str))
    final_train_mask = ~df["report_group"].astype(str).isin(final_heldout)

    states = state_dataframe(df)
    selection_pseudo, selection_conf, selection_cal = _teacher_arrays(
        df, states, gold, selection_train_mask, config, fold
    )
    selection_targets, selection_weights = combine_gold_and_pseudo(
        df, selection_pseudo, selection_conf, float(config.get("gold_weight", 8.0))
    )
    final_pseudo, final_conf, final_cal = _teacher_arrays(
        df, states, gold, final_train_mask, config, fold
    )
    final_targets, final_weights = combine_gold_and_pseudo(
        df, final_pseudo, final_conf, float(config.get("gold_weight", 8.0))
    )

    index = build_series_index(series, df["StudyInstanceUID"], mode="dual")
    si = np.flatnonzero(selection_train_mask.to_numpy())
    fi = np.flatnonzero(final_train_mask.to_numpy())
    ii = np.flatnonzero(inner_mask.to_numpy())
    oi = np.flatnonzero(outer_mask.to_numpy())
    wi = np.flatnonzero(weak_oof_mask.to_numpy())

    selection_ds = KneeStudyDataset(
        df.iloc[si]["StudyInstanceUID"].tolist(), index, _dataset_config(config, root, train=True),
        selection_targets[si], selection_weights[si], True
    )
    final_ds = KneeStudyDataset(
        df.iloc[fi]["StudyInstanceUID"].tolist(), index, _dataset_config(config, root, train=True),
        final_targets[fi], final_weights[fi], True
    )
    eval_cfg = _dataset_config(config, root, train=False)
    inner_ds = KneeStudyDataset(
        df.iloc[ii]["StudyInstanceUID"].tolist(), index, eval_cfg,
        df.iloc[ii][TARGETS].to_numpy(np.float32), None, False
    )
    outer_ds = KneeStudyDataset(
        df.iloc[oi]["StudyInstanceUID"].tolist(), index, eval_cfg,
        df.iloc[oi][TARGETS].to_numpy(np.float32), None, False
    )
    weak_ds = KneeStudyDataset(
        df.iloc[wi]["StudyInstanceUID"].tolist(), index, eval_cfg, train=False
    )

    selection_loader, selection_sampler, selection_trusted = _balanced_train_loader(
        selection_ds, gold.iloc[si].to_numpy(), selection_weights[si], config, runtime, seed + fold
    )
    batch_size = int(config.get("batch_size", 2))
    inner_loader = DataLoader(
        inner_ds, batch_size=batch_size, shuffle=False,
        **runtime.loader_kwargs(seed=seed + 10_000 + fold)
    )
    outer_loader = DataLoader(
        outer_ds, batch_size=batch_size, shuffle=False,
        **runtime.loader_kwargs(seed=seed + 20_000 + fold)
    )
    weak_loader = DataLoader(
        weak_ds, batch_size=batch_size, shuffle=False,
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
    assignments.loc[weak_oof_mask, "role"] = "weak_oof"
    assignments.to_csv(outdir / "fold_assignments.csv", index=False)
    (outdir / "metadata_repair.json").write_text(json.dumps(metadata_stats, indent=2))
    if preflight_payload is not None:
        (outdir / "preflight.json").write_text(json.dumps(preflight_payload, indent=2))
    if selection_cal is not None:
        (outdir / "calibration_selection.json").write_text(json.dumps(selection_cal.to_dict(), indent=2))
    if final_cal is not None:
        (outdir / "calibration.json").write_text(json.dumps(final_cal.to_dict(), indent=2))

    # Phase A: select only the epoch count. Continue only if there is enough
    # budget left for a conservative estimate of Phase B plus evaluation.
    model = _build_model(spec, config, runtime.device)
    max_epochs = int(config.get("epochs", 12))
    optimizer, scheduler, scaler = _optimizer_bundle(model, config, max_epochs, runtime)
    best_score, best_epoch, bad_epochs = -np.inf, 0, 0
    history = []
    pair_totals = {"selection": {t: 0 for t in TARGETS}, "retrain": {t: 0 for t in TARGETS}}
    epoch_times = []
    budget_limited = False

    for epoch in range(max_epochs):
        epoch_start = time.monotonic()
        metrics = _train_epoch(
            model, selection_loader, selection_sampler, optimizer, scaler, runtime, config, epoch
        )
        scheduler.step()
        _, inner_probability, inner_truth = predict(model, inner_loader, runtime.device, runtime)
        inner_score = macro_auc_from_arrays(inner_truth, inner_probability)[0]
        epoch_seconds = time.monotonic() - epoch_start
        epoch_times.append(epoch_seconds)
        for target, count in metrics["rank_pairs"].items():
            pair_totals["selection"][target] += int(count)
        row = {
            "phase": "selection",
            "epoch": epoch + 1,
            "train_loss": metrics["loss"],
            "inner_macro_auc": inner_score,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        print(row)
        if np.isfinite(inner_score) and inner_score > best_score:
            best_score, best_epoch, bad_epochs = float(inner_score), epoch + 1, 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(config.get("patience", 3)):
            break

        estimate = float(np.median(epoch_times))
        next_epoch = epoch + 2
        # Worst case: the next epoch becomes selected and Phase B must run that
        # many epochs. Reserve an additional evaluation-equivalent epoch.
        future_required = estimate * (1.0 + 1.15 * next_epoch + 0.75)
        if not budget.can_start(future_required):
            budget_limited = True
            print("[budget] stopping model selection early to reserve Phase-B/evaluation time")
            break

    if best_epoch == 0:
        raise RuntimeError("no finite inner-selection score")

    del model
    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()
    seed_everything(seed + 100_000 + fold)
    final_loader, final_sampler, final_trusted = _balanced_train_loader(
        final_ds, gold.iloc[fi].to_numpy(), final_weights[fi], config, runtime, seed + 50_000 + fold
    )
    estimate = float(np.median(epoch_times)) if epoch_times else 60.0
    budget.require(1.15 * best_epoch * estimate + 0.75 * estimate, label="Phase-B retraining and evaluation")

    model = _build_model(spec, config, runtime.device)
    optimizer, scheduler, scaler = _optimizer_bundle(model, config, best_epoch, runtime)
    for epoch in range(best_epoch):
        budget.require(1.15 * estimate, label=f"retrain epoch {epoch + 1}")
        epoch_start = time.monotonic()
        metrics = _train_epoch(model, final_loader, final_sampler, optimizer, scaler, runtime, config, epoch)
        scheduler.step()
        epoch_seconds = time.monotonic() - epoch_start
        for target, count in metrics["rank_pairs"].items():
            pair_totals["retrain"][target] += int(count)
        history.append(
            {
                "phase": "retrain",
                "epoch": epoch + 1,
                "train_loss": metrics["loss"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": epoch_seconds,
            }
        )

    torch.save(
        {
            "model": model.state_dict(),
            "model_spec": spec,
            "config": config,
            "stream_names": list(DUAL_STREAMS),
            "fold": fold,
            "inner_fold": inner_fold,
            "selected_epoch": best_epoch,
            "inner_score": best_score,
        },
        final_checkpoint,
    )

    budget.require(0.25 * estimate, label="outer/weak evaluation")
    outer_uids, outer_probability, outer_truth = predict(model, outer_loader, runtime.device, runtime)
    weak_uids, weak_probability, _ = predict(model, weak_loader, runtime.device, runtime)
    _save_predictions(outdir / "oof.csv", outer_uids, outer_probability)
    _save_predictions(outdir / "weak_oof.csv", weak_uids, weak_probability)
    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
    outer_score = macro_auc_from_arrays(outer_truth, outer_probability)[0]
    (outdir / "selection.json").write_text(
        json.dumps(
            {
                "outer_fold": fold,
                "inner_fold": inner_fold,
                "selected_epoch": best_epoch,
                "inner_macro_auc": best_score,
                "outer_macro_auc": float(outer_score),
                "selection_gold_train": int((selection_train_mask & gold).sum()),
                "final_gold_train": int((final_train_mask & gold).sum()),
                "budget_limited_selection": bool(budget_limited),
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
                "weak_oof": int(weak_oof_mask.sum()),
            },
            indent=2,
        )
    )
    (outdir / "ranking_pairs.json").write_text(json.dumps(pair_totals, indent=2))
    bootstrap = bootstrap_macro_auc(
        outer_truth,
        outer_probability,
        n_bootstrap=int(config.get("n_bootstrap", 2000)),
        seed=seed + fold,
    )
    (outdir / "bootstrap.json").write_text(json.dumps(bootstrap.to_dict(), indent=2))
    (outdir / "config.json").write_text(json.dumps(config, indent=2))
    runtime_payload = {
        "elapsed_seconds": float(time.time() - start),
        "device": runtime.device_name,
        "runtime": runtime.describe(),
        "num_workers": runtime.num_workers,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(runtime.device)) if runtime.device.type == "cuda" else 0,
        "git_sha_env": os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT"),
        "budget": budget.to_dict(),
    }
    (outdir / "runtime.json").write_text(json.dumps(runtime_payload, indent=2))
    return final_checkpoint
