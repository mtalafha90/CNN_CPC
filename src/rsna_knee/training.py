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

from .calibration import calibration_split_mask, fit_calibration
from .constants import DUAL_STREAMS, TARGETS
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
from .preflight import run_preflight
from .report_labels import combine_gold_and_pseudo, label_dataframe, state_dataframe
from .runtime import autocast, make_scaler, resolve_runtime


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def confidence_gated_ranking_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    pairs_per_target: int = 32,
    min_confidence: float = 0.35,
    positive_threshold: float = 0.75,
    negative_threshold: float = 0.25,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for j in range(logits.shape[1]):
        trusted = weight[:, j] >= float(min_confidence)
        pos = torch.nonzero(trusted & (target[:, j] >= positive_threshold), as_tuple=False).flatten()
        neg = torch.nonzero(trusted & (target[:, j] <= negative_threshold), as_tuple=False).flatten()
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        n = min(int(pairs_per_target), int(pos.numel() * neg.numel()))
        pi = pos[torch.randint(pos.numel(), (n,), device=logits.device)]
        ni = neg[torch.randint(neg.numel(), (n,), device=logits.device)]
        margin = logits[pi, j] - logits[ni, j]
        pair_weight = torch.minimum(weight[pi, j], weight[ni, j]).clamp(max=1.0)
        losses.append((nn.functional.softplus(-margin) * pair_weight).mean())
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


@torch.no_grad()
def predict(model, loader, device, runtime=None):
    model.eval()
    uids: list[str] = []
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        with (autocast(runtime) if runtime is not None else nullcontext()):
            logits = model(
                batch["volumes"].to(device, non_blocking=True),
                batch["present"].to(device, non_blocking=True),
            )
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        uids.extend(list(batch["study_uid"]))
        if "target" in batch:
            targets.append(batch["target"].numpy())
    p = np.concatenate(probs) if probs else np.empty((0, len(TARGETS)), np.float32)
    y = np.concatenate(targets) if targets else None
    return uids, p, y


def _dataset_config(config: dict, root: Path, *, train: bool) -> DatasetConfig:
    return DatasetConfig(
        data_root=str(root),
        split="train",
        n_slices=int(config.get("n_slices", 16)),
        image_size=int(config.get("image_size", 224)),
        noise_std=float(config.get("noise_std", 0.02)) if train else 0.0,
        slice_dropout=float(config.get("slice_dropout", 0.08)) if train else 0.0,
        triplet_gap=int(config.get("triplet_gap", 1)),
        strict_dicom=bool(config.get("strict_dicom", False)),
    )


def _model_spec(config: dict) -> dict:
    return {
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
    }


def train_fold(config: dict, fold: int) -> Path:
    start_time = time.time()
    seed = int(config.get("seed", 2026))
    n_folds = int(config.get("n_folds", 3))
    if fold < 0 or fold >= n_folds:
        raise ValueError(f"fold must be in [0,{n_folds - 1}]")
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
    print(f"[metadata] {metadata_stats}")

    df = add_report_groups(df)
    df["fold"] = make_balanced_gold_folds(df, n_folds, seed)
    val_mask = df["fold"].eq(fold) & gold_mask(df)
    if not val_mask.any():
        raise ValueError(f"fold {fold} contains no gold validation studies")

    val_groups = set(df.loc[val_mask, "report_group"].astype(str))
    train_mask = ~df["report_group"].astype(str).isin(val_groups)

    states = state_dataframe(df)
    calibration = None
    if bool(config.get("calibrate_teacher", True)):
        calib_mask = calibration_split_mask(gold_mask(df).to_numpy(), df["fold"].to_numpy(), fold)
        if calib_mask.sum() >= int(config.get("min_calibration_studies", 8)):
            calibration = fit_calibration(
                states[calib_mask],
                df.loc[calib_mask, TARGETS].to_numpy(dtype=np.float64),
                alpha=float(config.get("calibration_alpha", 5.0)),
            )
            pseudo = calibration.apply(states)
            confidence = calibration.confidence(states)
            print(f"[fold {fold}] teacher calibrated on {int(calib_mask.sum())} out-of-fold gold studies")
        else:
            pseudo, confidence = label_dataframe(df)
            print(f"[fold {fold}] insufficient gold calibration studies; using fixed report rules")
    else:
        pseudo, confidence = label_dataframe(df)

    train_targets, train_weights = combine_gold_and_pseudo(
        df, pseudo, confidence, float(config.get("gold_weight", 8.0))
    )

    series_index = build_series_index(series, df["StudyInstanceUID"], mode="dual")
    ti = np.flatnonzero(train_mask.to_numpy())
    vi = np.flatnonzero(val_mask.to_numpy())

    train_ds = KneeStudyDataset(
        df.iloc[ti]["StudyInstanceUID"].tolist(),
        series_index,
        _dataset_config(config, root, train=True),
        train_targets[ti],
        train_weights[ti],
        True,
    )
    val_ds = KneeStudyDataset(
        df.iloc[vi]["StudyInstanceUID"].tolist(),
        series_index,
        _dataset_config(config, root, train=False),
        df.iloc[vi][TARGETS].to_numpy(dtype=np.float32),
        None,
        False,
    )

    runtime = resolve_runtime(config)
    print(f"[fold {fold}] {runtime.describe()}")
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)

    loader_kwargs = runtime.loader_kwargs()
    batch_size = int(config.get("batch_size", 2))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    spec = _model_spec(config)
    if train_ds.stream_names != DUAL_STREAMS:
        raise RuntimeError("dataset stream contract does not match canonical DUAL_STREAMS")
    model = KneeMILNet(
        spec["n_streams"],
        spec["n_slices"],
        in_channels=spec["in_channels"],
        pretrained_weights=bool(config.get("pretrained", True)),
        normalize_input=spec["normalize_input"],
        dropout=spec["dropout"],
        encoder_batch_size=spec["encoder_batch_size"],
        gradient_checkpointing=spec["gradient_checkpointing"],
    ).to(runtime.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(config.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=float(config.get("min_lr", 1e-6))
    )
    scaler = make_scaler(runtime)

    outdir = Path(config.get("output_dir", "runs/model")) / f"fold{fold}"
    outdir.mkdir(parents=True, exist_ok=True)
    df[["StudyInstanceUID", "report_group", "fold"]].to_csv(outdir / "fold_assignments.csv", index=False)
    (outdir / "metadata_repair.json").write_text(json.dumps(metadata_stats, indent=2), encoding="utf-8")
    if preflight_payload is not None:
        (outdir / "preflight.json").write_text(json.dumps(preflight_payload, indent=2), encoding="utf-8")
    if calibration is not None:
        (outdir / "calibration.json").write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")

    best_score = -np.inf
    bad_epochs = 0
    history: list[dict] = []
    best_predictions = best_targets = None
    rank_weight = float(config.get("rank_loss_weight", 0.10))

    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            target = batch["target"].to(runtime.device, non_blocking=True)
            weight = batch["weight"].to(runtime.device, non_blocking=True)

            with autocast(runtime):
                logits = model(volumes, present)
                bce = weighted_bce(logits, target, weight)
                rank = (
                    confidence_gated_ranking_loss(
                        logits,
                        target,
                        weight,
                        pairs_per_target=int(config.get("rank_pairs_per_target", 32)),
                        min_confidence=float(config.get("rank_min_confidence", 0.35)),
                        positive_threshold=float(config.get("rank_positive_threshold", 0.75)),
                        negative_threshold=float(config.get("rank_negative_threshold", 0.25)),
                    )
                    if rank_weight > 0
                    else logits.new_zeros(())
                )
                loss = bce + rank_weight * rank

            scaler.scale(loss).backward()
            grad_clip = float(config.get("grad_clip", 1.0))
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * len(target)
            seen += len(target)

        scheduler.step()
        uids, probabilities, y_val = predict(model, val_loader, runtime.device, runtime)
        score = macro_auc_from_arrays(y_val, probabilities)[0] if y_val is not None else float("nan")
        row = {
            "epoch": epoch + 1,
            "train_loss": running / max(seen, 1),
            "macro_auc": score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(row)

        if np.isfinite(score) and score > best_score:
            best_score = score
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_spec": spec,
                    "config": config,
                    "stream_names": train_ds.stream_names,
                    "fold": fold,
                    "score": float(score),
                },
                outdir / "best.pt",
            )
            oof = pd.DataFrame(probabilities, columns=TARGETS)
            oof.insert(0, "StudyInstanceUID", uids)
            oof.to_csv(outdir / "oof.csv", index=False)
            best_predictions, best_targets = probabilities, y_val
        else:
            bad_epochs += 1
            if bad_epochs >= int(config.get("patience", 5)):
                break

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
    (outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if best_predictions is None or best_targets is None:
        raise RuntimeError(f"fold {fold} never produced a finite validation macro AUC")

    bootstrap = bootstrap_macro_auc(
        best_targets,
        best_predictions,
        n_bootstrap=int(config.get("n_bootstrap", 2000)),
        seed=seed + fold,
    )
    print(f"[fold {fold}] {bootstrap.summary()}")
    (outdir / "bootstrap.json").write_text(json.dumps(bootstrap.to_dict(), indent=2), encoding="utf-8")

    runtime_payload = {
        "elapsed_seconds": float(time.time() - start_time),
        "device": runtime.device_name,
        "visible_gpus": int(runtime.visible_gpus),
        "runtime": runtime.describe(),
        "num_workers": int(runtime.num_workers),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(runtime.device)) if runtime.device.type == "cuda" else 0
        ),
        "git_sha_env": os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT"),
    }
    (outdir / "runtime.json").write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")
    return outdir / "best.pt"
