from __future__ import annotations

import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .calibration import calibration_split_mask, fit_calibration
from .constants import TARGETS
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
from .evaluation import bootstrap_macro_auc
from .metrics import macro_auc
from .model import MultiSeriesKneeNet
from .preflight import run_preflight
from .report_labels import combine_gold_and_pseudo, label_dataframe, state_dataframe
from .runtime import autocast, make_scaler, resolve_runtime, unwrap, wrap_parallel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weighted_bce(logits, target, weight):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def pairwise_ranking_loss(logits, target, pairs_per_target: int = 32):
    """A small AUC-surrogate ranking term, disabled unless configured.

    Pseudo-label probabilities above/below 0.5 define positive/negative pools.
    The BCE term remains the primary objective; this term only encourages a
    positive study to rank above a negative study for each target.
    """
    losses = []
    for j in range(logits.shape[1]):
        pos = torch.nonzero(target[:, j] > 0.5, as_tuple=False).flatten()
        neg = torch.nonzero(target[:, j] <= 0.5, as_tuple=False).flatten()
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        n = min(int(pairs_per_target), int(pos.numel() * neg.numel()))
        if n <= 0:
            continue
        pi = pos[torch.randint(pos.numel(), (n,), device=logits.device)]
        ni = neg[torch.randint(neg.numel(), (n,), device=logits.device)]
        losses.append(nn.functional.softplus(-(logits[pi, j] - logits[ni, j])).mean())
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


@torch.no_grad()
def predict(model, loader, device, runtime=None):
    """Score a loader. Passing ``runtime`` enables mixed precision on GPU."""
    model.eval()
    uids, probs, targets = [], [], []
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


def _dataset_config(config: dict, root: Path, split: str) -> DatasetConfig:
    return DatasetConfig(
        data_root=str(root),
        split=split,
        n_slices=int(config.get("n_slices", 16)),
        image_size=int(config.get("image_size", 224)),
        noise_std=float(config.get("noise_std", 0.02)) if split == "train" else 0.0,
        slice_dropout=float(config.get("slice_dropout", 0.08)) if split == "train" else 0.0,
        input_mode=str(config.get("input_mode", "2d")),
        triplet_gap=int(config.get("triplet_gap", 1)),
        strict_dicom=bool(config.get("strict_dicom", False)),
    )


def train_fold(config: dict, fold: int) -> Path:
    seed = int(config.get("seed", 2026))
    seed_everything(seed + fold)
    root = Path(config["data_root"])
    df = load_train_csv(root / config.get("train_csv", "train.csv"))

    if bool(config.get("preflight_before_train", True)):
        result = run_preflight(
            root,
            split="train",
            series_csv=root / config.get("train_series_csv", "train_series.csv"),
            study_uids=df["StudyInstanceUID"].astype(str).tolist(),
            sample_size=int(config.get("preflight_sample_size", 24)),
            stream_mode=str(config.get("stream_mode", "best")),
            seed=seed,
            max_failure_rate=float(config.get("preflight_max_failure_rate", 0.05)),
            strict=True,
        )
        print(result.summary())

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    print(f"[metadata] {metadata_stats}")

    df = add_report_groups(df)
    df["fold"] = make_balanced_gold_folds(
        df,
        int(config.get("n_folds", 3)),
        seed,
    )
    if not (df["fold"].eq(fold) & gold_mask(df)).any():
        raise ValueError(f"fold {fold} contains no gold validation studies")

    val_mask = df["fold"].eq(fold) & gold_mask(df)
    val_groups = set(df.loc[val_mask, "report_group"].astype(str))
    train_mask = ~df["report_group"].astype(str).isin(val_groups)

    calibration = None
    if bool(config.get("calibrate_teacher", True)):
        states = state_dataframe(df)
        gold_values = df[TARGETS].to_numpy(dtype=np.float64)
        calib_mask = calibration_split_mask(gold_mask(df).to_numpy(), df["fold"].to_numpy(), fold)
        if calib_mask.sum() >= int(config.get("min_calibration_studies", 8)):
            calibration = fit_calibration(
                states[calib_mask],
                gold_values[calib_mask],
                alpha=float(config.get("calibration_alpha", 5.0)),
            )
            pseudo = calibration.apply(states)
            conf = calibration.confidence(states)
            print(
                f"[fold {fold}] teacher calibrated on {int(calib_mask.sum())} gold studies "
                "outside the validation fold"
            )
        else:
            print(
                f"[fold {fold}] only {int(calib_mask.sum())} gold studies available for "
                "calibration; falling back to fixed rule probabilities"
            )
            pseudo, conf = label_dataframe(df)
    else:
        pseudo, conf = label_dataframe(df)

    targets, weights = combine_gold_and_pseudo(
        df,
        pseudo,
        conf,
        float(config.get("gold_weight", 8.0)),
    )

    stream_mode = str(config.get("stream_mode", "best"))
    series_index = build_series_index(series, df["StudyInstanceUID"].astype(str), stream_mode)
    dcfg = _dataset_config(config, root, "train")
    val_dcfg = _dataset_config(config, root, "validation")
    val_dcfg.split = "train"  # validation cases live under the training DICOM tree

    ti = np.flatnonzero(train_mask.to_numpy())
    vi = np.flatnonzero(val_mask.to_numpy())
    train_ds = KneeStudyDataset(
        df.iloc[ti]["StudyInstanceUID"].astype(str).tolist(),
        series_index,
        dcfg,
        targets[ti],
        weights[ti],
        True,
    )
    val_ds = KneeStudyDataset(
        df.iloc[vi]["StudyInstanceUID"].astype(str).tolist(),
        series_index,
        val_dcfg,
        targets[vi],
        np.ones_like(weights[vi]),
        False,
    )

    runtime = resolve_runtime(config)
    print(f"[fold {fold}] {runtime.describe()}")
    batch_size = int(config.get("batch_size", 2))
    loader_kwargs = runtime.loader_kwargs()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, batch_size),
        shuffle=False,
        **loader_kwargs,
    )

    device = runtime.device
    model = MultiSeriesKneeNet(
        len(train_ds.stream_names),
        pretrained=bool(config.get("pretrained", False)),
        dropout=float(config.get("dropout", 0.25)),
        in_channels=train_ds.in_channels,
        backbone=str(config.get("backbone", "resnet18")),
        target_attention=bool(config.get("target_attention", False)),
    ).to(device)
    model = wrap_parallel(model, runtime, bool(config.get("data_parallel", True)))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 2e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scaler = make_scaler(runtime)

    outdir = Path(config.get("output_dir", "runs")) / f"fold{fold}"
    outdir.mkdir(parents=True, exist_ok=True)
    df[["StudyInstanceUID", "report_group", "fold"]].to_csv(
        outdir / "fold_assignments.csv", index=False
    )
    (outdir / "metadata_repair.json").write_text(
        json.dumps(metadata_stats, indent=2), encoding="utf-8"
    )

    best, bad, history = -np.inf, 0, []
    best_predictions = best_targets = None
    rank_weight = float(config.get("rank_loss_weight", 0.0))
    rank_pairs = int(config.get("rank_pairs_per_target", 32))

    for epoch in range(int(config.get("epochs", 10))):
        model.train()
        running = seen = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            v = batch["volumes"].to(device, non_blocking=True)
            present = batch["present"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            w = batch["weight"].to(device, non_blocking=True)
            with autocast(runtime):
                logits = model(v, present)
                bce = weighted_bce(logits, y, w)
                rank = pairwise_ranking_loss(logits, y, rank_pairs) if rank_weight > 0 else logits.new_zeros(())
                loss = bce + rank_weight * rank
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * len(y)
            seen += len(y)

        uids, p, yv = predict(model, val_loader, device, runtime)
        score = macro_auc(yv, p) if yv is not None and len(yv) else float("nan")
        row = {
            "epoch": epoch + 1,
            "train_loss": running / max(seen, 1),
            "macro_auc": score,
        }
        history.append(row)
        print(row)

        if np.isfinite(score) and score > best:
            best, bad = score, 0
            torch.save(
                {
                    "model": unwrap(model).state_dict(),
                    "config": config,
                    "stream_names": train_ds.stream_names,
                    "in_channels": train_ds.in_channels,
                    "fold": fold,
                    "score": score,
                },
                outdir / "best.pt",
            )
            oof = pd.DataFrame(p, columns=TARGETS)
            oof.insert(0, "StudyInstanceUID", uids)
            oof.to_csv(outdir / "oof.csv", index=False)
            best_predictions, best_targets = p, yv
        else:
            bad += 1
            if bad >= int(config.get("patience", 3)):
                break

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
    (outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if calibration is not None:
        (outdir / "calibration.json").write_text(
            json.dumps(calibration.to_dict(), indent=2), encoding="utf-8"
        )

    if best_predictions is not None and best_targets is not None and len(best_targets):
        result = bootstrap_macro_auc(
            best_targets,
            best_predictions,
            n_bootstrap=int(config.get("n_bootstrap", 2000)),
            seed=seed,
        )
        print(f"[fold {fold}] {result.summary()}")
        (outdir / "bootstrap.json").write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
    return outdir / "best.pt"
