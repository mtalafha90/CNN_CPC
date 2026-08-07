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
from .data import add_report_groups, build_series_index, gold_mask, load_series_csv, load_train_csv, make_balanced_gold_folds
from .dataset import DatasetConfig, KneeStudyDataset
from .evaluation import bootstrap_macro_auc
from .metrics import macro_auc
from .model import MultiSeriesKneeNet
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


@torch.no_grad()
def predict(model, loader, device, runtime=None):
    """Score a loader. Passing `runtime` enables mixed precision on the GPU."""
    model.eval()
    uids, probs, targets = [], [], []
    for batch in loader:
        # Built per batch rather than reused, so the context is never re-entered.
        with (autocast(runtime) if runtime is not None else nullcontext()):
            logits = model(
                batch["volumes"].to(device, non_blocking=True),
                batch["present"].to(device, non_blocking=True),
            )
        # Cast back to fp32 before sigmoid so bf16 rounding cannot create ties
        # between studies, which would flatten the ranking the metric depends on.
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        uids.extend(list(batch["study_uid"]))
        if "target" in batch:
            targets.append(batch["target"].numpy())
    p = np.concatenate(probs) if probs else np.empty((0, len(TARGETS)), np.float32)
    y = np.concatenate(targets) if targets else None
    return uids, p, y


def train_fold(config: dict, fold: int) -> Path:
    seed_everything(int(config.get("seed", 2026)) + fold)
    root = Path(config["data_root"])
    df = load_train_csv(root / config.get("train_csv", "train.csv"))
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    df = add_report_groups(df)
    df["fold"] = make_balanced_gold_folds(df, int(config.get("n_folds", 3)), int(config.get("seed", 2026)))

    val_mask = df["fold"].eq(fold) & gold_mask(df)
    val_groups = set(df.loc[val_mask, "report_group"].astype(str))
    train_mask = ~df["report_group"].astype(str).isin(val_groups)

    # Build the teacher's soft labels. When calibration is enabled the fixed
    # rule probabilities are replaced by P(y=1 | state) estimated from the gold
    # studies OUTSIDE this validation fold, so the teacher never sees the labels
    # it will be scored against.
    calibration = None
    if bool(config.get("calibrate_teacher", True)):
        states = state_dataframe(df)
        gold_values = df[TARGETS].to_numpy(dtype=np.float64)
        calib_mask = calibration_split_mask(
            gold_mask(df).to_numpy(), df["fold"].to_numpy(), fold
        )
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
                f"outside the validation fold"
            )
        else:
            print(
                f"[fold {fold}] only {int(calib_mask.sum())} gold studies available for "
                "calibration; falling back to the fixed rule probabilities"
            )
            pseudo, conf = label_dataframe(df)
    else:
        pseudo, conf = label_dataframe(df)

    targets, weights = combine_gold_and_pseudo(df, pseudo, conf, float(config.get("gold_weight", 8.0)))

    series_index = build_series_index(series, df["StudyInstanceUID"].astype(str), config.get("stream_mode", "best"))
    dcfg = DatasetConfig(
        data_root=str(root), split="train", n_slices=int(config.get("n_slices", 16)),
        image_size=int(config.get("image_size", 224)), noise_std=float(config.get("noise_std", 0.02)),
        slice_dropout=float(config.get("slice_dropout", 0.08)),
    )
    ti, vi = np.flatnonzero(train_mask.to_numpy()), np.flatnonzero(val_mask.to_numpy())
    train_ds = KneeStudyDataset(df.iloc[ti]["StudyInstanceUID"].astype(str).tolist(), series_index, dcfg, targets[ti], weights[ti], True)
    val_ds = KneeStudyDataset(df.iloc[vi]["StudyInstanceUID"].astype(str).tolist(), series_index, dcfg, targets[vi], np.ones_like(weights[vi]), False)
    runtime = resolve_runtime(config)
    print(f"[fold {fold}] {runtime.describe()}")
    batch_size = int(config.get("batch_size", 2))
    loader_kwargs = runtime.loader_kwargs()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=max(1, batch_size), shuffle=False, **loader_kwargs)

    device = runtime.device
    model = MultiSeriesKneeNet(len(train_ds.stream_names), bool(config.get("pretrained", False)), float(config.get("dropout", 0.25))).to(device)
    model = wrap_parallel(model, runtime, bool(config.get("data_parallel", True)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 2e-4)), weight_decay=float(config.get("weight_decay", 1e-4)))
    scaler = make_scaler(runtime)

    outdir = Path(config.get("output_dir", "runs")) / f"fold{fold}"
    outdir.mkdir(parents=True, exist_ok=True)
    best, bad, history = -np.inf, 0, []
    best_predictions = best_targets = None
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
                loss = weighted_bce(model(v, present), y, w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * len(y)
            seen += len(y)
        uids, p, yv = predict(model, val_loader, device, runtime)
        score = macro_auc(yv, p) if yv is not None and len(yv) else float("nan")
        row = {"epoch": epoch + 1, "train_loss": running / max(seen, 1), "macro_auc": score}
        history.append(row)
        print(row)
        if np.isfinite(score) and score > best:
            best, bad = score, 0
            # unwrap so the checkpoint has no `module.` prefix and loads
            # identically whether or not DataParallel was used to train it.
            torch.save({"model": unwrap(model).state_dict(), "config": config, "stream_names": train_ds.stream_names, "fold": fold, "score": score}, outdir / "best.pt")
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

    # A single fold's macro-AUC comes from a handful of studies, so quote it
    # with an interval rather than pretending to three decimal places.
    if best_predictions is not None and best_targets is not None and len(best_targets):
        result = bootstrap_macro_auc(
            best_targets, best_predictions,
            n_bootstrap=int(config.get("n_bootstrap", 2000)),
            seed=int(config.get("seed", 2026)),
        )
        print(f"[fold {fold}] {result.summary()}")
        (outdir / "bootstrap.json").write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
    return outdir / "best.pt"
