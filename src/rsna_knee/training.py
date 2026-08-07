from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .constants import TARGETS
from .data import add_report_groups, build_series_index, gold_mask, load_series_csv, load_train_csv, make_balanced_gold_folds
from .dataset import DatasetConfig, KneeStudyDataset
from .metrics import macro_auc
from .model import MultiSeriesKneeNet
from .report_labels import combine_gold_and_pseudo, label_dataframe


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weighted_bce(logits, target, weight):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    uids, probs, targets = [], [], []
    for batch in loader:
        logits = model(batch["volumes"].to(device, non_blocking=True), batch["present"].to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits).cpu().numpy())
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

    pseudo, conf = label_dataframe(df)
    targets, weights = combine_gold_and_pseudo(df, pseudo, conf, float(config.get("gold_weight", 8.0)))

    val_mask = df["fold"].eq(fold) & gold_mask(df)
    val_groups = set(df.loc[val_mask, "report_group"].astype(str))
    train_mask = ~df["report_group"].astype(str).isin(val_groups)

    series_index = build_series_index(series, df["StudyInstanceUID"].astype(str), config.get("stream_mode", "best"))
    dcfg = DatasetConfig(
        data_root=str(root), split="train", n_slices=int(config.get("n_slices", 16)),
        image_size=int(config.get("image_size", 224)), noise_std=float(config.get("noise_std", 0.02)),
        slice_dropout=float(config.get("slice_dropout", 0.08)),
    )
    ti, vi = np.flatnonzero(train_mask.to_numpy()), np.flatnonzero(val_mask.to_numpy())
    train_ds = KneeStudyDataset(df.iloc[ti]["StudyInstanceUID"].astype(str).tolist(), series_index, dcfg, targets[ti], weights[ti], True)
    val_ds = KneeStudyDataset(df.iloc[vi]["StudyInstanceUID"].astype(str).tolist(), series_index, dcfg, targets[vi], np.ones_like(weights[vi]), False)
    workers, batch_size = int(config.get("num_workers", 2)), int(config.get("batch_size", 2))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, batch_size), shuffle=False, num_workers=workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiSeriesKneeNet(len(train_ds.stream_names), bool(config.get("pretrained", False)), float(config.get("dropout", 0.25))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 2e-4)), weight_decay=float(config.get("weight_decay", 1e-4)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    outdir = Path(config.get("output_dir", "runs")) / f"fold{fold}"
    outdir.mkdir(parents=True, exist_ok=True)
    best, bad, history = -np.inf, 0, []
    for epoch in range(int(config.get("epochs", 10))):
        model.train()
        running = seen = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            v, present = batch["volumes"].to(device), batch["present"].to(device)
            y, w = batch["target"].to(device), batch["weight"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = weighted_bce(model(v, present), y, w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * len(y)
            seen += len(y)
        uids, p, yv = predict(model, val_loader, device)
        score = macro_auc(yv, p) if yv is not None and len(yv) else float("nan")
        row = {"epoch": epoch + 1, "train_loss": running / max(seen, 1), "macro_auc": score}
        history.append(row)
        print(row)
        if np.isfinite(score) and score > best:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "config": config, "stream_names": train_ds.stream_names, "fold": fold, "score": score}, outdir / "best.pt")
            oof = pd.DataFrame(p, columns=TARGETS)
            oof.insert(0, "StudyInstanceUID", uids)
            oof.to_csv(outdir / "oof.csv", index=False)
        else:
            bad += 1
            if bad >= int(config.get("patience", 3)):
                break
    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
    (outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return outdir / "best.pt"
