from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import load_train_csv
from .metrics import macro_auc
from .report_labels import label_dataframe


def tune_alpha(train_csv: str | Path, oof_csvs, grid: int = 101) -> dict:
    train = load_train_csv(train_csv)
    gold = train[TARGETS].notna().any(axis=1)
    merged = train.loc[gold, ["StudyInstanceUID", "Report", *TARGETS]].copy()
    y = merged[TARGETS].fillna(0).to_numpy(float)
    preds = [pd.read_csv(path) for path in oof_csvs]
    image = np.full((len(merged), len(TARGETS)), np.nan, float)
    for j, target in enumerate(TARGETS):
        vals = []
        for p in preds:
            vals.append(merged[["StudyInstanceUID"]].merge(p[["StudyInstanceUID", target]], on="StudyInstanceUID", how="left")[target].to_numpy(float))
        image[:, j] = np.nanmean(np.stack(vals), axis=0)
    report, _ = label_dataframe(merged)
    valid = np.isfinite(image).all(axis=1)
    if not valid.any():
        raise ValueError("no gold study has complete OOF predictions")
    y, image, report = y[valid], image[valid], report[valid]
    best = {"alpha": 1.0, "macro_auc": macro_auc(y, image)}
    for alpha in np.linspace(0, 1, grid):
        score = macro_auc(y, alpha * image + (1 - alpha) * report)
        if np.isfinite(score) and (not np.isfinite(best["macro_auc"]) or score > best["macro_auc"]):
            best = {"alpha": float(alpha), "macro_auc": float(score)}
    return best
