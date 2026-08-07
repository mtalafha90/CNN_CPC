from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import N_TARGETS, TARGETS


def load_train_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"StudyInstanceUID", "Report", *TARGETS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"train.csv missing columns: {missing}")
    return df


def load_test_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "StudyInstanceUID" not in df.columns:
        raise ValueError("test.csv must contain StudyInstanceUID")
    if "Report" not in df.columns:
        df["Report"] = ""
    return df


def gold_mask(df: pd.DataFrame) -> pd.Series:
    return df[TARGETS].notna().any(axis=1)


def load_series_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"series CSV missing columns: {missing}")
    df = df.copy()
    df["Fluid_Sensitive"] = df["Fluid_Sensitive"].astype(bool)
    df["Fat_Suppression"] = df["Fat_Suppression"].astype(bool)
    df["Anatomical_Plane"] = df["Anatomical_Plane"].astype(str).str.capitalize()
    return df


def normalize_report(text: str) -> str:
    text = "" if not isinstance(text, str) else text
    text = text.replace("İ", "I").replace("ı", "i").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def report_hash(text: str) -> str:
    return hashlib.sha1(normalize_report(text).encode("utf-8")).hexdigest()


def add_report_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["report_group"] = out["Report"].fillna("").map(report_hash)
    return out


def make_balanced_gold_folds(df: pd.DataFrame, n_splits: int = 3, seed: int = 2026) -> pd.Series:
    """Greedy multilabel folds for gold rows; duplicate normalized reports stay together."""
    out = pd.Series(-1, index=df.index, dtype=int)
    gold = add_report_groups(df.loc[gold_mask(df)]).copy()
    if gold.empty:
        return out
    rng = np.random.default_rng(seed)
    prevalence = np.maximum(gold[TARGETS].fillna(0).sum(axis=0).to_numpy(float), 1)
    groups = []
    for group, part in gold.groupby("report_group", sort=False):
        labels = part[TARGETS].fillna(0).max(axis=0).to_numpy(float)
        rarity = float(np.sum(labels / prevalence))
        groups.append((group, part.index.tolist(), labels, rarity, rng.random()))
    groups.sort(key=lambda x: (x[3], len(x[1]), x[4]), reverse=True)
    fold_pos = np.zeros((n_splits, N_TARGETS))
    fold_n = np.zeros(n_splits)
    target_pos = gold[TARGETS].fillna(0).sum(axis=0).to_numpy(float) / n_splits
    target_n = len(gold) / n_splits
    for _, indices, labels, _, _ in groups:
        scores = []
        for f in range(n_splits):
            pp = np.mean(((fold_pos[f] + labels - target_pos) / np.maximum(target_pos, 1.0)) ** 2)
            sp = ((fold_n[f] + len(indices) - target_n) / max(target_n, 1.0)) ** 2
            scores.append(pp + 0.2 * sp)
        f = int(np.argmin(scores))
        out.loc[indices] = f
        fold_pos[f] += labels
        fold_n[f] += len(indices)
    return out


def select_series(series_df: pd.DataFrame, study_uid: str, mode: str = "best") -> dict[str, str | None]:
    part = series_df[series_df["StudyInstanceUID"].astype(str) == str(study_uid)]
    result: dict[str, str | None] = {}
    for plane in ["Sagittal", "Coronal", "Axial"]:
        p = part[part["Anatomical_Plane"].astype(str).str.capitalize() == plane].copy()
        key = plane.lower()
        if p.empty:
            if mode == "dual":
                result[f"{key}_fluid"] = None
                result[f"{key}_structural"] = None
            else:
                result[key] = None
            continue
        fluid = 2 * p["Fluid_Sensitive"].astype(int) + 2 * p["Fat_Suppression"].astype(int)
        structural = 2 * (~p["Fat_Suppression"].astype(bool)).astype(int) + (~p["Fluid_Sensitive"].astype(bool)).astype(int)
        if mode == "dual":
            result[f"{key}_fluid"] = str(p.iloc[int(np.argmax(fluid.to_numpy()))]["SeriesInstanceUID"])
            result[f"{key}_structural"] = str(p.iloc[int(np.argmax(structural.to_numpy()))]["SeriesInstanceUID"])
        else:
            result[key] = str(p.iloc[int(np.argmax((fluid + 0.25 * structural).to_numpy()))]["SeriesInstanceUID"])
    return result


def build_series_index(series_df: pd.DataFrame, studies: Iterable[str], mode: str = "best") -> dict[str, dict[str, str | None]]:
    return {str(uid): select_series(series_df, str(uid), mode=mode) for uid in studies}
