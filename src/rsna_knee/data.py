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
    df["Fluid_Sensitive"] = coerce_bool(df["Fluid_Sensitive"])
    df["Fat_Suppression"] = coerce_bool(df["Fat_Suppression"])
    df["Anatomical_Plane"] = normalise_plane(df["Anatomical_Plane"])
    return df


TRUE_TOKENS = {"true", "t", "yes", "y", "1", "1.0"}
FALSE_TOKENS = {"false", "f", "no", "n", "0", "0.0"}


def coerce_bool(values: pd.Series) -> pd.Series:
    """Interpret a flag column as booleans, treating missing values as False.

    ``Series.astype(bool)`` is wrong here in two ways: NaN becomes True, and so
    does the *string* ``"False"``, because any non-empty string is truthy. Both
    would silently mark structural series as fluid sensitive and corrupt the
    routing in :func:`select_series`.
    """
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(float).ne(0.0)
    text = values.astype(str).str.strip().str.lower()
    result = pd.Series(False, index=values.index, dtype=bool)
    result[text.isin(TRUE_TOKENS)] = True
    # Anything that is neither a recognised true nor false token — including
    # blanks and "nan" — stays False, which is the safe default for routing.
    return result


def normalise_plane(values: pd.Series) -> pd.Series:
    """Normalise plane names to Sagittal / Coronal / Axial, else empty.

    Missing entries become ``""`` rather than the string ``"Nan"`` that
    ``astype(str)`` would produce, so :func:`backfill_series_metadata` can
    recognise them as gaps to fill.
    """
    text = values.astype(str).str.strip().str.lower()
    mapping = {
        "sagittal": "Sagittal", "sag": "Sagittal", "sagital": "Sagittal",
        "coronal": "Coronal", "cor": "Coronal",
        "axial": "Axial", "ax": "Axial", "transverse": "Axial",
    }
    normalised = text.map(mapping)
    return normalised.fillna("").astype(str)


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


def backfill_series_metadata(
    series_df: pd.DataFrame,
    data_root: str | Path,
    split: str = "train",
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill blank plane and sequence flags from the DICOM headers.

    `train_series.csv` is authoritative wherever it is populated, so only rows
    with a blank `Anatomical_Plane` are touched. Such a row is invisible to
    :func:`select_series`, which then returns ``None`` for that plane and hands
    the model a stream of zeros — a silent loss rather than an error, which is
    why it is worth repairing.

    Reads one DICOM per affected series. Returns the repaired frame and a count
    of what changed, so the caller can log how much was missing.
    """
    # Imported lazily: `dicom` pulls in torch, and `data` is otherwise usable
    # without it (the CLI loads CSVs long before any model is built).
    from .dicom import find_series_dir
    from .dicom_meta import read_series_metadata

    df = series_df.copy()
    if "Anatomical_Plane" not in df.columns:
        df["Anatomical_Plane"] = ""

    blank = df["Anatomical_Plane"].astype(str).str.strip().eq("")
    targets = df.index[blank]
    if limit is not None:
        targets = targets[:limit]

    stats = {"missing": int(blank.sum()), "inspected": len(targets), "repaired": 0}
    for index in targets:
        row = df.loc[index]
        series_dir = find_series_dir(
            data_root, split, str(row["StudyInstanceUID"]), str(row["SeriesInstanceUID"])
        )
        if series_dir is None:
            continue
        metadata = read_series_metadata(series_dir)
        if metadata["Anatomical_Plane"] is None:
            continue
        df.at[index, "Anatomical_Plane"] = metadata["Anatomical_Plane"]
        if metadata["Fluid_Sensitive"] is not None:
            df.at[index, "Fluid_Sensitive"] = bool(metadata["Fluid_Sensitive"])
        if metadata["Fat_Suppression"] is not None:
            df.at[index, "Fat_Suppression"] = bool(metadata["Fat_Suppression"])
        stats["repaired"] += 1

    return df, stats
