from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import N_TARGETS, TARGETS


TRUE_TOKENS = {"true", "t", "yes", "y", "1", "1.0"}
PLANES = ("Sagittal", "Coronal", "Axial")


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def load_train_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, {"StudyInstanceUID", "Report", *TARGETS}, "train.csv")
    if df["StudyInstanceUID"].isna().any():
        raise ValueError("train.csv contains missing StudyInstanceUID values")
    df = df.copy()
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    if df["StudyInstanceUID"].duplicated().any():
        raise ValueError("train.csv contains duplicate StudyInstanceUID values")
    return df


def load_test_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, {"StudyInstanceUID"}, "test.csv")
    if df["StudyInstanceUID"].isna().any():
        raise ValueError("test.csv contains missing StudyInstanceUID values")
    df = df.copy()
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    if df["StudyInstanceUID"].duplicated().any():
        raise ValueError("test.csv contains duplicate StudyInstanceUID values")
    if "Report" not in df.columns:
        df["Report"] = ""
    return df


def gold_mask(df: pd.DataFrame) -> pd.Series:
    return df[TARGETS].notna().any(axis=1)


def coerce_bool(values: pd.Series) -> pd.Series:
    """Parse metadata flags without unsafe generic string truthiness."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(float).ne(0.0)
    return values.astype(str).str.strip().str.lower().isin(TRUE_TOKENS)


def normalise_plane(values: pd.Series) -> pd.Series:
    mapping = {
        "sagittal": "Sagittal",
        "sag": "Sagittal",
        "sagital": "Sagittal",
        "coronal": "Coronal",
        "cor": "Coronal",
        "axial": "Axial",
        "ax": "Axial",
        "transverse": "Axial",
    }
    text = values.astype(str).str.strip().str.lower()
    return text.map(mapping).fillna("").astype(str)


def load_series_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Fluid_Sensitive",
        "Fat_Suppression",
        "Anatomical_Plane",
    }
    _require_columns(df, required, "series CSV")
    if df[["StudyInstanceUID", "SeriesInstanceUID"]].isna().any().any():
        raise ValueError("series CSV contains missing study/series UID values")
    df = df.copy()
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    df["SeriesInstanceUID"] = df["SeriesInstanceUID"].astype(str)
    df["Fluid_Sensitive"] = coerce_bool(df["Fluid_Sensitive"])
    df["Fat_Suppression"] = coerce_bool(df["Fat_Suppression"])
    df["Anatomical_Plane"] = normalise_plane(df["Anatomical_Plane"])
    if df[["StudyInstanceUID", "SeriesInstanceUID"]].duplicated().any():
        raise ValueError("series CSV contains duplicate study/series rows")
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
    """Greedy multilabel folds; duplicate normalized gold reports stay together."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
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
        costs = []
        for fold in range(n_splits):
            label_cost = np.mean(
                ((fold_pos[fold] + labels - target_pos) / np.maximum(target_pos, 1.0)) ** 2
            )
            size_cost = ((fold_n[fold] + len(indices) - target_n) / max(target_n, 1.0)) ** 2
            costs.append(label_cost + 0.2 * size_cost)
        chosen = int(np.argmin(costs))
        out.loc[indices] = chosen
        fold_pos[chosen] += labels
        fold_n[chosen] += len(indices)
    return out


def _rank_indices(score: np.ndarray) -> list[int]:
    return np.argsort(-score, kind="mergesort").astype(int).tolist()


def _select_from_study(part: pd.DataFrame, mode: str) -> dict[str, str | None]:
    if mode not in {"best", "dual"}:
        raise ValueError("stream mode must be 'best' or 'dual'")

    result: dict[str, str | None] = {}
    for plane in PLANES:
        p = part.loc[part["Anatomical_Plane"].eq(plane)].reset_index(drop=True)
        key = plane.lower()
        if p.empty:
            if mode == "dual":
                result[f"{key}_fluid"] = None
                result[f"{key}_structural"] = None
            else:
                result[key] = None
            continue

        fluid_score = (
            2 * p["Fluid_Sensitive"].to_numpy(dtype=int)
            + 2 * p["Fat_Suppression"].to_numpy(dtype=int)
        )
        structural_score = (
            2 * (~p["Fat_Suppression"]).to_numpy(dtype=int)
            + (~p["Fluid_Sensitive"]).to_numpy(dtype=int)
        )

        if mode == "best":
            idx = _rank_indices(fluid_score + 0.25 * structural_score)[0]
            result[key] = p.at[idx, "SeriesInstanceUID"]
            continue

        # If only one sequence exists in a plane, put it in the semantic slot
        # it best matches instead of decoding/encoding the exact same MRI twice.
        if len(p) == 1:
            uid = p.at[0, "SeriesInstanceUID"]
            if fluid_score[0] >= structural_score[0]:
                result[f"{key}_fluid"] = uid
                result[f"{key}_structural"] = None
            else:
                result[f"{key}_fluid"] = None
                result[f"{key}_structural"] = uid
            continue

        fluid_idx = _rank_indices(fluid_score)[0]
        fluid_uid = p.at[fluid_idx, "SeriesInstanceUID"]
        structural_idx = _rank_indices(structural_score)[0]
        if p.at[structural_idx, "SeriesInstanceUID"] == fluid_uid:
            for candidate in _rank_indices(structural_score)[1:]:
                if p.at[candidate, "SeriesInstanceUID"] != fluid_uid:
                    structural_idx = candidate
                    break

        result[f"{key}_fluid"] = fluid_uid
        result[f"{key}_structural"] = p.at[structural_idx, "SeriesInstanceUID"]
    return result


def select_series(series_df: pd.DataFrame, study_uid: str, mode: str = "best") -> dict[str, str | None]:
    """Select one study's series. `best` remains the stable public default."""
    uid = str(study_uid)
    part = series_df.loc[series_df["StudyInstanceUID"].astype(str).eq(uid)]
    return _select_from_study(part, mode)


def build_series_index(
    series_df: pd.DataFrame,
    studies: Iterable[str],
    mode: str = "dual",
) -> dict[str, dict[str, str | None]]:
    """Build routing with one groupby instead of one full-table scan per study."""
    if mode not in {"best", "dual"}:
        raise ValueError("stream mode must be 'best' or 'dual'")
    work = series_df.copy()
    work["StudyInstanceUID"] = work["StudyInstanceUID"].astype(str)
    grouped = {uid: part for uid, part in work.groupby("StudyInstanceUID", sort=False)}
    empty = work.iloc[0:0]
    return {
        str(uid): _select_from_study(grouped.get(str(uid), empty), mode)
        for uid in studies
    }


def backfill_series_metadata(
    series_df: pd.DataFrame,
    data_root: str | Path,
    split: str = "train",
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill blank plane/sequence metadata from one DICOM header per series."""
    from .dicom import find_series_dir
    from .dicom_meta import read_series_metadata

    df = series_df.copy()
    blank = df["Anatomical_Plane"].astype(str).str.strip().eq("")
    targets = df.index[blank]
    if limit is not None:
        targets = targets[: int(limit)]

    stats = {"missing": int(blank.sum()), "inspected": len(targets), "repaired": 0}
    for index in targets:
        row = df.loc[index]
        series_dir = find_series_dir(
            data_root,
            split,
            str(row["StudyInstanceUID"]),
            str(row["SeriesInstanceUID"]),
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
