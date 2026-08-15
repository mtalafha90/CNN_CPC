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
FALSE_TOKENS = {"false", "f", "no", "n", "0", "0.0"}
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


def coerce_bool(values: pd.Series, *, preserve_unknown: bool = False) -> pd.Series:
    """Parse metadata flags without converting missing/unknown values to True.

    ``preserve_unknown=True`` returns pandas' nullable Boolean dtype so DICOM
    metadata can repair unknown sequence flags before routing. The public helper
    keeps the historical conservative default of mapping unknown values to False.
    """
    result = pd.Series(pd.NA, index=values.index, dtype="boolean")
    if pd.api.types.is_bool_dtype(values):
        result.loc[values.notna()] = values.loc[values.notna()].astype(bool)
    elif pd.api.types.is_numeric_dtype(values):
        known = values.notna()
        result.loc[known] = values.loc[known].astype(float).ne(0.0)
    else:
        text = values.astype("string").str.strip().str.lower()
        result.loc[text.isin(TRUE_TOKENS)] = True
        result.loc[text.isin(FALSE_TOKENS)] = False
    return result if preserve_unknown else result.fillna(False).astype(bool)


def normalise_plane(values: pd.Series) -> pd.Series:
    mapping = {
        "sagittal": "Sagittal", "sag": "Sagittal", "sagital": "Sagittal",
        "coronal": "Coronal", "cor": "Coronal",
        "axial": "Axial", "ax": "Axial", "transverse": "Axial",
    }
    text = values.astype("string").str.strip().str.lower()
    return text.map(mapping).fillna("").astype(str)


def load_series_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"}
    _require_columns(df, required, "series CSV")
    if df[["StudyInstanceUID", "SeriesInstanceUID"]].isna().any().any():
        raise ValueError("series CSV contains missing study/series UID values")
    df = df.copy()
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    df["SeriesInstanceUID"] = df["SeriesInstanceUID"].astype(str)
    df["Fluid_Sensitive"] = coerce_bool(df["Fluid_Sensitive"], preserve_unknown=True)
    df["Fat_Suppression"] = coerce_bool(df["Fat_Suppression"], preserve_unknown=True)
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
        groups.append((group, part.index.tolist(), labels, float(np.sum(labels / prevalence)), rng.random()))
    if len(groups) < n_splits:
        raise ValueError(
            f"cannot create {n_splits} non-empty gold folds from only {len(groups)} report groups"
        )
    groups.sort(key=lambda x: (x[3], len(x[1]), x[4]), reverse=True)
    fold_pos = np.zeros((n_splits, N_TARGETS))
    fold_n = np.zeros(n_splits)
    target_pos = gold[TARGETS].fillna(0).sum(axis=0).to_numpy(float) / n_splits
    target_n = len(gold) / n_splits
    for group_index, (_, indices, labels, _, _) in enumerate(groups):
        # Seed every fold once, then score a hypothetical assignment by the
        # GLOBAL fold matrix. Scoring only the candidate fold can prefer a fold
        # already near its target and starve an empty fold.
        candidate_folds = (
            np.flatnonzero(fold_n == 0).astype(int).tolist()
            if group_index < n_splits
            else list(range(n_splits))
        )
        costs = []
        for fold in candidate_folds:
            trial_pos = fold_pos.copy()
            trial_n = fold_n.copy()
            trial_pos[fold] += labels
            trial_n[fold] += len(indices)
            label_cost = np.mean(
                ((trial_pos - target_pos[None, :]) / np.maximum(target_pos[None, :], 1.0)) ** 2
            )
            size_cost = np.mean(((trial_n - target_n) / max(target_n, 1.0)) ** 2)
            costs.append(label_cost + 0.2 * size_cost)
        chosen = int(candidate_folds[int(np.argmin(costs))])
        out.loc[indices] = chosen
        fold_pos[chosen] += labels
        fold_n[chosen] += len(indices)
    return out


def build_validation_manifest(
    df: pd.DataFrame,
    *,
    outer_fold: int,
    n_splits: int = 3,
    seed: int = 2026,
    inner_fold: int | None = None,
) -> pd.DataFrame:
    """Return the explicit gold nested-validation manifest for one outer fold.

    The manifest is for inspection/reproducibility only. ``outer_validation`` is
    the untouched final OOF fold; ``inner_selection`` chooses training duration;
    the remaining gold rows are ``gold_train``. Non-gold studies are excluded.
    """
    if n_splits < 3:
        raise ValueError("nested validation manifest requires at least 3 folds")
    if not 0 <= int(outer_fold) < n_splits:
        raise ValueError("outer_fold is outside the configured fold range")
    if inner_fold is None:
        inner_fold = (int(outer_fold) + 1) % n_splits
    if not 0 <= int(inner_fold) < n_splits or int(inner_fold) == int(outer_fold):
        raise ValueError("inner_fold must be valid and differ from outer_fold")

    folds = make_balanced_gold_folds(df, n_splits=n_splits, seed=seed)
    gold = gold_mask(df)
    manifest = df.loc[gold, ["StudyInstanceUID", *TARGETS]].copy()
    manifest.insert(1, "fold", folds.loc[gold].astype(int).to_numpy())
    manifest.insert(2, "role", "gold_train")
    manifest.loc[manifest["fold"].eq(int(inner_fold)), "role"] = "inner_selection"
    manifest.loc[manifest["fold"].eq(int(outer_fold)), "role"] = "outer_validation"
    manifest.insert(3, "outer_fold", int(outer_fold))
    manifest.insert(4, "inner_fold", int(inner_fold))
    return manifest.sort_values(["role", "StudyInstanceUID"]).reset_index(drop=True)


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

        fluid_flag = p["Fluid_Sensitive"].fillna(False).astype(bool).to_numpy()
        fat_flag = p["Fat_Suppression"].fillna(False).astype(bool).to_numpy()
        fluid_score = 2 * fluid_flag.astype(int) + 2 * fat_flag.astype(int)
        structural_score = 2 * (~fat_flag).astype(int) + (~fluid_flag).astype(int)

        if mode == "best":
            result[key] = p.at[_rank_indices(fluid_score + 0.25 * structural_score)[0], "SeriesInstanceUID"]
            continue
        if len(p) == 1:
            uid = p.at[0, "SeriesInstanceUID"]
            if fluid_score[0] >= structural_score[0]:
                result[f"{key}_fluid"], result[f"{key}_structural"] = uid, None
            else:
                result[f"{key}_fluid"], result[f"{key}_structural"] = None, uid
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
    uid = str(study_uid)
    return _select_from_study(series_df.loc[series_df["StudyInstanceUID"].astype(str).eq(uid)], mode)


def build_series_index(series_df: pd.DataFrame, studies: Iterable[str], mode: str = "dual") -> dict[str, dict[str, str | None]]:
    if mode not in {"best", "dual"}:
        raise ValueError("stream mode must be 'best' or 'dual'")
    work = series_df.copy()
    work["StudyInstanceUID"] = work["StudyInstanceUID"].astype(str)
    grouped = {uid: part for uid, part in work.groupby("StudyInstanceUID", sort=False)}
    empty = work.iloc[0:0]
    return {str(uid): _select_from_study(grouped.get(str(uid), empty), mode) for uid in studies}


def backfill_series_metadata(
    series_df: pd.DataFrame,
    data_root: str | Path,
    split: str = "train",
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Independently repair missing plane, fluid and fat-suppression metadata."""
    from .dicom import find_series_dir
    from .dicom_meta import read_series_metadata

    df = series_df.copy()
    missing_plane = df["Anatomical_Plane"].astype(str).str.strip().eq("")
    missing_fluid = df["Fluid_Sensitive"].isna()
    missing_fat = df["Fat_Suppression"].isna()
    needs = missing_plane | missing_fluid | missing_fat
    targets = df.index[needs]
    if limit is not None:
        targets = targets[: int(limit)]

    stats = {
        "rows_needing_metadata": int(needs.sum()),
        "missing_plane": int(missing_plane.sum()),
        "missing_fluid": int(missing_fluid.sum()),
        "missing_fat_suppression": int(missing_fat.sum()),
        "inspected": len(targets),
        "repaired_plane": 0,
        "repaired_fluid": 0,
        "repaired_fat_suppression": 0,
    }
    for index in targets:
        row = df.loc[index]
        series_dir = find_series_dir(data_root, split, str(row["StudyInstanceUID"]), str(row["SeriesInstanceUID"]))
        if series_dir is None:
            continue
        metadata = read_series_metadata(series_dir)
        if missing_plane.loc[index] and metadata["Anatomical_Plane"] is not None:
            df.at[index, "Anatomical_Plane"] = metadata["Anatomical_Plane"]
            stats["repaired_plane"] += 1
        if missing_fluid.loc[index] and metadata["Fluid_Sensitive"] is not None:
            df.at[index, "Fluid_Sensitive"] = bool(metadata["Fluid_Sensitive"])
            stats["repaired_fluid"] += 1
        if missing_fat.loc[index] and metadata["Fat_Suppression"] is not None:
            df.at[index, "Fat_Suppression"] = bool(metadata["Fat_Suppression"])
            stats["repaired_fat_suppression"] += 1
    return df, stats
