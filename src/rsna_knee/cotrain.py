"""Leakage-safe cross-fitted image/report co-training utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import add_report_groups


def assign_crossfit_folds(df: pd.DataFrame, n_folds: int = 3) -> pd.Series:
    if n_folds < 2:
        raise ValueError("n_folds must be >=2")
    work = add_report_groups(df) if "report_group" not in df.columns else df

    def fold(group: str) -> int:
        digest = hashlib.sha1(str(group).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % n_folds

    return work["report_group"].astype(str).map(fold).astype(int)


def _read_predictions(path: str | Path, study_uids: pd.Series) -> np.ndarray:
    frame = pd.read_csv(path)
    required = {"StudyInstanceUID", *TARGETS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame[["StudyInstanceUID", *TARGETS]].copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        dup = frame.loc[frame["StudyInstanceUID"].duplicated(), "StudyInstanceUID"].iloc[0]
        raise ValueError(f"cross-fitted image prediction repeated for study {dup}")
    ordered = pd.DataFrame({"StudyInstanceUID": study_uids.astype(str)}).merge(
        frame, on="StudyInstanceUID", how="left", validate="one_to_one"
    )
    return ordered[TARGETS].to_numpy(np.float32)


def load_fold_image_teacher(
    stage1_root: str | Path,
    fold: int,
    df: pd.DataFrame,
    gold_rows: np.ndarray,
) -> np.ndarray:
    """Load exactly the weak OOF predictions safe for one outer fold.

    Stage-1 model ``fold=k`` excludes outer-gold fold ``k`` and weak
    ``crossfit_fold=k``. Therefore only ``fold{k}/weak_oof.csv`` may supervise
    Stage-2 outer fold ``k``.
    """
    if "crossfit_fold" not in df.columns:
        raise ValueError("crossfit_fold must be assigned before loading Stage-2 teacher")
    path = Path(stage1_root) / f"fold{int(fold)}" / "weak_oof.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing leakage-safe Stage-1 teacher: {path}")
    image = _read_predictions(path, df["StudyInstanceUID"])
    gold_rows = np.asarray(gold_rows, dtype=bool)
    expected = (~gold_rows) & df["crossfit_fold"].eq(int(fold)).to_numpy()
    available = np.isfinite(image).any(axis=1)
    if np.any(available & ~expected):
        bad = df.loc[available & ~expected, "StudyInstanceUID"].astype(str).head(5).tolist()
        raise ValueError(f"fold {fold} Stage-1 teacher contains unsafe studies: {bad}")
    if np.any(expected & ~available):
        missing = int((expected & ~available).sum())
        raise ValueError(f"fold {fold} Stage-1 teacher is missing {missing} expected weak OOF studies")
    return image


def consensus_arrays(
    report_p: np.ndarray,
    report_conf: np.ndarray,
    image_p: np.ndarray,
    *,
    positive_threshold: float = 0.80,
    negative_threshold: float = 0.20,
    agreement_weight: float = 0.90,
    disagreement_weight: float = 0.05,
    blend: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse independent image/report teachers while preserving uncertainty."""
    report_p = np.asarray(report_p, np.float32)
    report_conf = np.asarray(report_conf, np.float32)
    image_p = np.asarray(image_p, np.float32)
    if report_p.shape != report_conf.shape or report_p.shape != image_p.shape:
        raise ValueError("teacher arrays must have identical shapes")
    probability = report_p.copy()
    confidence = report_conf.copy()
    available = np.isfinite(image_p)
    probability[available] = float(blend) * report_p[available] + (1 - float(blend)) * image_p[available]
    report_pos = report_p >= positive_threshold
    image_pos = image_p >= positive_threshold
    report_neg = report_p <= negative_threshold
    image_neg = image_p <= negative_threshold
    agree = available & ((report_pos & image_pos) | (report_neg & image_neg))
    disagree = available & ((report_pos & image_neg) | (report_neg & image_pos))
    confidence[agree] = float(agreement_weight)
    confidence[disagree] = np.minimum(confidence[disagree], float(disagreement_weight))
    return probability, confidence
