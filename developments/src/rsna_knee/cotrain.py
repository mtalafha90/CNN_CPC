"""Leakage-safe cross-fitted image/report co-training utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

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


def _candidate_roots(stage1_source: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(stage1_source, (str, Path)):
        roots = [Path(stage1_source)]
    else:
        roots = [Path(value) for value in stage1_source]
    if not roots:
        raise ValueError("at least one Stage-1 candidate root is required")
    return roots


def choose_fold_candidate_root(
    stage1_source: str | Path | Iterable[str | Path],
    fold: int,
) -> tuple[Path, dict]:
    """Choose a Stage-1 candidate using only the current fold's inner score."""
    candidates = []
    inner_fold = None
    validation_offsets = None
    for order, root in enumerate(_candidate_roots(stage1_source)):
        fold_dir = root / f"fold{int(fold)}"
        selection_path = fold_dir / "selection.json"
        weak_path = fold_dir / "weak_oof.csv"
        checkpoint = fold_dir / "best.pt"
        if not selection_path.is_file():
            raise FileNotFoundError(f"missing Stage-1 selection metadata: {selection_path}")
        if not weak_path.is_file():
            raise FileNotFoundError(f"missing Stage-1 weak OOF teacher: {weak_path}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing Stage-1 checkpoint: {checkpoint}")
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        if str(payload.get("stage", "")) != "stage1":
            raise ValueError(f"candidate {fold_dir} is not a Stage-1 fold")
        if int(payload.get("outer_fold", -1)) != int(fold):
            raise ValueError(f"candidate {fold_dir} has the wrong outer fold")
        score = float(payload.get("inner_macro_auc", float("nan")))
        if not np.isfinite(score):
            raise ValueError(f"candidate {fold_dir} has no finite inner_macro_auc")
        candidate_inner = int(payload.get("inner_fold", -1))
        if inner_fold is None:
            inner_fold = candidate_inner
        elif candidate_inner != inner_fold:
            raise ValueError("Stage-1 candidates used for nested selection have different inner folds")

        offsets_raw = payload.get("validation_tta_offsets")
        if not isinstance(offsets_raw, list) or not offsets_raw:
            raise ValueError(
                f"candidate {fold_dir} predates the validation-TTA contract; retrain it before Stage-2"
            )
        candidate_offsets = tuple(int(value) for value in offsets_raw)
        if validation_offsets is None:
            validation_offsets = candidate_offsets
        elif candidate_offsets != validation_offsets:
            raise ValueError("Stage-1 candidates were evaluated with different validation TTA policies")
        candidates.append((score, -order, root, payload))

    score, _, root, payload = max(candidates, key=lambda item: (item[0], item[1]))
    metadata = {
        "selected_root": str(root),
        "outer_fold": int(fold),
        "inner_fold": int(payload["inner_fold"]),
        "inner_macro_auc": float(score),
        "n_candidates": len(candidates),
        "criterion": "inner_macro_auc_only",
        "validation_tta_offsets": list(validation_offsets or ()),
    }
    return root, metadata


def load_fold_image_teacher(
    stage1_source: str | Path | Iterable[str | Path],
    fold: int,
    df: pd.DataFrame,
    gold_rows: np.ndarray,
    *,
    return_source: bool = False,
):
    if "crossfit_fold" not in df.columns:
        raise ValueError("crossfit_fold must be assigned before loading Stage-2 teacher")
    selected_root, source = choose_fold_candidate_root(stage1_source, fold)
    path = selected_root / f"fold{int(fold)}" / "weak_oof.csv"
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
    return (image, source) if return_source else image


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
    report_low_confidence: float = 0.10,
    image_only_positive_threshold: float = 0.95,
    image_only_negative_threshold: float = 0.05,
    image_only_weight: float = 0.20,
    image_only_blend: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse independent image/report teachers while preserving uncertainty."""
    report_p = np.asarray(report_p, np.float32)
    report_conf = np.asarray(report_conf, np.float32)
    image_p = np.asarray(image_p, np.float32)
    if report_p.shape != report_conf.shape or report_p.shape != image_p.shape:
        raise ValueError("teacher arrays must have identical shapes")
    for value, name in [
        (positive_threshold, "positive_threshold"),
        (negative_threshold, "negative_threshold"),
        (agreement_weight, "agreement_weight"),
        (disagreement_weight, "disagreement_weight"),
        (blend, "blend"),
        (report_low_confidence, "report_low_confidence"),
        (image_only_positive_threshold, "image_only_positive_threshold"),
        (image_only_negative_threshold, "image_only_negative_threshold"),
        (image_only_weight, "image_only_weight"),
        (image_only_blend, "image_only_blend"),
    ]:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    if negative_threshold >= positive_threshold:
        raise ValueError("negative_threshold must be below positive_threshold")
    if image_only_negative_threshold >= image_only_positive_threshold:
        raise ValueError("image-only negative threshold must be below positive threshold")

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

    image_only_confident = available & (
        (image_p >= image_only_positive_threshold) | (image_p <= image_only_negative_threshold)
    )
    image_only = image_only_confident & (report_conf <= report_low_confidence) & ~disagree
    if image_only.any():
        b = float(image_only_blend)
        probability[image_only] = (1.0 - b) * report_p[image_only] + b * image_p[image_only]
        confidence[image_only] = np.maximum(confidence[image_only], float(image_only_weight))
    return probability, confidence
