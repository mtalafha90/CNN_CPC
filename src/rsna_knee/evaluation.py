"""Gold-only ROC-AUC evaluation and study-level bootstrap uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import load_train_csv


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based binary AUC with ties and NaN masking."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = int(positives.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(y_score.size, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=np.float64)
    sorted_scores = y_score[order]
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [sorted_scores.size]))
    for start, stop in zip(starts, stops):
        if stop - start > 1:
            ranks[order[start:stop]] = ranks[order[start:stop]].mean()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_auc_from_arrays(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError(f"expected equal 2D shapes, got {y_true.shape} and {y_score.shape}")
    per_target = np.array(
        [fast_auc(y_true[:, j], y_score[:, j]) for j in range(y_true.shape[1])],
        dtype=np.float64,
    )
    usable = per_target[np.isfinite(per_target)]
    return (float(usable.mean()) if usable.size else float("nan")), per_target


@dataclass
class BootstrapResult:
    macro_auc: float
    lower: float
    upper: float
    per_target: dict[str, float]
    per_target_defined: dict[str, bool]
    n_studies: int
    n_bootstrap: int
    n_valid_replicates: int
    confidence_level: float

    def summary(self) -> str:
        undefined = [name for name, ok in self.per_target_defined.items() if not ok]
        text = (
            f"macro AUC {self.macro_auc:.4f} [{self.lower:.4f}, {self.upper:.4f}] "
            f"({int(self.confidence_level * 100)}% CI, n={self.n_studies}, "
            f"{self.n_valid_replicates}/{self.n_bootstrap} bootstrap replicates usable)"
        )
        if undefined:
            text += f"\nundefined on the full set: {', '.join(undefined)}"
        return text

    def to_dict(self) -> dict:
        return {
            "macro_auc": self.macro_auc,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "confidence_level": self.confidence_level,
            "n_studies": self.n_studies,
            "n_bootstrap": self.n_bootstrap,
            "n_valid_replicates": self.n_valid_replicates,
            "per_target_auc": self.per_target,
            "per_target_defined": self.per_target_defined,
        }


def bootstrap_macro_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 2026,
) -> BootstrapResult:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0,1)")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    n_studies = y_true.shape[0]
    if n_studies == 0:
        raise ValueError("no studies to evaluate")

    point, per_target = macro_auc_from_arrays(y_true, y_score)
    rng = np.random.default_rng(seed)
    replicates = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_studies, size=n_studies)
        replicates[b], _ = macro_auc_from_arrays(y_true[idx], y_score[idx])

    valid = replicates[np.isfinite(replicates)]
    if valid.size:
        tail = (1.0 - confidence_level) / 2.0
        lower, upper = np.percentile(valid, [100 * tail, 100 * (1 - tail)])
    else:
        lower = upper = float("nan")

    names = TARGETS[: y_true.shape[1]]
    return BootstrapResult(
        macro_auc=point,
        lower=float(lower),
        upper=float(upper),
        per_target={name: float(value) for name, value in zip(names, per_target)},
        per_target_defined={name: bool(np.isfinite(value)) for name, value in zip(names, per_target)},
        n_studies=n_studies,
        n_bootstrap=n_bootstrap,
        n_valid_replicates=int(valid.size),
        confidence_level=confidence_level,
    )


def _read_oof(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"StudyInstanceUID", *TARGETS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"OOF file {path} missing columns: {missing}")
    frame = frame[["StudyInstanceUID", *TARGETS]].copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"OOF file {path} contains duplicate StudyInstanceUID values")
    if not np.isfinite(frame[TARGETS].to_numpy(dtype=float)).all():
        raise ValueError(f"OOF file {path} contains non-finite predictions")
    return frame


def load_oof(
    train_csv: str | Path,
    oof_paths: list[str | Path],
    restrict_to: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Join OOF predictions to raw official gold target cells."""
    if not oof_paths:
        raise ValueError("at least one OOF file is required")
    oof = pd.concat([_read_oof(path) for path in oof_paths], ignore_index=True)
    duplicated = oof["StudyInstanceUID"].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(oof.loc[duplicated, "StudyInstanceUID"].unique().tolist())
        raise ValueError(f"studies appear in multiple OOF files: {duplicate_ids[:10]}")

    train = load_train_csv(train_csv)
    labelled = train.loc[train[TARGETS].notna().any(axis=1), ["StudyInstanceUID", *TARGETS]]
    merged = labelled.merge(oof, on="StudyInstanceUID", suffixes=("", "_pred"), how="inner")

    if restrict_to is not None:
        requested = [str(uid) for uid in restrict_to]
        merged = merged.loc[merged["StudyInstanceUID"].isin(set(requested))]
        order = {uid: i for i, uid in enumerate(requested)}
        merged = merged.sort_values("StudyInstanceUID", key=lambda s: s.map(order))
        missing = [uid for uid in requested if uid not in set(merged["StudyInstanceUID"])]
        if missing:
            raise ValueError(f"comparison OOF is missing {len(missing)} requested studies")
    else:
        merged = merged.sort_values("StudyInstanceUID")

    if merged.empty:
        raise ValueError("no gold-labelled studies overlap the OOF predictions")

    y_true = merged[TARGETS].to_numpy(dtype=np.float64)
    y_pred = merged[[f"{target}_pred" for target in TARGETS]].to_numpy(dtype=np.float64)
    return y_true, y_pred, merged["StudyInstanceUID"].tolist()


def compare_runs(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 2026,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score_a = np.asarray(y_score_a, dtype=np.float64)
    y_score_b = np.asarray(y_score_b, dtype=np.float64)
    if y_true.shape != y_score_a.shape or y_true.shape != y_score_b.shape:
        raise ValueError("paired comparison arrays must have identical shapes")
    if n_bootstrap < 1 or len(y_true) == 0:
        raise ValueError("paired comparison requires studies and bootstrap replicates")

    rng = np.random.default_rng(seed)
    differences = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        score_a, _ = macro_auc_from_arrays(y_true[idx], y_score_a[idx])
        score_b, _ = macro_auc_from_arrays(y_true[idx], y_score_b[idx])
        differences[b] = score_b - score_a

    valid = differences[np.isfinite(differences)]
    if valid.size == 0:
        return {
            "median_difference": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "probability_b_better": float("nan"),
            "n_valid_replicates": 0,
        }
    return {
        "median_difference": float(np.median(valid)),
        "ci_lower": float(np.percentile(valid, 2.5)),
        "ci_upper": float(np.percentile(valid, 97.5)),
        "probability_b_better": float((valid > 0).mean()),
        "n_valid_replicates": int(valid.size),
    }
