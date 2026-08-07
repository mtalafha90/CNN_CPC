"""Uncertainty around the gold macro-AUC.

Validation rests on 58 studies. A macro-AUC quoted to three decimals from that
many cases implies a precision that does not exist: resampling those studies
typically moves the figure by several points. Every entry in the experiment
matrix is a comparison against this number, so without an interval there is no
way to tell a real gain from noise.

This module bootstraps over studies — not over predictions — because the
studies are the sampling unit. Rare targets often become single-class inside a
resample, and their AUC is then undefined; those cells are dropped from that
replicate's mean rather than being invented, and the fraction of usable
replicates is reported so a hopelessly sparse target is visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import TARGETS


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC for one target, from ranks, with correct tie handling.

    Equivalent to ``sklearn.metrics.roc_auc_score`` (a test asserts this) but
    without the per-call validation overhead, which matters when a bootstrap
    makes tens of thousands of calls. Returns ``nan`` when only one class is
    present.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = int(positives.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(y_score.size, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=np.float64)

    # Average ranks within each run of tied scores.
    sorted_scores = y_score[order]
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    for start, stop in zip(
        np.concatenate(([0], boundaries)), np.concatenate((boundaries, [sorted_scores.size]))
    ):
        if stop - start > 1:
            ranks[order[start:stop]] = ranks[order[start:stop]].mean()

    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_auc_from_arrays(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the macro AUC and the per-target AUCs (which may contain nan)."""
    per_target = np.array(
        [fast_auc(y_true[:, j], y_score[:, j]) for j in range(y_true.shape[1])],
        dtype=np.float64,
    )
    usable = per_target[np.isfinite(per_target)]
    return (float(usable.mean()) if usable.size else float("nan")), per_target


@dataclass
class BootstrapResult:
    """Point estimate and interval for a macro-AUC."""

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
            f"macro AUC {self.macro_auc:.4f} "
            f"[{self.lower:.4f}, {self.upper:.4f}] "
            f"({int(self.confidence_level * 100)}% CI, n={self.n_studies} studies, "
            f"{self.n_valid_replicates}/{self.n_bootstrap} replicates usable)"
        )
        if undefined:
            text += f"\nundefined on the full set (single class): {', '.join(undefined)}"
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
    """Percentile bootstrap of the macro-AUC, resampling studies.

    Parameters
    ----------
    y_true, y_score:
        ``[n_studies, 12]`` arrays. ``NaN`` in ``y_true`` marks an unannotated
        cell and is ignored.
    n_bootstrap:
        Number of resamples. 2000 is enough to stabilise a 95% interval.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
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
    if valid.size == 0:
        lower = upper = float("nan")
    else:
        tail = (1.0 - confidence_level) / 2.0
        lower, upper = np.percentile(valid, [100 * tail, 100 * (1 - tail)])

    names = TARGETS[: y_true.shape[1]]
    return BootstrapResult(
        macro_auc=point,
        lower=float(lower),
        upper=float(upper),
        per_target={name: float(value) for name, value in zip(names, per_target)},
        per_target_defined={
            name: bool(np.isfinite(value)) for name, value in zip(names, per_target)
        },
        n_studies=n_studies,
        n_bootstrap=n_bootstrap,
        n_valid_replicates=int(valid.size),
        confidence_level=confidence_level,
    )


def load_oof(
    train_csv: str,
    oof_paths: list[str],
    restrict_to: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Join out-of-fold predictions to the gold labels.

    Concatenates one file per fold, keeps only studies that carry gold labels —
    scoring against the teacher's own pseudo-labels is not validation — and
    returns aligned arrays plus the study ids in a stable order.
    """
    import pandas as pd

    frames = [pd.read_csv(path) for path in oof_paths]
    oof = pd.concat(frames, ignore_index=True)
    oof["StudyInstanceUID"] = oof["StudyInstanceUID"].astype(str)
    # A study repeated across folds would otherwise be counted twice.
    oof = oof.drop_duplicates("StudyInstanceUID", keep="last")

    train = pd.read_csv(train_csv)
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    labelled = train[train[TARGETS].notna().any(axis=1)]

    merged = labelled.merge(oof, on="StudyInstanceUID", suffixes=("", "_pred"))
    if restrict_to is not None:
        merged = merged[merged["StudyInstanceUID"].isin(set(restrict_to))]
        # Preserve the caller's order so paired comparisons line up row by row.
        order = {uid: i for i, uid in enumerate(restrict_to)}
        merged = merged.sort_values("StudyInstanceUID", key=lambda s: s.map(order))
    else:
        merged = merged.sort_values("StudyInstanceUID")

    if merged.empty:
        raise ValueError(
            "no gold-labelled studies found in the OOF predictions; check that the "
            "OOF files come from the same data as train.csv"
        )

    y_true = merged[TARGETS].to_numpy(dtype=np.float64)
    y_pred = merged[[f"{t}_pred" for t in TARGETS]].to_numpy(dtype=np.float64)
    return y_true, y_pred, merged["StudyInstanceUID"].tolist()


def compare_runs(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 2026,
) -> dict:
    """Test whether run B beats run A, using paired resamples.

    Pairing matters: both runs are scored on the *same* resampled studies, so
    the shared study-selection noise cancels and the comparison is far tighter
    than two independent intervals would suggest. Overlapping individual
    intervals therefore do not imply the difference is insignificant.

    Returns the median difference, its interval, and the fraction of resamples
    in which B wins.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    n_studies = y_true.shape[0]
    rng = np.random.default_rng(seed)

    differences = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_studies, size=n_studies)
        a, _ = macro_auc_from_arrays(y_true[idx], np.asarray(y_score_a)[idx])
        c, _ = macro_auc_from_arrays(y_true[idx], np.asarray(y_score_b)[idx])
        differences[b] = c - a

    valid = differences[np.isfinite(differences)]
    if valid.size == 0:
        return {"median_difference": float("nan"), "wins": float("nan")}
    return {
        "median_difference": float(np.median(valid)),
        "ci_lower": float(np.percentile(valid, 2.5)),
        "ci_upper": float(np.percentile(valid, 97.5)),
        "probability_b_better": float((valid > 0).mean()),
        "n_valid_replicates": int(valid.size),
    }
