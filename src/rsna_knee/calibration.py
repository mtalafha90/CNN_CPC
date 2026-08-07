"""Fold-safe calibration of the report teacher.

The rule engine emits fixed probabilities — 0.92 for a positive mention, 0.06
for a negated one, 0.50 otherwise. Those numbers are guesses. The gold studies
can tell us what each state is actually worth: how often is a study with a
"positive ACL mention" really ACL positive?

The catch, and the reason this module exists, is *which* gold studies are
allowed to answer that question. Calibrating on all 58 and then validating on
a subset of the same 58 makes validation optimistic — the teacher has already
seen the answers. `docs/strategy.md` and the public-code review both flag this
as one of the easiest ways to fool yourself here.

So calibration is fitted per fold, on the gold studies **outside** the
validation fold:

    for each fold k:
        calibrate on gold studies not in fold k
        build soft labels for every study
        train
        score only on the gold studies in fold k

With so few studies per cell, raw frequencies are unusable — a state seen
three times would give 0.0 or 1.0. Estimates are therefore smoothed towards
the target's own prevalence, which is the standard empirical-Bayes shrinkage
and degrades gracefully to "we learned nothing, keep the prior".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import TARGETS
from .report_labels import STATE_UNMENTIONED, STATES

# Pseudo-count controlling how hard estimates are pulled towards the prior.
# At alpha = 5, a state needs about five gold examples before its own evidence
# outweighs the prior, which suits cells holding a handful of studies.
DEFAULT_ALPHA = 5.0

# Fallback prevalence when a target has no positive gold example at all.
FALLBACK_PRIOR = 0.1


@dataclass
class TeacherCalibration:
    """Maps ``(target, rule state)`` to an empirical probability."""

    table: dict[tuple[str, str], float] = field(default_factory=dict)
    prior: dict[str, float] = field(default_factory=dict)
    counts: dict[tuple[str, str], int] = field(default_factory=dict)
    alpha: float = DEFAULT_ALPHA
    n_calibration: int = 0

    def probability(self, target: str, state: str) -> float:
        if (target, state) in self.table:
            return self.table[(target, state)]
        return self.prior.get(target, FALLBACK_PRIOR)

    def apply(self, states: np.ndarray) -> np.ndarray:
        """Convert an ``[n, 12]`` state array into calibrated probabilities."""
        out = np.zeros(states.shape, dtype=np.float32)
        for j, target in enumerate(TARGETS):
            # One lookup per (target, state) rather than per cell.
            for state in STATES:
                mask = states[:, j] == state
                if mask.any():
                    out[mask, j] = self.probability(target, state)
            unknown = ~np.isin(states[:, j], list(STATES))
            if unknown.any():
                out[unknown, j] = self.prior.get(target, FALLBACK_PRIOR)
        return out

    def confidence(self, states: np.ndarray, floor: float = 0.05) -> np.ndarray:
        """Weight each pseudo-label by how much evidence backs its state.

        A cell calibrated from many gold studies is trusted more than one
        resting entirely on the prior. `unmentioned` is additionally damped:
        silence in a report is weak evidence, since radiologists routinely omit
        incidental findings.
        """
        out = np.full(states.shape, floor, dtype=np.float32)
        for j, target in enumerate(TARGETS):
            for state in STATES:
                mask = states[:, j] == state
                if not mask.any():
                    continue
                n = self.counts.get((target, state), 0)
                # Shrinkage factor: 0 with no evidence, approaching 1 with lots.
                weight = n / (n + self.alpha)
                if state == STATE_UNMENTIONED:
                    weight *= 0.25
                out[mask, j] = max(floor, float(weight))
        return out

    def to_dict(self) -> dict:
        return {
            "table": {f"{t}|{s}": v for (t, s), v in self.table.items()},
            "prior": self.prior,
            "counts": {f"{t}|{s}": v for (t, s), v in self.counts.items()},
            "alpha": self.alpha,
            "n_calibration": self.n_calibration,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TeacherCalibration":
        def split(key: str) -> tuple[str, str]:
            target, state = key.rsplit("|", 1)
            return target, state

        return cls(
            table={split(k): float(v) for k, v in payload.get("table", {}).items()},
            prior={k: float(v) for k, v in payload.get("prior", {}).items()},
            counts={split(k): int(v) for k, v in payload.get("counts", {}).items()},
            alpha=float(payload.get("alpha", DEFAULT_ALPHA)),
            n_calibration=int(payload.get("n_calibration", 0)),
        )


def fit_calibration(
    states: np.ndarray,
    gold: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> TeacherCalibration:
    """Learn ``P(y = 1 | target, state)`` from gold labels.

    Parameters
    ----------
    states:
        ``[n, 12]`` array of rule states, from
        :func:`rsna_knee.report_labels.state_dataframe`.
    gold:
        ``[n, 12]`` array of gold labels. ``NaN`` marks "not annotated" and is
        excluded from the counts — it must never be read as a negative.
    alpha:
        Smoothing strength towards the per-target prior.

    Pass only the calibration split. Feeding this the validation gold labels is
    exactly the leak the module exists to prevent.
    """
    states = np.asarray(states, dtype=object)
    gold = np.asarray(gold, dtype=np.float64)
    if states.shape != gold.shape:
        raise ValueError(f"states {states.shape} and gold {gold.shape} must match")

    calibration = TeacherCalibration(alpha=alpha, n_calibration=int(states.shape[0]))

    for j, target in enumerate(TARGETS):
        labelled = np.isfinite(gold[:, j])
        if labelled.sum() == 0:
            calibration.prior[target] = FALLBACK_PRIOR
            continue
        prior = float(gold[labelled, j].mean())
        # A prior of exactly 0 or 1 would make every cell degenerate.
        calibration.prior[target] = float(np.clip(prior, 0.01, 0.99))

        for state in STATES:
            cell = labelled & (states[:, j] == state)
            n = int(cell.sum())
            calibration.counts[(target, state)] = n
            if n == 0:
                continue
            positives = float(gold[cell, j].sum())
            smoothed = (positives + alpha * calibration.prior[target]) / (n + alpha)
            calibration.table[(target, state)] = float(np.clip(smoothed, 0.005, 0.995))

    return calibration


def calibration_split_mask(
    gold_present: np.ndarray,
    folds: np.ndarray,
    validation_fold: int,
) -> np.ndarray:
    """Select the gold studies that may be used to calibrate for one fold.

    Returns a boolean mask over all studies: those carrying gold labels and
    sitting outside the validation fold.
    """
    gold_present = np.asarray(gold_present, dtype=bool)
    folds = np.asarray(folds)
    return gold_present & (folds != validation_fold)
