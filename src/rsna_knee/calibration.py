"""Fold-safe calibration of the report teacher.

Report-derived targets are weak supervision. Calibration is fitted only on gold
studies outside the current validation fold, and supervision confidence reflects
both how much calibration evidence exists and how informative a report state is
relative to the target prevalence.

Crucially, an *unmentioned* finding is treated as unlabeled by default. Report
silence is not a negative label: radiologists routinely omit incidental findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import TARGETS
from .report_labels import STATE_UNCERTAIN, STATE_UNMENTIONED, STATES

DEFAULT_ALPHA = 5.0
FALLBACK_PRIOR = 0.1


@dataclass
class TeacherCalibration:
    """Maps ``(target, rule state)`` to fold-safe empirical probabilities."""

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
        out = np.zeros(states.shape, dtype=np.float32)
        for j, target in enumerate(TARGETS):
            for state in STATES:
                mask = states[:, j] == state
                if mask.any():
                    out[mask, j] = self.probability(target, state)
            unknown = ~np.isin(states[:, j], list(STATES))
            if unknown.any():
                out[unknown, j] = self.prior.get(target, FALLBACK_PRIOR)
        return out

    def confidence(
        self,
        states: np.ndarray,
        *,
        unmentioned_weight: float = 0.0,
        uncertain_weight_cap: float = 0.10,
        floor: float = 0.0,
    ) -> np.ndarray:
        """Return per-cell weak-label reliability weights.

        Reliability combines two quantities:

        1. evidence certainty ``n/(n+alpha)``;
        2. informativeness, measured by how far ``P(y|state)`` lies from the
           target prevalence.

        A frequent but uninformative state therefore does not become highly
        weighted merely because it was observed many times. ``unmentioned`` is
        capped at zero by default and behaves as positive-unlabeled data rather
        than a weak negative.
        """
        if not 0.0 <= unmentioned_weight <= 1.0:
            raise ValueError("unmentioned_weight must be in [0,1]")
        if not 0.0 <= uncertain_weight_cap <= 1.0:
            raise ValueError("uncertain_weight_cap must be in [0,1]")

        out = np.full(states.shape, float(floor), dtype=np.float32)
        for j, target in enumerate(TARGETS):
            prior = float(self.prior.get(target, FALLBACK_PRIOR))
            scale = max(prior, 1.0 - prior, 1e-6)
            for state in STATES:
                mask = states[:, j] == state
                if not mask.any():
                    continue
                n = self.counts.get((target, state), 0)
                evidence = n / (n + self.alpha) if n > 0 else 0.0
                probability = self.probability(target, state)
                information = min(1.0, abs(probability - prior) / scale)
                weight = float(evidence * information)
                if state == STATE_UNCERTAIN:
                    weight = min(weight, float(uncertain_weight_cap))
                elif state == STATE_UNMENTIONED:
                    weight = min(weight, float(unmentioned_weight))
                out[mask, j] = max(float(floor), weight)
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
    """Learn ``P(y=1 | target,state)`` from out-of-fold gold labels."""
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
    gold_present = np.asarray(gold_present, dtype=bool)
    folds = np.asarray(folds)
    return gold_present & (folds != validation_fold)
