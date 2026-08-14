from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .b21_acceptance_protocol import (
    B20_CANONICAL_GOLD_MACRO_AUC,
    B21_CROP_FRACTION,
    B21_GOLD_ACCEPTANCE_VARIANT,
)

B22_VARIANT = "b22_preresize_crop_full_b6_duration_audit_v1"
B22_EXPERIMENT = "B22_preresize_crop_full_b6_duration_audit"
B22_GOLD_AUDIT_VARIANT = "b22_preresize_crop_gold_duration_trajectory_audit_v1"

B22_EPOCHS = 5
B22_SCHEDULER_HORIZON = 5
B22_TRAIN_STUDIES = 3120
B22_TRAIN_SERIES = 17475
B22_TRAIN_CELLS = 14123
B22_POSITIVE_CELLS = 6871
B22_NEGATIVE_CELLS = 7252
B22_BATCHES = 1560
B22_CROP_FRACTION = B21_CROP_FRACTION
B22_E2_REPLAY_TOLERANCE = 0.005


def require_failed_b21_acceptance(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("variant") != B21_GOLD_ACCEPTANCE_VARIANT:
        raise ValueError("B22 requires the completed B21 gold-acceptance record")
    if payload.get("one_gold_look_consumed") is not True:
        raise ValueError("B21 acceptance record does not certify the completed gold look")
    if payload.get("promotion_rule_passed") is not False:
        raise ValueError("B22 duration audit is only defined after B21 failed promotion")
    candidate = payload.get("b21_candidate", {})
    score = float(candidate.get("macro_auc", float("nan")))
    if not np.isfinite(score):
        raise ValueError("B21 acceptance record lacks a finite candidate macro AUC")
    if score >= B20_CANONICAL_GOLD_MACRO_AUC:
        raise ValueError("B21 acceptance record conflicts with the failed-promotion state")
    return payload


def require_b22_e2_replay(new_e2_macro_auc: float, prior_b21_e2_macro_auc: float) -> float:
    new_value = float(new_e2_macro_auc)
    prior_value = float(prior_b21_e2_macro_auc)
    if not np.isfinite(new_value) or not np.isfinite(prior_value):
        raise ValueError("B22 E2 replay values must be finite")
    delta = new_value - prior_value
    if abs(delta) > B22_E2_REPLAY_TOLERANCE:
        raise RuntimeError(
            "B22 E2 does not reproduce B21 E2 closely enough; duration trajectory is not interpretable"
        )
    return float(delta)
