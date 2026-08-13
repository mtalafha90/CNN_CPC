from __future__ import annotations

import json
from pathlib import Path

import numpy as np

B21_FULL_VARIANT = "b21_preresize_crop_full_b6_fixed_e2_v1"
B21_FULL_EXPERIMENT = "B21_preresize_crop_full_b6_fixed_e2"
B21_GOLD_ACCEPTANCE_VARIANT = "b21_vs_b20_single_gold_acceptance_v1"

B21_FULL_TRAIN_STUDIES = 3120
B21_FULL_TRAIN_SERIES = 17475
B21_FULL_TRAIN_CELLS = 14123
B21_FULL_POSITIVE_CELLS = 6871
B21_FULL_NEGATIVE_CELLS = 7252
B21_FULL_BATCHES = 1560
B21_FIXED_EPOCHS = 2
B21_SCHEDULER_HORIZON = 5
B21_CROP_FRACTION = 0.90

B20_CANONICAL_GOLD_MACRO_AUC = 0.667159355531343
B20_CANONICAL_EPOCH = 2
B20_REPLAY_SANITY_TOLERANCE = 0.005
PROMOTION_RULE = "candidate_global_macro_auc_gt_canonical_b20"
SCIENTIFIC_SUPERIORITY_RULE = "paired_95pct_ci_lower_gt_zero"

WEAK_V2_GATE_VARIANT = "b21_preresize_crop_weak_v2_paired_comparison_v1"


def require_passed_weak_v2_gate(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("variant") != WEAK_V2_GATE_VARIANT:
        raise ValueError("B21 full-data training requires the frozen B21 weak-v2 comparison")
    if int(payload.get("n_holdout_studies", -1)) != 623:
        raise ValueError("B21 weak-v2 gate study count changed")
    if not np.isclose(float(payload.get("crop_fraction", -1)), B21_CROP_FRACTION, atol=1e-12, rtol=0):
        raise ValueError("B21 weak-v2 gate did not use the frozen 0.90 crop")
    if int(payload.get("fixed_epoch", -1)) != B21_FIXED_EPOCHS:
        raise ValueError("B21 weak-v2 gate did not use the fixed E2 endpoint")
    paired = payload.get("paired_candidate_minus_control", {})
    raw = float(paired.get("raw_difference_b_minus_a", float("nan")))
    lower = float(paired.get("ci_lower", float("nan")))
    upper = float(paired.get("ci_upper", float("nan")))
    if not np.isfinite(raw) or not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("B21 weak-v2 gate lacks finite paired evidence")
    if raw <= 0 or lower <= 0:
        raise ValueError("B21 weak-v2 gate was not favorable under the frozen paired rule")
    return payload


def promotion_decision(candidate_macro_auc: float) -> bool:
    value = float(candidate_macro_auc)
    if not np.isfinite(value):
        raise ValueError("candidate macro AUC must be finite")
    return bool(value > B20_CANONICAL_GOLD_MACRO_AUC)


def scientific_superiority_decision(paired_ci_lower: float) -> bool:
    value = float(paired_ci_lower)
    if not np.isfinite(value):
        raise ValueError("paired CI lower bound must be finite")
    return bool(value > 0.0)


def require_b20_replay_sanity(replayed_macro_auc: float) -> float:
    value = float(replayed_macro_auc)
    if not np.isfinite(value):
        raise ValueError("B20 replay macro AUC must be finite")
    delta = value - B20_CANONICAL_GOLD_MACRO_AUC
    if abs(delta) > B20_REPLAY_SANITY_TOLERANCE:
        raise RuntimeError(
            "historical B20 replay differs too much from its canonical score; "
            "abort the one-look acceptance decision"
        )
    return float(delta)
