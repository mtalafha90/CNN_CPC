"""Phase 9 supervision helpers for the matched B6 vs Phase-8 MRI experiment.

The central rule is that both arms expose the MRI model to the exact same 4,349
report-only studies.  The only difference is the supervision table:

CONTROL   frozen B6 v1.2.1
CANDIDATE frozen Phase-8 B6 + translation-rescue merge

B6-inactive studies therefore remain in the control dataloader with zero loss
weight rather than being removed from the MRI exposure surface.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .b7_weak_supervision import (
    B7_MIN_CONFIDENCE,
    B7_NEGATIVE_TARGET,
    B7_NEGATIVE_WEIGHT,
    B7_POSITIVE_TARGET,
    B7_POSITIVE_WEIGHT,
    load_frozen_b6_export,
)
from .constants import TARGETS
from .data import gold_mask

PHASE9_VERSION = "phase9_matched_b34_b6_vs_phase8_supervision_v1"
PHASE8_VERSION = "b6_v121_plus_phase7_translation_rescue_global_v1"
PHASE8_TRAINING_TARGETS_SHA256 = "c59d78c74743112f09946fd18b64d7726947e6f75b83aabd1f585389a89d045a"
PHASE8_PHASE7_RECOVERED_SHA256 = "ed094e5d6f77b1558fe63921f2f22b8e1006443c506f00f921d842cde72025d0"

REPORT_ONLY_STUDIES = 4349
CONTROL_ACTIVE_STUDIES = 3120
CONTROL_USABLE_CELLS = 14123
CONTROL_POSITIVE_CELLS = 6871
CONTROL_NEGATIVE_CELLS = 7252
CANDIDATE_ACTIVE_STUDIES = 4173
CANDIDATE_USABLE_CELLS = 18024
CANDIDATE_POSITIVE_CELLS = 9590
CANDIDATE_NEGATIVE_CELLS = 8434


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_frozen_phase8_export(root: str | Path) -> tuple[pd.DataFrame, dict, dict]:
    root = Path(root)
    targets_path = root / "training_targets.csv"
    audit_path = root / "merge_audit.json"
    policy_path = root / "policy.json"
    for path in (targets_path, audit_path, policy_path):
        if not path.is_file():
            raise FileNotFoundError(f"Phase 9 requires Phase-8 artifact: {path}")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if str(audit.get("version")) != PHASE8_VERSION or str(policy.get("version")) != PHASE8_VERSION:
        raise ValueError("Phase 9 requires the frozen Phase-8 merge version")
    observed_sha = sha256_file(targets_path)
    if observed_sha != PHASE8_TRAINING_TARGETS_SHA256:
        raise ValueError(
            "Phase-8 training_targets.csv fingerprint mismatch: "
            f"expected {PHASE8_TRAINING_TARGETS_SHA256}, got {observed_sha}"
        )
    if str(audit.get("output_training_targets_sha256")) != PHASE8_TRAINING_TARGETS_SHA256:
        raise ValueError("Phase-8 audit does not certify the frozen output fingerprint")
    if str(audit.get("phase7_recovered_cells_sha256")) != PHASE8_PHASE7_RECOVERED_SHA256:
        raise ValueError("Phase-8 audit does not point to the frozen Phase-7 recovered cells")
    if int(audit.get("report_only_studies", -1)) != REPORT_ONLY_STUDIES:
        raise ValueError("Phase-8 report-only study count changed")
    if int(audit.get("gold_studies_in_output", -1)) != 0:
        raise ValueError("Phase-8 artifact unexpectedly contains gold studies")

    candidate = audit.get("candidate", {})
    expected_candidate = (
        CANDIDATE_ACTIVE_STUDIES,
        CANDIDATE_USABLE_CELLS,
        CANDIDATE_POSITIVE_CELLS,
        CANDIDATE_NEGATIVE_CELLS,
    )
    observed_candidate = (
        int(candidate.get("active_studies", -1)),
        int(candidate.get("usable_cells", -1)),
        int(candidate.get("positive_cells", -1)),
        int(candidate.get("negative_cells", -1)),
    )
    if observed_candidate != expected_candidate:
        raise ValueError(f"Phase-8 candidate counts changed: {observed_candidate}")

    guard = audit.get("guardrails", {})
    required_true = {
        "all_original_b6_active_rows_preserved_exactly",
    }
    for key in required_true:
        if guard.get(key) is not True:
            raise ValueError(f"Phase-8 guardrail not certified: {key}")
    required_false = {
        "partially_silent_b6_active_cells_filled",
        "target_specific_filtering",
        "script_specific_filtering",
        "gold_in_training",
        "mri_model_trained",
    }
    for key in required_false:
        if guard.get(key) is not False:
            raise ValueError(f"Phase-8 guardrail changed: {key}")
    if int(guard.get("original_usable_b6_cells_overwritten", -1)) != 0:
        raise ValueError("Phase-8 overwrote original B6 cells")

    frame = pd.read_csv(targets_path)
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if len(frame) != REPORT_ONLY_STUDIES or frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("Phase-8 target table population changed")
    return frame, policy, audit


def prepare_all_report_only_supervision(
    train_df: pd.DataFrame,
    supervision_frame: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """Create B7-style targets/weights without dropping zero-cell studies."""
    train = train_df.copy()
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    frame = supervision_frame.copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("supervision table contains duplicate StudyInstanceUID values")

    gold = gold_mask(train)
    gold_uids = set(train.loc[gold, "StudyInstanceUID"])
    overlap = gold_uids.intersection(set(frame["StudyInstanceUID"]))
    if overlap:
        raise ValueError(f"Phase 9 supervision contains {len(overlap)} gold UID(s)")

    non_gold = train.loc[~gold, ["StudyInstanceUID"]].copy()
    if len(non_gold) != REPORT_ONLY_STUDIES:
        raise ValueError("Phase 9 requires exactly 4,349 report-only studies")
    expected = set(non_gold["StudyInstanceUID"])
    actual = set(frame["StudyInstanceUID"])
    if expected != actual:
        raise ValueError(
            f"Phase 9 report-only UID mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}"
        )

    ordered = non_gold.merge(frame, on="StudyInstanceUID", how="left", validate="one_to_one")
    n = len(ordered)
    y = np.full((n, len(TARGETS)), 0.5, dtype=np.float32)
    w = np.zeros((n, len(TARGETS)), dtype=np.float32)
    per_target: dict[str, dict] = {}

    for j, target in enumerate(TARGETS):
        state_col = f"{target}__state"
        conf_col = f"{target}__confidence"
        if state_col not in ordered or conf_col not in ordered:
            raise ValueError(f"supervision table missing state/confidence for {target}")
        state = ordered[state_col].fillna("").astype(str).to_numpy()
        conf = pd.to_numeric(ordered[conf_col], errors="coerce").fillna(0.0).to_numpy(float)
        positive = (state == "positive") & (conf >= B7_MIN_CONFIDENCE)
        negative = (state == "negated") & (conf >= B7_MIN_CONFIDENCE)
        y[positive, j] = B7_POSITIVE_TARGET
        y[negative, j] = B7_NEGATIVE_TARGET
        w[positive, j] = B7_POSITIVE_WEIGHT
        w[negative, j] = B7_NEGATIVE_WEIGHT
        per_target[target] = {
            "positive_cells": int(positive.sum()),
            "negative_cells": int(negative.sum()),
            "usable_cells": int((positive | negative).sum()),
            "base_weight_sum": float(w[:, j].sum()),
        }

    active = w.sum(axis=1) > 0
    summary = {
        "report_only_rows": n,
        "active_studies": int(active.sum()),
        "inactive_studies_zero_usable_cells": int((~active).sum()),
        "usable_cells": int((w > 0).sum()),
        "positive_cells": int(((w > 0) & (y > 0.5)).sum()),
        "negative_cells": int(((w > 0) & (y < 0.5)).sum()),
        "targets": per_target,
        "zero_weight_studies_retained_in_mri_exposure": True,
    }
    return ordered["StudyInstanceUID"].astype(str).tolist(), y, w, summary


def load_phase9_arm_supervision(
    train_df: pd.DataFrame,
    *,
    arm: str,
    b6_root: str | Path,
    phase8_root: str | Path,
) -> tuple[list[str], np.ndarray, np.ndarray, dict, dict]:
    arm = str(arm).lower()
    if arm not in {"control", "candidate"}:
        raise ValueError("Phase 9 arm must be 'control' or 'candidate'")

    if arm == "control":
        frame, policy, audit = load_frozen_b6_export(b6_root)
        source = {
            "arm": arm,
            "source": "frozen B6 v1.2.1",
            "b6_version": str(policy.get("version", audit.get("b6_version", ""))),
        }
    else:
        frame, policy, audit = load_frozen_phase8_export(phase8_root)
        source = {
            "arm": arm,
            "source": "frozen Phase-8 merged supervision",
            "phase8_version": PHASE8_VERSION,
            "training_targets_sha256": PHASE8_TRAINING_TARGETS_SHA256,
        }

    uids, targets, weights, summary = prepare_all_report_only_supervision(train_df, frame)
    observed = (
        summary["active_studies"],
        summary["usable_cells"],
        summary["positive_cells"],
        summary["negative_cells"],
    )
    expected = (
        (CONTROL_ACTIVE_STUDIES, CONTROL_USABLE_CELLS, CONTROL_POSITIVE_CELLS, CONTROL_NEGATIVE_CELLS)
        if arm == "control"
        else (CANDIDATE_ACTIVE_STUDIES, CANDIDATE_USABLE_CELLS, CANDIDATE_POSITIVE_CELLS, CANDIDATE_NEGATIVE_CELLS)
    )
    if observed != expected:
        raise ValueError(f"Phase 9 {arm} supervision counts changed: {observed} != {expected}")
    source["summary"] = summary
    return uids, targets, weights, summary, source
