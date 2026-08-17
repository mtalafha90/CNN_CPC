"""Phase 8: freeze the global B6 + Phase-7 translation-rescue supervision artifact.

This module does not train an MRI model. It takes frozen B6 v1.2.1 training
labels and the completed Phase-7 recovered-cell audit, then creates one global
merged supervision table over the same 4,349 report-only studies.

Governance is strict:
- every originally usable B6 cell is preserved exactly;
- no B6-active study may receive a translated cell;
- only Phase-7 cells from originally zero-usable-cell studies may be added;
- no target/script-specific filtering is allowed;
- the 58 official gold studies remain absent from the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .b7_weak_supervision import B7_MIN_CONFIDENCE, load_frozen_b6_export
from .constants import TARGETS

MERGE_VERSION = "b6_v121_plus_phase7_translation_rescue_global_v1"
REQUIRED_PHASE7_VERSION = "translation_to_frozen_b6_full_inactive_audit_v1"
REQUIRED_B6_VERSION = "1.2.1"
REQUIRED_RECOVERED_CELLS_SHA256 = "ed094e5d6f77b1558fe63921f2f22b8e1006443c506f00f921d842cde72025d0"
EXPECTED_REPORT_ONLY = 4349
EXPECTED_ORIGINAL_ACTIVE = 3120
EXPECTED_ORIGINAL_INACTIVE = 1229
EXPECTED_ORIGINAL_USABLE = 14123
EXPECTED_RECOVERED_STUDIES = 1053
EXPECTED_RECOVERED_CELLS = 3901
EXPECTED_RECOVERED_POSITIVE = 2719
EXPECTED_RECOVERED_NEGATIVE = 1182


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _usable(state: object, confidence: object) -> bool:
    try:
        conf = float(confidence)
    except Exception:
        conf = 0.0
    return str(state) in {"positive", "negated"} and conf >= B7_MIN_CONFIDENCE


def _study_usable_counts(frame: pd.DataFrame) -> pd.Series:
    counts = np.zeros(len(frame), dtype=np.int16)
    for target in TARGETS:
        state = frame[f"{target}__state"].fillna("").astype(str).to_numpy()
        conf = pd.to_numeric(frame[f"{target}__confidence"], errors="coerce").fillna(0.0).to_numpy(float)
        counts += ((np.isin(state, ["positive", "negated"])) & (conf >= B7_MIN_CONFIDENCE)).astype(np.int16)
    return pd.Series(counts, index=frame.index)


def _cell_counts(frame: pd.DataFrame) -> tuple[int, int, int]:
    usable = positive = negative = 0
    for target in TARGETS:
        state = frame[f"{target}__state"].fillna("").astype(str)
        conf = pd.to_numeric(frame[f"{target}__confidence"], errors="coerce").fillna(0.0)
        pos = state.eq("positive") & conf.ge(B7_MIN_CONFIDENCE)
        neg = state.eq("negated") & conf.ge(B7_MIN_CONFIDENCE)
        positive += int(pos.sum())
        negative += int(neg.sum())
        usable += int((pos | neg).sum())
    return usable, positive, negative


def build_merged_supervision(
    *,
    b6_root: str | Path,
    phase7_root: str | Path,
    out_root: str | Path,
) -> dict:
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    if str(b6_policy.get("version")) != REQUIRED_B6_VERSION:
        raise ValueError("Phase 8 requires frozen B6 v1.2.1")

    phase7 = Path(phase7_root)
    summary_path = phase7 / "full_population_summary.json"
    recovered_path = phase7 / "recovered_cells.csv"
    if not summary_path.is_file() or not recovered_path.is_file():
        raise FileNotFoundError("Phase 8 requires Phase-7 summary and recovered_cells.csv")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("version")) != REQUIRED_PHASE7_VERSION:
        raise ValueError(f"unexpected Phase-7 version: {summary.get('version')!r}")
    if not bool(summary.get("translator_matches_phase6_exactly", False)):
        raise ValueError("Phase-7 translator does not match frozen Phase 6")
    if _sha256_file(recovered_path) != REQUIRED_RECOVERED_CELLS_SHA256:
        raise ValueError("recovered_cells.csv does not match the frozen Phase-7 result")

    base = b6_frame.copy()
    base["StudyInstanceUID"] = base["StudyInstanceUID"].astype(str)
    if len(base) != EXPECTED_REPORT_ONLY or base["StudyInstanceUID"].duplicated().any():
        raise ValueError("unexpected B6 report-only population")

    original_counts = _study_usable_counts(base)
    original_active_mask = original_counts.gt(0)
    original_active = int(original_active_mask.sum())
    original_inactive = int((~original_active_mask).sum())
    original_usable, original_positive, original_negative = _cell_counts(base)
    if (original_active, original_inactive, original_usable) != (
        EXPECTED_ORIGINAL_ACTIVE, EXPECTED_ORIGINAL_INACTIVE, EXPECTED_ORIGINAL_USABLE
    ):
        raise ValueError(
            "B6 population does not match frozen Phase-7 contract: "
            f"active={original_active}, inactive={original_inactive}, usable={original_usable}"
        )

    recovered = pd.read_csv(recovered_path)
    required_cols = {
        "StudyInstanceUID", "target", "state", "confidence", "probability"
    }
    missing = required_cols.difference(recovered.columns)
    if missing:
        raise ValueError(f"recovered_cells.csv missing columns: {sorted(missing)}")
    recovered["StudyInstanceUID"] = recovered["StudyInstanceUID"].astype(str)
    if recovered.duplicated(["StudyInstanceUID", "target"]).any():
        raise ValueError("duplicate StudyInstanceUID/target recovered cells")
    if not recovered["target"].isin(TARGETS).all():
        raise ValueError("recovered_cells.csv contains an unknown target")
    if not recovered["state"].isin(["positive", "negated"]).all():
        raise ValueError("Phase-8 recovered cells must all be definite")
    if not pd.to_numeric(recovered["confidence"], errors="coerce").ge(B7_MIN_CONFIDENCE).all():
        raise ValueError("Phase-8 recovered cells must meet the frozen confidence threshold")

    inactive_uids = set(base.loc[~original_active_mask, "StudyInstanceUID"])
    recovered_uids = set(recovered["StudyInstanceUID"])
    if not recovered_uids.issubset(inactive_uids):
        raise ValueError("Phase-7 recovered cells include a B6-active or unknown study")

    n_pos = int(recovered["state"].eq("positive").sum())
    n_neg = int(recovered["state"].eq("negated").sum())
    if (len(recovered_uids), len(recovered), n_pos, n_neg) != (
        EXPECTED_RECOVERED_STUDIES,
        EXPECTED_RECOVERED_CELLS,
        EXPECTED_RECOVERED_POSITIVE,
        EXPECTED_RECOVERED_NEGATIVE,
    ):
        raise ValueError("Phase-7 recovered-cell totals differ from the frozen result")

    merged = base.copy()
    row_index = {uid: idx for idx, uid in zip(merged.index, merged["StudyInstanceUID"])}
    for row in recovered.itertuples(index=False):
        uid = str(row.StudyInstanceUID)
        target = str(row.target)
        idx = row_index[uid]
        # The study-level guard already guarantees zero original usable cells;
        # keep this cell-level guard as defense in depth.
        if _usable(base.at[idx, f"{target}__state"], base.at[idx, f"{target}__confidence"]):
            raise RuntimeError("attempted to overwrite an original usable B6 cell")
        merged.at[idx, target] = float(row.probability)
        merged.at[idx, f"{target}__confidence"] = float(row.confidence)
        merged.at[idx, f"{target}__state"] = str(row.state)

    # Strong preservation assertion: every column of every original active row
    # must be exactly unchanged.
    active_indices = base.index[original_active_mask]
    if not merged.loc[active_indices].equals(base.loc[active_indices]):
        raise RuntimeError("an original B6-active study changed during Phase-8 merge")

    candidate_counts = _study_usable_counts(merged)
    candidate_active = int(candidate_counts.gt(0).sum())
    candidate_usable, candidate_positive, candidate_negative = _cell_counts(merged)
    if candidate_active != EXPECTED_ORIGINAL_ACTIVE + EXPECTED_RECOVERED_STUDIES:
        raise RuntimeError("unexpected candidate active-study count")
    if candidate_usable != EXPECTED_ORIGINAL_USABLE + EXPECTED_RECOVERED_CELLS:
        raise RuntimeError("unexpected candidate usable-cell count")

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    targets_out = out / "training_targets.csv"
    merged.to_csv(targets_out, index=False)

    audit = {
        "version": MERGE_VERSION,
        "b6_version": REQUIRED_B6_VERSION,
        "phase7_version": REQUIRED_PHASE7_VERSION,
        "phase7_summary_sha256": _sha256_file(summary_path),
        "phase7_recovered_cells_sha256": _sha256_file(recovered_path),
        "output_training_targets_sha256": _sha256_file(targets_out),
        "report_only_studies": int(len(merged)),
        "gold_studies_in_output": 0,
        "original": {
            "active_studies": original_active,
            "inactive_studies": original_inactive,
            "usable_cells": original_usable,
            "positive_cells": original_positive,
            "negative_cells": original_negative,
        },
        "rescue": {
            "studies": int(len(recovered_uids)),
            "usable_cells": int(len(recovered)),
            "positive_cells": n_pos,
            "negative_cells": n_neg,
        },
        "candidate": {
            "active_studies": candidate_active,
            "inactive_studies": int(len(merged) - candidate_active),
            "usable_cells": candidate_usable,
            "positive_cells": candidate_positive,
            "negative_cells": candidate_negative,
        },
        "guardrails": {
            "all_original_b6_active_rows_preserved_exactly": True,
            "original_usable_b6_cells_overwritten": 0,
            "partially_silent_b6_active_cells_filled": False,
            "target_specific_filtering": False,
            "script_specific_filtering": False,
            "gold_in_training": False,
            "mri_model_trained": False,
        },
    }
    (out / "merge_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    policy = {
        "version": MERGE_VERSION,
        "purpose": "global candidate supervision for a future matched MRI experiment",
        "base": "frozen B6 v1.2.1",
        "addition": "all frozen Phase-7 recovered cells from originally zero-cell studies",
        "selection": "global; no target/script filtering",
        "gold_usage": "excluded from training_targets.csv",
        "status": "frozen supervision artifact only; no model promotion",
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    ap = argparse.ArgumentParser("Freeze B6 + Phase-7 global merged supervision")
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--phase7-root", required=True)
    ap.add_argument("--out-root", default="runs/translation_rescue_supervision_v1")
    args = ap.parse_args()
    build_merged_supervision(
        b6_root=args.b6_root,
        phase7_root=args.phase7_root,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
