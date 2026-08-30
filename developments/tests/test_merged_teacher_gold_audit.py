"""The reconstructed merge must be the merge that actually trained B51.

The gold rows of the fill-merged teacher exist in no artefact, so they are
rebuilt from the two source exports. A rebuild is only worth its verification:
if it does not reproduce the recorded training_targets.csv exactly, the numbers
would describe a merge that never trained anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from rsna_knee.b23_fill_merge import merge_fill_only  # noqa: E402
from rsna_knee.constants import TARGETS  # noqa: E402

from merged_teacher_gold_audit import _verify_against_recorded  # noqa: E402


def _frame(studies, state_for):
    fixed = {
        "positive": (0.97, 0.90),
        "negated": (0.03, 0.90),
        "unmentioned": (0.50, 0.00),
    }
    rows = []
    for study in studies:
        row = {"StudyInstanceUID": study}
        for target in TARGETS:
            state = state_for(study, target)
            probability, confidence = fixed[state]
            row[target] = probability
            row[f"{target}__confidence"] = confidence
            row[f"{target}__state"] = state
        rows.append(row)
    return pd.DataFrame(rows)


def _merged_export(tmp_path, frame, report_only):
    root = tmp_path / "merged"
    root.mkdir(parents=True, exist_ok=True)
    columns = ["StudyInstanceUID"]
    for target in TARGETS:
        columns.extend([target, f"{target}__confidence", f"{target}__state"])
    frame.loc[frame["StudyInstanceUID"].isin(report_only), columns].to_csv(
        root / "training_targets.csv", index=False
    )
    return root


def test_the_filler_only_speaks_where_the_base_is_silent():
    base = _frame(["a"], lambda s, t: "unmentioned" if t == TARGETS[0] else "positive")
    filler = _frame(["a"], lambda s, t: "negated")

    merged, audit = merge_fill_only(base, filler, min_confidence=0.75)
    assert merged.loc[0, f"{TARGETS[0]}__state"] == "negated", "the silent cell is filled"
    assert merged.loc[0, f"{TARGETS[1]}__state"] == "positive", "a committed cell is kept"
    assert audit["base_cells_overridden"] == 0


def test_a_faithful_reconstruction_is_accepted(tmp_path):
    studies = ["gold-0", "weak-0", "weak-1"]
    base = _frame(studies, lambda s, t: "unmentioned" if t == TARGETS[0] else "positive")
    filler = _frame(studies, lambda s, t: "negated")

    merged, _ = merge_fill_only(base, filler, min_confidence=0.75)
    root = _merged_export(tmp_path, merged, {"weak-0", "weak-1"})

    check = _verify_against_recorded(merged, root)
    assert check["studies_checked"] == 2


def test_a_different_rule_is_refused(tmp_path):
    """If the reconstruction disagrees, the gold numbers must not be reported."""
    studies = ["gold-0", "weak-0"]
    base = _frame(studies, lambda s, t: "unmentioned")
    filler = _frame(studies, lambda s, t: "negated")

    merged, _ = merge_fill_only(base, filler, min_confidence=0.75)
    root = _merged_export(tmp_path, merged, {"weak-0"})

    # A merge produced by a different threshold: nothing gets filled.
    other, _ = merge_fill_only(base, filler, min_confidence=0.99)
    with pytest.raises(RuntimeError, match="does not reproduce the recorded export"):
        _verify_against_recorded(other, root)


def test_a_source_mismatch_is_refused(tmp_path):
    studies = ["weak-0"]
    base = _frame(studies, lambda s, t: "positive")
    merged, _ = merge_fill_only(base, base, min_confidence=0.75)
    root = _merged_export(tmp_path, merged, {"weak-0"})

    fewer = merged.iloc[0:0]
    with pytest.raises(RuntimeError, match="the sources do not match this merge"):
        _verify_against_recorded(fewer, root)
