"""Coverage counting, and the split the rescue decision turns on.

The audit exists to answer one question before a merge is built: does the
translation rescue still have work to do once the LLM filler has run? Getting
the two landing categories the wrong way round would answer it wrongly in the
direction that invites an unmeasured policy change, so they are pinned here.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.teacher_coverage_audit import (
    audit,
    coverage,
    read_recovered_cells,
    read_teacher,
    rescue_headroom,
)


def _export(uids, states, *, gold=None, confidences=None):
    """Build an export frame: `states` maps target -> per-study state."""
    frame = {"StudyInstanceUID": list(uids)}
    if gold is not None:
        frame["is_gold"] = list(gold)
    for target in TARGETS:
        column = states.get(target, ["unmentioned"] * len(uids))
        conf = (confidences or {}).get(target)
        if conf is None:
            conf = [0.9 if s in ("positive", "negated") else 0.0 for s in column]
        frame[target] = [
            0.97 if s == "positive" else 0.03 if s == "negated" else 0.5 for s in column
        ]
        frame[f"{target}__confidence"] = conf
        frame[f"{target}__state"] = column
    return pd.DataFrame(frame)


def _recovered(rows):
    return pd.DataFrame(rows, columns=["StudyInstanceUID", "target", "state"])


# --- coverage ---------------------------------------------------------------


def test_a_silent_study_is_counted_as_silent():
    frame = _export(["a", "b"], {"ACL": ["positive", "unmentioned"]})
    result = coverage(frame)

    assert result["studies"] == 2
    assert result["answered_cells"] == 1
    assert result["active_studies"] == 1
    assert result["silent_studies"] == 1
    assert result["possible_cells"] == 2 * len(TARGETS)


def test_a_cell_below_the_threshold_does_not_count():
    frame = _export(["a"], {"ACL": ["positive"]}, confidences={"ACL": [0.5]})
    assert coverage(frame)["answered_cells"] == 0
    assert coverage(frame, min_confidence=0.4)["answered_cells"] == 1


def test_uncertain_is_not_an_answer():
    frame = _export(["a"], {"ACL": ["uncertain"]}, confidences={"ACL": [0.9]})
    assert coverage(frame)["answered_cells"] == 0


def test_gold_studies_are_excluded_from_every_count():
    """A gold row counted as coverage would flatter the teacher."""
    frame = _export(
        ["a", "g"], {"ACL": ["positive", "positive"]}, gold=[False, True]
    )
    result = coverage(frame)

    assert result["studies"] == 1
    assert result["gold_studies_carried"] == 1
    assert result["answered_cells"] == 1


def test_per_target_counts_sum_to_the_total():
    frame = _export(
        ["a", "b"],
        {"ACL": ["positive", "negated"], "MCL": ["negated", "unmentioned"]},
    )
    result = coverage(frame)

    assert result["per_target"]["ACL"] == 2
    assert result["per_target"]["MCL"] == 1
    assert sum(result["per_target"].values()) == result["answered_cells"]


# --- rescue headroom --------------------------------------------------------


def test_a_rescued_cell_in_a_wholly_silent_study_is_the_frozen_policy():
    frame = _export(["a"], {})
    result = rescue_headroom(frame, _recovered([("a", "ACL", "positive")]))

    assert result["frozen_policy"]["cells"] == 1
    assert result["frozen_policy"]["positive"] == 1
    assert result["new_policy"]["cells"] == 0
    assert result["silent_studies_that_would_become_active"] == 1
    assert result["silent_studies_left_after_rescue"] == 0


def test_a_rescued_cell_in_a_partly_filled_study_is_a_new_policy():
    """The filler reached this study, so Phase 6's clause no longer decides it."""
    frame = _export(["a"], {"MCL": ["negated"]})
    result = rescue_headroom(frame, _recovered([("a", "ACL", "positive")]))

    assert result["frozen_policy"]["cells"] == 0
    assert result["new_policy"]["cells"] == 1
    assert result["new_policy"]["studies"] == 1


def test_a_cell_the_teacher_already_answers_is_not_offered_again():
    frame = _export(["a"], {"ACL": ["negated"]})
    result = rescue_headroom(frame, _recovered([("a", "ACL", "positive")]))

    assert result["cells_the_teacher_already_answers"] == 1
    assert result["frozen_policy"]["cells"] == 0
    assert result["new_policy"]["cells"] == 0


def test_the_positive_and_negated_split_is_reported():
    """The rescue skews positive, and the teacher's known fault is false positives."""
    frame = _export(["a", "b"], {})
    result = rescue_headroom(
        frame,
        _recovered(
            [("a", "ACL", "positive"), ("a", "MCL", "positive"), ("b", "ACL", "negated")]
        ),
    )

    assert result["frozen_policy"]["positive"] == 2
    assert result["frozen_policy"]["negated"] == 1
    assert result["frozen_policy"]["studies"] == 2


def test_a_rescued_study_missing_from_the_export_is_reported_not_ignored():
    frame = _export(["a"], {})
    result = rescue_headroom(frame, _recovered([("z", "ACL", "positive")]))

    assert result["studies_absent_count"] == 1
    assert result["studies_absent_from_teacher"] == ["z"]
    assert result["frozen_policy"]["cells"] == 0


def test_gold_studies_never_receive_a_rescued_cell():
    """Not one of the 58 is eligible, which is why the expert audit cannot see this."""
    frame = _export(["g"], {}, gold=[True])
    result = rescue_headroom(frame, _recovered([("g", "ACL", "positive")]))

    assert result["studies_absent_count"] == 1
    assert result["frozen_policy"]["cells"] == 0
    assert result["new_policy"]["cells"] == 0


# --- reading artefacts ------------------------------------------------------


def test_a_directory_prefers_the_file_that_carries_gold(tmp_path):
    _export(["a"], {"ACL": ["positive"]}, gold=[False]).to_csv(
        tmp_path / "structured_labels.csv", index=False
    )
    _export(["a"], {"ACL": ["negated"]}).to_csv(
        tmp_path / "training_targets.csv", index=False
    )
    assert read_teacher(tmp_path)["ACL__state"].iloc[0] == "positive"


def test_a_directory_falls_back_to_training_targets(tmp_path):
    _export(["a"], {"ACL": ["negated"]}).to_csv(
        tmp_path / "training_targets.csv", index=False
    )
    assert read_teacher(tmp_path)["ACL__state"].iloc[0] == "negated"


def test_a_directory_with_neither_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="neither"):
        read_teacher(tmp_path)


def test_a_duplicated_study_is_refused(tmp_path):
    path = tmp_path / "training_targets.csv"
    _export(["a", "a"], {}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="more than once"):
        read_teacher(path)


def test_a_missing_state_column_is_named(tmp_path):
    path = tmp_path / "training_targets.csv"
    frame = _export(["a"], {})
    frame.drop(columns=["ACL__state"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="ACL__state"):
        read_teacher(path)


def test_an_unknown_target_in_the_rescue_is_refused(tmp_path):
    path = tmp_path / "recovered_cells.csv"
    _recovered([("a", "Kneecap", "positive")]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown target"):
        read_recovered_cells(path)


def test_a_duplicated_rescue_cell_is_refused(tmp_path):
    path = tmp_path / "recovered_cells.csv"
    _recovered([("a", "ACL", "positive"), ("a", "ACL", "negated")]).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="twice"):
        read_recovered_cells(path)


# --- the whole thing --------------------------------------------------------


def test_audit_writes_json_and_reports_both_halves(tmp_path):
    export = tmp_path / "teacher"
    export.mkdir()
    _export(["a", "b"], {"MCL": ["negated", "unmentioned"]}, gold=[False, False]).to_csv(
        export / "structured_labels.csv", index=False
    )
    rescue = tmp_path / "rescue"
    rescue.mkdir()
    _recovered([("a", "ACL", "positive"), ("b", "ACL", "negated")]).to_csv(
        rescue / "recovered_cells.csv", index=False
    )
    out = tmp_path / "coverage.json"

    result = audit(export=export, phase7_root=rescue, out_json=out)
    written = json.loads(out.read_text(encoding="utf-8"))

    assert written == result
    assert result["coverage"]["silent_studies"] == 1
    assert result["rescue_headroom"]["new_policy"]["cells"] == 1
    assert result["rescue_headroom"]["frozen_policy"]["cells"] == 1


def test_audit_without_the_rescue_reports_coverage_only(tmp_path):
    path = tmp_path / "training_targets.csv"
    _export(["a"], {"ACL": ["positive"]}).to_csv(path, index=False)

    result = audit(export=path)
    assert "rescue_headroom" not in result
    assert result["coverage"]["answered_cells"] == 1
