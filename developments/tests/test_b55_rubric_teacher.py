"""Rebuilding a teacher export with the severity rubric applied.

The failure this file exists to prevent: moving `<target>__state` and leaving
`<target>` at 1.0. The audit would say the cell was downgraded, the model would
train on an unchanged positive, and nothing would look wrong -- which is the
same shape as B52's augmentation flag setting fields nobody read.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rsna_knee.b55_rubric_teacher import (
    REPORT_COLUMNS,
    STATE_VALUE,
    melt_states,
    rebuild,
    report,
    report_column,
    teacher_anchors,
    write_states,
)
from rsna_knee.constants import TARGETS
from rsna_knee.report_labels import STATE_NEGATED, STATE_POSITIVE

UNCERTAIN = "uncertain"


def _wide(rows):
    """A minimal training_targets.csv: value, confidence and state per target."""
    built = []
    for uid, states in rows:
        row = {"StudyInstanceUID": uid}
        for target in TARGETS:
            state = states.get(target, UNCERTAIN)
            row[target] = STATE_VALUE.get(state, float("nan"))
            row[f"{target}__state"] = state
            row[f"{target}__confidence"] = 0.9
        built.append(row)
    return pd.DataFrame(built)


def _export(tmp_path, frame):
    root = tmp_path / "teacher"
    root.mkdir()
    frame.to_csv(root / "training_targets.csv", index=False)
    (root / "policy.json").write_text(json.dumps({"policy": "fill_only"}), "utf-8")
    (root / "audit.json").write_text(
        json.dumps({"base_cells_overridden": 0, "gold_rows_in_training_targets": 0}),
        "utf-8",
    )
    return root


# --- the wide/long round trip -------------------------------------------------


def test_melting_yields_one_row_per_study_and_target():
    frame = _wide([("a", {"Effusion": STATE_POSITIVE})])
    melted = melt_states(frame)

    assert len(melted) == len(TARGETS)
    assert set(melted.columns) == {"StudyInstanceUID", "target", "state"}


def test_writing_states_back_moves_the_label_value_too():
    """The whole point of this module."""
    frame = _wide([("a", {"Effusion": STATE_POSITIVE})])
    assert frame.loc[0, "Effusion"] == 1.0

    melted = melt_states(frame)
    melted.loc[melted["target"] == "Effusion", "state"] = STATE_NEGATED
    out = write_states(frame, melted)

    assert out.loc[0, "Effusion__state"] == STATE_NEGATED
    assert out.loc[0, "Effusion"] == 0.0, "the value must follow the state"


def test_the_state_and_the_value_can_never_disagree():
    frame = _wide([("a", {t: STATE_POSITIVE for t in TARGETS})])
    melted = melt_states(frame)
    melted["state"] = STATE_NEGATED
    out = write_states(frame, melted)

    for target in TARGETS:
        assert out.loc[0, f"{target}__state"] == STATE_NEGATED
        assert out.loc[0, target] == 0.0


def test_an_uncommitted_state_leaves_the_value_alone():
    """Only positive and negated have a training value."""
    frame = _wide([("a", {"Effusion": UNCERTAIN})])
    out = write_states(frame, melt_states(frame))

    assert out.loc[0, "Effusion__state"] == UNCERTAIN
    assert pd.isna(out.loc[0, "Effusion"])


def test_confidence_is_not_touched():
    """The rubric changes the answer, not how sure the reader was of the text."""
    frame = _wide([("a", {"Effusion": STATE_POSITIVE})])
    melted = melt_states(frame)
    melted.loc[melted["target"] == "Effusion", "state"] = STATE_NEGATED

    assert write_states(frame, melted).loc[0, "Effusion__confidence"] == 0.9


# --- the anchors --------------------------------------------------------------


def test_every_target_has_anchors():
    anchors = teacher_anchors()
    for target in TARGETS:
        assert len(anchors[target]) >= 2, target


def test_the_oa_targets_get_both_vocabularies():
    """LEXICON's phrases plus v1.3's regexes, which is where v1.3 did its work."""
    from rsna_knee.b6_v13_report_labels import V13_PATTERNS
    from rsna_knee.report_labels import LEXICON

    anchors = teacher_anchors()
    for target in ("Medial OA", "Lateral OA", "PF OA"):
        assert len(anchors[target]) == len(LEXICON.get(target, ())) + len(
            V13_PATTERNS.get(target, ())
        )


# --- the rebuild --------------------------------------------------------------


def _reports():
    return {
        "a": "There is a trace joint effusion.",
        "b": "Large joint effusion.",
        "c": "Moderate effusion, and a small Baker's cyst.",
    }


def _rows():
    return [
        ("a", {"Effusion": STATE_POSITIVE}),
        ("b", {"Effusion": STATE_POSITIVE}),
        ("c", {"Effusion": STATE_POSITIVE, "Baker's": STATE_POSITIVE}),
    ]


def test_a_rebuild_gates_only_the_sub_threshold_cells(tmp_path):
    source = _export(tmp_path, _wide(_rows()))
    audit = rebuild(source, _reports(), teacher_anchors(), tmp_path / "out")

    out = pd.read_csv(tmp_path / "out" / "training_targets.csv")
    by_uid = out.set_index("StudyInstanceUID")

    assert by_uid.loc["a", "Effusion__state"] == STATE_NEGATED, "trace"
    assert by_uid.loc["b", "Effusion__state"] == STATE_POSITIVE, "large"
    assert by_uid.loc["c", "Effusion__state"] == STATE_POSITIVE, "moderate"
    assert by_uid.loc["c", "Baker's__state"] == STATE_NEGATED, "small Baker's"
    assert audit["cells_downgraded"] == 2


def test_the_values_follow_in_the_written_file(tmp_path):
    source = _export(tmp_path, _wide(_rows()))
    rebuild(source, _reports(), teacher_anchors(), tmp_path / "out")

    out = pd.read_csv(tmp_path / "out" / "training_targets.csv").set_index(
        "StudyInstanceUID"
    )
    assert out.loc["a", "Effusion"] == 0.0
    assert out.loc["b", "Effusion"] == 1.0


def test_the_source_export_is_never_modified(tmp_path):
    source = _export(tmp_path, _wide(_rows()))
    before = (source / "training_targets.csv").read_text()

    rebuild(source, _reports(), teacher_anchors(), tmp_path / "out")

    assert (source / "training_targets.csv").read_text() == before


def test_the_result_still_loads_as_a_fill_merged_export(tmp_path):
    """policy.json and audit.json must survive, or the trainer refuses it."""
    from rsna_knee.phase9_supervision import load_fill_merged_export

    source = _export(tmp_path, _wide(_rows()))
    rebuild(source, _reports(), teacher_anchors(), tmp_path / "out")

    frame, policy, audit = load_fill_merged_export(tmp_path / "out")
    assert len(frame) == 3
    assert policy["policy"] == "fill_only"
    assert audit["severity_rubric"]["cells_downgraded"] == 2


def test_the_audit_records_the_rubric_without_rewriting_the_merge_claims(tmp_path):
    source = _export(tmp_path, _wide(_rows()))
    rebuild(source, _reports(), teacher_anchors(), tmp_path / "out")

    audit = json.loads((tmp_path / "out" / "audit.json").read_text("utf-8"))
    assert audit["base_cells_overridden"] == 0, "the merge's own claim survives"
    assert audit["severity_rubric"]["source_labels_root"].endswith("teacher")


def test_every_change_is_written_with_its_sentence(tmp_path):
    source = _export(tmp_path, _wide(_rows()))
    rebuild(source, _reports(), teacher_anchors(), tmp_path / "out")

    changes = pd.read_csv(tmp_path / "out" / "rubric_changes.csv")
    assert len(changes) == 2
    assert set(changes["from"]) == {STATE_POSITIVE}
    assert set(changes["to"]) == {STATE_NEGATED}


def test_the_downgrade_target_is_configurable(tmp_path):
    source = _export(tmp_path, _wide(_rows()))
    rebuild(
        source, _reports(), teacher_anchors(), tmp_path / "out", downgrade_to=UNCERTAIN
    )

    out = pd.read_csv(tmp_path / "out" / "training_targets.csv").set_index(
        "StudyInstanceUID"
    )
    assert out.loc["a", "Effusion__state"] == UNCERTAIN
    assert pd.isna(out.loc["a", "Effusion"]), "an uncertain cell has no value"


def test_a_study_with_no_report_survives_untouched(tmp_path):
    source = _export(tmp_path, _wide(_rows()))
    audit = rebuild(source, {}, teacher_anchors(), tmp_path / "out")

    assert audit["cells_downgraded"] == 0
    out = pd.read_csv(tmp_path / "out" / "training_targets.csv")
    assert (out["Effusion__state"] == STATE_POSITIVE).all()


def test_the_report_runs_on_a_real_audit(tmp_path, capsys):
    """It is the only thing a person reads before spending a day of GPU."""
    source = _export(tmp_path, _wide(_rows()))
    audit = rebuild(source, _reports(), teacher_anchors(), tmp_path / "out")

    report(audit)
    printed = capsys.readouterr().out
    assert "positive cells before" in printed
    assert "rubric_changes.csv" in printed


def test_an_uncommitted_state_clears_the_value(tmp_path):
    """The bug the configurable branch had: state moved, value did not.

    Only reachable via --downgrade-to uncertain, which is exactly the path a
    cautious operator would take, so it must not be the broken one.
    """
    frame = _wide([("a", {"Effusion": STATE_POSITIVE})])
    melted = melt_states(frame)
    melted.loc[melted["target"] == "Effusion", "state"] = UNCERTAIN

    out = write_states(frame, melted)
    assert out.loc[0, "Effusion__state"] == UNCERTAIN
    assert pd.isna(out.loc[0, "Effusion"])


# --- the evidence the runbook promises ----------------------------------------
#
# rubric_changes.csv carried only the study, target and states: the matched term
# and the sentence were computed inside apply_rubric and discarded. Since
# reading those sentences is the only way to tell a working gate from an
# over-firing one, the audit was useless for the job the runbook gives it.


def test_the_changes_csv_carries_the_matched_term_and_the_sentence(tmp_path):
    source = _export(tmp_path, _wide(_rows()))
    rebuild(source, _reports(), teacher_anchors(), tmp_path / "out")

    changes = pd.read_csv(tmp_path / "out" / "rubric_changes.csv")
    assert set(changes.columns) >= {
        "StudyInstanceUID", "target", "from", "to", "matched_term", "window",
    }

    trace = changes[changes["StudyInstanceUID"] == "a"].iloc[0]
    assert trace["matched_term"] == "trace"
    assert "trace joint effusion" in trace["window"]


def test_every_row_has_its_evidence_filled_in(tmp_path):
    """A blank column would be as useless as a missing one."""
    source = _export(tmp_path, _wide(_rows()))
    rebuild(source, _reports(), teacher_anchors(), tmp_path / "out")

    changes = pd.read_csv(tmp_path / "out" / "rubric_changes.csv")
    assert changes["matched_term"].notna().all()
    assert changes["window"].notna().all()
    assert (changes["window"].str.len() > 0).all()


def test_the_summary_json_is_not_buried_under_the_per_cell_rows(tmp_path):
    """Thousands of change rows in audit.json would hide the counts."""
    source = _export(tmp_path, _wide(_rows()))
    rebuild(source, _reports(), teacher_anchors(), tmp_path / "out")

    rubric = json.loads((tmp_path / "out" / "audit.json").read_text("utf-8"))[
        "severity_rubric"
    ]
    assert "changes" not in rubric
    assert rubric["cells_downgraded"] == 2


def test_apply_rubric_hands_the_evidence_to_its_caller():
    """Carried in the audit rather than returned separately: a caller that has
    to ask for the evidence is a caller that will not."""
    from rsna_knee.severity_rubric import apply_rubric

    frame = pd.DataFrame(
        [{"StudyInstanceUID": "a", "target": "Effusion", "state": STATE_POSITIVE}]
    )
    _, audit = apply_rubric(
        frame, {"a": "Trace effusion."}, {"Effusion": ("effusion",)}
    )

    assert audit["changes"][0]["matched_term"] == "trace"
    assert audit["changes"][0]["window"]


# --- finding the report text ---------------------------------------------------
#
# The first version looked for `report`, `report_text` and `text`, and refused
# this competition's own `train.csv` -- whose column is `Report`. The name was
# already written down in `data.py`, which requires it; nothing here checked
# against it. The test below does, so the two cannot drift apart again.


def test_the_real_column_name_is_the_one_data_py_requires():
    """Pinned against the loader that validates train.csv, not against memory."""
    import inspect

    from rsna_knee import data

    source = inspect.getsource(data.load_train_csv)
    assert '"Report"' in source, "data.py no longer requires a Report column"
    assert "Report" in REPORT_COLUMNS
    assert REPORT_COLUMNS[0] == "Report", "the real name should be tried first"


def test_the_competitions_own_column_is_found():
    frame = pd.DataFrame({"StudyInstanceUID": ["a"], "Report": ["Trace effusion."]})
    assert report_column(frame) == "Report"


@pytest.mark.parametrize("name", ["Report", "report", "REPORT", "Report_Text", "Text"])
def test_the_case_of_the_heading_does_not_matter(name):
    """A capital letter is what broke this; it must not be able to again."""
    frame = pd.DataFrame({"StudyInstanceUID": ["a"], name: ["Trace effusion."]})
    assert report_column(frame) == name


def test_the_name_that_is_returned_is_the_frames_own():
    """It indexes the frame, so a folded name would raise a KeyError."""
    frame = pd.DataFrame({"StudyInstanceUID": ["a"], "REPORT": ["Trace effusion."]})
    assert frame[report_column(frame)].iloc[0] == "Trace effusion."


def test_the_most_likely_name_wins_when_several_are_present():
    frame = pd.DataFrame(
        {"StudyInstanceUID": ["a"], "Report": ["real"], "text": ["something else"]}
    )
    assert report_column(frame) == "Report"


def test_a_frame_with_no_report_column_says_what_it_did_find():
    """The old message named only what it wanted, which left you guessing."""
    frame = pd.DataFrame({"StudyInstanceUID": ["a"], "Findings": ["x"]})

    with pytest.raises(ValueError, match="Findings") as problem:
        report_column(frame)
    assert "Report" in str(problem.value), "it should name what it looked for too"
