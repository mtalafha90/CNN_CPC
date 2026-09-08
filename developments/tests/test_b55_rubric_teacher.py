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
    STATE_VALUE,
    melt_states,
    rebuild,
    report,
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
