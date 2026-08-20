"""The parser-versus-person comparison must count agreement correctly.

This measurement exists to separate two failures the expert audit blends
together: the parser misreading a report, and the report disagreeing with the
images. Both readers see identical text here, so every disagreement is a
parsing mistake -- which only holds if the counting is right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.parser_vs_human import compare, read_hand_labels

TARGETS = ("ACL", "MCL", "Effusion")


def _parsed(states, confidences=None):
    """Build the parser's side: one state and confidence per target per study."""
    frame = {}
    for target, column in states.items():
        frame[f"{target}__state"] = column
        frame[f"{target}__confidence"] = (
            confidences.get(target, [0.9] * len(column)) if confidences else [0.9] * len(column)
        )
    return pd.DataFrame(frame)


def _human(**columns):
    uids = [f"study-{i}" for i in range(len(next(iter(columns.values()))))]
    return pd.DataFrame({"StudyInstanceUID": uids, **columns})


def test_full_agreement_scores_one():
    human = _human(ACL=[1.0, 0.0])
    parsed = _parsed({"ACL": ["positive", "negated"]})
    result = compare(human, parsed, TARGETS)
    assert result["overall"]["yes_agreement"] == pytest.approx(1.0)
    assert result["overall"]["no_agreement"] == pytest.approx(1.0)
    assert result["disagreements"] == []


def test_a_parser_yes_the_person_calls_no_is_counted_against_it():
    human = _human(ACL=[0.0])
    parsed = _parsed({"ACL": ["positive"]})
    result = compare(human, parsed, TARGETS)
    assert result["overall"]["yes_agreement"] == pytest.approx(0.0)
    assert len(result["disagreements"]) == 1
    assert result["disagreements"][0]["parser"] == "yes"
    assert result["disagreements"][0]["person"] == "no"


def test_a_low_confidence_answer_counts_as_silence_not_as_wrong():
    """Training ignores those cells, so scoring them as errors would mislead."""
    human = _human(ACL=[1.0])
    parsed = _parsed({"ACL": ["positive"]}, {"ACL": [0.5]})
    result = compare(human, parsed, TARGETS)
    assert result["overall"]["said_yes"] == 0
    assert result["overall"]["said_nothing"] == 1
    assert result["overall"]["missed_positives"] == 1


def test_a_cell_the_person_left_blank_is_ignored_entirely():
    human = _human(ACL=[np.nan, 1.0])
    parsed = _parsed({"ACL": ["positive", "positive"]})
    result = compare(human, parsed, TARGETS)
    assert result["overall"]["said_yes"] == 1
    assert result["per_target"][0]["hand_labelled"] == 1


def test_missed_positives_measure_what_the_parser_never_saw():
    """The parser's real weakness is silence, and silence has to be visible."""
    human = _human(ACL=[1.0, 1.0, 1.0])
    parsed = _parsed({"ACL": ["positive", "uncertain", "unmentioned"]})
    result = compare(human, parsed, TARGETS)
    overall = result["overall"]
    assert overall["said_yes"] == 1
    assert overall["said_nothing"] == 2
    assert overall["missed_positives"] == 2
    assert overall["recall_of_positives_stated_in_reports"] == pytest.approx(1 / 3)


def test_findings_the_person_did_not_label_are_skipped():
    human = _human(ACL=[1.0])
    parsed = _parsed({"ACL": ["positive"], "MCL": ["positive"]})
    result = compare(human, parsed, TARGETS)
    assert [row["target"] for row in result["per_target"]] == ["ACL"]


def test_a_sheet_without_a_uid_column_is_refused(tmp_path):
    path = tmp_path / "hand.csv"
    pd.DataFrame({"ACL": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="StudyInstanceUID"):
        read_hand_labels(path, TARGETS)


def test_a_sheet_with_no_finding_columns_is_refused(tmp_path):
    path = tmp_path / "hand.csv"
    pd.DataFrame({"StudyInstanceUID": ["a"], "notes": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="none of the 12 finding columns"):
        read_hand_labels(path, TARGETS)


def test_an_entirely_blank_sheet_is_refused(tmp_path):
    """Empty cells are the likely result of saving the wrong sheet."""
    path = tmp_path / "hand.csv"
    pd.DataFrame({"StudyInstanceUID": ["a", "b"], "ACL": [None, None]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="every cell is empty"):
        read_hand_labels(path, TARGETS)


def test_unlabelled_rows_are_dropped(tmp_path):
    path = tmp_path / "hand.csv"
    pd.DataFrame(
        {"StudyInstanceUID": ["a", "b", "c"], "ACL": [1, None, 0]}
    ).to_csv(path, index=False)
    frame = read_hand_labels(path, TARGETS)
    assert list(frame["StudyInstanceUID"]) == ["a", "c"]


def test_a_repeated_study_is_refused(tmp_path):
    path = tmp_path / "hand.csv"
    pd.DataFrame({"StudyInstanceUID": ["a", "a"], "ACL": [1, 0]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="more than once"):
        read_hand_labels(path, TARGETS)


def test_a_confident_uncertain_still_counts_as_silence():
    """It supervises nothing, so it must not vanish from the accounting."""
    human = _human(ACL=[1.0])
    parsed = _parsed({"ACL": ["uncertain"]}, {"ACL": [0.99]})
    result = compare(human, parsed, TARGETS)
    assert result["overall"]["said_nothing"] == 1
    assert result["overall"]["missed_positives"] == 1


def test_every_hand_labelled_cell_is_accounted_for():
    """yes + no + silent must equal what the person labelled, always."""
    human = _human(ACL=[1.0, 0.0, 1.0, 1.0, 0.0])
    parsed = _parsed(
        {"ACL": ["positive", "negated", "uncertain", "unmentioned", "positive"]},
        {"ACL": [0.9, 0.9, 0.99, 0.9, 0.4]},
    )
    row = compare(human, parsed, TARGETS)["per_target"][0]
    assert row["said_yes"] + row["said_no"] + row["said_nothing"] == row["hand_labelled"]
