"""What the export knows about which of its own cells are wrong.

The point of this audit is to find out whether the constant confidence column
could be replaced by something that discriminates. Two things have to be exactly
right for that answer to mean anything: which cells count as predictions, and
which labeller produced each one. Both are pinned here.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.teacher_confidence_audit import (
    CONTRADICTED,
    CORROBORATED,
    UNWITNESSED,
    audit,
    cell_table,
    discrimination,
)


def _train(uids, truths, gold=None):
    """train.csv shape: expert studies carry 0/1 in the twelve label columns."""
    gold = [True] * len(uids) if gold is None else gold
    frame = {"StudyInstanceUID": list(uids), "Report": ["report text"] * len(uids)}
    for target in TARGETS:
        frame[target] = [
            truths.get(target, [None] * len(uids))[i] if is_gold else None
            for i, is_gold in enumerate(gold)
        ]
    return pd.DataFrame(frame)


def _export(uids, states, *, confidences=None, model=None):
    frame = {"StudyInstanceUID": list(uids)}
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
        if model is not None and target in model:
            frame[f"{target}__model_confidence"] = model[target]
    return pd.DataFrame(frame)


def _write(tmp_path, name, frame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


# --- which cells count as predictions ---------------------------------------


def test_a_correct_call_and_a_wrong_one_are_told_apart(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1], "MCL": [0]}))
    teacher = _write(
        tmp_path, "teacher.csv", _export(["a"], {"ACL": ["positive"], "MCL": ["positive"]})
    )
    cells = cell_table(train_csv=train, teacher=teacher)

    assert cells.set_index("target").loc["ACL", "correct"]
    assert not cells.set_index("target").loc["MCL", "correct"]


def test_silence_is_not_a_prediction(tmp_path):
    """Report silence is never a negative, so it lowers coverage, not accuracy."""
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [0]}))
    teacher = _write(tmp_path, "teacher.csv", _export(["a"], {"ACL": ["unmentioned"]}))
    assert len(cell_table(train_csv=train, teacher=teacher)) == 0


def test_hedging_is_not_a_prediction(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [0]}))
    teacher = _write(
        tmp_path,
        "teacher.csv",
        _export(["a"], {"ACL": ["uncertain"]}, confidences={"ACL": [0.9]}),
    )
    assert len(cell_table(train_csv=train, teacher=teacher)) == 0


def test_a_cell_below_the_threshold_is_not_a_prediction(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    teacher = _write(
        tmp_path,
        "teacher.csv",
        _export(["a"], {"ACL": ["positive"]}, confidences={"ACL": [0.5]}),
    )
    assert len(cell_table(train_csv=train, teacher=teacher)) == 0
    assert len(cell_table(train_csv=train, teacher=teacher, min_confidence=0.4)) == 1


def test_a_study_with_no_expert_label_is_not_scored(tmp_path):
    train = _write(
        tmp_path,
        "train.csv",
        _train(["a", "r"], {"ACL": [1, 1]}, gold=[True, False]),
    )
    teacher = _write(tmp_path, "teacher.csv", _export(["a", "r"], {"ACL": ["positive"] * 2}))
    cells = cell_table(train_csv=train, teacher=teacher)

    assert cells["StudyInstanceUID"].tolist() == ["a"]


# --- which labeller produced the cell ----------------------------------------


def test_a_cell_with_no_self_report_came_from_the_parser(tmp_path):
    """The parser has no confidence of its own, so the column is absent there."""
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1], "MCL": [1]}))
    teacher = _write(
        tmp_path,
        "teacher.csv",
        _export(
            ["a"],
            {"ACL": ["positive"], "MCL": ["positive"]},
            model={"ACL": [float("nan")], "MCL": [0.8]},
        ),
    )
    cells = cell_table(train_csv=train, teacher=teacher).set_index("target")

    assert cells.loc["ACL", "source"] == "base"
    assert cells.loc["MCL", "source"] == "filled"


def test_an_export_without_the_column_at_all_is_read_as_all_base(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    teacher = _write(tmp_path, "teacher.csv", _export(["a"], {"ACL": ["positive"]}))
    assert cell_table(train_csv=train, teacher=teacher)["source"].tolist() == ["base"]


# --- what a second labeller says ---------------------------------------------


def test_a_second_labeller_saying_the_same_thing_corroborates(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    teacher = _write(tmp_path, "teacher.csv", _export(["a"], {"ACL": ["positive"]}))
    witness = _write(tmp_path, "witness.csv", _export(["a"], {"ACL": ["positive"]}))
    cells = cell_table(train_csv=train, teacher=teacher, witness=witness)

    assert cells["witness"].tolist() == [CORROBORATED]


def test_a_second_labeller_saying_the_opposite_contradicts(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    teacher = _write(tmp_path, "teacher.csv", _export(["a"], {"ACL": ["positive"]}))
    witness = _write(tmp_path, "witness.csv", _export(["a"], {"ACL": ["negated"]}))
    cells = cell_table(train_csv=train, teacher=teacher, witness=witness)

    assert cells["witness"].tolist() == [CONTRADICTED]


def test_a_silent_second_labeller_neither_corroborates_nor_contradicts(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    teacher = _write(tmp_path, "teacher.csv", _export(["a"], {"ACL": ["positive"]}))
    witness = _write(tmp_path, "witness.csv", _export(["a"], {"ACL": ["unmentioned"]}))
    cells = cell_table(train_csv=train, teacher=teacher, witness=witness)

    assert cells["witness"].tolist() == [UNWITNESSED]


def test_a_hedging_second_labeller_does_not_contradict(tmp_path):
    """`uncertain` is not a call, so it cannot disagree with one."""
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    teacher = _write(tmp_path, "teacher.csv", _export(["a"], {"ACL": ["positive"]}))
    witness = _write(
        tmp_path,
        "witness.csv",
        _export(["a"], {"ACL": ["uncertain"]}, confidences={"ACL": [0.9]}),
    )
    cells = cell_table(train_csv=train, teacher=teacher, witness=witness)

    assert cells["witness"].tolist() == [UNWITNESSED]


def test_no_second_labeller_leaves_the_verdict_empty(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    teacher = _write(tmp_path, "teacher.csv", _export(["a"], {"ACL": ["positive"]}))
    cells = cell_table(train_csv=train, teacher=teacher)

    assert cells["witness"].isna().all()


# --- discrimination ----------------------------------------------------------


def test_a_score_that_ranks_right_above_wrong_reaches_one():
    cells = pd.DataFrame(
        {"correct": [True, True, False, False], "model_confidence": [0.9, 0.8, 0.4, 0.3]}
    )
    assert discrimination(cells, "model_confidence") == 1.0


def test_a_constant_score_lands_at_a_half():
    cells = pd.DataFrame(
        {"correct": [True, False], "model_confidence": [0.9, 0.9]}
    )
    assert discrimination(cells, "model_confidence") == 0.5


def test_a_score_with_only_right_answers_cannot_be_scored():
    cells = pd.DataFrame({"correct": [True, True], "model_confidence": [0.9, 0.8]})
    assert discrimination(cells, "model_confidence") is None


# --- the whole thing ---------------------------------------------------------


def test_the_audit_splits_by_source_state_and_witness(tmp_path):
    train = _write(
        tmp_path,
        "train.csv",
        _train(["a"], {"ACL": [1], "MCL": [0], "Fracture": [1]}),
    )
    teacher = _write(
        tmp_path,
        "teacher.csv",
        _export(
            ["a"],
            {"ACL": ["positive"], "MCL": ["positive"], "Fracture": ["negated"]},
            model={"ACL": [float("nan")], "MCL": [0.6], "Fracture": [float("nan")]},
        ),
    )
    witness = _write(
        tmp_path,
        "witness.csv",
        _export(["a"], {"ACL": ["positive"], "Fracture": ["positive"]}),
    )
    out = tmp_path / "confidence.json"

    result = audit(train_csv=train, teacher=teacher, witness=witness, out_json=out)

    assert result["overall"] == {
        **result["overall"],
        "cells": 3,
        "wrong": 2,  # MCL called positive on a 0, Fracture negated on a 1
    }
    assert result["by_source"]["base"]["cells"] == 2
    assert result["by_source"]["filled"]["cells"] == 1
    assert result["by_state"]["positive"]["cells"] == 2
    assert result["by_witness"][CORROBORATED]["cells"] == 1
    assert result["by_witness"][CONTRADICTED]["cells"] == 1
    assert result["by_witness"][UNWITNESSED]["cells"] == 1
    assert json.loads(out.read_text())["overall"]["cells"] == 3


def test_dropping_contradicted_cells_is_reported_against_the_baseline(tmp_path):
    """The one filter that removes a cell without overriding its labeller."""
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1], "MCL": [0]}))
    teacher = _write(
        tmp_path, "teacher.csv", _export(["a"], {"ACL": ["positive"], "MCL": ["positive"]})
    )
    witness = _write(
        tmp_path, "witness.csv", _export(["a"], {"ACL": ["positive"], "MCL": ["negated"]})
    )

    result = audit(train_csv=train, teacher=teacher, witness=witness)

    assert result["filters"]["baseline"]["cells"] == 2
    assert result["filters"]["baseline"]["wrong"] == 1
    assert result["filters"]["drop_contradicted"]["cells"] == 1
    assert result["filters"]["drop_contradicted"]["wrong"] == 0


def test_a_small_bucket_reports_a_wide_standard_error(tmp_path):
    """Four cells cannot support a conclusion, and the number must say so."""
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1], "MCL": [1]}))
    teacher = _write(
        tmp_path, "teacher.csv", _export(["a"], {"ACL": ["positive"], "MCL": ["negated"]})
    )

    result = audit(train_csv=train, teacher=teacher)
    assert result["overall"]["standard_error"] > 0.3


def test_a_constant_self_report_is_named_as_such(tmp_path):
    """The whole reason this audit exists: 0.90 everywhere cannot discriminate."""
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1], "MCL": [0]}))
    teacher = _write(
        tmp_path,
        "teacher.csv",
        _export(
            ["a"],
            {"ACL": ["positive"], "MCL": ["positive"]},
            model={"ACL": [0.9], "MCL": [0.9]},
        ),
    )

    result = audit(train_csv=train, teacher=teacher)
    assert result["model_confidence_is_constant"] is True
    assert result["model_confidence_auc"] == 0.5


def test_training_targets_is_refused_with_a_useful_message(tmp_path):
    """It drops the 58 expert studies by design, so it can never be audited."""
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    teacher = _write(tmp_path, "teacher.csv", _export(["z"], {"ACL": ["positive"]}))

    with pytest.raises(ValueError, match="structured_labels.csv"):
        cell_table(train_csv=train, teacher=teacher)


def test_a_missing_export_names_the_file_it_wants(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    with pytest.raises(FileNotFoundError, match="structured_labels.csv"):
        cell_table(train_csv=train, teacher=tmp_path / "nowhere")
