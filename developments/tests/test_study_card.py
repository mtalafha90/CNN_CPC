"""Showing one study end to end, so a person can judge the join by eye.

The card exists to answer "does this label describe this case". That only works
if every column really comes from the study named at the top: a card that
silently showed a neighbour's report would be worse than no card, because it
would look like confirmation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.study_card import card


def _train(tmp_path, rows):
    frame = {
        "StudyInstanceUID": [r["uid"] for r in rows],
        "Report": [r.get("report", "") for r in rows],
    }
    for target in TARGETS:
        frame[target] = [r.get("truth", {}).get(target) for r in rows]
    path = tmp_path / "train.csv"
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def _export(tmp_path, name, rows, *, model=False):
    frame = {"StudyInstanceUID": [r["uid"] for r in rows]}
    for target in TARGETS:
        states = [r.get("states", {}).get(target, "unmentioned") for r in rows]
        frame[f"{target}__state"] = states
        frame[f"{target}__confidence"] = [
            0.9 if s in ("positive", "negated") else 0.0 for s in states
        ]
        frame[f"{target}__evidence"] = [
            r.get("evidence", {}).get(target, "") for r in rows
        ]
        frame[f"{target}__reason"] = [
            r.get("reason", {}).get(target, "no_target_evidence") for r in rows
        ]
        if model:
            frame[f"{target}__model_confidence"] = [
                r.get("model", {}).get(target, float("nan")) for r in rows
            ]
    path = tmp_path / name
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


@pytest.fixture
def world(tmp_path):
    rows = [
        {
            "uid": "alpha",
            "report": "complete acl tear. menisci intact.",
            "truth": {"ACL": 1.0, "MCL": 0.0},
            "states": {"ACL": "positive", "MCL": "negated"},
            "evidence": {"ACL": "complete acl tear", "MCL": "the mcl is intact"},
        },
        {
            "uid": "beta",
            "report": "unremarkable knee.",
            "truth": {"ACL": 0.0},
            "states": {"ACL": "negated"},
            "evidence": {"ACL": "acl unremarkable"},
        },
    ]
    return {
        "root": tmp_path,
        "train": _train(tmp_path, rows),
        "teacher": _export(tmp_path, "teacher.csv", rows),
        "b6": _export(tmp_path, "b6.csv", rows),
    }


# --- the join ----------------------------------------------------------------


def test_the_card_shows_the_named_study_and_not_its_neighbour(world):
    result = card(data_root=world["root"], study="alpha", teacher=world["teacher"])

    assert result["study"] == "alpha"
    assert "complete acl tear" in result["report"]
    assert "unremarkable" not in result["report"]


def test_the_second_study_gets_its_own_report(world):
    result = card(data_root=world["root"], study="beta", teacher=world["teacher"])
    assert result["report"] == "unremarkable knee."


def test_a_study_that_is_not_in_train_csv_is_refused(world):
    with pytest.raises(KeyError, match="no study"):
        card(data_root=world["root"], study="gamma", teacher=world["teacher"])


def test_a_study_missing_from_the_teacher_is_refused(world, tmp_path):
    """Silently blank labels would look like 'the report said nothing'."""
    short = _export(tmp_path, "short.csv", [{"uid": "beta"}])
    with pytest.raises(KeyError, match="teacher export"):
        card(data_root=world["root"], study="alpha", teacher=short)


# --- what each column says ---------------------------------------------------


def test_the_teacher_and_expert_columns_are_read_separately(world):
    result = card(data_root=world["root"], study="alpha", teacher=world["teacher"])
    by_target = {item["target"]: item for item in result["findings"]}

    assert by_target["ACL"]["teacher"] == "positive"
    assert by_target["ACL"]["expert"] == "positive"
    assert by_target["MCL"]["teacher"] == "negated"
    assert by_target["MCL"]["expert"] == "negative"


def test_an_unanswered_finding_shows_a_dash_not_a_negative(world):
    """Report silence is never a negative, here as everywhere else."""
    result = card(data_root=world["root"], study="alpha", teacher=world["teacher"])
    by_target = {item["target"]: item for item in result["findings"]}

    assert by_target["Effusion"]["teacher"] == "-"
    assert by_target["Effusion"]["expert"] == "-"
    assert by_target["Effusion"]["agrees"] is None


def test_a_disagreement_is_flagged(world, tmp_path):
    rows = [{"uid": "alpha", "truth": {"ACL": 0.0}, "states": {"ACL": "positive"}}]
    teacher = _export(tmp_path, "wrong.csv", rows)
    train = _train(tmp_path, rows)

    result = card(data_root=train.parent, study="alpha", teacher=teacher)
    by_target = {item["target"]: item for item in result["findings"]}

    assert by_target["ACL"]["agrees"] is False
    assert result["disagreements"] == 1


def test_the_evidence_clause_comes_from_the_parser_export(world):
    result = card(
        data_root=world["root"], study="alpha", teacher=world["teacher"], b6_export=world["b6"]
    )
    by_target = {item["target"]: item for item in result["findings"]}

    assert by_target["ACL"]["evidence"] == "complete acl tear"
    assert by_target["MCL"]["evidence"] == "the mcl is intact"


def test_without_the_parser_export_there_is_simply_no_clause(world):
    result = card(data_root=world["root"], study="alpha", teacher=world["teacher"])
    assert all(item["evidence"] == "" for item in result["findings"])


# --- which labeller ----------------------------------------------------------


def test_a_cell_with_no_self_report_is_attributed_to_the_parser(tmp_path):
    rows = [
        {
            "uid": "alpha",
            "states": {"ACL": "positive", "MCL": "negated"},
            "model": {"ACL": float("nan"), "MCL": 0.8},
        }
    ]
    train = _train(tmp_path, rows)
    teacher = _export(tmp_path, "merged.csv", rows, model=True)

    result = card(data_root=train.parent, study="alpha", teacher=teacher)
    by_target = {item["target"]: item for item in result["findings"]}

    assert by_target["ACL"]["from"] == "parser"
    assert by_target["MCL"]["from"] == "LLM"


def test_a_raw_parser_export_attributes_everything_to_the_parser(world):
    """It has no __model_confidence column at all, and every cell is its own."""
    result = card(data_root=world["root"], study="alpha", teacher=world["teacher"])
    answered = [item for item in result["findings"] if item["teacher"] != "-"]

    assert answered and all(item["from"] == "parser" for item in answered)


def test_an_unanswered_cell_has_no_labeller(world):
    result = card(data_root=world["root"], study="alpha", teacher=world["teacher"])
    by_target = {item["target"]: item for item in result["findings"]}

    assert by_target["Effusion"]["from"] == ""


# --- gold and counts ---------------------------------------------------------


def test_an_expert_labelled_study_is_named_as_one(world):
    assert card(data_root=world["root"], study="alpha", teacher=world["teacher"])["is_gold"]


def test_a_report_only_study_is_not(tmp_path):
    rows = [{"uid": "alpha", "states": {"ACL": "positive"}}]
    train = _train(tmp_path, rows)
    teacher = _export(tmp_path, "t.csv", rows)

    result = card(data_root=train.parent, study="alpha", teacher=teacher)
    assert result["is_gold"] is False
    assert result["disagreements"] == 0


def test_the_answered_count_matches_the_rows(world):
    result = card(data_root=world["root"], study="alpha", teacher=world["teacher"])
    assert result["answered"] == sum(1 for i in result["findings"] if i["teacher"] != "-")
    assert result["answered"] == 2


def test_a_low_confidence_cell_is_not_answered(tmp_path):
    rows = [{"uid": "alpha", "states": {"ACL": "positive"}}]
    train = _train(tmp_path, rows)
    teacher = _export(tmp_path, "t.csv", rows)

    assert card(data_root=train.parent, study="alpha", teacher=teacher, min_confidence=0.95)[
        "answered"
    ] == 0
