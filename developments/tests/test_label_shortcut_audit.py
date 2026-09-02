"""Finding what predicts a label without reading the anatomy.

The number this produces will be used to decide whether Synovitis at 0.9954 has
an explanation other than the model reading synovium. Two things must be exactly
right for that: the AUC, and the permutation null that says how much of it a
column of this shape reaches on shuffled labels alone.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.label_shortcut_audit import (
    audit,
    auc,
    group_rates,
    permutation_ceiling,
)


def _teacher(uids, states, *, gold=None, extra=None):
    frame = {"StudyInstanceUID": list(uids)}
    if gold is not None:
        frame["is_gold"] = list(gold)
    for target in TARGETS:
        column = states.get(target, ["unmentioned"] * len(uids))
        frame[f"{target}__state"] = column
        frame[f"{target}__confidence"] = [
            0.9 if s in ("positive", "negated") else 0.0 for s in column
        ]
    frame.update(extra or {})
    return pd.DataFrame(frame)


def _table(uids, **columns):
    return pd.DataFrame({"StudyInstanceUID": list(uids), **columns})


def _write(tmp_path, frame, name):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


# --- the AUC -----------------------------------------------------------------


def test_a_perfect_ranking_is_one():
    assert auc(np.array([1.0, 1.0, 0.0, 0.0]), np.array([9.0, 8.0, 2.0, 1.0])) == 1.0


def test_a_reversed_ranking_is_zero():
    assert auc(np.array([1.0, 1.0, 0.0, 0.0]), np.array([1.0, 2.0, 8.0, 9.0])) == 0.0


def test_all_ties_are_a_half():
    assert auc(np.array([1.0, 0.0]), np.array([5.0, 5.0])) == 0.5


def test_one_tie_costs_half_a_pair():
    # 2 positives x 2 negatives = 4 pairs; one pair tied, the rest correct.
    value = auc(np.array([1.0, 1.0, 0.0, 0.0]), np.array([9.0, 5.0, 5.0, 1.0]))
    assert value == pytest.approx(3.5 / 4.0)


def test_one_class_only_is_not_a_number():
    assert np.isnan(auc(np.array([1.0, 1.0]), np.array([1.0, 2.0])))


# --- group rates and their null ----------------------------------------------


def test_a_group_is_scored_by_its_own_positive_rate():
    rates = group_rates(np.array([1.0, 1.0, 0.0]), np.array(["a", "a", "b"]))
    assert rates.tolist() == [1.0, 1.0, 0.0]


def test_groups_of_one_reach_a_perfect_score_on_noise(tmp_path):
    """Which is exactly why the raw number cannot be read on its own."""
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    unique = pd.Series(["a", "b", "c", "d"])
    assert auc(labels, group_rates(labels, unique.to_numpy())) == 1.0
    assert permutation_ceiling(
        labels, unique, categorical=True, draws=50, seed=1
    ) == pytest.approx(1.0)


def test_a_balanced_two_group_column_has_a_modest_null():
    labels = np.array([1.0] * 10 + [0.0] * 10)
    scanner = pd.Series(["A"] * 10 + ["B"] * 10)
    ceiling = permutation_ceiling(labels, scanner, categorical=True, draws=200, seed=3)

    assert 0.5 < ceiling < 0.85


def test_the_null_is_reproducible_for_a_seed():
    labels = np.array([1.0] * 6 + [0.0] * 6)
    values = pd.Series(list("aabbccddeeff"))
    first = permutation_ceiling(labels, values, categorical=True, draws=40, seed=7)
    second = permutation_ceiling(labels, values, categorical=True, draws=40, seed=7)

    assert first == second


# --- what the audit finds ----------------------------------------------------


def test_a_scanner_that_matches_the_label_is_found(tmp_path):
    uids = [f"s{i}" for i in range(20)]
    states = {"Synovitis": ["positive"] * 10 + ["negated"] * 10}
    teacher = _write(tmp_path, _teacher(uids, states), "teacher.csv")
    table = _write(
        tmp_path, _table(uids, scanner=["A"] * 10 + ["B"] * 10), "study_domain_table.csv"
    )

    best = audit(teacher=teacher, study_table=table, draws=100)["targets"]["Synovitis"]["best"]
    assert best["kind"] == "metadata"
    assert best["column"] == "scanner"
    assert best["auc"] > 0.95
    assert best["clears_null"] is True
    assert best["is_shortcut"] is True


def test_a_scanner_unrelated_to_the_label_is_not_a_shortcut(tmp_path):
    uids = [f"s{i}" for i in range(20)]
    states = {"Synovitis": ["positive", "negated"] * 10}
    teacher = _write(tmp_path, _teacher(uids, states), "teacher.csv")
    table = _write(
        tmp_path, _table(uids, scanner=["A"] * 10 + ["B"] * 10), "study_domain_table.csv"
    )

    best = audit(teacher=teacher, study_table=table, draws=100)["targets"]["Synovitis"]["best"]
    assert best["is_shortcut"] is False
    assert best["auc"] <= best["null_p95"]


def test_a_sibling_target_that_copies_the_label_is_found(tmp_path):
    uids = [f"s{i}" for i in range(20)]
    column = ["positive"] * 10 + ["negated"] * 10
    teacher = _write(
        tmp_path, _teacher(uids, {"Synovitis": column, "Effusion": column}), "teacher.csv"
    )

    best = audit(teacher=teacher, draws=50)["targets"]["Synovitis"]["best"]
    assert best["kind"] == "sibling"
    assert best["column"] == "Effusion"
    assert best["auc"] == 1.0
    assert best["is_shortcut"] is True


def test_a_backwards_predictor_counts_just_as_much(tmp_path):
    """A column that ranks the label upside down is still a shortcut."""
    uids = [f"s{i}" for i in range(20)]
    teacher = _write(
        tmp_path,
        _teacher(
            uids,
            {
                "Synovitis": ["positive"] * 10 + ["negated"] * 10,
                "Effusion": ["negated"] * 10 + ["positive"] * 10,
            },
        ),
        "teacher.csv",
    )
    assert audit(teacher=teacher, draws=50)["targets"]["Synovitis"]["best"]["auc"] == 1.0


def test_a_silent_sibling_cell_is_not_read_as_a_negative(tmp_path):
    """Silence is never a negative anywhere else in this project."""
    uids = [f"s{i}" for i in range(4)]
    teacher = _write(
        tmp_path,
        _teacher(
            uids,
            {
                "Synovitis": ["positive", "positive", "negated", "negated"],
                "Effusion": ["positive", "unmentioned", "unmentioned", "negated"],
            },
        ),
        "teacher.csv",
    )
    # Effusion scores 1.0, 0.5, 0.5, 0.0 against Synovitis 1,1,0,0: 3.5 of 4 pairs.
    siblings = audit(teacher=teacher, draws=20)["targets"]["Synovitis"]["sibling_targets"]
    assert siblings["Effusion"]["auc"] == pytest.approx(0.875)


# --- what is and is not scored -----------------------------------------------


def test_gold_studies_are_excluded(tmp_path):
    uids = ["a", "b", "g"]
    teacher = _write(
        tmp_path,
        _teacher(
            uids,
            {"Synovitis": ["positive", "negated", "positive"]},
            gold=[False, False, True],
        ),
        "teacher.csv",
    )
    assert audit(teacher=teacher, draws=0)["targets"]["Synovitis"]["cells"] == 2


def test_silence_is_not_a_scored_cell(tmp_path):
    teacher = _write(
        tmp_path,
        _teacher(["a", "b", "c"], {"Synovitis": ["positive", "negated", "unmentioned"]}),
        "teacher.csv",
    )
    item = audit(teacher=teacher, draws=0)["targets"]["Synovitis"]

    assert item["cells"] == 2
    assert item["positives"] == 1
    assert item["negatives"] == 1


def test_a_target_with_one_class_gets_no_verdict(tmp_path):
    teacher = _write(
        tmp_path, _teacher(["a", "b"], {"Synovitis": ["positive", "positive"]}), "teacher.csv"
    )
    item = audit(teacher=teacher, draws=0)["targets"]["Synovitis"]

    assert item["cells"] == 2
    assert "best" not in item


def test_the_study_id_is_never_offered_as_a_predictor(tmp_path):
    """It identifies the study rather than describing it, and would score 1.0."""
    uids = [f"s{i}" for i in range(20)]
    teacher = _write(
        tmp_path,
        _teacher(uids, {"Synovitis": ["positive"] * 10 + ["negated"] * 10}),
        "teacher.csv",
    )
    table = _write(tmp_path, _table(uids, scanner=["A"] * 20), "study_domain_table.csv")

    metadata = audit(teacher=teacher, study_table=table, draws=20)["targets"]["Synovitis"]["metadata"]
    assert "StudyInstanceUID" not in metadata


def test_a_missing_study_table_names_the_file_it_wants(tmp_path):
    teacher = _write(
        tmp_path, _teacher(["a", "b"], {"ACL": ["positive", "negated"]}), "teacher.csv"
    )
    with pytest.raises(FileNotFoundError, match="study_domain_table.csv"):
        audit(teacher=teacher, study_table=tmp_path / "nowhere")


def test_the_result_is_written_as_json(tmp_path):
    teacher = _write(
        tmp_path, _teacher(["a", "b"], {"ACL": ["positive", "negated"]}), "teacher.csv"
    )
    out = tmp_path / "shortcut.json"

    result = audit(teacher=teacher, draws=20, out_json=out)
    assert json.loads(out.read_text()) == result
