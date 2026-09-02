"""Splitting the parser's wrong calls by the rule that produced them.

The value of this diagnostic rests on one thing: that each wrong cell is
attributed to the rule that actually fired and carries the clause that fired it.
Get that wrong and the map points at the wrong rule, which is worse than no map.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.parser_error_anatomy import anatomy, cell_table


def _train(uids, truths, reports=None, gold=None):
    gold = [True] * len(uids) if gold is None else gold
    frame = {
        "StudyInstanceUID": list(uids),
        "Report": list(reports) if reports else ["report text"] * len(uids),
    }
    for target in TARGETS:
        frame[target] = [
            truths.get(target, [None] * len(uids))[i] if is_gold else None
            for i, is_gold in enumerate(gold)
        ]
    return pd.DataFrame(frame)


def _b6(uids, states, *, reasons=None, evidence=None, confidences=None):
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
        frame[f"{target}__reason"] = (reasons or {}).get(
            target, ["no_target_evidence"] * len(uids)
        )
        frame[f"{target}__evidence"] = (evidence or {}).get(target, [""] * len(uids))
    return pd.DataFrame(frame)


def _write(tmp_path, name, frame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


# --- attribution -------------------------------------------------------------


def test_a_wrong_call_is_attributed_to_the_rule_that_fired(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [0], "MCL": [0]}))
    export = _write(
        tmp_path,
        "b6.csv",
        _b6(
            ["a"],
            {"ACL": ["positive"], "MCL": ["positive"]},
            reasons={
                "ACL": ["explicit_pathology_mention"],
                "MCL": ["explicit_structural_abnormality"],
            },
        ),
    )
    result = anatomy(train_csv=train, b6_export=export)

    assert result["by_reason"]["explicit_pathology_mention"]["wrong"] == 1
    assert result["by_reason"]["explicit_structural_abnormality"]["wrong"] == 1


def test_the_evidence_clause_travels_with_the_cell(tmp_path):
    """Without it the map says where to look and gives nothing to read."""
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [0]}))
    export = _write(
        tmp_path,
        "b6.csv",
        _b6(
            ["a"],
            {"ACL": ["positive"]},
            evidence={"ACL": ["complete tear of the anterior cruciate ligament"]},
        ),
    )
    cells = cell_table(train_csv=train, b6_export=export)

    assert cells["evidence"].iloc[0].startswith("complete tear")


def test_the_report_travels_too_so_a_misread_sentence_is_visible(tmp_path):
    train = _write(
        tmp_path,
        "train.csv",
        _train(["a"], {"ACL": [0]}, reports=["the ACL is intact throughout"]),
    )
    export = _write(tmp_path, "b6.csv", _b6(["a"], {"ACL": ["positive"]}))
    cells = cell_table(train_csv=train, b6_export=export)

    assert "intact" in cells["report"].iloc[0]


def test_positive_and_negated_calls_are_counted_apart(tmp_path):
    """52 of the teacher's 57 errors are positives; the split is the finding."""
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [0], "MCL": [1]}))
    export = _write(
        tmp_path, "b6.csv", _b6(["a"], {"ACL": ["positive"], "MCL": ["negated"]})
    )
    result = anatomy(train_csv=train, b6_export=export)

    assert result["positive_calls"] == {**result["positive_calls"], "cells": 1, "wrong": 1}
    assert result["negated_calls"] == {**result["negated_calls"], "cells": 1, "wrong": 1}


def test_the_positive_only_view_excludes_negated_calls(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [0], "MCL": [1]}))
    export = _write(
        tmp_path,
        "b6.csv",
        _b6(
            ["a"],
            {"ACL": ["positive"], "MCL": ["negated"]},
            reasons={"ACL": ["explicit_pathology_mention"], "MCL": ["explicit_negation"]},
        ),
    )
    result = anatomy(train_csv=train, b6_export=export)

    assert "explicit_negation" in result["by_reason"]
    assert "explicit_negation" not in result["by_reason_positive_calls_only"]


def test_a_correct_call_is_not_an_error(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    export = _write(tmp_path, "b6.csv", _b6(["a"], {"ACL": ["positive"]}))
    result = anatomy(train_csv=train, b6_export=export)

    assert result["overall"]["wrong"] == 0
    assert result["overall"]["cells"] == 1


# --- what counts as a call ---------------------------------------------------


def test_silence_is_not_a_call(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    export = _write(tmp_path, "b6.csv", _b6(["a"], {"ACL": ["unmentioned"]}))
    assert len(cell_table(train_csv=train, b6_export=export)) == 0


def test_hedging_is_not_a_call(tmp_path):
    """`uncertainty_scope` lowers coverage; it is not an error."""
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    export = _write(
        tmp_path,
        "b6.csv",
        _b6(
            ["a"],
            {"ACL": ["uncertain"]},
            reasons={"ACL": ["uncertainty_scope"]},
            confidences={"ACL": [0.25]},
        ),
    )
    assert len(cell_table(train_csv=train, b6_export=export)) == 0


def test_a_study_without_an_expert_label_is_not_scored(tmp_path):
    train = _write(
        tmp_path, "train.csv", _train(["a", "r"], {"ACL": [0, 0]}, gold=[True, False])
    )
    export = _write(tmp_path, "b6.csv", _b6(["a", "r"], {"ACL": ["positive"] * 2}))
    cells = cell_table(train_csv=train, b6_export=export)

    assert cells["StudyInstanceUID"].tolist() == ["a"]


# --- reading the output ------------------------------------------------------


def test_buckets_are_ordered_worst_first(tmp_path):
    train = _write(
        tmp_path,
        "train.csv",
        _train(["a", "b"], {"ACL": [0, 0], "MCL": [0, 1]}),
    )
    export = _write(
        tmp_path,
        "b6.csv",
        _b6(
            ["a", "b"],
            {"ACL": ["positive", "positive"], "MCL": ["positive", "positive"]},
            reasons={
                "ACL": ["explicit_pathology_mention"] * 2,
                "MCL": ["explicit_structural_abnormality"] * 2,
            },
        ),
    )
    result = anatomy(train_csv=train, b6_export=export)

    assert list(result["by_reason"]) == [
        "explicit_pathology_mention",  # 2 wrong
        "explicit_structural_abnormality",  # 1 wrong
    ]


def test_a_small_bucket_reports_a_wide_standard_error(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [0], "MCL": [1]}))
    export = _write(
        tmp_path, "b6.csv", _b6(["a"], {"ACL": ["positive"], "MCL": ["positive"]})
    )
    result = anatomy(train_csv=train, b6_export=export)

    assert result["overall"]["standard_error"] > 0.3


def test_the_disagreements_file_holds_only_the_wrong_cells(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1], "MCL": [0]}))
    export = _write(
        tmp_path, "b6.csv", _b6(["a"], {"ACL": ["positive"], "MCL": ["positive"]})
    )
    out = tmp_path / "anatomy"

    result = anatomy(train_csv=train, b6_export=export, out_root=out)
    wrong = pd.read_csv(out / "disagreements.csv")

    assert result["disagreements_written"] == 1
    assert wrong["target"].tolist() == ["MCL"]
    assert len(pd.read_csv(out / "all_expert_cells.csv")) == 2
    assert json.loads((out / "summary.json").read_text())["overall"]["cells"] == 2


# --- refusing the wrong file -------------------------------------------------


def test_a_merged_export_is_refused_because_it_drops_the_reason(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    merged = _b6(["a"], {"ACL": ["positive"]}).drop(
        columns=[f"{target}__reason" for target in TARGETS]
    )
    path = _write(tmp_path, "merged.csv", merged)

    with pytest.raises(ValueError, match="__reason"):
        cell_table(train_csv=train, b6_export=path)


def test_training_targets_is_refused_with_a_useful_message(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    export = _write(tmp_path, "b6.csv", _b6(["z"], {"ACL": ["positive"]}))

    with pytest.raises(ValueError, match="structured_labels.csv"):
        cell_table(train_csv=train, b6_export=export)


def test_a_missing_export_names_the_file_it_wants(tmp_path):
    train = _write(tmp_path, "train.csv", _train(["a"], {"ACL": [1]}))
    with pytest.raises(FileNotFoundError, match="__reason"):
        cell_table(train_csv=train, b6_export=tmp_path / "nowhere")
