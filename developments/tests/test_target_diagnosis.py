"""Whether 58 studies can support the claim being made about a target.

The whole point is to stop a training run being spent on a target that is merely
under-sampled. That only works if the interval is right and if "distinguishable
from chance" means strictly what it says, so both are pinned here.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.target_diagnosis import (
    best_epoch_per_target,
    diagnose,
    expert_class_counts,
    hanley_mcneil_se,
)


def _train(uids, truths, gold=None):
    gold = [True] * len(uids) if gold is None else gold
    frame = {
        "StudyInstanceUID": list(uids),
        "Report": ["report text"] * len(uids),
    }
    for target in TARGETS:
        frame[target] = [
            truths.get(target, [None] * len(uids))[i] if is_gold else None
            for i, is_gold in enumerate(gold)
        ]
    return pd.DataFrame(frame)


def _write(tmp_path, frame, name="train.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


# --- the standard error ------------------------------------------------------


def test_more_data_narrows_the_interval():
    wide = hanley_mcneil_se(0.55, 8, 50)
    narrow = hanley_mcneil_se(0.55, 80, 500)
    assert wide > narrow > 0


def test_a_perfect_auc_has_almost_no_spread():
    assert hanley_mcneil_se(1.0, 20, 20) == pytest.approx(0.0, abs=1e-9)


def test_a_target_with_no_positives_has_no_standard_error():
    assert math.isnan(hanley_mcneil_se(0.7, 0, 50))
    assert math.isnan(hanley_mcneil_se(0.7, 50, 0))


def test_the_spread_at_realistic_expert_counts_is_large():
    """Eight positives against fifty negatives cannot separate 0.55 from 0.70."""
    se = hanley_mcneil_se(0.55, 8, 50)
    assert se > 0.09


# --- counting the expert surface ---------------------------------------------


def test_positives_and_negatives_are_counted_per_target(tmp_path):
    path = _write(tmp_path, _train(["a", "b", "c"], {"ACL": [1, 0, 0], "MCL": [1, 1, 0]}))
    counts = expert_class_counts(path)

    assert counts["ACL"] == {"positives": 1, "negatives": 2, "labelled": 3, "pairs": 2}
    assert counts["MCL"]["positives"] == 2


def test_report_only_studies_are_not_counted(tmp_path):
    path = _write(
        tmp_path, _train(["g", "r"], {"ACL": [1, 1]}, gold=[True, False])
    )
    assert expert_class_counts(path)["ACL"]["labelled"] == 1


def test_a_file_with_no_expert_studies_is_refused(tmp_path):
    path = _write(tmp_path, _train(["r"], {"ACL": [None]}, gold=[False]))
    with pytest.raises(ValueError, match="no expert-labelled studies"):
        expert_class_counts(path)


# --- the verdict -------------------------------------------------------------


def test_a_target_whose_interval_spans_a_half_is_not_called_a_failure(tmp_path):
    """The claim this tool exists to check."""
    path = _write(
        tmp_path,
        _train([f"s{i}" for i in range(58)], {"ACL": [1] * 8 + [0] * 50}),
    )
    result = diagnose(train_csv=path, expert_auc={"ACL": 0.5478})
    acl = result["targets"]["ACL"]

    assert acl["positives"] == 8
    assert acl["ci_low"] < 0.5
    assert acl["distinguishable_from_chance"] is False


def test_a_strong_target_on_the_same_counts_is_distinguishable(tmp_path):
    path = _write(
        tmp_path,
        _train([f"s{i}" for i in range(58)], {"Effusion": [1] * 25 + [0] * 33}),
    )
    result = diagnose(train_csv=path, expert_auc={"Effusion": 0.8981})

    assert result["targets"]["Effusion"]["distinguishable_from_chance"] is True
    assert result["targets"]["Effusion"]["ci_low"] > 0.5


def test_the_interval_is_symmetric_about_the_estimate(tmp_path):
    path = _write(
        tmp_path, _train([f"s{i}" for i in range(20)], {"ACL": [1] * 6 + [0] * 14})
    )
    acl = diagnose(train_csv=path, expert_auc={"ACL": 0.7})["targets"]["ACL"]

    assert acl["expert_auc"] - acl["ci_low"] == pytest.approx(acl["ci_high"] - acl["expert_auc"])


def test_counts_are_reported_even_with_no_auc_given(tmp_path):
    path = _write(tmp_path, _train(["a", "b"], {"ACL": [1, 0]}))
    acl = diagnose(train_csv=path)["targets"]["ACL"]

    assert acl["pairs"] == 1
    assert "expert_auc" not in acl


# --- the second surface ------------------------------------------------------


def test_the_selected_epoch_is_the_one_with_the_best_macro(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            [
                {"epoch": 1, "validation_macro_auc": 0.70, "validation_per_target_auc": {"ACL": 0.60}},
                {"epoch": 2, "validation_macro_auc": 0.83, "validation_per_target_auc": {"ACL": 0.78}},
                {"epoch": 3, "validation_macro_auc": 0.81, "validation_per_target_auc": {"ACL": 0.75}},
            ]
        )
    )
    assert best_epoch_per_target(path) == {"ACL": 0.78}


def test_the_macro_is_derived_when_the_run_did_not_record_it(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            [
                {"epoch": 1, "validation_per_target_auc": {"ACL": 0.60, "MCL": 0.60}},
                {"epoch": 2, "validation_per_target_auc": {"ACL": 0.90, "MCL": 0.90}},
            ]
        )
    )
    assert best_epoch_per_target(path)["ACL"] == 0.90


def test_a_high_report_auc_beside_a_low_expert_auc_is_reported_as_a_gap(tmp_path):
    """That shape is a teacher/expert disagreement, not a model defect."""
    train = _write(
        tmp_path, _train([f"s{i}" for i in range(58)], {"Contusion": [1] * 12 + [0] * 46})
    )
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps([{"validation_macro_auc": 0.83, "validation_per_target_auc": {"Contusion": 0.82}}])
    )

    result = diagnose(
        train_csv=train, expert_auc={"Contusion": 0.5735}, history=history
    )
    assert result["targets"]["Contusion"]["report_minus_expert"] == pytest.approx(0.2465)


def test_a_history_without_validation_scores_is_refused(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps([{"epoch": 1, "train_loss": 0.5}]))
    with pytest.raises(ValueError, match="validation_per_target_auc"):
        best_epoch_per_target(path)


def test_a_missing_history_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="history"):
        best_epoch_per_target(tmp_path / "nowhere")


# --- output ------------------------------------------------------------------


def test_the_result_is_written_as_json(tmp_path):
    train = _write(tmp_path, _train(["a", "b"], {"ACL": [1, 0]}))
    out = tmp_path / "diagnosis.json"

    result = diagnose(train_csv=train, expert_auc={"ACL": 0.75}, out_json=out)
    assert json.loads(out.read_text()) == result
    assert result["expert_studies"] == 2
