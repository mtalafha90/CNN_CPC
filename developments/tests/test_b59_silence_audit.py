"""B59 tests.

The rule is three clauses and each one exists because a different way of being
wrong is cheap to fall into. Each clause therefore gets a test that fails if the
clause is removed, built from a hand-made surface where the right answer is
known by construction rather than by running the code and writing down what it
said.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.b59_silence_audit import (
    B59_VERSION,
    CONVERTIBLE_STATES,
    SILENCE_MAX_POSITIVE_RATE,
    SILENCE_MIN_GOLD_CELLS,
    SILENCE_NEGATIVE_TARGET,
    SILENCE_NEGATIVE_WEIGHT,
    VERDICT_CONVERT,
    VERDICT_NO_EVIDENCE,
    VERDICT_NOT_INFORMATIVE,
    VERDICT_TOO_MANY_FALSE_NEGATIVES,
    apply_policy,
    audit,
    decide_target,
    gold_surface,
    policy_from,
    state_columns,
    state_truth_table,
    training_cells,
    wilson_interval,
)
from rsna_knee.b7_weak_supervision import B7_NEGATIVE_TARGET, B7_NEGATIVE_WEIGHT
from rsna_knee.constants import TARGETS


# --- Wilson, which is the whole reason a tiny surface can be read at all -------


def test_a_proportion_with_no_positives_still_has_width():
    """The normal approximation gives zero width here and waves everything through."""
    low, high = wilson_interval(0, 20)
    assert low == 0.0
    assert high > 0.1, "0/20 must not look like certainty"


def test_a_proportion_with_every_cell_positive_stays_inside_one():
    low, high = wilson_interval(20, 20)
    assert high == 1.0
    assert low < 1.0


def test_fewer_cells_means_a_wider_interval():
    _, wide = wilson_interval(1, 10)
    _, narrow = wilson_interval(10, 100)
    assert wide > narrow


def test_the_interval_brackets_the_point_estimate():
    low, high = wilson_interval(3, 25)
    assert low <= 3 / 25 <= high


def test_an_impossible_count_is_refused():
    with pytest.raises(ValueError, match="not a proportion"):
        wilson_interval(5, 3)


def test_an_empty_state_returns_no_interval():
    low, high = wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)


# --- the three clauses of the rule --------------------------------------------


def _row(*, n, positives, prevalence):
    low, high = wilson_interval(positives, n)
    return pd.Series(
        {
            "target": "ACL",
            "state": "unmentioned",
            "n": n,
            "gold_positive": positives,
            "p_gold_positive": positives / n if n else float("nan"),
            "wilson_low": low,
            "wilson_high": high,
            "prevalence": prevalence,
        }
    )


def test_too_few_cells_gets_no_verdict_at_all():
    """A handful of rows cannot decide anything on a 58-study surface."""
    verdict, reason = decide_target(
        _row(n=SILENCE_MIN_GOLD_CELLS - 1, positives=0, prevalence=0.5)
    )
    assert verdict == VERDICT_NO_EVIDENCE
    assert str(SILENCE_MIN_GOLD_CELLS) in reason


def test_a_perfect_but_tiny_state_is_still_refused():
    """0 of 5 positive looks ideal and is worth nothing. This is the trap."""
    verdict, _ = decide_target(_row(n=5, positives=0, prevalence=0.6))
    assert verdict == VERDICT_NO_EVIDENCE


def test_a_state_that_matches_its_base_rate_says_nothing():
    """Silence as likely to be positive as any study carries no information."""
    verdict, reason = decide_target(_row(n=40, positives=4, prevalence=0.10))
    assert verdict == VERDICT_NOT_INFORMATIVE
    assert "base rate" in reason


def test_a_rare_finding_cannot_be_converted_on_its_base_rate_alone():
    """The failure this clause exists for.

    A finding positive in 5% of knees will be silent-and-negative about 95% of
    the time no matter what silence means. Without the prevalence comparison
    that looks like strong evidence, and converting would add thousands of
    cells carrying no signal.
    """
    verdict, _ = decide_target(_row(n=50, positives=2, prevalence=0.05))
    assert verdict == VERDICT_NOT_INFORMATIVE


def test_an_informative_state_above_the_ceiling_is_still_refused():
    """Beating the base rate is not enough when the cells are asserted negatives."""
    verdict, reason = decide_target(_row(n=60, positives=15, prevalence=0.70))
    assert verdict == VERDICT_TOO_MANY_FALSE_NEGATIVES
    assert "ceiling" in reason


def test_a_state_that_passes_every_clause_converts():
    verdict, reason = decide_target(_row(n=50, positives=1, prevalence=0.55))
    assert verdict == VERDICT_CONVERT
    assert "1/50" in reason


def test_the_ceiling_is_read_off_the_interval_not_the_point_estimate():
    """20 of 100 sits exactly on the ceiling; its upper bound does not."""
    verdict, _ = decide_target(
        _row(n=100, positives=int(SILENCE_MAX_POSITIVE_RATE * 100), prevalence=0.9)
    )
    assert verdict == VERDICT_TOO_MANY_FALSE_NEGATIVES


# --- a whole surface, built so the answer is known ----------------------------


def _surface(tmp_path, *, states_for_acl, acl_truth, n_report_only=40):
    """58 gold studies plus report-only rows, with ACL controlled by hand.

    Every other target is filled with a fixed, uninformative pattern so that
    only ACL's verdict is under test.
    """
    gold_rows, label_rows = [], []
    for i, (state, truth) in enumerate(zip(states_for_acl, acl_truth)):
        uid = f"gold{i:03d}"
        row = {"StudyInstanceUID": uid, "Report": "text"}
        for target in TARGETS:
            row[target] = float(truth) if target == "ACL" else float(i % 2)
        gold_rows.append(row)

        label = {"StudyInstanceUID": uid}
        for target in TARGETS:
            label[f"{target}__state"] = state if target == "ACL" else "positive"
            label[f"{target}__confidence"] = 0.90
        label_rows.append(label)

    for i in range(n_report_only):
        uid = f"weak{i:03d}"
        row = {"StudyInstanceUID": uid, "Report": "text"}
        for target in TARGETS:
            row[target] = np.nan
        gold_rows.append(row)

        label = {"StudyInstanceUID": uid}
        for target in TARGETS:
            label[f"{target}__state"] = "unmentioned" if target == "ACL" else "negated"
            label[f"{target}__confidence"] = 0.90
        label_rows.append(label)

    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    train_csv = tmp_path / "train.csv"
    structured_csv = tmp_path / "structured_labels.csv"
    pd.DataFrame(gold_rows).to_csv(train_csv, index=False)
    pd.DataFrame(label_rows).to_csv(structured_csv, index=False)
    return train_csv, structured_csv


def test_the_gold_surface_is_the_labelled_rows_only(tmp_path):
    train_csv, _ = _surface(
        tmp_path, states_for_acl=["unmentioned"] * 58, acl_truth=[0] * 58
    )
    uids, truth = gold_surface(pd.read_csv(train_csv))
    assert len(uids) == 58
    assert truth.shape == (58, len(TARGETS))
    assert all(uid.startswith("gold") for uid in uids)


def test_a_train_csv_with_no_expert_rows_is_refused(tmp_path):
    frame = pd.DataFrame(
        [{"StudyInstanceUID": "a", "Report": "x", **{t: np.nan for t in TARGETS}}]
    )
    with pytest.raises(ValueError, match="no expert-labelled studies"):
        gold_surface(frame)


def test_the_training_export_is_named_when_the_gold_rows_are_missing(tmp_path):
    """`training_targets.csv` drops the gold rows by design; say so."""
    structured = pd.DataFrame([{"StudyInstanceUID": "weak000"}])
    with pytest.raises(ValueError, match="training_targets.csv"):
        state_columns(structured, ["gold000"])


def test_an_unknown_state_is_refused(tmp_path):
    states = pd.DataFrame(
        [{f"{t}__state": "probably" for t in TARGETS}], index=["gold000"]
    )
    truth = np.ones((1, len(TARGETS)))
    with pytest.raises(ValueError, match="unknown states"):
        state_truth_table(truth, states)


def test_silence_that_really_indicates_absence_is_converted(tmp_path):
    """29 silent studies, one positive among them, against a 50% base rate."""
    states = ["unmentioned"] * 29 + ["positive"] * 29
    truth = [1] + [0] * 28 + [1] * 28 + [0]
    train_csv, structured_csv = _surface(
        tmp_path, states_for_acl=states, acl_truth=truth
    )

    result = audit(train_csv, structured_csv)
    acl = result["decisions"].query("target == 'ACL' and state == 'unmentioned'").iloc[0]

    assert acl["gold_cells"] == 29
    assert acl["gold_positive"] == 1
    assert acl["verdict"] == VERDICT_CONVERT
    assert acl["training_cells_added"] == 40


def test_silence_that_hides_positives_is_refused(tmp_path):
    """The same shape, with silence half positive. Nothing may be converted."""
    states = ["unmentioned"] * 29 + ["positive"] * 29
    truth = [1, 0] * 14 + [1] + [1] * 29
    train_csv, structured_csv = _surface(
        tmp_path, states_for_acl=states, acl_truth=truth
    )

    result = audit(train_csv, structured_csv)
    acl = result["decisions"].query("target == 'ACL' and state == 'unmentioned'").iloc[0]

    assert acl["verdict"] != VERDICT_CONVERT
    assert acl["training_cells_added"] == 0


def test_a_refusal_reports_the_cells_it_declined(tmp_path):
    """`available` stays populated so the refusal can be read as a cost."""
    states = ["unmentioned"] * 29 + ["positive"] * 29
    truth = [1, 0] * 14 + [1] + [1] * 29
    train_csv, structured_csv = _surface(
        tmp_path, states_for_acl=states, acl_truth=truth
    )
    acl = (
        audit(train_csv, structured_csv)["decisions"]
        .query("target == 'ACL' and state == 'unmentioned'")
        .iloc[0]
    )
    assert acl["training_cells_available"] == 40
    assert acl["training_cells_added"] == 0


def test_coverage_before_and_after_are_both_reported(tmp_path):
    states = ["unmentioned"] * 29 + ["positive"] * 29
    truth = [1] + [0] * 28 + [1] * 28 + [0]
    train_csv, structured_csv = _surface(
        tmp_path, states_for_acl=states, acl_truth=truth
    )
    coverage = audit(train_csv, structured_csv)["summary"]["coverage"]

    assert coverage["coverage_after"] > coverage["coverage_now"]
    assert coverage["cells_added"] == 40
    assert coverage["grid_cells"] == 40 * len(TARGETS)


def test_every_state_appears_for_every_target(tmp_path):
    train_csv, structured_csv = _surface(
        tmp_path, states_for_acl=["unmentioned"] * 58, acl_truth=[0] * 58
    )
    evidence = audit(train_csv, structured_csv)["evidence"]
    assert len(evidence) == len(TARGETS) * 4
    assert set(evidence["state"]) == {"positive", "negated", "uncertain", "unmentioned"}


def test_uncertain_gets_its_own_verdict_never_silence(tmp_path):
    """Two different claims, measured side by side, decided separately."""
    train_csv, structured_csv = _surface(
        tmp_path, states_for_acl=["unmentioned"] * 58, acl_truth=[0] * 58
    )
    decisions = audit(train_csv, structured_csv)["decisions"]
    assert set(decisions["state"]) == set(CONVERTIBLE_STATES)


# --- the policy ----------------------------------------------------------------


def test_an_empty_policy_is_still_written():
    """"Silence says nothing" is a result and must be recorded, not dropped."""
    decisions = pd.DataFrame(
        [{"target": "ACL", "state": "unmentioned", "verdict": VERDICT_NOT_INFORMATIVE}]
    )
    policy = policy_from(decisions)
    assert policy["convert"] == {}
    assert policy["version"] == B59_VERSION


def test_the_policy_round_trips_as_json():
    decisions = pd.DataFrame(
        [{"target": "ACL", "state": "unmentioned", "verdict": VERDICT_CONVERT}]
    )
    policy = json.loads(json.dumps(policy_from(decisions)))
    assert policy["convert"]["unmentioned"] == ["ACL"]


def test_the_frozen_weight_is_a_quarter_of_an_explicit_negation():
    assert SILENCE_NEGATIVE_TARGET == B7_NEGATIVE_TARGET
    assert SILENCE_NEGATIVE_WEIGHT * 4 == pytest.approx(B7_NEGATIVE_WEIGHT)


# --- applying it ----------------------------------------------------------------


def test_applying_a_policy_writes_only_the_named_cells():
    n = 4
    structured = pd.DataFrame(
        {
            **{f"{t}__state": ["negated"] * n for t in TARGETS},
            "ACL__state": ["unmentioned", "negated", "unmentioned", "positive"],
        }
    )
    targets = np.full((n, len(TARGETS)), 0.5, dtype=np.float32)
    weights = np.zeros((n, len(TARGETS)), dtype=np.float32)
    policy = policy_from(
        pd.DataFrame([{"target": "ACL", "state": "unmentioned", "verdict": VERDICT_CONVERT}])
    )

    report = apply_policy(targets, weights, structured, policy)

    j = TARGETS.index("ACL")
    assert report["cells_written"] == 2
    assert weights[[0, 2], j].tolist() == pytest.approx([SILENCE_NEGATIVE_WEIGHT] * 2)
    assert targets[[0, 2], j].tolist() == pytest.approx([SILENCE_NEGATIVE_TARGET] * 2)
    assert weights[[1, 3], j].tolist() == [0.0, 0.0]
    assert weights[:, TARGETS.index("MCL")].sum() == 0.0, "untouched targets stay untouched"


def test_a_cell_the_teacher_already_answers_is_never_overwritten():
    n = 2
    structured = pd.DataFrame(
        {
            **{f"{t}__state": ["negated"] * n for t in TARGETS},
            "ACL__state": ["unmentioned", "unmentioned"],
        }
    )
    targets = np.full((n, len(TARGETS)), 0.5, dtype=np.float32)
    weights = np.zeros((n, len(TARGETS)), dtype=np.float32)
    j = TARGETS.index("ACL")
    targets[0, j], weights[0, j] = 0.85, 0.50  # an existing positive call

    apply_policy(targets, weights, structured, policy_from(
        pd.DataFrame([{"target": "ACL", "state": "unmentioned", "verdict": VERDICT_CONVERT}])
    ))

    assert (targets[0, j], weights[0, j]) == (pytest.approx(0.85), pytest.approx(0.50))
    assert weights[1, j] == pytest.approx(SILENCE_NEGATIVE_WEIGHT)


def test_a_policy_from_another_version_is_refused():
    structured = pd.DataFrame({f"{t}__state": ["unmentioned"] for t in TARGETS})
    with pytest.raises(ValueError, match="not a b59"):
        apply_policy(
            np.zeros((1, len(TARGETS))),
            np.zeros((1, len(TARGETS))),
            structured,
            {"version": "something_else", "convert": {}},
        )


def test_a_policy_naming_a_trainable_state_is_refused():
    """Converting `positive` or `negated` would overwrite the parser, not extend it."""
    structured = pd.DataFrame({f"{t}__state": ["positive"] for t in TARGETS})
    with pytest.raises(ValueError, match="unconvertible state"):
        apply_policy(
            np.zeros((1, len(TARGETS))),
            np.zeros((1, len(TARGETS))),
            structured,
            {
                "version": B59_VERSION,
                "negative_target": SILENCE_NEGATIVE_TARGET,
                "negative_weight": SILENCE_NEGATIVE_WEIGHT,
                "convert": {"positive": ["ACL"]},
            },
        )


def test_misaligned_rows_are_refused():
    structured = pd.DataFrame({f"{t}__state": ["unmentioned"] * 3 for t in TARGETS})
    with pytest.raises(ValueError, match="same studies in the same order"):
        apply_policy(
            np.zeros((2, len(TARGETS))),
            np.zeros((2, len(TARGETS))),
            structured,
            policy_from(pd.DataFrame(columns=["target", "state", "verdict"])),
        )


# --- the ruler is not negotiable -------------------------------------------------


def test_only_expert_rows_reach_the_evidence(tmp_path):
    """The ruler is the 58, and no amount of report-only data may dilute it.

    Coverage predicts the report surface's AUC at Pearson -0.931, so that
    surface cannot referee a change to coverage. This checks the behaviour
    rather than the wording: adding report-only studies must change the size
    of the intervention and nothing about the evidence or the verdict.
    """
    states = ["unmentioned"] * 29 + ["positive"] * 29
    truth = [1] + [0] * 28 + [1] * 28 + [0]

    small = audit(*_surface(tmp_path / "a", states_for_acl=states,
                            acl_truth=truth, n_report_only=10))
    large = audit(*_surface(tmp_path / "b", states_for_acl=states,
                            acl_truth=truth, n_report_only=500))

    def acl(result):
        return result["decisions"].query(
            "target == 'ACL' and state == 'unmentioned'"
        ).iloc[0]

    assert acl(small)["gold_cells"] == acl(large)["gold_cells"] == 29
    assert acl(small)["verdict"] == acl(large)["verdict"]
    assert acl(small)["wilson_high"] == acl(large)["wilson_high"]
    assert acl(small)["training_cells_added"] == 10
    assert acl(large)["training_cells_added"] == 500
    assert small["summary"]["gold_studies"] == large["summary"]["gold_studies"] == 58


def test_the_summary_names_its_surface(tmp_path):
    train_csv, structured_csv = _surface(
        tmp_path, states_for_acl=["unmentioned"] * 58, acl_truth=[0] * 58
    )
    summary = audit(train_csv, structured_csv)["summary"]
    assert summary["surface"] == "expert_58_only"
    assert summary["gold_studies"] == 58


def test_the_frozen_rule_is_recorded_in_the_output(tmp_path):
    """A threshold that is not written down beside the result is not frozen."""
    train_csv, structured_csv = _surface(
        tmp_path, states_for_acl=["unmentioned"] * 58, acl_truth=[0] * 58
    )
    rule = audit(train_csv, structured_csv)["summary"]["rule"]
    assert rule["min_gold_cells"] == SILENCE_MIN_GOLD_CELLS
    assert rule["max_positive_rate"] == SILENCE_MAX_POSITIVE_RATE
    assert rule["negative_weight"] == SILENCE_NEGATIVE_WEIGHT


def test_the_training_size_table_counts_report_only_studies(tmp_path):
    train_csv, structured_csv = _surface(
        tmp_path, states_for_acl=["unmentioned"] * 58, acl_truth=[0] * 58, n_report_only=17
    )
    sizes = training_cells(pd.read_csv(structured_csv), pd.read_csv(train_csv))
    assert set(sizes["report_only_studies"]) == {17}
    assert int(sizes.query("target == 'ACL'")["unmentioned_cells"].iloc[0]) == 17
