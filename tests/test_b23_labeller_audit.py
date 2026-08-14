import numpy as np
import pandas as pd
import pytest

from rsna_knee.b23_labeller_audit import (
    audit_labeller,
    confusion_summary,
    format_audit,
    paired_state_only_bootstrap,
    state_only_scores,
    state_truth_table,
)
from rsna_knee.constants import TARGETS

N_STUDIES = 24


def _truth():
    rng = np.random.default_rng(11)
    # Alternate labels so every target has both classes on the gold surface.
    base = np.tile(np.array([0.0, 1.0]), N_STUDIES // 2)
    return np.column_stack([rng.permutation(base) for _ in TARGETS])


def _labeller_frame(truth, *, accuracy, coverage, confidence=0.9, seed=0):
    """Build a structured_labels frame that agrees with truth `accuracy` of the time."""
    rng = np.random.default_rng(seed)
    data = {"StudyInstanceUID": [f"uid-{i}" for i in range(truth.shape[0])]}
    data["is_gold"] = np.ones(truth.shape[0], dtype=bool)
    for j, target in enumerate(TARGETS):
        states, confs = [], []
        for i in range(truth.shape[0]):
            if rng.random() > coverage:
                states.append("unmentioned")
                confs.append(0.0)
                continue
            correct = rng.random() < accuracy
            positive = bool(truth[i, j] == 1)
            if not correct:
                positive = not positive
            states.append("positive" if positive else "negated")
            confs.append(confidence)
        data[f"{target}__state"] = states
        data[f"{target}__confidence"] = np.asarray(confs, dtype=np.float32)
    return pd.DataFrame(data)


def test_state_only_scores_map_the_four_states():
    frame = pd.DataFrame(
        {f"{target}__state": ["positive", "negated", "uncertain", "unmentioned"] for target in TARGETS}
    )
    scores = state_only_scores(frame)
    assert scores.shape == (4, len(TARGETS))
    np.testing.assert_allclose(scores[:, 0], [0.85, 0.05, 0.50, 0.50])


def test_state_only_scores_rejects_an_unknown_state():
    frame = pd.DataFrame({f"{target}__state": ["maybe"] for target in TARGETS})
    with pytest.raises(ValueError, match="unknown states"):
        state_only_scores(frame)


def test_confusion_summary_counts_only_confident_positive_or_negated_cells():
    # Two studies; only the first two targets carry expert labels, the rest are
    # NaN so they contribute nothing to either the numerator or the denominator.
    truth = np.full((2, len(TARGETS)), np.nan)
    truth[:, 0] = [1.0, 0.0]  # ACL
    truth[:, 1] = [0.0, 1.0]  # MCL

    states = pd.DataFrame({f"{target}__state": ["unmentioned", "unmentioned"] for target in TARGETS})
    states[f"{TARGETS[0]}__state"] = ["positive", "negated"]
    states[f"{TARGETS[1]}__state"] = ["uncertain", "positive"]

    confidences = np.zeros((2, len(TARGETS)))
    confidences[:, 0] = [0.9, 0.9]
    confidences[:, 1] = [0.6, 0.9]

    summary = confusion_summary(truth, states, confidences, min_confidence=0.75)
    # ACL: positive/gold=1 (TP) and negated/gold=0 (TN). MCL row 0 is a hedge so
    # it is dropped; MCL row 1 is a confident positive with gold=1 (TP).
    assert summary["usable_cells"] == 3
    assert summary["labelled_cells"] == 4
    assert summary["true_positive"] == 2
    assert summary["true_negative"] == 1
    assert summary["false_positive"] == 0
    assert summary["false_negative"] == 0
    assert summary["coverage"] == pytest.approx(0.75)


def test_state_truth_table_reports_every_state_and_a_pooled_row():
    truth = _truth()
    frame = _labeller_frame(truth, accuracy=0.8, coverage=0.7, seed=3)
    table = state_truth_table(truth, frame)
    assert set(table["state"]) == {"positive", "negated", "uncertain", "unmentioned"}
    assert "__pooled__" in set(table["target"])
    pooled = table.loc[table["target"] == "__pooled__"]
    assert len(pooled) == 4
    # Pooled counts must equal the sum of the per-target counts.
    per_target = table.loc[table["target"] != "__pooled__"]
    assert int(pooled["n"].sum()) == int(per_target["n"].sum())


def test_a_better_labeller_scores_higher_and_the_paired_bootstrap_agrees():
    truth = _truth()
    weak = _labeller_frame(truth, accuracy=0.62, coverage=0.40, seed=1)
    strong = _labeller_frame(truth, accuracy=0.90, coverage=0.90, seed=2)

    weak_scores = state_only_scores(weak)
    strong_scores = state_only_scores(strong)
    boot = paired_state_only_bootstrap(truth, weak_scores, strong_scores, n_bootstrap=400, seed=5)

    assert boot["median_difference"] > 0
    assert boot["probability_candidate_better"] > 0.9
    assert boot["valid_replicates"] > 0


def test_audit_labeller_end_to_end_writes_artifacts(tmp_path):
    truth = _truth()
    train = pd.DataFrame({"StudyInstanceUID": [f"uid-{i}" for i in range(truth.shape[0])]})
    train["Report"] = "report text"
    for j, target in enumerate(TARGETS):
        train[target] = truth[:, j]
    train_csv = tmp_path / "train.csv"
    train.to_csv(train_csv, index=False)

    candidate_csv = tmp_path / "candidate.csv"
    baseline_csv = tmp_path / "baseline.csv"
    _labeller_frame(truth, accuracy=0.92, coverage=0.90, seed=7).to_csv(candidate_csv, index=False)
    _labeller_frame(truth, accuracy=0.62, coverage=0.36, seed=8).to_csv(baseline_csv, index=False)

    payload = audit_labeller(
        train_csv,
        candidate_csv,
        baseline_structured=baseline_csv,
        out_root=tmp_path / "audit",
        n_bootstrap=200,
    )

    assert payload["n_gold_studies"] == truth.shape[0]
    assert payload["candidate"]["state_only_macro_auc"] > payload["baseline"]["state_only_macro_auc"]
    assert payload["candidate"]["confusion"]["coverage"] > payload["baseline"]["confusion"]["coverage"]
    assert (tmp_path / "audit" / "labeller_audit.json").is_file()
    assert (tmp_path / "audit" / "candidate_state_truth.csv").is_file()
    assert (tmp_path / "audit" / "baseline_state_truth.csv").is_file()

    text = format_audit(payload)
    assert "state-only macro AUC" in text
    assert "P(candidate better)" in text


def test_audit_labeller_rejects_an_export_missing_gold_studies(tmp_path):
    truth = _truth()
    train = pd.DataFrame({"StudyInstanceUID": [f"uid-{i}" for i in range(truth.shape[0])]})
    train["Report"] = "report text"
    for j, target in enumerate(TARGETS):
        train[target] = truth[:, j]
    train_csv = tmp_path / "train.csv"
    train.to_csv(train_csv, index=False)

    short = _labeller_frame(truth, accuracy=0.9, coverage=0.9, seed=9).iloc[:-3]
    short_csv = tmp_path / "short.csv"
    short.to_csv(short_csv, index=False)

    with pytest.raises(ValueError, match="absent from the labeller export"):
        audit_labeller(train_csv, short_csv, out_root=tmp_path / "audit")
