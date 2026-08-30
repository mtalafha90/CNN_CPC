"""The label audit measures the teacher, and says so honestly.

Two properties matter here. Report silence and hedging must lower coverage
rather than counting as errors, because "the radiologist did not write about it"
is not a wrong answer. And the audit must accept B23's export shape, which omits
the `__reason` column B6 writes -- the reason `b6_gold_audit` could not be used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.report_label_gold_audit import audit_export_against_gold

GOLD = ["gold-0", "gold-1", "gold-2", "gold-3"]


def _train_csv(tmp_path, truths):
    """train.csv with four expert-labelled studies and one report-only study."""
    rows = []
    for index, study in enumerate(GOLD):
        row = {"StudyInstanceUID": study, "Report": "knee mri"}
        row.update({target: float(truths[index]) for target in TARGETS})
        rows.append(row)
    weak = {"StudyInstanceUID": "weak-0", "Report": "knee mri"}
    weak.update({target: np.nan for target in TARGETS})
    rows.append(weak)

    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _structured(tmp_path, states, *, with_reason=False):
    """A B23-shaped structured_labels.csv, which has no __reason column."""
    fixed = {
        "positive": (0.97, 0.90),
        "negated": (0.03, 0.90),
        "uncertain": (0.50, 0.00),
        "unmentioned": (0.50, 0.00),
    }
    rows = []
    for index, study in enumerate(GOLD + ["weak-0"]):
        row = {"StudyInstanceUID": study}
        for target in TARGETS:
            state = states[index] if index < len(states) else "unmentioned"
            probability, confidence = fixed[state]
            row[target] = probability
            row[f"{target}__confidence"] = confidence
            row[f"{target}__state"] = state
            row[f"{target}__evidence"] = ""
            if with_reason:
                row[f"{target}__reason"] = ""
        rows.append(row)

    path = tmp_path / "structured_labels.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_a_perfect_export_scores_perfectly(tmp_path):
    train = _train_csv(tmp_path, [1, 1, 0, 0])
    structured = _structured(tmp_path, ["positive", "positive", "negated", "negated"])

    result = audit_export_against_gold(train, structured, label="perfect")
    assert result["macro"]["precision_positive"] == pytest.approx(1.0)
    assert result["macro"]["recall_sensitivity"] == pytest.approx(1.0)
    assert result["macro"]["specificity"] == pytest.approx(1.0)
    assert result["macro"]["coverage"] == pytest.approx(1.0)
    assert result["n_mismatched_cells"] == 0


def test_silence_lowers_coverage_and_is_not_an_error(tmp_path):
    """The single most important rule: a report not mentioning a finding is not wrong."""
    train = _train_csv(tmp_path, [1, 1, 0, 0])
    structured = _structured(tmp_path, ["positive", "unmentioned", "negated", "uncertain"])

    result = audit_export_against_gold(train, structured, label="quiet")
    assert result["macro"]["coverage"] == pytest.approx(0.5), "two of four cells were called"
    assert result["n_mismatched_cells"] == 0, "silence must never count as a mistake"
    assert result["macro"]["precision_positive"] == pytest.approx(1.0)


def test_a_disagreement_is_counted(tmp_path):
    train = _train_csv(tmp_path, [1, 0, 0, 0])
    structured = _structured(tmp_path, ["positive", "positive", "negated", "negated"])

    result = audit_export_against_gold(train, structured, label="wrong")
    # gold-1 is an expert negative the report called positive.
    assert result["macro"]["precision_positive"] == pytest.approx(0.5)
    assert result["n_mismatched_cells"] == len(TARGETS)


def test_a_b23_export_without_a_reason_column_is_accepted(tmp_path):
    """b6_gold_audit requires __reason; B23 never writes it. That is why this exists."""
    train = _train_csv(tmp_path, [1, 1, 0, 0])
    structured = _structured(tmp_path, ["positive", "positive", "negated", "negated"])

    frame = pd.read_csv(structured)
    assert not any(name.endswith("__reason") for name in frame.columns)
    result = audit_export_against_gold(train, structured, label="b23")
    assert result["pooled"]["n_usable_cells"] > 0


def test_training_targets_is_refused_with_an_explanation(tmp_path):
    """It excludes the expert studies by design, so it cannot answer this question."""
    train = _train_csv(tmp_path, [1, 1, 0, 0])
    structured = _structured(tmp_path, ["positive", "positive", "negated", "negated"])

    frame = pd.read_csv(structured)
    report_only = frame.loc[frame["StudyInstanceUID"] == "weak-0"]
    path = tmp_path / "training_targets.csv"
    report_only.to_csv(path, index=False)

    with pytest.raises(ValueError, match="structured_labels.csv"):
        audit_export_against_gold(train, path, label="wrong file")


def test_low_confidence_cells_are_not_scored(tmp_path):
    train = _train_csv(tmp_path, [1, 1, 0, 0])
    structured = _structured(tmp_path, ["positive", "positive", "negated", "negated"])

    frame = pd.read_csv(structured)
    frame[f"{TARGETS[0]}__confidence"] = 0.10
    frame.to_csv(structured, index=False)

    result = audit_export_against_gold(train, structured, label="hedged")
    assert result["per_target"][TARGETS[0]]["n_usable"] == 0
    assert result["per_target"][TARGETS[1]]["n_usable"] == 4


def test_the_result_carries_the_b6_reference(tmp_path):
    """The number B23 was built to beat, so the comparison needs no memory."""
    train = _train_csv(tmp_path, [1, 1, 0, 0])
    structured = _structured(tmp_path, ["positive", "positive", "negated", "negated"])

    result = audit_export_against_gold(train, structured, label="b23")
    assert result["b6_v121_reference"]["precision_positive"] == pytest.approx(0.6905)
    assert result["b6_v121_reference"]["coverage"] == pytest.approx(0.3606)


def test_the_written_output_records_the_mismatches(tmp_path):
    train = _train_csv(tmp_path, [1, 0, 0, 0])
    structured = _structured(tmp_path, ["positive", "positive", "negated", "negated"])
    out = tmp_path / "audit"

    audit_export_against_gold(train, structured, label="b23", out_root=out)
    assert (out / "gold_audit.json").is_file()
    mismatches = pd.read_csv(out / "mismatches.csv")
    assert set(mismatches["StudyInstanceUID"]) == {"gold-1"}
    assert set(mismatches.columns) >= {"expert", "report_label", "state", "confidence"}
