"""The confidence probe must be able to return "this lever does not exist".

A probe that only ever reports a usable signal is worthless. These tests pin
both outcomes: a confidence that genuinely ranks correct labels above wrong ones
is called informative, and one that is pure noise is called uninformative.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from rsna_knee.constants import TARGETS  # noqa: E402

from model_confidence_probe import probe  # noqa: E402

STUDIES = [f"gold-{index}" for index in range(8)]


def _train_csv(tmp_path, truth_for):
    rows = []
    for study in STUDIES:
        row = {"StudyInstanceUID": study, "Report": "knee mri"}
        row.update({target: float(truth_for(study, target)) for target in TARGETS})
        rows.append(row)
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _structured(tmp_path, state_for, model_confidence_for):
    rows = []
    for study in STUDIES:
        row = {"StudyInstanceUID": study}
        for target in TARGETS:
            state = state_for(study, target)
            row[target] = 0.97 if state == "positive" else 0.03
            row[f"{target}__confidence"] = 0.90
            row[f"{target}__model_confidence"] = model_confidence_for(study, target)
            row[f"{target}__state"] = state
        rows.append(row)
    path = tmp_path / "structured_labels.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _wrong_on_odd(study, target):
    """Half the labels disagree with the expert, deterministically."""
    return int(study[-1]) % 2 == 1


def test_a_confidence_that_knows_is_called_informative(tmp_path):
    train = _train_csv(tmp_path, lambda s, t: 1)
    # Says "negated" (wrong) on odd studies, and is honestly unsure when wrong.
    structured = _structured(
        tmp_path,
        lambda s, t: "negated" if _wrong_on_odd(s, t) else "positive",
        lambda s, t: 0.30 if _wrong_on_odd(s, t) else 0.95,
    )
    result = probe(train, structured)

    assert result["confidence_auc_for_correctness"] == pytest.approx(1.0)
    assert result["verdict"] == "informative"
    # Thresholding away the low-confidence cells must remove every mistake.
    strict = [row for row in result["sweep"] if row["threshold"] >= 0.9][0]
    assert strict["accuracy"] == pytest.approx(1.0)


def test_a_confidence_that_does_not_know_is_called_uninformative(tmp_path):
    train = _train_csv(tmp_path, lambda s, t: 1)
    # Equally confident whether right or wrong: the lever does not exist.
    structured = _structured(
        tmp_path,
        lambda s, t: "negated" if _wrong_on_odd(s, t) else "positive",
        lambda s, t: 0.90,
    )
    result = probe(train, structured)

    assert result["confidence_auc_for_correctness"] is None or result[
        "confidence_auc_for_correctness"
    ] == pytest.approx(0.5)
    assert result["verdict"] in {"uninformative", "undefined"}


def test_the_baseline_precision_is_reported(tmp_path):
    train = _train_csv(tmp_path, lambda s, t: 1)
    structured = _structured(
        tmp_path,
        lambda s, t: "negated" if _wrong_on_odd(s, t) else "positive",
        lambda s, t: 0.9,
    )
    result = probe(train, structured)
    # Every positive call is correct here; the negated ones are the errors.
    assert result["baseline_precision_positive"] == pytest.approx(1.0)
    assert result["baseline_accuracy"] == pytest.approx(0.5)


def test_an_export_without_model_confidence_is_refused(tmp_path):
    """B6 does not record it, and a silent wrong answer would be worse."""
    train = _train_csv(tmp_path, lambda s, t: 1)
    structured = _structured(tmp_path, lambda s, t: "positive", lambda s, t: 0.9)
    frame = pd.read_csv(structured)
    frame = frame.drop(columns=[c for c in frame.columns if c.endswith("__model_confidence")])
    frame.to_csv(structured, index=False)

    with pytest.raises(ValueError, match="no model_confidence columns"):
        probe(train, structured)
