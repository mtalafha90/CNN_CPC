"""Tests for run-time schema discovery and submission writing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsnaknee.schema import (  # noqa: E402
    DataSchema,
    discover_schema,
    infer_row_id_template,
    write_submission,
)

LABELS = ["acl_tear", "meniscus_tear", "effusion"]


def _write_wide(directory: Path) -> None:
    pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b", "c"],
            "PatientID": ["p1", "p1", "p2"],
            "acl_tear": [0, 1, 0],
            "meniscus_tear": [1, 0, 0],
            "effusion": [0, 0, 1],
        }
    ).to_csv(directory / "train.csv", index=False)


def test_discovers_wide_schema(tmp_path: Path) -> None:
    _write_wide(tmp_path)
    pd.DataFrame(
        {"StudyInstanceUID": ["x"], "acl_tear": [0.5], "meniscus_tear": [0.5], "effusion": [0.5]}
    ).to_csv(tmp_path / "sample_submission.csv", index=False)

    schema = discover_schema(tmp_path)

    assert schema.id_column == "StudyInstanceUID"
    assert schema.labels == LABELS
    assert schema.submission_format == "wide"
    # The patient column must be found so folds never split a patient.
    assert schema.group_column == "PatientID"


def test_discovers_long_schema(tmp_path: Path) -> None:
    _write_wide(tmp_path)
    pd.DataFrame(
        {
            "row_id": [f"x_{label}" for label in LABELS],
            "prediction": [0.5, 0.5, 0.5],
        }
    ).to_csv(tmp_path / "sample_submission.csv", index=False)

    schema = discover_schema(tmp_path)

    assert schema.submission_format == "long"
    assert schema.row_id_column == "row_id"
    assert schema.value_column == "prediction"
    assert schema.row_id_template == "{id}_{label}"


def test_patient_and_metadata_columns_are_not_labels(tmp_path: Path) -> None:
    """Binary-looking metadata such as sex must not be mistaken for a target."""
    pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b"],
            "sex": [0, 1],
            "fold": [0, 1],
            "acl_tear": [0, 1],
        }
    ).to_csv(tmp_path / "train.csv", index=False)

    schema = discover_schema(tmp_path)

    assert schema.labels == ["acl_tear"]


def test_infer_row_id_template_handles_separators() -> None:
    row_ids = pd.Series(["study1-acl_tear"])
    assert infer_row_id_template(row_ids, ["acl_tear"]) == "{id}-{label}"


def test_write_submission_long_format(tmp_path: Path) -> None:
    schema = DataSchema(
        id_column="StudyInstanceUID",
        labels=LABELS,
        submission_format="long",
        row_id_column="row_id",
        value_column="prediction",
        row_id_template="{id}_{label}",
    )
    predictions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    frame = write_submission(predictions, ["a", "b"], schema, tmp_path / "sub.csv")

    assert list(frame.columns) == ["row_id", "prediction"]
    assert len(frame) == 6
    assert frame.loc[0, "row_id"] == "a_acl_tear"
    assert frame.loc[5, "prediction"] == pytest.approx(0.6)


def test_write_submission_matches_sample_row_order(tmp_path: Path) -> None:
    """Rows must come back in the sample submission's order, not ours."""
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame(
        {"StudyInstanceUID": ["b", "a"], "acl_tear": [0.5, 0.5], "meniscus_tear": [0.5, 0.5],
         "effusion": [0.5, 0.5]}
    ).to_csv(sample_path, index=False)

    schema = DataSchema(id_column="StudyInstanceUID", labels=LABELS, submission_format="wide")
    predictions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    frame = write_submission(
        predictions, ["a", "b"], schema, tmp_path / "sub.csv", sample_path
    )

    assert frame["StudyInstanceUID"].tolist() == ["b", "a"]
    assert frame.loc[0, "acl_tear"] == pytest.approx(0.4)
