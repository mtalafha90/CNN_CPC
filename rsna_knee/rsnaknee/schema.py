"""Discover the competition's data schema at run time.

Kaggle competitions vary in how they lay out labels and submissions, and the
RSNA challenge data is not readable from this development environment. Rather
than hard-coding column names that might be wrong, the pipeline inspects the
CSV files that ship with the data and works out:

* which column identifies an exam (study);
* which columns are the binary targets;
* whether the submission is *wide* (one row per exam, one column per finding)
  or *long* (one row per exam-and-finding pair).

Everything downstream consumes the resulting :class:`DataSchema`, so a change
in the organisers' naming never requires a code change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import get_logger

LOGGER = get_logger()

# Column names that identify an exam, in order of preference.
ID_CANDIDATES = (
    "study_instance_uid",
    "studyinstanceuid",
    "exam_id",
    "study_id",
    "examid",
    "studyuid",
    "patient_study",
    "id",
)

# Columns that are never targets even when numeric.
NON_TARGET_HINTS = (
    "fold",
    "split",
    "site",
    "institution",
    "language",
    "lang",
    "age",
    "sex",
    "gender",
    "laterality",
    "side",
    "weight",
    "sample_weight",
    "patient",
    "series",
    "instance",
    "path",
    "report",
    "text",
    "impression",
    "findings_text",
)

# Column names used for the value in a long-format submission.
VALUE_CANDIDATES = ("prediction", "target", "probability", "value", "label", "score")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _pick_id_column(columns: list[str]) -> str:
    """Choose the exam identifier column from a list of column names."""
    normalised = {_norm(c): c for c in columns}
    for candidate in ID_CANDIDATES:
        if candidate in normalised:
            return normalised[candidate]
    # Fall back to the first column that looks like an identifier.
    for column in columns:
        norm = _norm(column)
        if norm.endswith("id") or "uid" in norm:
            return column
    return columns[0]


def _looks_binary(series: pd.Series) -> bool:
    """True when a column holds only 0/1 style values (missing allowed)."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = pd.unique(series.dropna())
    if values.size == 0 or values.size > 3:
        return False
    return bool(np.all(np.isin(values, [0, 1, -1])))


@dataclass
class DataSchema:
    """Everything the pipeline needs to know about the label layout."""

    id_column: str
    labels: list[str]
    submission_format: str = "wide"  # "wide" or "long"
    row_id_column: str | None = None
    value_column: str | None = None
    row_id_template: str | None = None
    group_column: str | None = None
    extra_columns: list[str] = field(default_factory=list)

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict:
        return {
            "id_column": self.id_column,
            "labels": self.labels,
            "submission_format": self.submission_format,
            "row_id_column": self.row_id_column,
            "value_column": self.value_column,
            "row_id_template": self.row_id_template,
            "group_column": self.group_column,
            "extra_columns": self.extra_columns,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DataSchema":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


def infer_row_id_template(row_ids: pd.Series, labels: list[str]) -> str | None:
    """Work out how a long-format row id joins an exam id to a label.

    Returns a format string such as ``"{id}_{label}"``. Returns ``None`` when
    no separator reproduces the observed row ids.
    """
    sample = str(row_ids.iloc[0])
    for label in sorted(labels, key=len, reverse=True):
        if not sample.endswith(label):
            continue
        prefix = sample[: -len(label)]
        for separator in ("_", "-", "__", "/", " "):
            if prefix.endswith(separator):
                return "{id}" + separator + "{label}"
        return "{id}{label}"
    return None


def discover_schema(
    data_dir: str | Path,
    train_csv: str | None = None,
    sample_submission_csv: str | None = None,
) -> DataSchema:
    """Inspect the competition CSVs and return the inferred schema.

    Parameters
    ----------
    data_dir:
        Directory holding the competition CSV files.
    train_csv, sample_submission_csv:
        Optional explicit paths. When omitted the usual Kaggle names are tried.
    """
    data_dir = Path(data_dir)
    train_path = Path(train_csv) if train_csv else _find_first(
        data_dir, ["train.csv", "train_labels.csv", "labels.csv", "train_metadata.csv"]
    )
    if train_path is None:
        raise FileNotFoundError(
            f"No training label CSV found under {data_dir}. Pass train_csv explicitly."
        )
    train = pd.read_csv(train_path, nrows=20_000)
    id_column = _pick_id_column(list(train.columns))

    labels = [
        column
        for column in train.columns
        if column != id_column
        and not any(hint in _norm(column) for hint in NON_TARGET_HINTS)
        and _looks_binary(train[column])
    ]

    sub_path = Path(sample_submission_csv) if sample_submission_csv else _find_first(
        data_dir, ["sample_submission.csv", "sample_solution.csv"]
    )

    submission_format = "wide"
    row_id_column: str | None = None
    value_column: str | None = None
    row_id_template: str | None = None

    if sub_path is not None:
        sub = pd.read_csv(sub_path, nrows=5_000)
        sub_columns = list(sub.columns)
        value_matches = [c for c in sub_columns if _norm(c) in VALUE_CANDIDATES]
        if len(sub_columns) == 2 and value_matches:
            submission_format = "long"
            value_column = value_matches[0]
            row_id_column = next(c for c in sub_columns if c != value_column)
            if labels:
                row_id_template = infer_row_id_template(sub[row_id_column], labels)
            else:
                labels = _labels_from_row_ids(sub[row_id_column])
                row_id_template = infer_row_id_template(sub[row_id_column], labels)
        else:
            submission_format = "wide"
            row_id_column = _pick_id_column(sub_columns)
            sub_labels = [c for c in sub_columns if c != row_id_column]
            if sub_labels:
                # The submission is authoritative about label order.
                labels = sub_labels

    if not labels:
        raise ValueError(
            "Could not infer any label columns. Inspect the CSVs and pass a schema "
            "explicitly via --schema-json."
        )

    schema = DataSchema(
        id_column=id_column,
        labels=labels,
        submission_format=submission_format,
        row_id_column=row_id_column,
        value_column=value_column,
        row_id_template=row_id_template,
        group_column=_find_group_column(train, id_column),
    )
    LOGGER.info(
        "Discovered schema: id=%s, %d labels, %s submission",
        schema.id_column,
        schema.num_labels,
        schema.submission_format,
    )
    return schema


def _labels_from_row_ids(row_ids: pd.Series) -> list[str]:
    """Recover label names from long-format row ids by finding a shared suffix set."""
    seen: list[str] = []
    for raw in row_ids.astype(str):
        for separator in ("_", "-"):
            if separator in raw:
                suffix = raw.rsplit(separator, 1)[1]
                if suffix not in seen:
                    seen.append(suffix)
                break
    return seen


def _find_group_column(frame: pd.DataFrame, id_column: str) -> str | None:
    """Find a patient-level column so folds never split one patient across folds."""
    for column in frame.columns:
        norm = _norm(column)
        if column != id_column and ("patient" in norm or norm.endswith("patientid")):
            return column
    return None


def _find_first(directory: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    # Recursive search as a fallback, useful when the data sits one level deeper.
    for name in names:
        matches = sorted(directory.rglob(name))
        if matches:
            return matches[0]
    return None


def write_submission(
    predictions: np.ndarray,
    exam_ids: list[str],
    schema: DataSchema,
    output_path: str | Path,
    sample_submission_path: str | Path | None = None,
) -> pd.DataFrame:
    """Write predictions in exactly the layout the sample submission uses."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if schema.submission_format == "long":
        template = schema.row_id_template or "{id}_{label}"
        rows = {
            schema.row_id_column or "row_id": [
                template.format(id=exam, label=label)
                for exam in exam_ids
                for label in schema.labels
            ],
            schema.value_column or "prediction": predictions.reshape(-1).tolist(),
        }
        frame = pd.DataFrame(rows)
    else:
        frame = pd.DataFrame(predictions, columns=schema.labels)
        frame.insert(0, schema.row_id_column or schema.id_column, exam_ids)

    if sample_submission_path is not None and Path(sample_submission_path).exists():
        sample = pd.read_csv(sample_submission_path)
        key = frame.columns[0]
        if key in sample.columns:
            # Re-order to the sample's row order; the grader is usually tolerant
            # but matching exactly avoids any doubt.
            frame = (
                sample[[key]]
                .merge(frame, on=key, how="left")
                .fillna(0.5)
            )
        frame = frame[[c for c in sample.columns if c in frame.columns]]

    frame.to_csv(output_path, index=False)
    LOGGER.info("Wrote submission with %d rows to %s", len(frame), output_path)
    return frame
