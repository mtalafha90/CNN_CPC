from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rsna_knee.b6_gold_audit import _binary_metrics, run_b6_gold_audit
from rsna_knee.constants import TARGETS
from rsna_knee.report_labels import STATE_NEGATED, STATE_POSITIVE, STATE_UNMENTIONED


def test_binary_metrics_counts_and_rates():
    result = _binary_metrics(
        np.asarray([1, 1, 0, 0], dtype=bool),
        np.asarray([1, 0, 1, 0], dtype=bool),
    )
    assert result["tp"] == 1
    assert result["tn"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["precision_positive"] == 0.5
    assert result["recall_sensitivity"] == 0.5
    assert result["specificity"] == 0.5
    assert result["negative_predictive_value"] == 0.5
    assert result["balanced_accuracy"] == 0.5


def test_gold_audit_uses_only_high_confidence_definite_cells(tmp_path):
    train_rows = []
    for i in range(4):
        row = {"StudyInstanceUID": f"study-{i}", "Report": f"report {i}"}
        for target in TARGETS:
            row[target] = float(i % 2 == 0)
        train_rows.append(row)
    train_csv = tmp_path / "train.csv"
    pd.DataFrame(train_rows).to_csv(train_csv, index=False)

    structured_rows = []
    for i in range(4):
        row = {"StudyInstanceUID": f"study-{i}"}
        for target in TARGETS:
            if i == 0:
                state, prob, conf = STATE_POSITIVE, 0.97, 0.90
            elif i == 1:
                state, prob, conf = STATE_NEGATED, 0.03, 0.90
            elif i == 2:
                state, prob, conf = STATE_NEGATED, 0.03, 0.90  # one false negative
            else:
                state, prob, conf = STATE_UNMENTIONED, 0.50, 0.0
            row[target] = prob
            row[f"{target}__confidence"] = conf
            row[f"{target}__state"] = state
            row[f"{target}__reason"] = "synthetic"
            row[f"{target}__evidence"] = "synthetic evidence"
        structured_rows.append(row)

    structured_csv = tmp_path / "structured_labels.csv"
    pd.DataFrame(structured_rows).to_csv(structured_csv, index=False)
    (tmp_path / "policy.json").write_text(json.dumps({"version": "1.2.1"}))

    out = tmp_path / "audit"
    payload = run_b6_gold_audit(
        train_csv,
        structured_csv,
        out_root=out,
        min_confidence=0.75,
    )

    acl = payload["targets"]["ACL"]
    assert payload["b6_version"] == "1.2.1"
    assert payload["n_gold_studies"] == 4
    assert acl["n_usable"] == 3
    assert acl["tp"] == 1
    assert acl["tn"] == 1
    assert acl["fp"] == 0
    assert acl["fn"] == 1
    assert acl["precision_positive"] == 1.0
    assert acl["recall_sensitivity"] == 0.5
    assert acl["specificity"] == 1.0
    assert acl["negative_predictive_value"] == 0.5
    assert payload["parser_change_after_this_audit_allowed"] is False

    mismatches = pd.read_csv(out / "gold_mismatches.csv")
    assert len(mismatches) == len(TARGETS)
    assert set(mismatches["StudyInstanceUID"]) == {"study-2"}
    assert (out / "gold_audit.json").exists()
    assert (out / "gold_usable_cells.csv").exists()
