"""Tests for the post-run B37 visual diagnostic report."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tools.plot_b37_results import (
    B37ReportError,
    PREDICTION_FILES,
    STUDY_UID,
    TARGETS,
    generate_b37_report,
)


def _history_row(epoch: int, gate: float) -> dict:
    return {
        "epoch": epoch,
        "loss_total": 1.5 - epoch * 0.1,
        "loss_combined": 0.8 - epoch * 0.05,
        "loss_local_aux": 0.7 - epoch * 0.05,
        "gate": {"gate_effective": [gate] * len(TARGETS)},
    }


def _make_fixture(tmp_path):
    run_root = tmp_path / "b37_highres_sparse_mil"
    evaluation_root = run_root / "expert58"
    data_root = tmp_path / "data"
    evaluation_root.mkdir(parents=True)
    data_root.mkdir()
    (run_root / "history.json").write_text(
        json.dumps([_history_row(1, 0.01), _history_row(2, 0.03)]),
        encoding="utf-8",
    )

    uids = ["study-a", "study-b", "study-c", "study-d"]
    truth = pd.DataFrame({STUDY_UID: uids})
    for index, target in enumerate(TARGETS):
        truth[target] = [(row + index) % 2 for row in range(len(uids))]
    truth.to_csv(data_root / "train.csv", index=False)

    prediction = pd.DataFrame({STUDY_UID: uids})
    for index, target in enumerate(TARGETS):
        labels = truth[target].to_numpy(dtype=float)
        prediction[target] = np.clip(0.15 + labels * 0.70 + index * 0.002, 0.0, 1.0)
    for file_name in PREDICTION_FILES.values():
        prediction.to_csv(evaluation_root / file_name, index=False)

    per_target = {
        target: {
            "base_224_auc": 0.55,
            "b37_global_448_auc": 0.60,
            "b37_combined_auc": 0.65,
        }
        for target in TARGETS
    }
    (evaluation_root / "expert58.json").write_text(
        json.dumps(
            {
                "evaluation_role": "reused expert development diagnostic",
                "per_target": per_target,
                "base_224_macro_auc": 0.55,
                "b37_global_448_macro_auc": 0.60,
                "b37_combined_macro_auc": 0.65,
                "macro_delta_primary": 0.10,
            }
        ),
        encoding="utf-8",
    )
    return run_root, data_root, evaluation_root


def test_generate_b37_report_writes_training_and_expert_diagnostics(tmp_path):
    run_root, data_root, evaluation_root = _make_fixture(tmp_path)

    outputs = generate_b37_report(
        run_root=run_root,
        data_root=data_root,
        evaluation_root=evaluation_root,
        threshold=0.50,
        n_examples=3,
    )

    assert all(path.is_file() for path in outputs.values())
    matrix = pd.read_csv(outputs["confusion_table"])
    assert len(matrix) == len(TARGETS)
    assert set(matrix["model"]) == {"b37_combined"}
    assert (matrix["true_positive"] == 2).all()
    case_table = pd.read_csv(outputs["case_table"])
    assert len(case_table) == 3 * len(TARGETS)
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["n_expert_studies"] == 4
    assert summary["classification_threshold"] == 0.50


def test_generate_b37_report_refuses_an_invalid_threshold(tmp_path):
    run_root, data_root, evaluation_root = _make_fixture(tmp_path)

    with pytest.raises(B37ReportError, match="threshold"):
        generate_b37_report(
            run_root=run_root,
            data_root=data_root,
            evaluation_root=evaluation_root,
            threshold=1.0,
        )
