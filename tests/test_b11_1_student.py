import json

import pandas as pd
import pytest

from rsna_knee.b11_1_quantile_pseudo import B11_1_POLICY
from rsna_knee.b11_1_teacher_student import _load_pseudo_artifacts
from rsna_knee.b11_pseudo_labels import _sha256_file


def _write_artifacts(tmp_path, *, viable=True, gold_free=True):
    frame = pd.DataFrame({"StudyInstanceUID": ["study-a"], "dummy": [1.0]})
    csv_path = tmp_path / "pseudo_labels.csv"
    frame.to_csv(csv_path, index=False)
    policy = {
        "policy": B11_1_POLICY,
        "viability_passed": viable,
        "uses_gold_labels_to_choose_pseudo_cells": gold_free,
        "pseudo_labels_sha256": _sha256_file(csv_path),
    }
    (tmp_path / "pseudo_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    return csv_path


def test_b11_1_loader_accepts_frozen_viable_gold_free_artifacts(tmp_path):
    _write_artifacts(tmp_path)
    frame, policy = _load_pseudo_artifacts(tmp_path)
    assert frame["StudyInstanceUID"].tolist() == ["study-a"]
    assert policy["viability_passed"] is True
    assert policy["uses_gold_labels_to_choose_pseudo_cells"] is False


def test_b11_1_loader_rejects_failed_viability(tmp_path):
    _write_artifacts(tmp_path, viable=False)
    with pytest.raises(ValueError, match="viability"):
        _load_pseudo_artifacts(tmp_path)


def test_b11_1_loader_rejects_modified_pseudo_csv(tmp_path):
    csv_path = _write_artifacts(tmp_path)
    pd.DataFrame({"StudyInstanceUID": ["study-a"], "dummy": [2.0]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="SHA-256"):
        _load_pseudo_artifacts(tmp_path)
