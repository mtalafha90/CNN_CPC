import json

import pytest

from rsna_knee.b15_gold_eval import load_passing_b15_gate
from rsna_knee.b15_ssl import WEAK_V2_MANIFEST_SHA256, WEAK_V2_SURFACE


def _passing_gate():
    return {
        "surface": WEAK_V2_SURFACE,
        "macro_auc_a": 0.5652498117985745,
        "macro_auc_b": 0.7319060415162949,
        "raw_difference_b_minus_a": 0.16665622971772043,
        "median_difference": 0.16752458387082447,
        "ci_lower": 0.11244332081629649,
        "ci_upper": 0.2165156305365904,
        "probability_b_better": 1.0,
        "n_bootstrap": 5000,
        "n_valid_replicates": 4921,
        "valid_replicate_fraction": 0.9842,
        "strict_all_12_targets": True,
        "comparison": "B15 minus B13-v2-control",
        "model_a": "B13-v2-control",
        "model_b": "B15",
        "weak_holdout_manifest_sha256": WEAK_V2_MANIFEST_SHA256,
        "predeclared_gate": {
            "raw_difference_positive": True,
            "median_difference_positive": True,
            "probability_b_better_at_least_0_95": True,
        },
        "passes_gate": True,
    }


def test_load_passing_b15_gate_accepts_exact_frozen_contract(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(_passing_gate()), encoding="utf-8")
    payload = load_passing_b15_gate(path)
    assert payload["passes_gate"] is True
    assert payload["probability_b_better"] == 1.0


def test_load_passing_b15_gate_rejects_failed_gate(tmp_path):
    payload = _passing_gate()
    payload["passes_gate"] = False
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        load_passing_b15_gate(path)


def test_load_passing_b15_gate_rejects_manifest_change(tmp_path):
    payload = _passing_gate()
    payload["weak_holdout_manifest_sha256"] = "wrong"
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA"):
        load_passing_b15_gate(path)
