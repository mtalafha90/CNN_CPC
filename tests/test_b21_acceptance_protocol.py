import json

import pytest

from rsna_knee.b21_acceptance_protocol import (
    B20_CANONICAL_GOLD_MACRO_AUC,
    promotion_decision,
    require_b20_replay_sanity,
    require_passed_weak_v2_gate,
    scientific_superiority_decision,
)


def _gate_payload():
    return {
        "variant": "b21_preresize_crop_weak_v2_paired_comparison_v1",
        "n_holdout_studies": 623,
        "crop_fraction": 0.9,
        "fixed_epoch": 2,
        "paired_candidate_minus_control": {
            "raw_difference_b_minus_a": 0.011136250028058625,
            "ci_lower": 0.00016240697238880116,
            "ci_upper": 0.022634658973656623,
        },
    }


def test_b21_weak_v2_gate_accepts_frozen_favorable_result(tmp_path):
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(_gate_payload()), encoding="utf-8")
    payload = require_passed_weak_v2_gate(path)
    assert payload["fixed_epoch"] == 2


def test_b21_weak_v2_gate_rejects_nonpositive_lower_bound(tmp_path):
    payload = _gate_payload()
    payload["paired_candidate_minus_control"]["ci_lower"] = 0.0
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        require_passed_weak_v2_gate(path)


def test_b21_promotion_rule_is_strictly_above_canonical_b20():
    assert promotion_decision(B20_CANONICAL_GOLD_MACRO_AUC + 1e-8)
    assert not promotion_decision(B20_CANONICAL_GOLD_MACRO_AUC)


def test_b21_scientific_superiority_requires_positive_paired_lower_bound():
    assert scientific_superiority_decision(1e-6)
    assert not scientific_superiority_decision(0.0)
    assert not scientific_superiority_decision(-1e-6)


def test_b20_replay_sanity_guard():
    assert require_b20_replay_sanity(B20_CANONICAL_GOLD_MACRO_AUC + 1e-4) == pytest.approx(1e-4)
    with pytest.raises(RuntimeError):
        require_b20_replay_sanity(B20_CANONICAL_GOLD_MACRO_AUC + 0.006)
