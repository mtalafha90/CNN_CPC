import json

import pytest

from rsna_knee.b21_acceptance_protocol import B21_GOLD_ACCEPTANCE_VARIANT
from rsna_knee.b22_duration_contract import require_b22_duration_contract
from rsna_knee.b22_duration_protocol import (
    require_b22_e2_replay,
    require_failed_b21_acceptance,
)


def _acceptance_payload():
    return {
        "variant": B21_GOLD_ACCEPTANCE_VARIANT,
        "one_gold_look_consumed": True,
        "promotion_rule_passed": False,
        "b21_candidate": {"macro_auc": 0.6573196516459231},
    }


def test_b22_requires_failed_b21_acceptance(tmp_path):
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(_acceptance_payload()), encoding="utf-8")
    payload = require_failed_b21_acceptance(path)
    assert payload["promotion_rule_passed"] is False


def test_b22_rejects_promoted_b21(tmp_path):
    payload = _acceptance_payload()
    payload["promotion_rule_passed"] = True
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        require_failed_b21_acceptance(path)


def test_b22_e2_replay_guard_accepts_small_delta():
    assert require_b22_e2_replay(0.659, 0.6573) == pytest.approx(0.0017)


def test_b22_e2_replay_guard_rejects_large_delta():
    with pytest.raises(RuntimeError):
        require_b22_e2_replay(0.665, 0.6573)


def test_b22_contract_freezes_five_epochs():
    assert require_b22_duration_contract({"b7_epochs": 5, "b22_scheduler_horizon": 5}) == pytest.approx(0.90)
    with pytest.raises(ValueError):
        require_b22_duration_contract({"b7_epochs": 4, "b22_scheduler_horizon": 5})
