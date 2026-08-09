from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from rsna_knee.b7_weak_supervision import (
    B7_NEGATIVE_TARGET,
    B7_NEGATIVE_WEIGHT,
    B7_POSITIVE_TARGET,
    B7_POSITIVE_WEIGHT,
    load_b5_encoder_payload,
    load_frozen_b6_export,
    prepare_b7_supervision,
    target_balance_multipliers,
    target_balanced_weak_bce,
)
from rsna_knee.constants import TARGETS
from rsna_knee.report_ssl import B5_VARIANT
from rsna_knee.ssl import SSL_SOURCE


def _train_frame():
    rows = []
    for uid in ("weak-a", "weak-b", "gold"):
        row = {"StudyInstanceUID": uid, "Report": "report"}
        for target in TARGETS:
            row[target] = np.nan
        if uid == "gold":
            for target in TARGETS:
                row[target] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _b6_frame():
    rows = []
    for uid in ("weak-a", "weak-b"):
        row = {"StudyInstanceUID": uid}
        for target in TARGETS:
            row[target] = 0.5
            row[f"{target}__confidence"] = 0.0
            row[f"{target}__state"] = "unmentioned"
        rows.append(row)
    rows[0]["ACL__state"] = "positive"
    rows[0]["ACL__confidence"] = 0.9
    rows[1]["ACL__state"] = "negated"
    rows[1]["ACL__confidence"] = 0.9
    rows[1]["MCL__state"] = "uncertain"
    rows[1]["MCL__confidence"] = 0.25
    return pd.DataFrame(rows)


def test_prepare_b7_supervision_uses_frozen_asymmetric_policy_and_excludes_gold():
    uids, target, weight, summary = prepare_b7_supervision(_train_frame(), _b6_frame())
    assert uids == ["weak-a", "weak-b"]
    acl = TARGETS.index("ACL")
    assert target[0, acl] == pytest.approx(B7_POSITIVE_TARGET)
    assert weight[0, acl] == pytest.approx(B7_POSITIVE_WEIGHT)
    assert target[1, acl] == pytest.approx(B7_NEGATIVE_TARGET)
    assert weight[1, acl] == pytest.approx(B7_NEGATIVE_WEIGHT)
    assert summary["usable_cells"] == 2
    assert summary["positive_cells"] == 1
    assert summary["negative_cells"] == 1


def test_prepare_b7_supervision_rejects_gold_uid_leakage():
    b6 = _b6_frame()
    leaked = b6.iloc[[0]].copy()
    leaked["StudyInstanceUID"] = "gold"
    b6 = pd.concat([b6, leaked], ignore_index=True)
    with pytest.raises(ValueError, match="gold"):
        prepare_b7_supervision(_train_frame(), b6)


def test_target_balance_equalizes_total_supervision_mass():
    weight = np.zeros((4, len(TARGETS)), dtype=np.float32)
    weight[:, 0] = [0.5, 0.5, 0.0, 0.0]
    weight[:, 1] = [1.0, 1.0, 0.0, 0.0]
    for j in range(2, len(TARGETS)):
        weight[0, j] = 1.0
    multiplier = target_balance_multipliers(weight)
    adjusted = (weight * multiplier[None, :]).sum(axis=0)
    assert np.allclose(adjusted, adjusted[0], rtol=1e-6, atol=1e-6)


def test_target_balanced_weak_bce_ignores_zero_weight_cells():
    logits = torch.zeros((2, len(TARGETS)), dtype=torch.float32)
    target = torch.full_like(logits, 0.5)
    weight = torch.zeros_like(logits)
    target[0, 0] = B7_POSITIVE_TARGET
    weight[0, 0] = B7_POSITIVE_WEIGHT
    multiplier = torch.ones(len(TARGETS))
    first = target_balanced_weak_bce(logits, target, weight, multiplier)

    target[1, 1] = 0.99
    second = target_balanced_weak_bce(logits, target, weight, multiplier)
    assert first.item() == pytest.approx(second.item())


def test_load_b5_encoder_payload_enforces_b5_contract(tmp_path):
    good = tmp_path / "good.pt"
    torch.save(
        {
            "source": SSL_SOURCE,
            "variant": B5_VARIANT,
            "gold_studies_used": 0,
            "external_image_pretraining": False,
            "encoder": {},
        },
        good,
    )
    payload = load_b5_encoder_payload(good)
    assert payload["variant"] == B5_VARIANT

    bad = tmp_path / "bad.pt"
    torch.save(
        {
            "source": SSL_SOURCE,
            "variant": "not_b5",
            "gold_studies_used": 0,
            "external_image_pretraining": False,
            "encoder": {},
        },
        bad,
    )
    with pytest.raises(ValueError, match="variant"):
        load_b5_encoder_payload(bad)


def test_load_frozen_b6_export_requires_v121_and_zero_gold(tmp_path):
    root = tmp_path / "b6"
    root.mkdir()
    _b6_frame().to_csv(root / "training_targets.csv", index=False)
    (root / "policy.json").write_text(
        json.dumps({"version": "1.2.1", "gold_usage": "audit only"}),
        encoding="utf-8",
    )
    (root / "audit.json").write_text(
        json.dumps(
            {
                "b6_version": "1.2.1",
                "gold_rows_in_training_targets": 0,
                "min_confidence_for_usable_cell": 0.75,
            }
        ),
        encoding="utf-8",
    )
    frame, _, audit = load_frozen_b6_export(root)
    assert len(frame) == 2
    assert audit["b6_version"] == "1.2.1"
