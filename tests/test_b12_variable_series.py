import json

import pandas as pd
import pytest
import torch

from rsna_knee.b12_training import _load_series_policy
from rsna_knee.b12_variable_series import (
    B12_SERIES_POLICY,
    audit_variable_series_surface,
    build_variable_series_index,
    collate_variable_series,
)


def _series_frame(n_studies=12):
    rows = []
    for i in range(n_studies):
        study = f"study-{i:03d}"
        # Three real acquisitions with the same plane. Historical dual routing
        # can retain only two; B12 must retain all three.
        rows.extend(
            [
                {
                    "StudyInstanceUID": study,
                    "SeriesInstanceUID": f"{study}-a",
                    "Anatomical_Plane": "Sagittal",
                    "Fluid_Sensitive": True,
                    "Fat_Suppression": True,
                },
                {
                    "StudyInstanceUID": study,
                    "SeriesInstanceUID": f"{study}-b",
                    "Anatomical_Plane": "Sagittal",
                    "Fluid_Sensitive": False,
                    "Fat_Suppression": False,
                },
                {
                    "StudyInstanceUID": study,
                    "SeriesInstanceUID": f"{study}-c",
                    "Anatomical_Plane": "Sagittal",
                    "Fluid_Sensitive": True,
                    "Fat_Suppression": True,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_variable_series_keeps_repeated_acquisitions():
    frame = _series_frame(1)
    index = build_variable_series_index(frame, ["study-000"])
    records = index["study-000"]
    assert len(records) == 3
    assert {r["series_uid"] for r in records} == {
        "study-000-a",
        "study-000-b",
        "study-000-c",
    }
    # a and c have identical coarse metadata but remain separate real series.
    a = next(r for r in records if r["series_uid"].endswith("-a"))
    c = next(r for r in records if r["series_uid"].endswith("-c"))
    assert (a["plane_id"], a["fluid_id"], a["fat_id"]) == (
        c["plane_id"], c["fluid_id"], c["fat_id"]
    )


def test_b12_label_free_series_audit_detects_extra_information():
    frame = _series_frame(12)
    studies = [f"study-{i:03d}" for i in range(12)]
    summary, _ = audit_variable_series_surface(frame, studies)
    assert summary["viability_passed"] is True
    assert summary["eligible_recognized_plane_series"] == 36
    assert summary["historical_dual_unique_series"] == 24
    assert summary["extra_series_retained"] == 12
    assert summary["studies_with_extra_series"] == 12
    assert summary["studies_with_zero_eligible_series"] == 0
    assert summary["historical_selected_series_missing_from_b12"] == 0


def test_collate_variable_series_pads_only_to_batch_max():
    s, c, h, w = 2, 3, 4, 4
    batch = [
        {
            "study_uid": "a",
            "volumes": torch.ones(2, s, c, h, w),
            "present": torch.ones(2),
            "series_meta": torch.tensor([[1, 2, 2], [2, 1, 1]]),
            "target": torch.zeros(12),
            "weight": torch.ones(12),
        },
        {
            "study_uid": "b",
            "volumes": torch.ones(5, s, c, h, w),
            "present": torch.ones(5),
            "series_meta": torch.tensor([[3, 2, 1]] * 5),
            "target": torch.ones(12),
            "weight": torch.ones(12),
        },
    ]
    out = collate_variable_series(batch)
    assert out["volumes"].shape == (2, 5, s, c, h, w)
    assert out["present"].shape == (2, 5)
    assert out["series_meta"].shape == (2, 5, 3)
    assert out["present"][0].tolist() == [1, 1, 0, 0, 0]
    assert out["study_uid"] == ["a", "b"]


def test_collate_variable_series_preserves_tta_axis():
    v, s, c, h, w = 3, 2, 3, 4, 4
    batch = [
        {
            "study_uid": "a",
            "volumes": torch.ones(v, 2, s, c, h, w),
            "present": torch.ones(2),
            "series_meta": torch.tensor([[1, 2, 2], [2, 1, 1]]),
        },
        {
            "study_uid": "b",
            "volumes": torch.ones(v, 4, s, c, h, w),
            "present": torch.ones(4),
            "series_meta": torch.tensor([[3, 2, 1]] * 4),
        },
    ]
    out = collate_variable_series(batch)
    assert out["volumes"].shape == (2, v, 4, s, c, h, w)
    assert out["present"].shape == (2, 4)


def _write_policy(tmp_path, *, viable=True, gold_free=True):
    payload = {
        "policy": B12_SERIES_POLICY,
        "uses_gold_labels": not gold_free,
        "viability_passed": viable,
        "b6_active_studies": 3120,
        "b6_usable_cells": 14123,
    }
    path = tmp_path / "series_policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_b12_loader_accepts_viable_gold_free_policy(tmp_path):
    policy = _load_series_policy(_write_policy(tmp_path))
    assert policy["uses_gold_labels"] is False
    assert policy["viability_passed"] is True


def test_b12_loader_rejects_gold_informed_policy(tmp_path):
    with pytest.raises(ValueError, match="label-free"):
        _load_series_policy(_write_policy(tmp_path, gold_free=False))


def test_b12_loader_rejects_failed_viability(tmp_path):
    with pytest.raises(ValueError, match="viability"):
        _load_series_policy(_write_policy(tmp_path, viable=False))
