import numpy as np
import pandas as pd
import pytest
import torch

from rsna_knee.b16_gold_eval import _read_reference_predictions
from rsna_knee.b16_report_ssl import (
    B16_REPORT_SSL_OBJECTIVE,
    B16_REPORT_SSL_VARIANT,
    _require_b16_report_contract,
    b16_report_study_pool,
    load_b16_report_encoder,
)
from rsna_knee.b15_ssl import B15_SSL_VARIANT
from rsna_knee.constants import TARGETS


def _competition_frame():
    rows = []
    for i in range(58):
        row = {"StudyInstanceUID": f"G{i:03d}", "Report": "gold report"}
        row.update({target: float(i % 2) for target in TARGETS})
        rows.append(row)
    for i in range(4349):
        row = {"StudyInstanceUID": f"N{i:04d}", "Report": f"report {i}"}
        row.update({target: np.nan for target in TARGETS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_b16_report_pool_is_exact_all_non_gold_surface():
    frame = _competition_frame()
    uids, stats = b16_report_study_pool(frame)
    assert len(uids) == 4349
    assert stats["excluded_gold_studies"] == 58
    assert stats["report_alignment_studies"] == 4349
    assert not any(uid.startswith("G") for uid in uids)


def test_b16_report_contract_rejects_protocol_drift():
    config = {"allow_external_pretrained": True, "pretrained": True}
    _require_b16_report_contract(config)
    bad = dict(config)
    bad["b16_report_temperature"] = 0.2
    with pytest.raises(ValueError, match="temperature"):
        _require_b16_report_contract(bad)


def test_load_b16_report_encoder_accepts_frozen_metadata(tmp_path):
    history = [
        {"epoch": i + 1, "full_coverage": True, "budget_limited": False}
        for i in range(4)
    ]
    path = tmp_path / "b16_report_encoder.pt"
    torch.save(
        {
            "variant": B16_REPORT_SSL_VARIANT,
            "objective": B16_REPORT_SSL_OBJECTIVE,
            "initialization": B15_SSL_VARIANT,
            "input_normalization": "imagenet_mean_std",
            "gold_studies_used": 0,
            "report_alignment_studies": 4349,
            "completed_epochs": 4,
            "history": history,
            "encoder": {},
        },
        path,
    )
    payload = load_b16_report_encoder(path)
    assert payload["completed_epochs"] == 4


def test_reference_predictions_require_exact_gold_surface(tmp_path):
    uids = [f"G{i:03d}" for i in range(58)]
    frame = pd.DataFrame({"StudyInstanceUID": uids})
    for target in TARGETS:
        frame[target] = np.linspace(0.01, 0.99, len(frame))
    path = tmp_path / "b13.csv"
    frame.to_csv(path, index=False)
    values = _read_reference_predictions(path, uids)
    assert values.shape == (58, len(TARGETS))

    bad = frame.iloc[:-1].copy()
    bad_path = tmp_path / "bad.csv"
    bad.to_csv(bad_path, index=False)
    with pytest.raises(ValueError, match="exact 58-study"):
        _read_reference_predictions(bad_path, uids)
