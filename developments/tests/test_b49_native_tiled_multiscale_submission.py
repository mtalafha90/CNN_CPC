from __future__ import annotations

import pytest

from rsna_knee.b49_native_tiled_multiscale_submission import (
    B49_SUBMISSION_TTA_OFFSETS,
    require_b49_candidate_submission_contract,
)


def _config() -> dict:
    return {
        "b7_eval_tta_offsets": list(B49_SUBMISSION_TTA_OFFSETS),
        "b7_eval_batch_size": 1,
        "num_workers": 0,
        "pin_memory": False,
        "series_cache_mb_per_worker": 0,
        "strict_dicom_inference": True,
    }


def test_b49_candidate_submission_contract_accepts_frozen_execution():
    assert require_b49_candidate_submission_contract(_config()) == B49_SUBMISSION_TTA_OFFSETS


def test_b49_candidate_submission_rejects_changed_tta():
    config = _config()
    config["b7_eval_tta_offsets"] = [0]
    with pytest.raises(ValueError, match="tta_offsets"):
        require_b49_candidate_submission_contract(config)
