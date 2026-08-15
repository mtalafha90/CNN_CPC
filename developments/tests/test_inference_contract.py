from pathlib import Path

import pytest

from rsna_knee.inference import _validate_ensemble_contract


def _payload(fold: int, stage: str = "stage2", offsets=(-1, 0, 1)) -> dict:
    return {
        "fold": fold,
        "stage": stage,
        "validation_tta_offsets": list(offsets),
        "model_spec": {"architecture": "same"},
    }


def _config() -> dict:
    return {
        "n_folds": 3,
        "tta_center_offsets": [-1, 0, 1],
        "expected_checkpoint_stage": "stage2",
    }


def test_ensemble_contract_orders_complete_fold_set():
    paths = [Path("fold2.pt"), Path("fold0.pt"), Path("fold1.pt")]
    payloads = [_payload(2), _payload(0), _payload(1)]
    ordered_paths, ordered_payloads = _validate_ensemble_contract(paths, payloads, _config())
    assert [payload["fold"] for payload in ordered_payloads] == [0, 1, 2]
    assert [path.name for path in ordered_paths] == ["fold0.pt", "fold1.pt", "fold2.pt"]


def test_ensemble_contract_rejects_duplicate_or_missing_fold():
    paths = [Path("a.pt"), Path("b.pt"), Path("c.pt")]
    payloads = [_payload(0), _payload(0), _payload(2)]
    with pytest.raises(ValueError, match="folds must be exactly"):
        _validate_ensemble_contract(paths, payloads, _config())


def test_ensemble_contract_rejects_mixed_stage():
    paths = [Path("a.pt"), Path("b.pt"), Path("c.pt")]
    payloads = [_payload(0, "stage2"), _payload(1, "stage1"), _payload(2, "stage2")]
    with pytest.raises(ValueError, match="mix checkpoint stages"):
        _validate_ensemble_contract(paths, payloads, _config())


def test_ensemble_contract_rejects_posthoc_tta_change():
    paths = [Path("a.pt"), Path("b.pt"), Path("c.pt")]
    payloads = [_payload(0), _payload(1, offsets=(0,)), _payload(2)]
    with pytest.raises(ValueError, match="validated TTA offsets"):
        _validate_ensemble_contract(paths, payloads, _config())
