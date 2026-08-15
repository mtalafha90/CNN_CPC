import numpy as np
import pytest

from rsna_knee.b21_contract import require_b21_contract
from rsna_knee.b21_protocol import B21_FIXED_EPOCHS, mode_identity
from rsna_knee.preresize_crop import center_crop_raw_volume


def test_center_crop_raw_volume_uses_native_samples():
    raw = np.arange(2 * 320 * 320, dtype=np.float32).reshape(2, 320, 320)
    cropped = center_crop_raw_volume(raw, 0.90)
    assert cropped.shape == (2, 288, 288)
    assert np.array_equal(cropped, raw[:, 16:304, 16:304])


def test_b21_protocol_is_fixed_to_epoch_two():
    assert B21_FIXED_EPOCHS == 2
    assert require_b21_contract({}) == pytest.approx(0.90)
    with pytest.raises(ValueError):
        require_b21_contract({"b7_epochs": 5})


def test_b21_protocol_disables_expert_selection():
    with pytest.raises(ValueError):
        require_b21_contract({"b18_expert_selection": True})


def test_matched_arms_have_distinct_crop_stages():
    assert mode_identity("control")[2] == "post_resize_224"
    assert mode_identity("preresize")[2] == "native_array_pre_resize"
