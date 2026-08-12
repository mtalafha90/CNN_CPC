from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.b17_submission import _test_dataset_config, _validate_submission
from rsna_knee.constants import SUBMISSION_COLUMNS, TARGETS


def test_b17_test_dataset_uses_hidden_test_split_and_frozen_tta():
    config = {
        "b7_n_slices": 16,
        "b7_image_size": 224,
        "b7_triplet_gap": 1,
        "strict_dicom_inference": True,
        "b7_train_gap_choices": [1, 2],
        "series_cache_mb_per_worker": 256,
    }
    ds = _test_dataset_config(config, Path("/tmp/data"), (-1, 0, 1))
    assert ds.split == "test"
    assert ds.n_slices == 16
    assert ds.image_size == 224
    assert ds.tta_center_offsets == (-1, 0, 1)
    assert ds.noise_std == 0.0
    assert ds.slice_dropout == 0.0
    assert ds.center_jitter == 0
    assert ds.rotation_deg == 0.0


def test_b17_submission_validation_accepts_exact_schema_and_order():
    uids = ["a", "b", "c"]
    frame = pd.DataFrame(np.full((3, len(TARGETS)), 0.5), columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    _validate_submission(frame, uids)
    assert list(frame.columns) == SUBMISSION_COLUMNS


def test_b17_submission_validation_rejects_wrong_uid_order():
    uids = ["a", "b"]
    frame = pd.DataFrame(np.full((2, len(TARGETS)), 0.5), columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", ["b", "a"])
    with pytest.raises(ValueError, match="order"):
        _validate_submission(frame, uids)


def test_b17_submission_validation_rejects_invalid_probability():
    uids = ["a"]
    frame = pd.DataFrame(np.full((1, len(TARGETS)), 0.5), columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    frame.loc[0, TARGETS[0]] = 1.2
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        _validate_submission(frame, uids)
