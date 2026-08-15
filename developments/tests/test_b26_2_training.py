from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.b26_2_training import apply_b26_2_fill_to_arrays
from rsna_knee.constants import TARGETS


def _base():
    uids = ["u0", "u1", "u2"]
    y = np.full((3, len(TARGETS)), 0.5, dtype=np.float32)
    w = np.zeros_like(y)
    j = TARGETS.index("Synovitis")
    # Existing B6 cell must survive exactly.
    y[0, j] = 0.85
    w[0, j] = 0.50
    return uids, y, w


def test_b26_2_training_adds_only_silent_synovitis_cells():
    uids, y, w = _base()
    frame = pd.DataFrame(
        {
            "StudyInstanceUID": ["u1", "u2"],
            "target": ["Synovitis", "Synovitis"],
            "b26_2_accept": [True, True],
            "b26_2_state": ["positive", "negated"],
        }
    )
    y2, w2, diag = apply_b26_2_fill_to_arrays(uids, y, w, frame)
    j = TARGETS.index("Synovitis")
    assert y2[0, j] == y[0, j] and w2[0, j] == w[0, j]
    assert y2[1, j] == pytest.approx(0.85) and w2[1, j] == pytest.approx(0.50)
    assert y2[2, j] == pytest.approx(0.05) and w2[2, j] == pytest.approx(1.00)
    assert diag["accepted_positive"] == 1
    assert diag["accepted_negated"] == 1
    assert diag["base_cells_dropped"] == 0
    assert diag["base_cells_overridden"] == 0


def test_b26_2_training_refuses_override_of_existing_b6_cell():
    uids, y, w = _base()
    frame = pd.DataFrame(
        {
            "StudyInstanceUID": ["u0"],
            "target": ["Synovitis"],
            "b26_2_accept": [True],
            "b26_2_state": ["negated"],
        }
    )
    with pytest.raises(RuntimeError, match="overwrite"):
        apply_b26_2_fill_to_arrays(uids, y, w, frame)


def test_b26_2_training_ignores_rejected_candidate():
    uids, y, w = _base()
    frame = pd.DataFrame(
        {
            "StudyInstanceUID": ["u1"],
            "target": ["Synovitis"],
            "b26_2_accept": [False],
            "b26_2_state": ["unmentioned"],
        }
    )
    y2, w2, diag = apply_b26_2_fill_to_arrays(uids, y, w, frame)
    assert np.array_equal(y2, y)
    assert np.array_equal(w2, w)
    assert diag["accepted_total"] == 0


def test_b26_2_training_refuses_non_synovitis_scope():
    uids, y, w = _base()
    frame = pd.DataFrame(
        {
            "StudyInstanceUID": ["u1"],
            "target": ["ACL"],
            "b26_2_accept": [True],
            "b26_2_state": ["positive"],
        }
    )
    with pytest.raises(ValueError, match="Synovitis"):
        apply_b26_2_fill_to_arrays(uids, y, w, frame)


def test_b26_2_training_refuses_unknown_uid():
    uids, y, w = _base()
    frame = pd.DataFrame(
        {
            "StudyInstanceUID": ["outside"],
            "target": ["Synovitis"],
            "b26_2_accept": [True],
            "b26_2_state": ["positive"],
        }
    )
    with pytest.raises(ValueError, match="outside the exact B20 surface"):
        apply_b26_2_fill_to_arrays(uids, y, w, frame)


def test_b26_2_training_refuses_duplicate_uid_rows():
    uids, y, w = _base()
    frame = pd.DataFrame(
        {
            "StudyInstanceUID": ["u1", "u1"],
            "target": ["Synovitis", "Synovitis"],
            "b26_2_accept": [True, False],
            "b26_2_state": ["positive", "unmentioned"],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        apply_b26_2_fill_to_arrays(uids, y, w, frame)
