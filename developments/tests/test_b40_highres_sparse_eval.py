from __future__ import annotations

import numpy as np

from rsna_knee.b40_highres_sparse_eval import _global_macro_auc


def test_b40_global_macro_auc_is_a_scalar_not_the_per_target_array():
    """Protect the B40 JSON summary from reversing macro_auc_from_arrays outputs."""
    truth = np.array([[0.0] * 12, [1.0] * 12], dtype=np.float64)
    prediction = np.array([[0.1] * 12, [0.9] * 12], dtype=np.float64)

    score = _global_macro_auc(truth, prediction)

    assert isinstance(score, float)
    assert score == 1.0
