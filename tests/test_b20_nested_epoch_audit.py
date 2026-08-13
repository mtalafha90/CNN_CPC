import numpy as np

from rsna_knee.b20_nested_epoch_audit import nested_epoch_audit_from_predictions
from rsna_knee.constants import TARGETS


def test_b20_nested_epoch_audit_selects_earliest_tie_and_builds_complete_oof():
    n = 18
    folds = np.arange(n) % 3
    within_fold = np.arange(n) // 3
    truth = np.zeros((n, len(TARGETS)), dtype=float)
    for j in range(len(TARGETS)):
        truth[:, j] = ((within_fold + j) % 2).astype(float)

    # Epochs 1 and 2 are identical perfect rankers. The frozen tie-break must
    # choose epoch 1 everywhere. Later epochs are progressively less useful.
    perfect = truth * 0.9 + (1.0 - truth) * 0.1
    reverse = 1.0 - perfect
    flat = np.full_like(truth, 0.5)
    mixed = perfect.copy()
    mixed[::2] = reverse[::2]

    predictions = {
        1: perfect,
        2: perfect.copy(),
        3: mixed,
        4: flat,
        5: reverse,
    }

    result = nested_epoch_audit_from_predictions(
        truth=truth,
        predictions=predictions,
        folds=folds,
    )

    assert result["global_selected_epoch"] == 1
    assert result["global_selected_macro_auc"] == 1.0
    assert result["crossfit_oof_score"]["macro_auc"] == 1.0
    assert result["crossfit_oof_score"]["all_12_targets_defined"] is True
    assert [row["selected_epoch"] for row in result["crossfit_rows"]] == [1, 1, 1]
    assert result["strict_complete"] is True
    assert [row["selected_epoch"] for row in result["strict_rows"]] == [1, 1, 1]
    assert np.isfinite(result["crossfit_oof_predictions"]).all()
    assert np.isfinite(result["strict_oof_predictions"]).all()
