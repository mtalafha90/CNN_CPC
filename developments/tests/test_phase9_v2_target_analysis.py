import numpy as np

from rsna_knee.phase9_v2_target_analysis import _hard_truth, paired_target_bootstrap


def test_hard_truth_ignores_zero_weight_cells():
    target = np.array([0.85, 0.05, 0.85, 0.05, 0.5], dtype=float)
    weight = np.array([0.5, 1.0, 0.0, 0.0, 0.0], dtype=float)
    truth = _hard_truth(target, weight)
    assert truth[0] == 1.0
    assert truth[1] == 0.0
    assert np.isnan(truth[2:]).all()


def test_paired_target_bootstrap_recovers_positive_direction():
    # Four active examples, plus two inactive NaN truth entries. Control has one
    # cross-class inversion; candidate ranks both positives above both negatives.
    truth = np.array([0, 0, 1, 1, np.nan, np.nan], dtype=float)
    control = np.array([0.4, 0.7, 0.6, 0.5, 0.2, 0.8], dtype=float)
    candidate = np.array([0.1, 0.2, 0.8, 0.9, 0.9, 0.1], dtype=float)
    out = paired_target_bootstrap(truth, control, candidate, n_bootstrap=500, seed=7)
    assert out["candidate_auc"] > out["control_auc"]
    assert out["point_difference"] > 0
    assert out["n_valid_replicates"] > 0
    assert 0.0 <= out["probability_candidate_better"] <= 1.0
