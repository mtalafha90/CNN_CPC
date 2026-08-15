import numpy as np

from rsna_knee.b26_2_mechanism_audit import _binary_contingency, _target_mass_table
from rsna_knee.constants import TARGETS


def test_binary_contingency_counts_and_conditionals():
    a = np.array([1, 1, 0, 0], dtype=bool)
    b = np.array([1, 0, 1, 0], dtype=bool)
    out = _binary_contingency(a, b, label_a="A", label_b="B")
    assert out["n_both_defined"] == 4
    assert out["a_pos_b_pos"] == 1
    assert out["a_pos_b_neg"] == 1
    assert out["a_neg_b_pos"] == 1
    assert out["a_neg_b_neg"] == 1
    assert out["phi"] == 0.0
    assert out["p_a_positive_given_b_positive"] == 0.5
    assert out["p_a_positive_given_b_negative"] == 0.5


def test_target_balance_keeps_equal_total_target_share():
    n = 4
    y = np.zeros((n, len(TARGETS)), dtype=np.float32)
    w = np.ones((n, len(TARGETS)), dtype=np.float32)
    # Give target 0 half the raw mass while keeping it non-empty.
    w[:, 0] = 0.5
    y[:2, :] = 1.0
    out = _target_mass_table(y, w)
    shares = np.array([out[t]["normalized_total_loss_share"] for t in TARGETS])
    np.testing.assert_allclose(shares, np.full(len(TARGETS), 1.0 / len(TARGETS)), atol=1e-7)
    assert out[TARGETS[0]]["target_balance_multiplier"] > out[TARGETS[1]]["target_balance_multiplier"]
