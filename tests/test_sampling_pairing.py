from __future__ import annotations

import numpy as np

from rsna_knee.sampling import TwoPoolBatchSampler


def _trusted_counts(batches, trusted_mask):
    return [sum(bool(trusted_mask[i]) for i in batch) for batch in batches]


def test_batch_size_two_groups_trusted_rows_in_pairs():
    trusted = np.array([True] * 20 + [False] * 80)
    sampler = TwoPoolBatchSampler(
        trusted,
        batch_size=2,
        trusted_fraction=0.30,
        seed=7,
        max_batches=20,
    )
    batches = list(iter(sampler))
    counts = _trusted_counts(batches, trusted)

    assert len(batches) == 20
    assert set(counts) <= {0, 2}
    assert sum(counts) == 12  # 20 batches * 2 rows * 30%
    assert 2 in counts and 0 in counts


def test_pair_grouping_is_epoch_deterministic():
    trusted = np.array([True] * 20 + [False] * 80)
    sampler = TwoPoolBatchSampler(
        trusted,
        batch_size=2,
        trusted_fraction=0.30,
        seed=11,
        max_batches=20,
    )
    first = list(iter(sampler))
    sampler.set_epoch(0)
    assert first == list(iter(sampler))
    sampler.set_epoch(1)
    assert first != list(iter(sampler))


def test_even_larger_batch_preserves_requested_trusted_fraction():
    trusted = np.array([True] * 30 + [False] * 90)
    sampler = TwoPoolBatchSampler(
        trusted,
        batch_size=8,
        trusted_fraction=0.25,
        seed=5,
        max_batches=10,
    )
    batches = list(iter(sampler))
    counts = _trusted_counts(batches, trusted)

    assert all(count % 2 == 0 for count in counts)
    assert sum(counts) == 20  # 10 * 8 * 25%
