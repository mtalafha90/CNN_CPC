from __future__ import annotations

import torch

from rsna_knee.ssl import _ssl_examples, ssl_position_indices


def test_ssl_position_indices_are_distributed():
    assert ssl_position_indices(9, 1).tolist() == [4]
    assert ssl_position_indices(9, 2).tolist() == [0, 8]
    assert ssl_position_indices(9, 3).tolist() == [0, 4, 8]
    assert ssl_position_indices(5, 10).tolist() == [0, 1, 2, 3, 4]


def test_ssl_position_indices_validate_inputs():
    for n_slices, positions in [(0, 1), (5, 0)]:
        try:
            ssl_position_indices(n_slices, positions)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid SSL position arguments must fail")


def test_ssl_examples_expand_only_present_streams():
    # B=2, K=2, S=5. Three streams are present, and two positions are
    # requested, so six active 2.5D examples must be returned.
    volumes = torch.arange(2 * 2 * 5 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 2, 5, 3, 2, 2)
    present = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    x, stream_idx, study_ids, position_idx, positions = _ssl_examples(
        volumes,
        present,
        positions_per_stream=2,
    )

    assert positions.tolist() == [0, 4]
    assert tuple(x.shape) == (6, 3, 2, 2)
    assert stream_idx.tolist() == [0, 0, 0, 0, 1, 1]
    assert study_ids.tolist() == [0, 0, 1, 1, 1, 1]
    assert position_idx.tolist() == [0, 1, 0, 1, 0, 1]


def test_ssl_examples_reject_bad_shapes():
    volumes = torch.zeros(2, 6, 5, 3, 8, 8)
    bad_present = torch.zeros(2, 5)
    try:
        _ssl_examples(volumes, bad_present, positions_per_stream=2)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched present mask must fail")
