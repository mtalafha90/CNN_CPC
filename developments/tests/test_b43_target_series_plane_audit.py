import math

import torch

from rsna_knee.b43_target_series_plane_audit import _lme, decode_sparse_index


def test_b43_decode_sparse_index_matches_b42_flatten_order():
    assert decode_sparse_index(
        0, n_slices=32, n_regions=36, grid_size=6
    ) == (0, 0, 0, 0, 0)
    assert decode_sparse_index(
        35, n_slices=32, n_regions=36, grid_size=6
    ) == (0, 0, 35, 5, 5)
    assert decode_sparse_index(
        36, n_slices=32, n_regions=36, grid_size=6
    ) == (0, 1, 0, 0, 0)
    assert decode_sparse_index(
        32 * 36, n_slices=32, n_regions=36, grid_size=6
    ) == (1, 0, 0, 0, 0)


def test_b43_lme_matches_logmeanexp_definition():
    values = torch.tensor([0.0, 1.0, 2.0, 3.0])
    observed = float(_lme(values, 1.0).item())
    expected = float(torch.logsumexp(values, dim=0).item() - math.log(4.0))
    assert abs(observed - expected) < 1e-7
