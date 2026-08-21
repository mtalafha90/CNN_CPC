from __future__ import annotations

import numpy as np
import torch

from rsna_knee.b35_target_spatial_residual import (
    B35_BASE_SLICES,
    B35_DENSE_SLICES,
    B35_GRID_SIZE,
    TargetSpatialHead,
    b35_centers,
)
from rsna_knee.dicom import _centers


def test_b35_first_16_centers_are_exact_historical_centers():
    for n_frames in (8, 17, 31, 78, 320):
        combined, position = b35_centers(n_frames, gap=1, center_offset=0)
        historical = _centers(n_frames, B35_BASE_SLICES, 1, center_offset=0, jitter=0)
        assert combined.shape == (B35_DENSE_SLICES,)
        assert position.shape == (B35_DENSE_SLICES,)
        np.testing.assert_array_equal(combined[:B35_BASE_SLICES], historical)
        assert np.all(position >= 0)
        assert np.all(position <= 1)


def test_b35_center_offsets_preserve_historical_tta_centers():
    for offset in (-1, 0, 1):
        combined, _ = b35_centers(123, gap=1, center_offset=offset)
        historical = _centers(123, B35_BASE_SLICES, 1, center_offset=offset, jitter=0)
        np.testing.assert_array_equal(combined[:B35_BASE_SLICES], historical)


def test_b35_zero_gate_is_exact_base_passthrough():
    torch.manual_seed(7)
    head = TargetSpatialHead(dim=24, grid_size=B35_GRID_SIZE, token_dropout=0.0)
    head.eval()
    spatial = torch.randn(2, 3, B35_DENSE_SLICES, B35_GRID_SIZE**2, 24)
    present = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    meta = torch.zeros(2, 3, 3, dtype=torch.long)
    position = torch.linspace(0, 1, B35_DENSE_SLICES).view(1, 1, -1).repeat(2, 3, 1)
    local, _ = head(spatial, present, meta, position)
    base = torch.randn(2, 12)
    combined = base + head.effective_gate()[None, :] * local
    torch.testing.assert_close(combined, base, rtol=0, atol=0)


def test_b35_local_attention_masks_missing_series():
    torch.manual_seed(8)
    head = TargetSpatialHead(dim=16, grid_size=2, token_dropout=0.0)
    head.eval()
    spatial = torch.randn(1, 2, 4, 4, 16)
    present = torch.tensor([[1, 0]], dtype=torch.float32)
    meta = torch.zeros(1, 2, 3, dtype=torch.long)
    position = torch.linspace(0, 1, 4).view(1, 1, 4).repeat(1, 2, 1)
    _, attention = head(spatial, present, meta, position)
    # Tokens belonging to the second series occupy the final half of the flat memory.
    assert torch.count_nonzero(attention[..., 16:]).item() == 0
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.ones_like(attention.sum(dim=-1)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_b35_nonzero_gate_can_change_predictions():
    torch.manual_seed(9)
    head = TargetSpatialHead(dim=12, grid_size=1, token_dropout=0.0)
    head.eval()
    with torch.no_grad():
        head.gate.fill_(0.25)
    spatial = torch.randn(1, 1, 3, 1, 12)
    present = torch.ones(1, 1)
    meta = torch.zeros(1, 1, 3, dtype=torch.long)
    position = torch.tensor([[[0.0, 0.5, 1.0]]])
    local, _ = head(spatial, present, meta, position)
    base = torch.zeros(1, 12)
    combined = base + head.effective_gate()[None, :] * local
    assert torch.count_nonzero(combined).item() > 0
