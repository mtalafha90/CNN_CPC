from __future__ import annotations

import torch

from rsna_knee.b12_1_hierarchical import HierarchicalSeriesKneeMILNet
from rsna_knee.b27_pathology_routing import (
    B27_ROUTE_PARAMETER_COUNT,
    PathologyMetadataRoutedKneeMILNet,
)
from rsna_knee.constants import N_TARGETS


def _model(cls):
    return cls(
        1,
        pretrained_weights=False,
        normalize_input=False,
        dropout=0.0,
        encoder_batch_size=4,
        gradient_checkpointing=False,
        transformer_layers=1,
        transformer_heads=8,
        transformer_ff_mult=1.0,
        pathology_layers=1,
        series_pool_heads=8,
    )


def test_b27_has_exactly_84_zero_initialized_routing_parameters():
    model = _model(PathologyMetadataRoutedKneeMILNet)
    route = [model.route_plane_bias, model.route_fluid_bias, model.route_fat_bias]
    assert sum(x.numel() for x in route) == B27_ROUTE_PARAMETER_COUNT == N_TARGETS * 7
    assert all(torch.count_nonzero(x).item() == 0 for x in route)


def test_b27_unknown_metadata_is_permanently_zero_bias():
    model = _model(PathologyMetadataRoutedKneeMILNet)
    with torch.no_grad():
        model.route_plane_bias.fill_(1.0)
        model.route_fluid_bias.fill_(2.0)
        model.route_fat_bias.fill_(3.0)

    meta = torch.tensor(
        [
            [
                [0, 0, 0],  # all unknown -> exactly zero
                [1, 1, 1],  # known -> 1 + 2 + 3
                [3, 2, 2],  # known -> 1 + 2 + 3
            ]
        ],
        dtype=torch.long,
    )
    bias = model.metadata_route_bias(meta)
    assert bias.shape == (1, N_TARGETS, 3)
    assert torch.allclose(bias[:, :, 0], torch.zeros_like(bias[:, :, 0]))
    assert torch.allclose(bias[:, :, 1], torch.full_like(bias[:, :, 1], 6.0))
    assert torch.allclose(bias[:, :, 2], torch.full_like(bias[:, :, 2], 6.0))


def test_b27_zero_route_is_functionally_b20_equivalent():
    torch.manual_seed(123)
    base = _model(HierarchicalSeriesKneeMILNet).eval()
    routed = _model(PathologyMetadataRoutedKneeMILNet).eval()

    missing, unexpected = routed.load_state_dict(base.state_dict(), strict=False)
    assert set(missing) == {
        "route_plane_bias",
        "route_fluid_bias",
        "route_fat_bias",
    }
    assert unexpected == []

    # One present and one padded series exercises both zero route bias and the
    # B27 additive -inf padding mask.
    volumes = torch.randn(1, 2, 1, 3, 64, 64)
    present = torch.tensor([[1, 0]], dtype=torch.bool)
    meta = torch.tensor([[[1, 2, 2], [0, 0, 0]]], dtype=torch.long)

    with torch.no_grad():
        y_base = base(volumes, present, meta)
        y_routed = routed(volumes, present, meta)
    assert torch.allclose(y_base, y_routed, atol=1e-5, rtol=1e-5)


def test_b27_metadata_bias_is_pathology_specific():
    model = _model(PathologyMetadataRoutedKneeMILNet)
    with torch.no_grad():
        model.route_plane_bias.zero_()
        model.route_fluid_bias.zero_()
        model.route_fat_bias.zero_()
        model.route_plane_bias[0, 0] = 1.25  # target 0, Sagittal only
        model.route_plane_bias[1, 1] = -0.75  # target 1, Coronal only

    meta = torch.tensor([[[1, 1, 1], [2, 1, 1], [3, 1, 1]]], dtype=torch.long)
    bias = model.metadata_route_bias(meta)
    assert float(bias[0, 0, 0]) == 1.25
    assert float(bias[0, 0, 1]) == 0.0
    assert float(bias[0, 1, 1]) == -0.75
    assert float(bias[0, 1, 0]) == 0.0
