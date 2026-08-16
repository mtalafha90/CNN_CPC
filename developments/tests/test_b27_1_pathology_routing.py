from __future__ import annotations

import torch

from rsna_knee.b12_1_hierarchical import HierarchicalSeriesKneeMILNet
from rsna_knee.b27_1_pathology_routing import (
    B27_1_ROUTE_PARAMETER_COUNT,
    PathologyPairedMetadataRoutedKneeMILNet,
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


def _routing_batch():
    volumes = torch.randn(2, 4, 1, 3, 64, 64)
    present = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    meta = torch.tensor(
        [
            [[1, 1, 1], [2, 2, 2], [3, 1, 1], [0, 0, 0]],
            [[2, 2, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.long,
    )
    return volumes, present, meta


def test_b27_1_has_exactly_60_zero_initialized_routing_parameters():
    model = _model(PathologyPairedMetadataRoutedKneeMILNet)
    route = [model.route_plane_bias, model.route_sequence_bias]
    assert sum(x.numel() for x in route) == B27_1_ROUTE_PARAMETER_COUNT == N_TARGETS * 5
    assert all(torch.count_nonzero(x).item() == 0 for x in route)


def test_b27_1_paired_sequence_mapping_is_conservative():
    model = _model(PathologyPairedMetadataRoutedKneeMILNet)
    meta = torch.tensor(
        [[
            [1, 1, 1],  # category 1
            [1, 2, 2],  # category 2
            [1, 1, 2],  # discordant -> 0
            [1, 2, 1],  # discordant -> 0
            [1, 0, 0],  # unknown -> 0
            [1, 0, 2],  # partial unknown -> 0
        ]],
        dtype=torch.long,
    )
    got = model.paired_sequence_ids(meta)
    assert got.tolist() == [[1, 2, 0, 0, 0, 0]]


def test_b27_1_unknown_or_discordant_sequence_bias_is_fixed_zero():
    model = _model(PathologyPairedMetadataRoutedKneeMILNet)
    with torch.no_grad():
        model.route_plane_bias.zero_()
        model.route_sequence_bias.fill_(2.0)
    meta = torch.tensor(
        [[[1, 0, 0], [1, 1, 2], [1, 1, 1], [1, 2, 2]]],
        dtype=torch.long,
    )
    bias = model.metadata_route_bias(meta)
    assert torch.allclose(bias[:, :, 0], torch.zeros_like(bias[:, :, 0]))
    assert torch.allclose(bias[:, :, 1], torch.zeros_like(bias[:, :, 1]))
    assert torch.allclose(bias[:, :, 2], torch.full_like(bias[:, :, 2], 2.0))
    assert torch.allclose(bias[:, :, 3], torch.full_like(bias[:, :, 3], 2.0))


def test_b27_1_zero_route_is_functionally_b20_equivalent():
    torch.manual_seed(123)
    base = _model(HierarchicalSeriesKneeMILNet).eval()
    routed = _model(PathologyPairedMetadataRoutedKneeMILNet).eval()
    missing, unexpected = routed.load_state_dict(base.state_dict(), strict=False)
    assert set(missing) == {"route_plane_bias", "route_sequence_bias"}
    assert unexpected == []

    volumes = torch.randn(1, 2, 1, 3, 64, 64)
    present = torch.tensor([[1, 0]], dtype=torch.bool)
    meta = torch.tensor([[[1, 2, 2], [0, 0, 0]]], dtype=torch.long)
    with torch.no_grad():
        y_base = base(volumes, present, meta)
        y_routed = routed(volumes, present, meta)
    assert torch.allclose(y_base, y_routed, atol=1e-5, rtol=1e-5)


def test_b27_1_routing_is_pathology_specific():
    model = _model(PathologyPairedMetadataRoutedKneeMILNet)
    with torch.no_grad():
        model.route_plane_bias.zero_()
        model.route_sequence_bias.zero_()
        model.route_plane_bias[0, 0] = 1.25
        model.route_sequence_bias[1, 1] = -0.75
    meta = torch.tensor([[[1, 1, 1], [2, 2, 2], [3, 1, 1]]], dtype=torch.long)
    bias = model.metadata_route_bias(meta)
    assert float(bias[0, 0, 0]) == 1.25
    assert float(bias[0, 0, 1]) == 0.0
    assert float(bias[0, 1, 1]) == -0.75
    assert float(bias[0, 1, 0]) == 0.0


def test_b27_1_backward_reaches_every_routing_parameter():
    model = _model(PathologyPairedMetadataRoutedKneeMILNet)
    logits = model(*_routing_batch())
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    for name in ("route_plane_bias", "route_sequence_bias"):
        grad = getattr(model, name).grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad).item() == grad.numel()


def test_b27_1_additive_inf_mask_survives_bf16_autocast():
    model = _model(PathologyPairedMetadataRoutedKneeMILNet)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(*_routing_batch())
    assert torch.isfinite(logits).all()
    logits.float().sum().backward()
    assert torch.isfinite(model.route_plane_bias.grad).all()
    assert torch.isfinite(model.route_sequence_bias.grad).all()


def test_b27_1_empty_study_stays_finite():
    model = _model(PathologyPairedMetadataRoutedKneeMILNet)
    with torch.no_grad():
        model.route_plane_bias.fill_(3.0)
        model.route_sequence_bias.fill_(3.0)
    volumes, _, meta = _routing_batch()
    empty = torch.zeros(volumes.shape[0], volumes.shape[1], dtype=torch.bool)
    logits = model(volumes, empty, torch.zeros_like(meta))
    assert torch.isfinite(logits).all()
