from __future__ import annotations

import pytest
import torch

from rsna_knee.b7_weak_supervision import B7_VARIANT
from rsna_knee.b8_anatomy_spatial import (
    B8_REQUIRED_B71_EXPERIMENT,
    build_anatomy_attention_bias,
    load_b71_payload,
)
from rsna_knee.constants import DUAL_STREAMS, TARGETS
from rsna_knee.ssl import SSL_SOURCE


def _token_index(stream: str, slice_index: int, region: int, *, n_slices: int, grid: int) -> int:
    r = grid * grid
    return DUAL_STREAMS.index(stream) * n_slices * r + slice_index * r + region


def test_b8_anatomy_bias_shape_and_preferred_stream_direction():
    n_slices, grid = 16, 2
    bias = build_anatomy_attention_bias(n_slices=n_slices, spatial_grid_size=grid)
    assert bias.shape == (len(TARGETS), len(DUAL_STREAMS) * n_slices * grid * grid)

    acl = TARGETS.index("ACL")
    center = n_slices // 2
    sagittal_fluid = bias[acl, _token_index("sagittal_fluid", center, 0, n_slices=n_slices, grid=grid)]
    axial_structural = bias[acl, _token_index("axial_structural", center, 0, n_slices=n_slices, grid=grid)]
    assert sagittal_fluid > axial_structural


def test_b8_acl_center_prior_is_soft_and_regions_are_not_hard_coded():
    n_slices, grid = 16, 2
    bias = build_anatomy_attention_bias(n_slices=n_slices, spatial_grid_size=grid)
    acl = TARGETS.index("ACL")
    center = n_slices // 2
    edge = 0
    center_value = bias[acl, _token_index("sagittal_fluid", center, 0, n_slices=n_slices, grid=grid)]
    edge_value = bias[acl, _token_index("sagittal_fluid", edge, 0, n_slices=n_slices, grid=grid)]
    assert center_value > edge_value
    # Fixed prior is uniform across the 2x2 in-plane regions; region location is learned.
    region_values = [
        bias[acl, _token_index("sagittal_fluid", center, region, n_slices=n_slices, grid=grid)].item()
        for region in range(grid * grid)
    ]
    assert region_values == pytest.approx([region_values[0]] * (grid * grid))
    assert float(edge_value) > -1.0  # soft bias, not a hard mask


def test_b8_diffuse_effusion_prior_is_slice_neutral():
    n_slices, grid = 16, 2
    bias = build_anatomy_attention_bias(n_slices=n_slices, spatial_grid_size=grid)
    effusion = TARGETS.index("Effusion")
    values = [
        bias[effusion, _token_index("axial_fluid", s, 0, n_slices=n_slices, grid=grid)].item()
        for s in range(n_slices)
    ]
    assert values == pytest.approx([values[0]] * n_slices)


def test_b8_zero_strength_removes_fixed_attention_bias():
    bias = build_anatomy_attention_bias(n_slices=16, spatial_grid_size=2, strength=0.0)
    assert torch.count_nonzero(bias).item() == 0


def _b71_payload(*, batches: int = 1560, experiment: str = B8_REQUIRED_B71_EXPERIMENT):
    return {
        "variant": B7_VARIANT,
        "source": SSL_SOURCE,
        "completed_epochs": 4,
        "gold_studies_used_in_gradient": 0,
        "gold_studies_used_for_early_stopping": 0,
        "config": {
            "b7_experiment_name": experiment,
            "b7_max_batches_per_epoch": batches,
        },
        "supervision": {
            "training_studies": 3120,
            "training_usable_cells": 14123,
        },
        "model_state": {},
    }


def test_load_b71_payload_requires_full_coverage_named_checkpoint(tmp_path):
    good = tmp_path / "b71.pt"
    torch.save(_b71_payload(), good)
    payload = load_b71_payload(good)
    assert payload["config"]["b7_experiment_name"] == B8_REQUIRED_B71_EXPERIMENT

    short = tmp_path / "short.pt"
    torch.save(_b71_payload(batches=500, experiment="B7-v1"), short)
    with pytest.raises(ValueError, match="B7.1_full_coverage|1560"):
        load_b71_payload(short)
