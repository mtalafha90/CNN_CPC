from __future__ import annotations

import copy

import pytest
import torch

from rsna_knee.b36_sparse_mil import B36SparseMILHead
from rsna_knee.b45_plane_calibrated_sparse_mil import (
    B45_PLANE_POOLING,
    B45_PLANE_ROUTER_INIT,
    B45PlaneCalibratedSparseMILHead,
    require_b45_contract,
)


def _config() -> dict:
    return {
        "b7_image_size": 448,
        "b20_crop_focus_enabled": True,
        "b20_crop_focus_version": "joint_focus_center_crop_only_v1",
        "b20_crop_focus_crop_fraction": 0.90,
        "b37_grid_size": 6,
        "b37_top_k": 8,
        "b37_temperature": 1.0,
        "b37_local_aux_weight": 1.0,
        "b37_encoder_trainable_stages": 1,
        "b37_encoder_lr_scale": 0.05,
        "b37_encoder_chunk_size": 4,
        "b42_resize_policy": "constant_area_aspect_rectangular",
        "b42_reference_area": 448 * 448,
        "b42_stride_alignment": 32,
        "b42_padding_mode": "reflect",
        "b42_effective_batch": 2,
        "b45_plane_pooling": B45_PLANE_POOLING,
        "b45_plane_count": 3,
        "b45_top_k_per_plane": 8,
        "b45_plane_router_init": B45_PLANE_ROUTER_INIT,
        "b45_plane_router_temperature": 1.0,
        "b45_remove_plane_embedding_from_token_score": True,
    }


def _head() -> B45PlaneCalibratedSparseMILHead:
    torch.manual_seed(11)
    return B45PlaneCalibratedSparseMILHead(
        dim=8,
        grid_size=2,
        top_k=2,
        temperature=1.0,
        router_temperature=1.0,
    ).eval()


def test_b45_contract_rejects_cross_plane_global_pooling() -> None:
    config = _config()
    require_b45_contract(config)
    bad = copy.deepcopy(config)
    bad["b45_plane_pooling"] = "global_topk"
    with pytest.raises(ValueError):
        require_b45_contract(bad)


def test_b45_uniform_router_masks_only_missing_planes() -> None:
    head = _head()
    spatial = torch.randn(1, 3, 2, 4, 8)
    present = torch.ones(1, 3)
    meta = torch.tensor([[[1, 1, 1], [2, 1, 1], [3, 1, 1]]], dtype=torch.long)
    position = torch.tensor([[[0.2, 0.4], [0.2, 0.4], [0.2, 0.4]]])
    out = head.forward_details(spatial, present, meta, position)
    expected = torch.full((1, 12, 3), 1.0 / 3.0)
    assert torch.allclose(out.plane_weights, expected, atol=1e-7, rtol=0)
    assert out.plane_available.tolist() == [[True, True, True]]

    present_missing = torch.tensor([[1.0, 0.0, 1.0]])
    out_missing = head.forward_details(spatial, present_missing, meta, position)
    assert torch.allclose(out_missing.plane_weights[:, :, 0], torch.full((1, 12), 0.5))
    assert torch.count_nonzero(out_missing.plane_weights[:, :, 1]).item() == 0
    assert torch.allclose(out_missing.plane_weights[:, :, 2], torch.full((1, 12), 0.5))


def test_b45_plane_embedding_cannot_shift_token_evidence() -> None:
    head = _head()
    spatial = torch.randn(1, 3, 2, 4, 8)
    present = torch.ones(1, 3)
    meta = torch.tensor([[[1, 2, 2], [2, 2, 2], [3, 2, 2]]], dtype=torch.long)
    position = torch.tensor([[[0.1, 0.8], [0.1, 0.8], [0.1, 0.8]]])
    first = head.forward_details(spatial, present, meta, position)
    with torch.no_grad():
        head.plane_embedding.weight.fill_(1000.0)
    second = head.forward_details(spatial, present, meta, position)
    assert torch.equal(first.plane_top_indices, second.plane_top_indices)
    assert torch.allclose(first.plane_top_values, second.plane_top_values, atol=0, rtol=0)
    assert torch.allclose(first.local_logits, second.local_logits, atol=0, rtol=0)
    assert head.plane_embedding.weight.requires_grad is False


def test_b45_inherits_exact_b42_head_initialization_then_adds_zero_router() -> None:
    torch.manual_seed(2026)
    b42_head = B36SparseMILHead(dim=8, grid_size=2, top_k=2, temperature=1.0)
    b45_head = B45PlaneCalibratedSparseMILHead(
        dim=8,
        grid_size=2,
        top_k=2,
        temperature=1.0,
        initial_b42_head=b42_head,
    )
    b42_state = b42_head.state_dict()
    b45_state = b45_head.state_dict()
    for key, value in b42_state.items():
        assert key in b45_state
        assert torch.equal(value, b45_state[key])
    assert torch.count_nonzero(b45_head.plane_router_logits).item() == 0
