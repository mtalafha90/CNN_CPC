from __future__ import annotations

import math

import torch

from rsna_knee.b36_sparse_mil import B36SparseMILHead


def _fixture(dim: int = 16):
    torch.manual_seed(11)
    head = B36SparseMILHead(
        dim=dim,
        grid_size=3,
        top_k=8,
        temperature=1.0,
        token_dropout=0.0,
    )
    # B=2, K=2, S=4, R=9 => 72 tokens/study, 36 valid in study 2.
    spatial = torch.randn(2, 2, 4, 9, dim)
    present = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    meta = torch.tensor(
        [
            [[1, 1, 1], [2, 2, 2]],
            [[3, 1, 2], [0, 0, 0]],
        ],
        dtype=torch.long,
    )
    position = torch.linspace(0.0, 1.0, 4).repeat(2, 2, 1)
    return head, spatial, present, meta, position


def test_b36_topk_is_explicitly_sparse_and_valid():
    head, spatial, present, meta, position = _fixture()
    local, indices, values = head(spatial, present, meta, position)

    assert local.shape == (2, 12)
    assert indices.shape == (2, 12, 8)
    assert values.shape == (2, 12, 8)
    assert torch.isfinite(local).all()
    assert torch.isfinite(values).all()

    # Study 2 has only its first series valid: 4 slices * 9 regions = 36 tokens.
    assert int(indices[1].max().item()) < 36


def test_b36_logmeanexp_pool_matches_manual_formula():
    head, spatial, present, meta, position = _fixture()
    local, _, values = head(spatial, present, meta, position)
    manual = torch.logsumexp(values, dim=-1) - math.log(8.0)
    torch.testing.assert_close(local, manual, rtol=1e-6, atol=1e-6)


def test_b36_direct_local_loss_gives_evidence_gradient_at_zero_gate():
    head, spatial, present, meta, position = _fixture()
    assert torch.count_nonzero(head.gate).item() == 0

    local, _, _ = head(spatial, present, meta, position)
    target = torch.randint(0, 2, local.shape, dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(local, target)
    loss.backward()

    assert head.evidence_weight.grad is not None
    assert torch.count_nonzero(head.evidence_weight.grad).item() > 0


def test_b36_geometry_parameters_start_zero():
    head, *_ = _fixture()
    assert torch.count_nonzero(head.position_projection.weight).item() == 0
    assert torch.count_nonzero(head.region_embedding).item() == 0
    assert torch.count_nonzero(head.plane_embedding.weight).item() == 0
    assert torch.count_nonzero(head.fluid_embedding.weight).item() == 0
    assert torch.count_nonzero(head.fat_embedding.weight).item() == 0
