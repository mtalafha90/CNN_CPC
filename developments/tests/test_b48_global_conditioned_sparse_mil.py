"""Unit contracts for B48 global-query-conditioned sparse MIL.

These tests intentionally operate on small tensors.  They establish the
mechanism's causal invariants without needing DICOM data or a ConvNeXt pass.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from rsna_knee.b36_sparse_mil import B36SparseMILHead
from rsna_knee.b37_highres_sparse_mil import B37HighResSparseMILResidual
from rsna_knee.b48_global_conditioned_sparse_mil import (
    B48_ARMS,
    B48_CONTEXT_DIM,
    B48_POST_CROSS_ATTENTION_CANDIDATE,
    B48_STATIC_PRIOR_CONTROL,
    B48GlobalConditionedSparseMILHead,
    B48GlobalConditionedSparseMILResidual,
    b48_state,
    require_b48_contract,
)
from rsna_knee.b48_global_conditioned_sparse_eval import (
    _masked_target_matrix,
    _matched_pair_seed,
)
from rsna_knee.b48_global_conditioned_sparse_training import b48_fill_artifacts
from rsna_knee.constants import N_TARGETS


def _inputs(*, batch: int = 1, series: int = 2, slices: int = 3, regions: int = 4, dim: int = 8):
    torch.manual_seed(71)
    return {
        "spatial": torch.randn(batch, series, slices, regions, dim),
        "present": torch.ones(batch, series),
        "series_meta": torch.tensor(
            [[[1, 1, 1], [2, 2, 2]]], dtype=torch.long
        ).expand(batch, -1, -1).clone(),
        "slice_position": torch.linspace(0.0, 1.0, slices).reshape(1, 1, slices).expand(
            batch, series, -1
        ).clone(),
        "global_query": torch.randn(batch, N_TARGETS, dim),
    }


def _heads(dim: int = 8, regions_side: int = 2, top_k: int = 2):
    torch.manual_seed(2026)
    b42 = B36SparseMILHead(dim, grid_size=regions_side, top_k=top_k, temperature=1.0)
    b48 = B48GlobalConditionedSparseMILHead(
        dim,
        grid_size=regions_side,
        top_k=top_k,
        temperature=1.0,
        context_dim=4,
        initial_b42_head=b42,
    )
    return b42, b48


def test_b48_copies_the_entire_b42_sparse_head_before_adding_its_zero_branch():
    b42, b48 = _heads()
    for key, value in b42.state_dict().items():
        assert key in b48.state_dict()
        assert torch.equal(value, b48.state_dict()[key])
    assert torch.count_nonzero(b48.context_gate).item() == 0
    assert b48.context_dim == 4


def test_zero_context_gate_reproduces_b42_local_logits_and_topk_exactly():
    b42, b48 = _heads()
    b42.eval()
    b48.eval()
    data = _inputs()
    reference = b42(
        data["spatial"],
        data["present"],
        data["series_meta"],
        data["slice_position"],
    )
    candidate = b48.forward_details(**data)
    torch.testing.assert_close(candidate.local_logits, reference[0], rtol=0, atol=0)
    assert torch.equal(candidate.top_indices, reference[1])
    torch.testing.assert_close(candidate.top_values, reference[2], rtol=0, atol=0)
    assert torch.count_nonzero(candidate.context_abs_mean).item() == 0


def test_global_query_is_stopped_before_the_local_head():
    _, head = _heads()
    head.train()
    data = _inputs()
    data["global_query"] = data["global_query"].requires_grad_(True)
    with torch.no_grad():
        head.context_gate.fill_(0.25)
    out = head.forward_details(**data)
    probe = torch.linspace(-1.0, 1.0, N_TARGETS)
    (out.local_logits * probe[None, :]).sum().backward()
    assert data["global_query"].grad is None
    assert head.context_gate.grad is not None
    assert torch.count_nonzero(head.context_gate.grad).item() > 0
    assert head.context_query.weight.grad is not None
    assert torch.count_nonzero(head.context_query.weight.grad).item() > 0
    assert head.context_key.weight.grad is not None
    assert torch.count_nonzero(head.context_key.weight.grad).item() > 0


def test_local_aux_path_opens_gate_before_low_rank_projections_receive_gradient():
    _, head = _heads()
    head.train()
    data = _inputs()
    output = head.forward_details(**data)
    target = torch.arange(N_TARGETS, dtype=torch.float32).remainder(2).unsqueeze(0)
    F.binary_cross_entropy_with_logits(output.local_logits, target).backward()
    assert head.context_gate.grad is not None
    assert torch.count_nonzero(head.context_gate.grad).item() > 0
    # The residual is zero exactly at initialization, so the context projections
    # correctly wait until the gate has taken one optimizer step.
    assert head.context_query.weight.grad is not None
    assert torch.count_nonzero(head.context_query.weight.grad).item() == 0
    assert head.context_key.weight.grad is not None
    assert torch.count_nonzero(head.context_key.weight.grad).item() == 0


def test_nonzero_global_context_changes_local_evidence_without_removing_masks():
    _, head = _heads()
    head.eval()
    data = _inputs()
    with torch.no_grad():
        head.context_gate.fill_(0.5)
    first = head.forward_details(**data, audit_context=True)
    second_query = -data["global_query"]
    second = head.forward_details(
        data["spatial"],
        data["present"],
        data["series_meta"],
        data["slice_position"],
        second_query,
        audit_context=True,
    )
    assert not torch.allclose(first.local_logits, second.local_logits)
    assert first.topk_overlap_with_static is not None
    assert torch.all(first.topk_overlap_with_static >= 0)
    assert torch.all(first.topk_overlap_with_static <= 1)

    missing = dict(data)
    missing["present"] = data["present"].clone()
    missing["present"][:, 1] = 0
    missing["spatial"] = data["spatial"].clone()
    missing["spatial"][:, 1] = 10_000.0
    masked = head.forward_details(**missing)
    # The missing series occupies the second contiguous token block.
    first_block = data["spatial"].shape[2] * data["spatial"].shape[3]
    assert torch.all(masked.top_indices < first_block)


class _IdentityContext(nn.Module):
    def forward(self, x, **_kwargs):
        return x


class _TinyB34(nn.Module):
    """Only the frozen B34 hierarchy pieces used by query extraction."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.plane_embedding = nn.Embedding(4, dim)
        self.fluid_embedding = nn.Embedding(3, dim)
        self.fat_embedding = nn.Embedding(3, dim)
        self.slice_position = nn.Parameter(torch.randn(16, dim))
        self.pathology_tokens = nn.Parameter(torch.randn(N_TARGETS, dim))
        self.pathology_context = _IdentityContext()
        self.context = _IdentityContext()
        self.cross_attention = nn.MultiheadAttention(dim, 2, batch_first=True)
        self.query_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.0)
        self.target_weight = nn.Parameter(torch.randn(N_TARGETS, dim))
        self.target_bias = nn.Parameter(torch.randn(N_TARGETS))

    @staticmethod
    def _pool_real_series_b31(x, present):
        # A deterministic compatible stand-in for B34's one token per real
        # series pooling; masking is performed by the caller exactly as B34.
        return x.mean(dim=2)


def test_post_attention_query_reconstructs_the_unchanged_b42_base_logit():
    torch.manual_seed(99)
    base = _TinyB34().eval()
    fake = SimpleNamespace(base=base)
    global_feature = torch.randn(1, 3, 32, 8, requires_grad=True)
    present = torch.tensor([[1.0, 1.0, 0.0]])
    meta = torch.tensor([[[1, 1, 1], [2, 2, 2], [0, 0, 0]]], dtype=torch.long)
    static, post, reconstructed = B48GlobalConditionedSparseMILResidual._global_query_states(
        fake,
        global_feature,
        present,
        meta,
    )
    expected = B37HighResSparseMILResidual._base_logits_from_global(
        fake,
        global_feature,
        present,
        meta,
    )
    torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)
    assert static.shape == post.shape == (1, N_TARGETS, 8)
    assert not static.requires_grad and not post.requires_grad
    assert not torch.allclose(static, post)


def test_context_reconstruction_accepts_an_unbatched_real_study_metadata_item():
    """The preflight supplies one dataset item, not a collated batch."""
    torch.manual_seed(99)
    base = _TinyB34().eval()
    fake = SimpleNamespace(base=base)
    fake._base_logits_from_global = lambda global_feature, present, meta: (
        B37HighResSparseMILResidual._base_logits_from_global(
            fake, global_feature, present, meta
        )
    )
    fake._global_query_states = lambda global_feature, present, meta: (
        B48GlobalConditionedSparseMILResidual._global_query_states(
            fake, global_feature, present, meta
        )
    )
    global_feature = torch.randn(1, 3, 32, 8)
    present = torch.tensor([1.0, 1.0, 0.0])
    meta = torch.tensor([[1, 1, 1], [2, 2, 2], [0, 0, 0]], dtype=torch.long)

    error = B48GlobalConditionedSparseMILResidual.context_reconstruction_error(
        fake,
        global_feature,
        present,
        meta,
    )
    assert error == 0.0


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
        "b48_context_dim": B48_CONTEXT_DIM,
        "b48_fixed_epochs": 2,
        "b48_context_metric": "cosine_low_rank_query_token_compatibility",
        "b48_context_eps": 1e-6,
        "b48_context_gate_init": "zero_tanh_targetwise",
        "b48_context_query_gradient": "detached_before_local_head",
        "b48_supervision": "report_only_weak_no_official_gold",
        "b48_checkpoint_selection": "none_fixed_epoch_2",
        "b48_validation_surface": "frozen_scanner_grouped_domain_split_v1",
    }


def test_b48_contract_has_two_distinct_matched_arms_and_rejects_drift():
    config = _config()
    for arm in B48_ARMS:
        state = b48_state(arm)
        resolved = require_b48_contract(config, arm=arm)
        assert resolved["arm"] == arm
        assert state["arm"] == arm
        assert state["context_dim"] == B48_CONTEXT_DIM
    assert B48_STATIC_PRIOR_CONTROL != B48_POST_CROSS_ATTENTION_CANDIDATE
    # Matched on the frozen key rather than on prose. The guard is a generic
    # loop that reports "B48 freezes <key>=<frozen>; got <value>", the same
    # wording B37 and B42 use, and the original expectation of "context
    # dimension" never matched it.
    with pytest.raises(ValueError, match="B48 freezes b48_context_dim"):
        require_b48_contract({**config, "b48_context_dim": 64}, arm=B48_STATIC_PRIOR_CONTROL)
    with pytest.raises(ValueError, match="B48 freezes b48_fixed_epochs"):
        require_b48_contract({**config, "b48_fixed_epochs": 3}, arm=B48_STATIC_PRIOR_CONTROL)
    with pytest.raises(ValueError, match="B42 freezes"):
        require_b48_contract({**config, "b42_reference_area": 100_000}, arm=B48_STATIC_PRIOR_CONTROL)


def test_fill_artifact_fingerprints_are_complete_and_content_sensitive(tmp_path):
    contents = {
        "training_targets.csv": "StudyInstanceUID\nuid-a\n",
        "policy.json": '{"version":"fill-only"}\n',
        "audit.json": '{"base_cells_overridden":0}\n',
    }
    for name, value in contents.items():
        (tmp_path / name).write_text(value, encoding="utf-8")
    observed = b48_fill_artifacts(tmp_path)
    expected = {
        name: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for name, value in contents.items()
    }
    assert observed == expected
    (tmp_path / "audit.json").write_text('{"base_cells_overridden":1}\n', encoding="utf-8")
    assert b48_fill_artifacts(tmp_path) != expected


def test_paired_checkpoint_identity_rejects_mixed_arms_or_seeds():
    identity = {
        "seed": 2026,
        "config_sha256": "config",
        "base_checkpoint_sha256": "base",
        "training_uids_sha256": "uids",
        "target_balance_multiplier": {"ACL": 1.0},
        "domain_split_sha256": "domain",
        "domain_rows_sha256": "rows",
        "fill_artifacts": {"training_targets.csv": "labels"},
        "series_policy_signature": "series",
        "source_sha256": {"model": "model", "training": "training"},
    }

    def checkpoint(arm, pair_identity):
        return {
            "arm": arm,
            "seed": 2026,
            "config_sha256": "config",
            "training_uids_sha256": "uids",
            "base_checkpoint_sha256": "base",
            "matched_pair_identity": pair_identity,
        }

    control = checkpoint(B48_STATIC_PRIOR_CONTROL, dict(identity))
    candidate = checkpoint(B48_POST_CROSS_ATTENTION_CANDIDATE, dict(identity))
    assert _matched_pair_seed(control, candidate) == 2026

    candidate["matched_pair_identity"] = {**identity, "domain_rows_sha256": "other"}
    with pytest.raises(ValueError, match="frozen pair identity"):
        _matched_pair_seed(control, candidate)


def test_weak_auc_surface_binarizes_soft_targets_at_the_frozen_boundary():
    targets = np.array([[0.85, 0.05], [0.05, 0.85], [0.50, 0.05]], dtype=np.float32)
    weights = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    observed = _masked_target_matrix(targets, weights)
    np.testing.assert_array_equal(observed[:2], np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert observed[2, 0] == 0.0
    assert np.isnan(observed[2, 1])
