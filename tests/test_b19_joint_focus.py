from __future__ import annotations

import numpy as np
import pytest
import torch

from rsna_knee.joint_focus import (
    DEFAULT_JOINT_FOCUS_POLICY,
    apply_joint_focus,
    joint_focus_mask,
    validate_joint_focus_policy,
)
from rsna_knee.b19_joint_focus import b19_joint_focus_policy


def test_joint_focus_mask_preserves_center_and_zeroes_extreme_edges():
    mask = joint_focus_mask(224, 224, DEFAULT_JOINT_FOCUS_POLICY)
    assert mask.shape == (224, 224)
    assert torch.isclose(mask[112, 112], torch.tensor(1.0), atol=1e-6)
    assert float(mask[0, 112]) == 0.0
    assert float(mask[112, 0]) == 0.0
    assert float(mask[-1, 112]) == 0.0
    assert float(mask[112, -1]) == 0.0
    assert float(mask[0, 0]) == 0.0
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0


def test_joint_focus_transform_keeps_shape_and_suppresses_edges():
    x = torch.ones(16, 3, 224, 224)
    y = apply_joint_focus(x, DEFAULT_JOINT_FOCUS_POLICY)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.allclose(y[:, :, 112, 112], torch.ones(16, 3), atol=1e-6)
    assert torch.count_nonzero(y[:, :, 0, :]) == 0
    assert torch.count_nonzero(y[:, :, :, 0]) == 0
    assert float(y.mean()) < 1.0
    assert float(y.mean()) > 0.4


def test_joint_focus_supports_tta_shape():
    x = torch.rand(3, 5, 16, 3, 224, 224)
    y = apply_joint_focus(x, DEFAULT_JOINT_FOCUS_POLICY)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.count_nonzero(y[..., 0, :]) == 0
    assert torch.count_nonzero(y[..., :, 0]) == 0


def test_joint_focus_policy_is_frozen_for_b19_v1():
    config = {
        "b19_joint_focus_enabled": True,
        "b19_joint_focus_version": "joint_focus_center_crop_cosine_v1",
        "b19_joint_focus_crop_fraction": 0.90,
        "b19_joint_focus_full_weight_fraction": 0.72,
        "b19_joint_focus_outer_zero_fraction": 0.90,
    }
    policy = b19_joint_focus_policy(config)
    assert policy == validate_joint_focus_policy(DEFAULT_JOINT_FOCUS_POLICY)

    changed = dict(config)
    changed["b19_joint_focus_crop_fraction"] = 0.85
    with pytest.raises(ValueError, match="freezes crop_fraction"):
        b19_joint_focus_policy(changed)


def test_joint_focus_crop_changes_spatial_content_not_only_border_values():
    # Horizontal ramp makes the centered 90% crop/rescale observable.
    ramp = torch.linspace(0.0, 1.0, 224).view(1, 1, 1, 224).expand(1, 3, 224, 224)
    y = apply_joint_focus(ramp, DEFAULT_JOINT_FOCUS_POLICY)
    # The crop removes the original extremes before resizing, so the first
    # non-zero interior values should no longer correspond to the raw 0/1 ends.
    center_row = y[0, 0, 112]
    positive = center_row[center_row > 0]
    assert positive.numel() > 0
    assert float(positive.max()) < 1.0
    assert float(positive.min()) > 0.0
