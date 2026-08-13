from __future__ import annotations

import torch

from rsna_knee.crop_focus import (
    CROP_FOCUS_VERSION,
    DEFAULT_CROP_FOCUS_POLICY,
    apply_crop_focus,
    validate_crop_focus_policy,
)


def test_b20_crop_focus_policy_is_frozen_90_percent_crop():
    policy = validate_crop_focus_policy(DEFAULT_CROP_FOCUS_POLICY)
    assert policy == {
        "version": CROP_FOCUS_VERSION,
        "crop_fraction": 0.90,
    }


def test_b20_crop_focus_preserves_shape_and_has_no_artificial_dark_border():
    x = torch.ones(2, 16, 3, 224, 224)
    y = apply_crop_focus(x, DEFAULT_CROP_FOCUS_POLICY)
    assert y.shape == x.shape
    # A crop+resize of a constant image remains constant. This specifically
    # guards against reintroducing B19's multiplicative cosine/vignette mask.
    assert torch.allclose(y, torch.ones_like(y), atol=0.0, rtol=0.0)
    assert torch.all(y[..., 0, :] == 1)
    assert torch.all(y[..., -1, :] == 1)
    assert torch.all(y[..., :, 0] == 1)
    assert torch.all(y[..., :, -1] == 1)


def test_b20_crop_focus_zoom_changes_nonconstant_image_without_black_frame():
    yy = torch.linspace(0, 1, 224).view(1, 1, 224, 1)
    xx = torch.linspace(0, 1, 224).view(1, 1, 1, 224)
    x = (yy + xx).expand(1, 3, 224, 224).clone() / 2
    y = apply_crop_focus(x, DEFAULT_CROP_FOCUS_POLICY)
    assert y.shape == x.shape
    assert not torch.allclose(x, y)
    # The resized crop carries actual image signal to every output edge rather
    # than forcing a synthetic zero-valued frame.
    assert float(y[..., 0, :].mean()) > 0.0
    assert float(y[..., -1, :].mean()) > 0.0
    assert float(y[..., :, 0].mean()) > 0.0
    assert float(y[..., :, -1].mean()) > 0.0
