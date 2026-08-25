"""What the model is shown at test time must be what it was trained on.

B42 has two preprocessing paths. Training builds each view separately through
`preprocess_dense_triplets_b42`. The Kaggle submission uses
`preprocess_three_offsets_b42_normalize_once`, which normalises the native
volume once and then builds all three test-time views from it, because
normalising the same volume three times is wasted work under a runtime budget.

They share their primitives -- the same normaliser, the same native crop, the
same constant-area resize, in the same order -- so today they agree. Nothing
enforces that. If they ever drift, the model is fed different images at test
time than it saw in training, the score falls, and **no error is raised**: both
paths still return well-formed tensors of the right shape. That failure is
invisible in a training log and invisible in a submission manifest.

There are already equivalence tests tying the streaming views to the B39 and
B41 normalise-once helpers. This closes the remaining link, and it is the one
that matters most, because it is the only one that crosses from training to
inference rather than between two inference implementations.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rsna_knee.b37_highres_sparse_mil import B37_CROP_FRACTION
from rsna_knee.b42_constant_area_aspect_sparse_mil import (
    b42_preprocessing_state,
    preprocess_dense_triplets_b42,
)
from rsna_knee.b42_kaggle_fast_preprocess import (
    B42_FAST_TTA_OFFSETS,
    preprocess_three_offsets_b42_normalize_once,
)


def _raw(seed: int = 31, shape: tuple[int, int, int] = (47, 73, 121)) -> np.ndarray:
    """Deliberately rectangular and non-power-of-two, like the real series."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float32)


def test_the_submission_builds_exactly_what_training_built():
    """Bit-for-bit, not merely close."""
    raw = _raw()
    submission_images, submission_positions = preprocess_three_offsets_b42_normalize_once(
        raw,
        gap=1,
        crop_fraction=B37_CROP_FRACTION,
    )
    assert submission_images.shape[0] == len(B42_FAST_TTA_OFFSETS)

    for view, offset in enumerate(B42_FAST_TTA_OFFSETS):
        training_image, training_position = preprocess_dense_triplets_b42(
            raw,
            gap=1,
            center_offset=int(offset),
            crop_fraction=B37_CROP_FRACTION,
        )
        assert torch.equal(training_image, submission_images[view]), (
            f"view {offset} differs between the training and submission paths"
        )
        assert torch.equal(
            torch.from_numpy(training_position), submission_positions[view]
        )


@pytest.mark.parametrize(
    "shape",
    [
        (47, 73, 121),  # rectangular, taller than wide
        (47, 121, 73),  # rectangular, wider than tall
        (47, 96, 96),  # square
        (5, 64, 64),  # fewer slices than centres, so indices clamp
        (400, 48, 48),  # the long-series tail
    ],
)
def test_the_two_paths_agree_on_every_series_shape(shape):
    """Including the shapes where clamping and aspect handling actually bite."""
    raw = _raw(seed=7, shape=shape)
    submission_images, _ = preprocess_three_offsets_b42_normalize_once(
        raw, gap=1, crop_fraction=B37_CROP_FRACTION
    )
    for view, offset in enumerate(B42_FAST_TTA_OFFSETS):
        training_image, _ = preprocess_dense_triplets_b42(
            raw, gap=1, center_offset=int(offset), crop_fraction=B37_CROP_FRACTION
        )
        assert torch.equal(training_image, submission_images[view])


def test_the_three_test_time_views_are_genuinely_different():
    """A TTA whose views coincide is a no-op dressed up as an ensemble."""
    raw = _raw(seed=11)
    images, _ = preprocess_three_offsets_b42_normalize_once(
        raw, gap=1, crop_fraction=B37_CROP_FRACTION
    )
    assert not torch.equal(images[0], images[1])
    assert not torch.equal(images[1], images[2])


def test_the_offsets_the_submission_uses_are_the_frozen_ones():
    assert B42_FAST_TTA_OFFSETS == (-1, 0, 1)


def test_the_recorded_preprocessing_description_still_matches_the_code():
    """The manifest asserts this description; nothing else checks it.

    `b42_preprocessing_state()` is written into every submission manifest but
    never verified against behaviour, so it can outlive the code it describes.
    These are the claims that can be checked cheaply from constants.
    """
    state = b42_preprocessing_state()
    assert state["crop_fraction"] == B37_CROP_FRACTION
    assert state["crop_stage"] == "native resolution before deterministic resize"
    assert state["normalization"] == "full native volume before crop"
    assert state["deterministic_resize_count"] == 1
    assert state["preserves_in_plane_aspect_ratio"] is True
    assert state["padding"]["square_padding"] is False
