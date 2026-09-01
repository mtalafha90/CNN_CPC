import numpy as np
import pytest
import torch

from rsna_knee.b39_kaggle_fast_preprocess import (
    B39_FAST_TTA_OFFSETS,
    preprocess_five_offsets_b39_normalize_once,
)
from rsna_knee.b41_kaggle_fast_preprocess import (
    B41_FAST_TTA_OFFSETS,
    preprocess_three_offsets_b41_normalize_once,
)
from rsna_knee.dicom import _normalise_volume
from rsna_knee.b42_kaggle_fast_preprocess import (
    B42_FAST_TTA_OFFSETS,
    preprocess_three_offsets_b42_normalize_once,
)
from rsna_knee.kaggle_hidden_streaming_highres import (
    normalized_view_b39,
    normalized_view_b41,
    normalized_view_b42,
)


def _raw(seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Deliberately rectangular and non-power-of-two so both crop/resize paths are
    # exercised rather than only a trivial square input.
    return rng.normal(size=(47, 73, 121)).astype(np.float32)


def test_b39_streamed_views_exactly_match_existing_normalize_once_helper():
    raw = _raw(17)
    historical_images, historical_positions = preprocess_five_offsets_b39_normalize_once(
        raw,
        gap=1,
        crop_fraction=0.90,
    )
    normalized = _normalise_volume(raw)
    assert historical_images.shape[0] == len(B39_FAST_TTA_OFFSETS)
    for view, offset in enumerate(B39_FAST_TTA_OFFSETS):
        image, position = normalized_view_b39(
            normalized,
            gap=1,
            center_offset=offset,
            crop_fraction=0.90,
        )
        assert torch.equal(image, historical_images[view])
        assert torch.equal(position, historical_positions[view])


def test_b41_streamed_views_exactly_match_existing_normalize_once_helper():
    raw = _raw(23)
    historical_images, historical_positions = preprocess_three_offsets_b41_normalize_once(
        raw,
        gap=1,
        crop_fraction=0.90,
    )
    normalized = _normalise_volume(raw)
    assert historical_images.shape[0] == len(B41_FAST_TTA_OFFSETS)
    for view, offset in enumerate(B41_FAST_TTA_OFFSETS):
        image, position = normalized_view_b41(
            normalized,
            gap=1,
            center_offset=offset,
            crop_fraction=0.90,
        )
        assert torch.equal(image, historical_images[view])
        assert torch.equal(position, historical_positions[view])


def test_streamed_helpers_materialize_one_view_not_an_all_tta_tensor():
    normalized = _normalise_volume(_raw(29))
    b39_image, b39_position = normalized_view_b39(
        normalized,
        gap=1,
        center_offset=0,
        crop_fraction=0.90,
    )
    b41_image, b41_position = normalized_view_b41(
        normalized,
        gap=1,
        center_offset=0,
        crop_fraction=0.90,
    )
    assert b39_image.shape == (32, 3, 448, 448)
    assert b41_image.shape == (32, 3, 448, 448)
    assert b39_position.shape == (32,)
    assert b41_position.shape == (32,)


def test_b42_streamed_views_exactly_match_existing_normalize_once_helper():
    """The gate on submitting B42-family endpoints through the streaming path.

    B42 is the ragged one: constant pixel area, native aspect, stride-padded. A
    resize that differed by a rounding step would still produce a plausible
    submission, so this compares tensors with torch.equal, not a tolerance.
    """
    raw = _raw(23)
    historical_images, historical_positions = preprocess_three_offsets_b42_normalize_once(
        raw,
        gap=1,
        crop_fraction=0.90,
    )
    normalized = _normalise_volume(raw)
    assert historical_images.shape[0] == len(B42_FAST_TTA_OFFSETS)
    for view, offset in enumerate(B42_FAST_TTA_OFFSETS):
        image, position = normalized_view_b42(
            normalized,
            gap=1,
            center_offset=offset,
            crop_fraction=0.90,
        )
        assert torch.equal(image, historical_images[view]), f"offset {offset} differs"
        assert torch.equal(position, historical_positions[view])


def test_the_b42_streamed_view_is_rectangular_and_not_padded_to_a_square():
    """If it ever came back square, it would be B39/B41 geometry, not B42's."""
    normalized = _normalise_volume(_raw(23))
    image, _ = normalized_view_b42(normalized, gap=1, center_offset=0, crop_fraction=0.90)
    height, width = int(image.shape[-2]), int(image.shape[-1])
    assert height != width, "a 73x121 native volume must not resize to a square"
    assert height % 32 == 0 and width % 32 == 0, "stride alignment is part of the contract"


def test_a_zero_gap_is_refused_rather_than_silently_producing_a_view():
    normalized = _normalise_volume(_raw(23))
    with pytest.raises(ValueError, match="gap must be positive"):
        normalized_view_b42(normalized, gap=0, center_offset=0, crop_fraction=0.90)
