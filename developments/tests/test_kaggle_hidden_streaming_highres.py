import numpy as np
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
from rsna_knee.kaggle_hidden_streaming_highres import (
    normalized_view_b39,
    normalized_view_b41,
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
