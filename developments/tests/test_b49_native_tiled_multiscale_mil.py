from __future__ import annotations

import numpy as np
import pytest
import torch

from rsna_knee.b49_native_tiled_multiscale_mil import (
    B49NativeTiledMILHead,
    coordinate_basis,
    native_tile_layout,
    native_tile_triplet_chunks,
    require_b49_contract,
    tile_feature_coordinates,
    verify_native_tile_coverage,
)


@pytest.mark.parametrize(
    ("height", "width", "expected_tiles"),
    (
        (320, 300, 1),
        (512, 512, 1),
        (640, 640, 1),
        (1024, 1024, 4),
        (640, 1280, 3),
    ),
)
def test_b49_native_tiles_cover_every_pixel_once(height: int, width: int, expected_tiles: int):
    layout = native_tile_layout(height, width)
    assert len(layout) == expected_tiles
    verify_native_tile_coverage(height, width, layout)


def test_b49_small_native_source_is_reflection_padded_without_resize():
    # Four simple source frames make every triplet channel independently
    # identifiable without allocating a 32-frame 640px test volume.
    raw = np.stack(
        [np.full((512, 512), float(index), dtype=np.float32) for index in range(4)], axis=0
    )
    centres = np.ones(32, dtype=np.int64)
    layout = native_tile_layout(512, 512)
    tile, chosen, batch = next(
        native_tile_triplet_chunks(raw, centres, layout, gap=1, chunk_size=2)
    )
    assert tile.y0 == -64 and tile.x0 == -64
    assert chosen.tolist() == [0, 1]
    assert tuple(batch.shape) == (2, 3, 640, 640)
    # The central 512x512 area is an exact source crop.  The surrounding values
    # are reflected context only; no interpolated source values are introduced.
    expected = torch.from_numpy(raw[np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64)])
    assert torch.equal(batch[:, :, 64:576, 64:576], expected)


def test_b49_feature_coordinates_mask_padding_and_overlap_ownership():
    layout = native_tile_layout(512, 512)
    coordinates, valid = tile_feature_coordinates(
        layout[0],
        native_height=512,
        native_width=512,
        feature_height=20,
        feature_width=20,
    )
    assert tuple(coordinates.shape) == (400, 2)
    # A centred 512-pixel image inside one 640 tile has a 16x16 valid lattice.
    assert int(valid.sum().item()) == 16 * 16
    assert torch.all((coordinates >= 0) & (coordinates <= 1))
    assert tuple(coordinate_basis(coordinates).shape) == (400, 12)


def test_b49_zero_context_gate_exactly_preserves_static_scores():
    torch.manual_seed(8)
    head = B49NativeTiledMILHead(dim=16, top_k=2, temperature=1.0, context_dim=4)
    head.eval()
    fmap = torch.randn(2, 16, 2, 2)
    positions = torch.tensor([0.25, 0.75])
    meta = torch.tensor([1, 2, 1], dtype=torch.long)
    coordinates = torch.tensor([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    valid = torch.ones(4, dtype=torch.bool)
    query = torch.randn(1, 12, 16)
    static, conditioned, returned_valid = head.score_tile_features(
        fmap,
        slice_positions=positions,
        series_meta=meta,
        coordinates=coordinates,
        coordinate_valid=valid,
        global_query=query,
    )
    assert torch.equal(returned_valid, valid.repeat(2))
    assert torch.equal(static, conditioned)


def test_b49_online_pool_is_exact_global_topk():
    head = B49NativeTiledMILHead(dim=8, top_k=3, temperature=1.0, context_dim=4)
    first = torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4)
    second = torch.arange(12 * 5, dtype=torch.float32).reshape(12, 5) + 100.0
    pool = head.update_pool(head.empty_pool(), first, torch.arange(4))
    pool = head.update_pool(pool, second, torch.arange(4, 9))
    expected_values, expected_index = torch.topk(torch.cat((first, second), dim=-1), k=3, dim=-1)
    expected_ids = torch.gather(torch.arange(9)[None, :].expand(12, -1), 1, expected_index)
    assert torch.equal(pool.values, expected_values)
    assert torch.equal(pool.ids, expected_ids)


def test_b49_contract_refuses_a_local_resize_policy():
    with pytest.raises(ValueError, match="b49_local_preprocessing"):
        require_b49_contract(
            {"b49_local_preprocessing": "constant_area_resize"},
            arm="static_prior_control",
        )
