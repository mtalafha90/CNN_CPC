"""B47 scores evidence on the grid the encoder actually produced.

The first test states the fault B47 exists to fix, in numbers, so that if
anybody later changes the resolution or the backbone the arithmetic is checked
rather than remembered.
"""

from __future__ import annotations

import math

import pytest
import torch

from rsna_knee.b36_sparse_mil import B36SparseMILHead
from rsna_knee.b37_highres_sparse_mil import (
    B37_GRID_SIZE,
    B37_IMAGE_SIZE,
    B37_TEMPERATURE,
    B37_TOP_K,
)
from rsna_knee.b47_native_grid_sparse_mil import (
    B47_ARM_BUDGETS,
    B47_ARMS,
    B47_CONTROL_REGION_BUDGET,
    B47_ENCODER_STRIDE,
    B47_NATIVE_REGION_BUDGET,
    B47_REGION_BASIS,
    B47_SQUARE_REFERENCE_CELLS,
    B47NativeGridSparseMILHead,
    b47_state,
    grid_for_budget,
    region_basis,
    require_b47_contract,
)


# --- the fault being fixed ------------------------------------------------


def test_the_pooling_B47_removes_really_did_discard_most_of_the_map():
    """448 at stride 32 gives 14x14; B37/B42 scored evidence on 6x6."""
    native_side = B37_IMAGE_SIZE // B47_ENCODER_STRIDE
    assert native_side == 14
    native_cells = native_side * native_side
    assert native_cells == B47_SQUARE_REFERENCE_CELLS == 196
    assert B37_GRID_SIZE * B37_GRID_SIZE == B47_CONTROL_REGION_BUDGET == 36
    assert native_cells / B47_CONTROL_REGION_BUDGET == pytest.approx(5.44, abs=0.01)


def test_each_old_cell_covered_a_large_piece_of_the_knee():
    """A 6x6 cell spans a sixth of the crop in each direction."""
    pixels_per_cell = B37_IMAGE_SIZE / B37_GRID_SIZE
    assert pixels_per_cell == pytest.approx(74.7, abs=0.1)
    # At the audited median in-plane sampling of 0.3125 mm/pixel.
    assert pixels_per_cell * 0.3125 == pytest.approx(23.3, abs=0.5)


# --- the grid --------------------------------------------------------------


def test_a_map_inside_the_budget_is_kept_exactly_as_the_encoder_made_it():
    assert grid_for_budget(14, 14, B47_NATIVE_REGION_BUDGET) == (14, 14)
    assert grid_for_budget(10, 20, B47_NATIVE_REGION_BUDGET) == (10, 20)
    assert grid_for_budget(6, 40, B47_NATIVE_REGION_BUDGET) == (6, 40)
    assert grid_for_budget(6, 6, 36) == (6, 6)


def test_the_native_budget_covers_every_geometry_b42_can_produce():
    """The budget is measured, not guessed, and 196 would have been too small.

    B42 holds the anatomical area near 448^2 and then reflection-pads up to
    stride alignment, so a rectangular series carries padding a square one does
    not. If the budget were set to the square reference of 196, almost every
    non-square series would be pooled and the native arm would not be native.
    """
    from rsna_knee.b42_constant_area_aspect_sparse_mil import constant_area_shape

    counts = []
    for height in range(120, 1300, 40):
        for width in range(120, 1300, 40):
            shape = constant_area_shape(height, width)
            counts.append(
                (shape["aligned_height"] // 32) * (shape["aligned_width"] // 32)
            )

    assert min(counts) >= B47_SQUARE_REFERENCE_CELLS
    assert max(counts) <= B47_NATIVE_REGION_BUDGET
    assert max(counts) > B47_SQUARE_REFERENCE_CELLS, "196 alone would pool rectangles"

    # Nothing the geometry can produce gets pooled by the native arm.
    for height in range(120, 1300, 40):
        for width in range(120, 1300, 40):
            shape = constant_area_shape(height, width)
            cells = (shape["aligned_height"] // 32, shape["aligned_width"] // 32)
            assert grid_for_budget(*cells, B47_NATIVE_REGION_BUDGET) == cells


def test_a_map_above_the_budget_keeps_its_shape_while_shrinking():
    """The point of budgeting by cell count rather than by a fixed side."""
    for height, width in [(14, 14), (10, 20), (20, 10), (8, 32)]:
        grid_h, grid_w = grid_for_budget(height, width, 36)
        assert grid_h * grid_w <= 36
        assert grid_h <= height and grid_w <= width
        native_aspect = height / width
        grid_aspect = grid_h / grid_w
        # Rounding to whole cells cannot preserve the ratio exactly; it must not
        # squash a 1:4 acquisition onto a square the way a fixed grid does.
        assert grid_aspect == pytest.approx(native_aspect, rel=0.5)


def test_a_rectangle_is_not_forced_square():
    grid_h, grid_w = grid_for_budget(8, 32, 36)
    assert grid_h != grid_w, "a 1:4 map must not become a square grid"


def test_the_budget_is_never_exceeded():
    for height in range(1, 25):
        for width in range(1, 25):
            for budget in (8, 36, 100, 196):
                grid_h, grid_w = grid_for_budget(height, width, budget)
                assert 1 <= grid_h <= height
                assert 1 <= grid_w <= width
                assert grid_h * grid_w <= max(budget, 1)


@pytest.mark.parametrize("bad", [(0, 4), (4, 0), (-1, 4)])
def test_an_empty_feature_map_is_refused(bad):
    with pytest.raises(ValueError, match="positive extent"):
        grid_for_budget(bad[0], bad[1], 36)


def test_a_budget_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        grid_for_budget(14, 14, 0)


# --- how a cell says where it is ------------------------------------------


def test_the_region_basis_means_the_same_thing_at_any_grid():
    """The property a position-indexed lookup table cannot have."""
    coarse = region_basis(2, 2)
    fine = region_basis(6, 6)
    assert coarse.shape == (4, B47_REGION_BASIS)
    assert fine.shape == (36, B47_REGION_BASIS)
    # A 2x2 cell has its centre at 0.25 or 0.75; a 6x6 cell centre lands on
    # 0.25 at index 1 and 0.75 at index 4. Where the coordinates coincide the
    # description must too -- that is the property a lookup table cannot have.
    assert torch.allclose(coarse[0], fine[1 * 6 + 1], atol=1e-6)
    assert torch.allclose(coarse[3], fine[4 * 6 + 4], atol=1e-6)


def test_the_region_basis_is_ordered_row_major_like_the_feature_map():
    basis = region_basis(3, 5)
    assert basis.shape == (15, B47_REGION_BASIS)
    # First column is the normalised row coordinate, constant along a row.
    rows = basis[:, 0].reshape(3, 5)
    assert torch.allclose(rows[0], rows[0, 0].expand(5))
    assert rows[0, 0] < rows[1, 0] < rows[2, 0]


def test_a_rectangular_grid_normalises_each_axis_separately():
    basis = region_basis(2, 8)
    ys, xs = basis[:, 0], basis[:, 6]
    assert float(ys.min()) == pytest.approx(0.25)
    assert float(xs.min()) == pytest.approx(1 / 16)


# --- the head --------------------------------------------------------------


def _head(budget=36, dim=16, top_k=4):
    return B47NativeGridSparseMILHead(
        dim, region_budget=budget, top_k=top_k, temperature=B37_TEMPERATURE
    )


def test_the_region_encoding_starts_at_zero_like_the_table_it_replaces():
    """At step zero B47 asks the pretrained features the same question B36 does."""
    head = _head()
    assert torch.count_nonzero(head.region_projection.weight) == 0
    reference = B36SparseMILHead(16, grid_size=6, top_k=4, temperature=B37_TEMPERATURE)
    assert torch.count_nonzero(reference.region_embedding) == 0


def test_the_position_indexed_table_is_gone():
    assert not hasattr(_head(), "region_embedding")


def test_a_budget_below_top_k_is_refused():
    with pytest.raises(ValueError, match="at least top_k"):
        B47NativeGridSparseMILHead(16, region_budget=4, top_k=8)


def _inputs(b=1, k=2, s=3, r=36, d=16):
    torch.manual_seed(0)
    return {
        "spatial": torch.randn(b, k, s, r, d),
        "present": torch.ones(b, k),
        "series_meta": torch.zeros(b, k, 3, dtype=torch.long),
        "slice_position": torch.rand(b, k, s),
    }


def test_the_head_runs_and_returns_one_logit_per_target():
    head = _head()
    logits, indices, values = head(**_inputs())
    assert logits.shape == (1, 12)
    assert indices.shape == (1, 12, head.top_k)
    assert values.shape == (1, 12, head.top_k)
    assert torch.isfinite(logits).all()


def test_padding_cells_never_enter_the_evidence_pool():
    """A short series must not contribute padded tokens to the top-k.

    Set up so the leak would be unmissable rather than probabilistic. The head
    layer-normalises every token, so padding has to be attractive in *direction*
    rather than magnitude -- a constant-valued token is flattened to nothing,
    which is why simply making the padding large does not test anything.
    """
    head = _head(top_k=2, dim=16)
    data = _inputs(k=2, r=36, d=16)
    with torch.no_grad():
        # Only the first feature counts towards evidence.
        head.evidence_weight.zero_()
        head.evidence_weight[:, 0] = 1.0

    # Real cells carry their signal in feature 1, so their feature 0 normalises
    # to a low value. Series 1 has 4 real cells; the remaining 32 are padding
    # and carry their signal in feature 0, so they score far higher.
    data["spatial"][:] = 0.0
    data["spatial"][..., 1] = 1.0
    data["spatial"][:, 1, :, 4:, :] = 0.0
    data["spatial"][:, 1, :, 4:, 0] = 1.0

    valid = torch.ones(2, 36, dtype=torch.bool)
    valid[1, 4:] = False

    leaked, leaked_index, _ = head(**data)
    guarded, guarded_index, _ = head(**data, region_valid=valid)

    # Unmasked, the top-k is drawn entirely from the padded cells.
    leaked_regions = leaked_index[0, 0] % 36
    assert bool((leaked_regions >= 4).all()), "the setup must make padding win"
    guarded_regions = guarded_index[0, 0] % 36
    assert bool((guarded_regions < 36).all())
    assert not torch.allclose(leaked, guarded), "the mask must change the answer"

    # Masked, the answer must equal a study whose padding never carried anything.
    trimmed = {key: value.clone() for key, value in data.items()}
    trimmed["spatial"][:, 1, :, 4:, :] = 0.0
    valid_only = head(**trimmed, region_valid=valid)[0]
    assert torch.allclose(guarded, valid_only, atol=1e-5)


def test_an_absent_series_is_still_excluded():
    head = _head(top_k=2)
    data = _inputs(k=2)
    data["present"][0, 1] = 0.0
    data["spatial"][:, 1] = 50.0
    logits = head(**data)[0]
    assert torch.isfinite(logits).all()
    assert float(logits.detach().abs().max()) < 40.0


def test_a_study_with_nothing_readable_is_refused():
    head = _head()
    data = _inputs()
    data["present"][:] = 0.0
    with pytest.raises(RuntimeError, match="no readable MRI series"):
        head(**data)


def test_too_few_valid_cells_for_the_top_k_is_refused():
    head = _head(top_k=8)
    data = _inputs(k=1, s=1, r=36)
    valid = torch.zeros(1, 36, dtype=torch.bool)
    valid[0, :3] = True
    with pytest.raises(RuntimeError, match="fewer valid local tokens"):
        head(**data, region_valid=valid)


def test_more_regions_than_the_budget_is_refused():
    head = _head(budget=36)
    with pytest.raises(ValueError, match="above the frozen budget"):
        head(**_inputs(r=64))


def test_the_evidence_ranking_is_done_in_fp32():
    """bfloat16 orders tightly clustered scores arbitrarily; fp32 does not."""
    head = _head(top_k=2, dim=8)
    data = _inputs(k=1, s=1, r=36, d=8)
    with torch.no_grad():
        head.evidence_weight.zero_()
        head.evidence_weight[0, 0] = 1.0
    # Two cells differing by far less than bfloat16 can represent.
    data["spatial"][:] = 0.0
    data["spatial"][0, 0, 0, 5, 0] = 1.0
    data["spatial"][0, 0, 0, 9, 0] = 1.0 + 1e-3
    _, indices, _ = head(**data)
    assert int(indices[0, 0, 0]) == 9, "the larger score must rank first"


# --- the contract ----------------------------------------------------------


@pytest.fixture
def config():
    from model._implementation import read_config

    return read_config("config/b42_constant_area_aspect_sparse.yaml")


def test_the_native_arm_clears_the_inherited_b42_contract(config):
    resolved = require_b47_contract({**config, "b47_arm": "native"})
    assert resolved["region_budget"] == B47_NATIVE_REGION_BUDGET
    assert resolved["arm"] == "native"


def test_the_control_arm_reproduces_the_old_resolution(config):
    resolved = require_b47_contract({**config, "b47_arm": "control"})
    assert resolved["region_budget"] == B47_CONTROL_REGION_BUDGET == 36


def test_an_unknown_arm_is_refused(config):
    with pytest.raises(ValueError, match="B47 arm must be one of"):
        require_b47_contract({**config, "b47_arm": "bigger"})


def test_an_arm_cannot_quietly_change_its_own_budget(config):
    """Otherwise the control is not a control."""
    with pytest.raises(ValueError, match="freezes b47_region_budget"):
        require_b47_contract({**config, "b47_arm": "control", "b47_region_budget": 100})


def test_b47_does_not_loosen_anything_b42_froze(config):
    with pytest.raises(ValueError, match="B42 freezes b42_reference_area"):
        require_b47_contract(
            {**config, "b47_arm": "native", "b42_reference_area": 100_000}
        )


@pytest.mark.parametrize("key,value", [("b37_top_k", 16), ("b37_temperature", 2.0)])
def test_the_inherited_sparse_constants_stay_frozen(config, key, value):
    """B42's own contract catches these before B47's check is reached."""
    with pytest.raises(ValueError, match=f"freezes {key}"):
        require_b47_contract({**config, "b47_arm": "native", key: value})


def test_both_arms_are_declared_and_differ(config):
    assert B47_ARMS == ("control", "native")
    assert B47_ARM_BUDGETS["control"] != B47_ARM_BUDGETS["native"]
    for arm in B47_ARMS:
        state = b47_state(arm)
        assert state["arm"] == arm
        assert state["region_budget"] == B47_ARM_BUDGETS[arm]
        assert state["n_targets"] == 12
