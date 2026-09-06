"""Did the spacing term learn nothing, or learn something too small to matter?

The claim this module rests on is that **spread**, not size, is what can carry
information: a constant added to every series is a bias the hierarchy absorbs.
So the tests that matter check the verdict follows the spread ratio even when
the mean is large, and that an untrained weight reads as never really tested.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from rsna_knee.spacing_conditioning import SPACING_BASIS
from rsna_knee.spacing_conditioning_probe import (
    CONDITIONING_KEY,
    CORPUS_P05_MM,
    CORPUS_P95_MM,
    SPREAD_PRESENT,
    SPREAD_REAL,
    _report,
    _verdict,
    contributions,
    load_conditioning,
    metadata_scale,
    probe,
)

D_MODEL = 16


def _base_state(weight: torch.Tensor | None = None, embedding_scale: float = 1.0) -> dict:
    torch.manual_seed(0)
    state = {}
    for name, rows in (
        ("plane_embedding.weight", 4),
        ("fluid_embedding.weight", 3),
        ("fat_embedding.weight", 3),
    ):
        values = torch.randn(rows, D_MODEL) * embedding_scale
        values[0] = 0.0  # padding_idx, permanently zero
        state[name] = values
    if weight is not None:
        state[CONDITIONING_KEY] = weight
    return state


def _checkpoint(tmp_path, weight, *, embedding_scale: float = 1.0, name: str = "c.pt"):
    path = tmp_path / name
    torch.save({"base_state": _base_state(weight, embedding_scale)}, path)
    return path


# --- reading the weight out ----------------------------------------------------


def test_it_reads_the_trained_weight(tmp_path):
    weight = torch.randn(D_MODEL, SPACING_BASIS)
    loaded, state, scale = load_conditioning(_checkpoint(tmp_path, weight))

    assert torch.allclose(loaded, weight)
    assert "plane_embedding.weight" in state
    assert scale == 1.0, "a checkpoint with no recorded scale was trained at 1.0"


def test_the_recorded_scale_is_read_back(tmp_path):
    """It is not in the state dict, so it has to come from the audit."""
    path = tmp_path / "scaled.pt"
    torch.save(
        {
            "base_state": _base_state(torch.zeros(D_MODEL, SPACING_BASIS)),
            "spacing": {"conditioning_scale": 90.5},
        },
        path,
    )
    assert load_conditioning(path)[2] == pytest.approx(90.5)


def test_the_probe_applies_the_recorded_scale(tmp_path):
    """The ratio only means what it claims if the scale is included."""
    weight = torch.zeros(D_MODEL, SPACING_BASIS)
    weight[:, 0] = 0.001

    plain = tmp_path / "plain.pt"
    torch.save({"base_state": _base_state(weight)}, plain)
    scaled = tmp_path / "scaled.pt"
    torch.save(
        {"base_state": _base_state(weight), "spacing": {"conditioning_scale": 100.0}},
        scaled,
    )

    assert probe(scaled)["spread_p05_to_p95"] == pytest.approx(
        probe(plain)["spread_p05_to_p95"] * 100.0, rel=1e-5
    )


def test_a_checkpoint_without_the_conditioning_is_refused(tmp_path):
    with pytest.raises(ValueError, match="carries no"):
        load_conditioning(_checkpoint(tmp_path, None))


def test_a_checkpoint_with_no_base_state_is_refused(tmp_path):
    path = tmp_path / "empty.pt"
    torch.save({"nothing": 1}, path)
    with pytest.raises(ValueError, match="no base_state"):
        load_conditioning(path)


def test_a_wrong_basis_width_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unexpected conditioning shape"):
        load_conditioning(_checkpoint(tmp_path, torch.randn(D_MODEL, 3)))


# --- the yardstick -------------------------------------------------------------


def test_the_padding_row_is_left_out_of_the_metadata_scale():
    """Row 0 is permanently zero and would drag the mean down."""
    state = _base_state(torch.zeros(D_MODEL, SPACING_BASIS))
    scale = metadata_scale(state)

    plane = state["plane_embedding.weight"]
    assert scale["per_embedding"]["plane_embedding.weight"] == pytest.approx(
        float(plane[1:].norm(dim=-1).mean())
    )


def test_the_yardstick_is_the_sum_of_all_three():
    scale = metadata_scale(_base_state(torch.zeros(D_MODEL, SPACING_BASIS)))
    assert scale["typical_metadata_norm"] == pytest.approx(
        sum(scale["per_embedding"].values())
    )


def test_a_missing_embedding_is_refused():
    state = _base_state(torch.zeros(D_MODEL, SPACING_BASIS))
    del state["fat_embedding.weight"]
    with pytest.raises(ValueError, match="fat_embedding"):
        metadata_scale(state)


# --- spread, not size ----------------------------------------------------------


def test_an_untrained_weight_has_no_spread_at_all(tmp_path):
    """The zero case: nothing was learned, so nothing varies."""
    result = probe(_checkpoint(tmp_path, torch.zeros(D_MODEL, SPACING_BASIS)))

    assert result["weight_is_exactly_zero"] is True
    assert result["spread_p05_to_p95"] == pytest.approx(0.0)
    assert "not really tested" in result["verdict"]


def test_the_basis_has_no_constant_feature():
    """So a genuinely flat contribution is unreachable, which is a property of
    the design worth knowing: every one of the eight features moves with the
    spacing, so any non-zero weight varies at least a little."""
    thin, thick = contributions(
        torch.eye(SPACING_BASIS), (CORPUS_P05_MM, CORPUS_P95_MM)
    )
    moved = (thin - thick).abs()

    assert bool((moved > 1e-6).all()), moved.tolist()


def test_the_verdict_follows_the_spread_and_not_the_mean():
    """Driven through `_finish` with the two ratios put in different bands.

    Going through `probe` cannot test this: every basis feature varies with the
    spacing, so the mean and the spread ratio always land close enough together
    to fall in the same band, and the assertion would pass without meaning
    anything.
    """
    from rsna_knee.spacing_conditioning_probe import _finish

    result = _finish(
        {"spread_over_metadata": 0.001, "mean_over_metadata": 5.0}, None
    )
    assert "not really tested" in result["verdict"]
    assert "a real term" not in result["verdict"]


def test_a_large_average_does_not_rescue_a_flat_term(tmp_path):
    """The mean is reported for context, but it is not what is judged."""
    weight = torch.zeros(D_MODEL, SPACING_BASIS)
    weight[:, 0] = 100.0
    result = probe(_checkpoint(tmp_path, weight))

    assert result["mean_contribution_norm"] > result["spread_p05_to_p95"]
    assert result["verdict"] == _verdict(result["spread_over_metadata"])


def test_a_term_that_varies_with_spacing_shows_spread(tmp_path):
    weight = torch.zeros(D_MODEL, SPACING_BASIS)
    weight[:, 0] = 1.0  # basis feature 0 is the normalised log spacing
    result = probe(_checkpoint(tmp_path, weight))

    assert result["spread_p05_to_p95"] > 0.0


def test_the_spread_is_measured_across_the_corpus_range():
    weight = torch.zeros(D_MODEL, SPACING_BASIS)
    weight[:, 0] = 1.0
    thin, thick = contributions(weight, (CORPUS_P05_MM, CORPUS_P95_MM))

    assert not torch.allclose(thin, thick)
    assert CORPUS_P05_MM == 0.80
    assert CORPUS_P95_MM == 5.00


# --- the verdict bands ---------------------------------------------------------


def test_the_bands_are_ordered():
    assert SPREAD_REAL > SPREAD_PRESENT > 0


def test_a_large_ratio_reads_as_a_real_term():
    assert "a real term" in _verdict(SPREAD_REAL)
    assert "a real term" in _verdict(1.0)


def test_a_middling_ratio_reads_as_underpowered():
    assert "underpowered" in _verdict(0.05)
    assert "underpowered" in _verdict(SPREAD_PRESENT)


def test_a_tiny_ratio_reads_as_never_really_tested():
    assert "not really tested" in _verdict(0.0001)
    assert "not really tested" in _verdict(0.0)


def test_an_unmeasurable_ratio_is_named():
    assert _verdict(float("nan")) == "unmeasurable"


# --- output --------------------------------------------------------------------


def test_the_result_is_written_and_reports(tmp_path, capsys):
    weight = torch.zeros(D_MODEL, SPACING_BASIS)
    weight[:, 0] = 0.5
    out = tmp_path / "probe.json"
    result = probe(_checkpoint(tmp_path, weight), out_json=out)

    assert json.loads(out.read_text()) == result
    _report(result)
    printed = capsys.readouterr().out
    assert "the spread, not the size" in printed
    assert "typical plane+fluid+fat" in printed


def test_every_probed_spacing_is_reported(tmp_path):
    from rsna_knee.spacing_conditioning_probe import CORPUS_SPACINGS_MM

    result = probe(_checkpoint(tmp_path, torch.randn(D_MODEL, SPACING_BASIS)))
    assert len(result["contribution_norm_by_spacing_mm"]) == len(CORPUS_SPACINGS_MM)


def test_the_ratio_scales_with_the_yardstick(tmp_path):
    """Doubling the embeddings halves the ratio, since it is a comparison."""
    weight = torch.zeros(D_MODEL, SPACING_BASIS)
    weight[:, 0] = 1.0

    small = probe(_checkpoint(tmp_path, weight, embedding_scale=1.0, name="a.pt"))
    large = probe(_checkpoint(tmp_path, weight, embedding_scale=2.0, name="b.pt"))

    assert small["spread_p05_to_p95"] == pytest.approx(large["spread_p05_to_p95"])
    assert large["spread_over_metadata"] < small["spread_over_metadata"]


# --- it must agree with the module it probes -----------------------------------


def test_the_contribution_matches_what_the_model_actually_adds():
    """The probe reimplements nothing: same basis, same projection."""
    from rsna_knee.spacing_conditioning import SpacingConditioning

    torch.manual_seed(1)
    module = SpacingConditioning(D_MODEL)
    with torch.no_grad():
        module.projection.weight.normal_()

    spacings = torch.tensor([0.8, 3.3, 5.0])
    through_module = module(spacings)
    through_probe = contributions(module.projection.weight.detach(), spacings.tolist())

    assert torch.allclose(through_module, through_probe, atol=1e-6)


def test_a_disabled_module_is_irrelevant_to_the_probe():
    """The probe reads weights, not the runtime switch, which is what we want:
    the checkpoint's `enabled` flag says nothing about what was learned."""
    from rsna_knee.spacing_conditioning import SpacingConditioning

    module = SpacingConditioning(D_MODEL, enabled=False)
    with torch.no_grad():
        module.projection.weight.fill_(0.25)

    values = contributions(module.projection.weight.detach(), [3.3])
    assert float(values.norm()) > 0.0


class _Anchor(nn.Module):
    """Present only so the test module imports nn for a realistic shape check."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(SPACING_BASIS, D_MODEL, bias=False)


def test_the_key_matches_where_the_weight_actually_lives():
    anchor = _Anchor()
    assert "projection.weight" in dict(anchor.named_parameters())
    assert CONDITIONING_KEY.endswith("projection.weight")
