"""B54: the plumbing from a measured spacing to the model's metadata sum.

Two contracts carry this module. First, with nothing installed
`spacing_metadata` must equal the frozen `plane + fluid + fat` expression
exactly — otherwise switching B54 on changes the model twice over. Second, a
series with no measurable spacing, and every padded slot, must contribute
exactly zero rather than a stand-in value.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from rsna_knee.b54_spacing_run import (
    B54_VERSION,
    SPACING_KEY,
    attach_spacing,
    b54_state,
    collate_b54,
    install_spacing_conditioning,
    preflight,
    set_spacing_enabled,
    spacing_metadata,
    spacing_summary,
)
from rsna_knee.spacing_conditioning import SpacingConditioning


class _MetaModule(nn.Module):
    """The three embeddings every model in the lineage carries."""

    def __init__(self, d: int = 8):
        super().__init__()
        self.plane_embedding = nn.Embedding(4, d, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, d, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, d, padding_idx=0)
        for embedding in (
            self.plane_embedding,
            self.fluid_embedding,
            self.fat_embedding,
        ):
            nn.init.normal_(embedding.weight)


def _meta(batch: int = 2, series: int = 3) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.stack(
        [
            torch.randint(0, 4, (batch, series)),
            torch.randint(0, 3, (batch, series)),
            torch.randint(0, 3, (batch, series)),
        ],
        dim=-1,
    )


def _records(**spacings) -> dict[str, list[dict]]:
    return {
        study: [
            {"series_uid": uid, SPACING_KEY: value} for uid, value in entries.items()
        ]
        for study, entries in spacings.items()
    }


# --- the frozen expression must not move --------------------------------------


def test_with_nothing_installed_it_is_the_frozen_sum():
    module, meta = _MetaModule(), _meta()
    expected = (
        module.plane_embedding(meta[:, :, 0])
        + module.fluid_embedding(meta[:, :, 1])
        + module.fat_embedding(meta[:, :, 2])
    )
    assert torch.allclose(spacing_metadata(module, meta), expected)


def test_supplying_a_spacing_without_installing_changes_nothing():
    module, meta = _MetaModule(), _meta()
    spacing = torch.full((2, 3), 3.3)

    assert torch.allclose(
        spacing_metadata(module, meta, spacing), spacing_metadata(module, meta)
    )


def test_a_freshly_installed_conditioning_also_changes_nothing():
    """Zero-initialised: installing it cannot move the starting point."""
    module, meta = _MetaModule(), _meta()
    before = spacing_metadata(module, meta)
    install_spacing_conditioning(module)

    assert torch.allclose(
        spacing_metadata(module, meta, torch.full((2, 3), 3.3)), before
    )


def test_a_trained_conditioning_does_change_it():
    module, meta = _MetaModule(), _meta()
    conditioning = install_spacing_conditioning(module)
    with torch.no_grad():
        conditioning.projection.weight.normal_()

    assert not torch.allclose(
        spacing_metadata(module, meta, torch.full((2, 3), 3.3)),
        spacing_metadata(module, meta),
    )


# --- installation -------------------------------------------------------------


def test_the_width_is_taken_from_the_embeddings():
    module = _MetaModule(d=16)
    assert install_spacing_conditioning(module).d_model == 16


def test_a_module_without_the_embeddings_is_refused():
    with pytest.raises(ValueError, match="not one of the modules"):
        install_spacing_conditioning(nn.Linear(2, 2))


def test_the_refusal_names_what_is_missing():
    module = _MetaModule()
    del module.fluid_embedding
    with pytest.raises(ValueError, match="fluid_embedding"):
        install_spacing_conditioning(module)


def test_it_can_be_installed_at_more_than_one_site():
    """The base and the sparse head both sum metadata; both may be conditioned."""
    model = nn.Module()
    model.base = _MetaModule()
    model.head = _MetaModule()
    install_spacing_conditioning(model.base)
    install_spacing_conditioning(model.head)

    assert b54_state(model)["conditioning_sites"] == 2


# --- the free ablation --------------------------------------------------------


def test_switching_every_site_off_is_one_call():
    model = nn.Module()
    model.base, model.head = _MetaModule(), _MetaModule()
    install_spacing_conditioning(model.base)
    install_spacing_conditioning(model.head)

    assert set_spacing_enabled(model, False) == 2
    assert b54_state(model)["conditioning_enabled"] == [False, False]


def test_the_ablated_model_reproduces_the_frozen_sum():
    """One checkpoint, both arms."""
    module, meta = _MetaModule(), _meta()
    conditioning = install_spacing_conditioning(module)
    with torch.no_grad():
        conditioning.projection.weight.normal_()
    spacing = torch.full((2, 3), 3.3)

    set_spacing_enabled(module, False)
    assert torch.allclose(
        spacing_metadata(module, meta, spacing), spacing_metadata(module, meta)
    )


def test_a_model_with_no_conditioning_reports_no_sites():
    assert set_spacing_enabled(_MetaModule(), False) == 0


# --- unresolved and padded series ---------------------------------------------


def test_a_series_with_no_spacing_contributes_nothing():
    module, meta = _MetaModule(), _meta(batch=1, series=2)
    conditioning = install_spacing_conditioning(module)
    with torch.no_grad():
        conditioning.projection.weight.normal_()

    spacing = torch.tensor([[math.nan, 3.3]])
    conditioned = spacing_metadata(module, meta, spacing)
    frozen = spacing_metadata(module, meta)

    assert torch.allclose(conditioned[0, 0], frozen[0, 0])
    assert not torch.allclose(conditioned[0, 1], frozen[0, 1])


def test_the_collate_pads_with_nan_not_zero():
    batch = [
        {
            "study_uid": "a",
            "volumes": torch.zeros(2, 1, 3, 4, 4),
            "present": torch.ones(2),
            "series_meta": torch.zeros(2, 3, dtype=torch.long),
            "series_spacing": torch.tensor([3.3, 4.0]),
        },
        {
            "study_uid": "b",
            "volumes": torch.zeros(1, 1, 3, 4, 4),
            "present": torch.ones(1),
            "series_meta": torch.zeros(1, 3, dtype=torch.long),
            "series_spacing": torch.tensor([0.6]),
        },
    ]
    out = collate_b54(batch)

    assert out["series_spacing"].shape == (2, 2)
    assert out["series_spacing"][1, 0].item() == pytest.approx(0.6)
    assert math.isnan(out["series_spacing"][1, 1].item())


def test_the_collate_is_a_no_op_when_no_spacing_was_supplied():
    batch = [
        {
            "study_uid": "a",
            "volumes": torch.zeros(1, 1, 3, 4, 4),
            "present": torch.ones(1),
            "series_meta": torch.zeros(1, 3, dtype=torch.long),
        }
    ]
    assert "series_spacing" not in collate_b54(batch)


# --- attaching the number to the records --------------------------------------


def _geometry_csv(path, rows):
    pd.DataFrame(
        [
            {
                "StudyInstanceUID": study,
                "SeriesInstanceUID": series,
                "slice_spacing_mm": spacing,
            }
            for study, series, spacing in rows
        ]
    ).to_csv(path, index=False)
    return path


def test_the_table_populates_the_records(tmp_path):
    path = _geometry_csv(tmp_path / "g.csv", [("a", "s1", 3.3), ("a", "s2", 0.6)])
    records = {"a": [{"series_uid": "s1"}, {"series_uid": "s2"}]}

    stats = attach_spacing(records, series_geometry_csv=path)

    assert stats["from_table"] == 2
    assert stats["unresolved"] == 0
    assert records["a"][0][SPACING_KEY] == pytest.approx(3.3)


def test_a_series_the_table_misses_is_counted_as_unresolved(tmp_path):
    path = _geometry_csv(tmp_path / "g.csv", [("a", "s1", 3.3)])
    records = {"a": [{"series_uid": "s1"}, {"series_uid": "missing"}]}

    stats = attach_spacing(records, series_geometry_csv=path)

    assert stats["unresolved"] == 1
    assert np.isnan(records["a"][1][SPACING_KEY])
    assert stats["resolved_fraction"] == pytest.approx(0.5)


def test_the_headers_are_read_when_the_table_misses_a_series(tmp_path):
    from test_slice_geometry_scan import _write_series

    root = tmp_path / "data"
    _write_series(root / "train_series" / "a" / "s2", 8, pitch=4.0)
    path = _geometry_csv(tmp_path / "g.csv", [("a", "s1", 3.3)])
    records = {"a": [{"series_uid": "s1"}, {"series_uid": "s2"}]}

    stats = attach_spacing(records, series_geometry_csv=path, data_root=root)

    assert stats["from_table"] == 1
    assert stats["from_headers"] == 1
    assert records["a"][1][SPACING_KEY] == pytest.approx(4.0)


# --- the summary and the preflight --------------------------------------------


def test_the_summary_reports_the_depth_the_run_will_see():
    records = _records(a={"s1": 0.6, "s2": 3.3, "s3": 5.0})
    summary = spacing_summary(records, gap=1)

    assert summary["series"] == 3
    assert summary["with_a_spacing"] == 3
    assert summary["triplet_depth_mm"]["p50"] == pytest.approx(6.6)


def test_the_summary_survives_a_corpus_with_no_spacing_at_all():
    summary = spacing_summary(_records(a={"s1": math.nan}))
    assert summary["with_a_spacing"] == 0


def test_the_preflight_passes_a_well_resolved_corpus():
    module = _MetaModule()
    install_spacing_conditioning(module)
    result = preflight(_records(a={"s1": 3.3, "s2": 0.6}), model=module)

    assert result["passed"] is True
    assert result["problems"] == []
    assert result["version"] == B54_VERSION


def test_the_preflight_refuses_a_corpus_it_could_not_measure():
    records = _records(a={"s1": 3.3, "s2": math.nan, "s3": math.nan})
    result = preflight(records)

    assert result["passed"] is False
    assert any("measurable spacing" in problem for problem in result["problems"])


def test_the_preflight_refuses_a_model_with_nothing_installed():
    """Otherwise the run is a plain re-run under a new name."""
    result = preflight(_records(a={"s1": 3.3}), model=_MetaModule())

    assert result["passed"] is False
    assert any("no SpacingConditioning" in problem for problem in result["problems"])


def test_the_state_records_that_the_weights_are_still_zero():
    module = _MetaModule()
    install_spacing_conditioning(module)
    assert b54_state(module)["conditioning_is_still_zero"] == [True]

    with torch.no_grad():
        next(
            child
            for child in module.modules()
            if isinstance(child, SpacingConditioning)
        ).projection.weight.normal_()
    assert b54_state(module)["conditioning_is_still_zero"] == [False]
