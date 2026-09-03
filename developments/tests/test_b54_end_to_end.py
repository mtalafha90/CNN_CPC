"""The whole B54 chain on one synthetic corpus, from DICOM headers to the sum.

Every piece is unit-tested on its own. This exists because unit tests do not
catch the failure that actually happens: two modules that each work, agreeing
on a column name that neither one owns. It runs the real scan over real DICOM
files, feeds its real CSV to the rollup and to the conditioning, and ends at
the tensor the model would add.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from rsna_knee.b54_spacing_run import (
    SPACING_KEY,
    attach_spacing,
    collate_b54,
    install_spacing_conditioning,
    preflight,
    set_spacing_enabled,
    spacing_metadata,
)
from rsna_knee.slice_geometry_scan import scan
from rsna_knee.spacing_conditioning import spacing_lookup
from rsna_knee.study_geometry_rollup import rollup

pytest.importorskip("pydicom")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Two studies: one thick 2D pair, one that mixes a thin volume with a stack."""
    from test_slice_geometry_scan import _write_series

    root = tmp_path_factory.mktemp("corpus") / "data"
    plan = {
        "studyA": [("a1", 30, 4.0, "2D"), ("a2", 26, 4.5, "2D")],
        "studyB": [("b1", 200, 0.6, "3D"), ("b2", 24, 4.0, "2D")],
    }
    rows = []
    for study, entries in plan.items():
        for name, frames, pitch, acquisition in entries:
            _write_series(
                root / "train_series" / study / name,
                frames,
                pitch=pitch,
                thickness=pitch,
                acquisition=acquisition,
            )
            rows.append(
                {
                    "StudyInstanceUID": study,
                    "SeriesInstanceUID": name,
                    "Anatomical_Plane": "Sagittal",
                    "Fluid_Sensitive": "Yes",
                    "Fat_Suppression": "No",
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "train_series.csv", index=False)
    return root, plan


@pytest.fixture(scope="module")
def geometry_csv(tmp_path_factory, corpus):
    root, _ = corpus
    out = tmp_path_factory.mktemp("scan") / "series_geometry.csv"
    scan(data_root=root, workers=1, out_csv=out)
    return out


# --- the scan writes what the others read -------------------------------------


def test_the_scan_measures_the_pitch_that_was_written(geometry_csv, corpus):
    _, plan = corpus
    frame = pd.read_csv(geometry_csv)
    expected = {
        name: pitch for entries in plan.values() for name, _, pitch, _ in entries
    }

    for _, row in frame.iterrows():
        assert row["slice_spacing_mm"] == pytest.approx(
            expected[row["SeriesInstanceUID"]]
        ), row["SeriesInstanceUID"]


def test_the_rollup_reads_the_scans_csv_without_translation(geometry_csv):
    """The two modules must agree on column names neither of them owns."""
    result = rollup(series_csv=geometry_csv)

    assert result["all_studies"]["studies"] == 2
    assert result["all_studies"]["studies_mixing_thin_and_thick"]["studies"] == 1


def test_the_lookup_reads_the_same_csv(geometry_csv):
    lookup = spacing_lookup(geometry_csv)

    assert lookup[("studyA", "a1")] == pytest.approx(4.0)
    assert lookup[("studyB", "b1")] == pytest.approx(0.6)


# --- the number reaches the records -------------------------------------------


def _records(plan):
    return {
        study: [{"series_uid": name} for name, _, _, _ in entries]
        for study, entries in plan.items()
    }


def test_every_series_resolves_from_the_table(geometry_csv, corpus):
    _, plan = corpus
    records = _records(plan)
    stats = attach_spacing(records, series_geometry_csv=geometry_csv)

    assert stats["series"] == 4
    assert stats["from_table"] == 4
    assert stats["unresolved"] == 0
    assert stats["resolved_fraction"] == pytest.approx(1.0)


def test_headers_cover_a_series_the_table_never_saw(corpus, geometry_csv):
    """The submission path: no table row, so read the DICOMs."""
    root, plan = corpus
    trimmed = pd.read_csv(geometry_csv)
    trimmed = trimmed.loc[trimmed["SeriesInstanceUID"] != "b1"]
    partial = geometry_csv.parent / "partial.csv"
    trimmed.to_csv(partial, index=False)

    records = _records(plan)
    stats = attach_spacing(records, series_geometry_csv=partial, data_root=root)

    assert stats["from_headers"] == 1
    assert stats["unresolved"] == 0
    b1 = next(r for r in records["studyB"] if r["series_uid"] == "b1")
    assert b1[SPACING_KEY] == pytest.approx(0.6)


def test_the_preflight_passes_the_assembled_corpus(geometry_csv, corpus):
    _, plan = corpus
    records = _records(plan)
    attach_spacing(records, series_geometry_csv=geometry_csv)

    module = _MetaModule()
    install_spacing_conditioning(module)
    result = preflight(records, model=module)

    assert result["passed"] is True
    assert result["spacing_summary"]["triplet_depth_mm"]["p50"] > 0


# --- the number reaches the model ---------------------------------------------


class _MetaModule(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.plane_embedding = nn.Embedding(4, d, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, d, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, d, padding_idx=0)


def _batch(records):
    """What `collate_b54` would produce for these two studies."""
    items = []
    for study, entries in records.items():
        k = len(entries)
        items.append(
            {
                "study_uid": study,
                "volumes": torch.zeros(k, 1, 3, 4, 4),
                "present": torch.ones(k),
                "series_meta": torch.ones(k, 3, dtype=torch.long),
                "series_spacing": torch.tensor(
                    [float(r[SPACING_KEY]) for r in entries], dtype=torch.float32
                ),
            }
        )
    return collate_b54(items)


def test_the_spacing_survives_the_collate(geometry_csv, corpus):
    _, plan = corpus
    records = _records(plan)
    attach_spacing(records, series_geometry_csv=geometry_csv)
    batch = _batch(records)

    assert batch["series_spacing"].shape == (2, 2)
    assert torch.isfinite(batch["series_spacing"]).all()


def test_a_ragged_batch_pads_with_nan_that_contributes_nothing(geometry_csv, corpus):
    _, plan = corpus
    records = _records(plan)
    attach_spacing(records, series_geometry_csv=geometry_csv)
    records["studyA"] = records["studyA"][:1]  # make the batch ragged
    batch = _batch(records)

    assert math.isnan(batch["series_spacing"][0, 1].item())

    module = _MetaModule()
    conditioning = install_spacing_conditioning(module)
    with torch.no_grad():
        conditioning.projection.weight.normal_()

    conditioned = spacing_metadata(
        module, batch["series_meta"], batch["series_spacing"]
    )
    frozen = spacing_metadata(module, batch["series_meta"])
    assert torch.allclose(conditioned[0, 1], frozen[0, 1])


def test_the_untrained_chain_is_numerically_the_old_model(geometry_csv, corpus):
    """Nothing about switching B54 on may move the starting point."""
    _, plan = corpus
    records = _records(plan)
    attach_spacing(records, series_geometry_csv=geometry_csv)
    batch = _batch(records)

    module = _MetaModule()
    before = spacing_metadata(module, batch["series_meta"])
    install_spacing_conditioning(module)
    after = spacing_metadata(module, batch["series_meta"], batch["series_spacing"])

    assert torch.allclose(before, after)


def test_a_thin_and_a_thick_series_get_different_conditioning(geometry_csv, corpus):
    """The point of the whole chain, asserted on real measured spacings."""
    _, plan = corpus
    records = _records(plan)
    attach_spacing(records, series_geometry_csv=geometry_csv)
    batch = _batch(records)

    module = _MetaModule()
    conditioning = install_spacing_conditioning(module)
    with torch.no_grad():
        conditioning.projection.weight.normal_()

    out = spacing_metadata(module, batch["series_meta"], batch["series_spacing"])
    # studyB row: b1 at 0.6 mm and b2 at 4.0 mm, same plane/fluid/fat.
    assert not torch.allclose(out[1, 0], out[1, 1])


def test_the_ablation_recovers_the_old_model_from_a_trained_one(geometry_csv, corpus):
    _, plan = corpus
    records = _records(plan)
    attach_spacing(records, series_geometry_csv=geometry_csv)
    batch = _batch(records)

    module = _MetaModule()
    conditioning = install_spacing_conditioning(module)
    with torch.no_grad():
        conditioning.projection.weight.normal_()

    set_spacing_enabled(module, False)
    assert torch.allclose(
        spacing_metadata(module, batch["series_meta"], batch["series_spacing"]),
        spacing_metadata(module, batch["series_meta"]),
    )


def test_the_contribution_keeps_the_metadata_dtype(geometry_csv, corpus):
    """Half precision must not be silently promoted out from under the model."""
    _, plan = corpus
    records = _records(plan)
    attach_spacing(records, series_geometry_csv=geometry_csv)
    batch = _batch(records)

    module = _MetaModule().half()
    conditioning = install_spacing_conditioning(module)
    with torch.no_grad():
        conditioning.projection.weight.normal_()

    out = spacing_metadata(module, batch["series_meta"], batch["series_spacing"])
    assert out.dtype == torch.float16


# --- the two measurements must agree ------------------------------------------


def test_the_scan_and_the_run_see_the_same_depths(geometry_csv, corpus):
    """A silent disagreement here would invalidate the whole finding."""
    from rsna_knee.b54_spacing_run import spacing_summary

    _, plan = corpus
    records = _records(plan)
    attach_spacing(records, series_geometry_csv=geometry_csv)

    scanned = pd.read_csv(geometry_csv)["triplet_span_mm"].dropna().to_numpy(float)
    summary = spacing_summary(records, gap=1)

    assert summary["with_a_spacing"] == len(scanned)
    assert summary["triplet_depth_mm"]["p50"] == pytest.approx(
        float(np.percentile(scanned, 50))
    )
