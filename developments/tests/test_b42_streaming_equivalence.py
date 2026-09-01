"""Streaming changes when memory is held, not what the model is shown.

`normalized_view_b42` is already asserted bit-identical to the audited
normalize-once helper in `test_kaggle_hidden_streaming_highres.py`. That leaves
the wiring: whether `_infer_one_study_streamed` hands the model the same series
in the same order, with the same positions and metadata, for the same three
offsets. A transposition there would still produce a plausible submission and a
plausible score, so it is checked against the audited helper rather than argued.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rsna_knee import b42_constant_area_aspect_sparse_submission_dualgpu_fast as launcher
from rsna_knee.b42_kaggle_fast_preprocess import (
    B42_FAST_TTA_OFFSETS,
    preprocess_three_offsets_b42_normalize_once,
)
from rsna_knee.constants import TARGETS

GAP = 1
CROP = 0.90


class _Config:
    data_root = "/does/not/matter"
    split = "test"


class _Reader:
    """Stands in for the B42 dataset, which streaming uses only to read volumes."""

    def __init__(self, volumes: dict[str, np.ndarray]):
        self.config = _Config()
        self.volumes = volumes

    def _read_volume(self, path, plane):
        return self.volumes[str(path)]


class _Recorder:
    """Captures exactly what the model was given, per view."""

    def __init__(self):
        self.calls = []

    def __call__(self, volumes, present, series_meta, position):
        self.calls.append({
            "volumes": [v.clone() for v in volumes],
            "present": present.clone(),
            "series_meta": series_meta.clone(),
            "position": position.clone(),
        })
        return type("Out", (), {"logits": torch.zeros(1, len(TARGETS))})()


@pytest.fixture
def study(monkeypatch):
    """Three deliberately different native series, so an ordering slip shows."""
    rng = np.random.default_rng(5)
    shapes = [(41, 73, 121), (37, 64, 64), (45, 101, 83)]
    volumes, records = {}, []
    for position, shape in enumerate(shapes):
        series_uid = f"series-{position}"
        volumes[f"/fake/{series_uid}"] = rng.normal(size=shape).astype(np.float32)
        records.append({
            "series_uid": series_uid,
            "plane": "Sagittal",
            "plane_id": position + 1,
            "fluid_id": position + 2,
            "fat_id": position + 3,
        })
    monkeypatch.setattr(
        launcher, "find_series_dir", lambda root, split, uid, series_uid: f"/fake/{series_uid}"
    )
    return _Reader(volumes), records, [volumes[f"/fake/{r['series_uid']}"] for r in records]


def test_the_streamed_study_gives_the_model_the_audited_tensors(study):
    reader, records, raws = study
    model = _Recorder()

    launcher._infer_one_study_streamed(
        "study-a", records, reader, model, torch.device("cpu"), gap=GAP, crop_fraction=CROP
    )

    assert len(model.calls) == len(B42_FAST_TTA_OFFSETS), "one model call per TTA view"

    expected = [
        preprocess_three_offsets_b42_normalize_once(raw, gap=GAP, crop_fraction=CROP)
        for raw in raws
    ]
    for view, call in enumerate(model.calls):
        assert len(call["volumes"]) == len(records)
        for series, volume in enumerate(call["volumes"]):
            images, positions = expected[series]
            assert torch.equal(volume, images[view]), (
                f"view {view} series {series} differs from the audited helper"
            )
            assert torch.equal(call["position"][series], positions[view])


def test_the_series_metadata_travels_with_its_own_series(study):
    """Plane, fluid and fat identifiers are per series and are easy to transpose."""
    reader, records, _ = study
    model = _Recorder()

    launcher._infer_one_study_streamed(
        "study-a", records, reader, model, torch.device("cpu"), gap=GAP, crop_fraction=CROP
    )

    expected = torch.tensor(
        [[r["plane_id"], r["fluid_id"], r["fat_id"]] for r in records], dtype=torch.long
    )
    for call in model.calls:
        assert torch.equal(call["series_meta"], expected)
        assert torch.equal(call["present"], torch.ones(len(records)))


def test_the_views_are_rectangular_and_differ_between_series(study):
    """If every series came back the same shape, this would not be B42 geometry."""
    reader, records, _ = study
    model = _Recorder()

    launcher._infer_one_study_streamed(
        "study-a", records, reader, model, torch.device("cpu"), gap=GAP, crop_fraction=CROP
    )
    shapes = {tuple(v.shape[-2:]) for v in model.calls[0]["volumes"]}
    assert len(shapes) > 1, "three different native shapes must not collapse to one"


# --- what happens when part of the study cannot be read --------------------


def test_one_unreadable_series_is_dropped_and_the_rest_still_predict(study):
    """A study with five good series out of six is still a real prediction."""
    reader, records, _ = study
    del reader.volumes["/fake/series-1"]
    model = _Recorder()

    uid, probability, shapes, dropped = launcher._infer_one_study_streamed(
        "study-a", records, reader, model, torch.device("cpu"), gap=GAP, crop_fraction=CROP
    )

    assert len(dropped) == 1
    assert dropped[0]["series_uid"] == "series-1"
    assert len(model.calls[0]["volumes"]) == 2, "the two readable series still ran"
    assert probability.shape == (len(TARGETS),)


def test_a_missing_series_directory_is_dropped_rather_than_raised(study, monkeypatch):
    reader, records, _ = study
    monkeypatch.setattr(
        launcher,
        "find_series_dir",
        lambda root, split, uid, series_uid: (
            None if series_uid == "series-2" else f"/fake/{series_uid}"
        ),
    )
    model = _Recorder()
    _, _, _, dropped = launcher._infer_one_study_streamed(
        "study-a", records, reader, model, torch.device("cpu"), gap=GAP, crop_fraction=CROP
    )
    assert [record["series_uid"] for record in dropped] == ["series-2"]
    assert "not found" in dropped[0]["error"]


def test_a_study_with_nothing_readable_still_raises(study, monkeypatch):
    """Dropping every series would silently submit a prediction from no data."""
    reader, records, _ = study
    monkeypatch.setattr(launcher, "find_series_dir", lambda *args: None)
    with pytest.raises(RuntimeError, match="has no readable MRI series"):
        launcher._infer_one_study_streamed(
            "study-a", records, reader, _Recorder(), torch.device("cpu"),
            gap=GAP, crop_fraction=CROP,
        )
