"""B51 trains the whole population, and its checkpoint reaches the proven path.

Two properties carry this one. B51 must train on all 4,349 report-only studies,
because its whole claim is that it differs from B42 -- the endpoint behind the
0.714 hidden score -- by exactly one thing. And the conversion that lets the
B42 submission path load a B51 checkpoint must not alter a single weight, which
is asserted bit-for-bit rather than argued.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from rsna_knee.b42_constant_area_aspect_sparse_mil import B42_EXPERIMENT, B42_VERSION
from rsna_knee.b51_checkpoint_to_b42_format import (
    WEIGHT_KEYS,
    convert,
    convert_file,
    weights_are_identical,
)
from rsna_knee.b51_full_population_training import (
    B51_EXPERIMENT,
    B51_REPORT_ONLY_STUDIES,
    full_population_rows,
)
from rsna_knee.constants import TARGETS


# --- the population --------------------------------------------------------


def _train_frame(report_only: int, gold: int = 58) -> pd.DataFrame:
    rows = []
    for index in range(report_only):
        row = {"StudyInstanceUID": f"weak-{index}"}
        row.update({name: None for name in TARGETS})
        rows.append(row)
    for index in range(gold):
        row = {"StudyInstanceUID": f"gold-{index}"}
        row.update({name: 1.0 for name in TARGETS})
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def patched(monkeypatch):
    """Stand in for train.csv, with gold identified by its label columns."""
    import rsna_knee.b51_full_population_training as module

    def install(frame):
        monkeypatch.setattr(module, "load_train_csv", lambda path: frame)
        monkeypatch.setattr(
            module, "gold_mask", lambda f: f[list(TARGETS)].notna().all(axis=1)
        )

    return install


def test_the_full_report_only_population_is_used(patched):
    patched(_train_frame(B51_REPORT_ONLY_STUDIES))
    rows = full_population_rows(Path("."), {})
    assert len(rows) == B51_REPORT_ONLY_STUDIES == 4349
    assert set(rows["split"]) == {"train"}
    assert not any(uid.startswith("gold-") for uid in rows["StudyInstanceUID"])


def test_the_gold_studies_are_excluded(patched):
    patched(_train_frame(B51_REPORT_ONLY_STUDIES, gold=58))
    rows = full_population_rows(Path("."), {})
    assert len(rows) == B51_REPORT_ONLY_STUDIES, "58 gold studies must not enter training"


def test_a_short_population_is_refused(patched):
    """B51's claim depends on training the same studies B42 did."""
    patched(_train_frame(B51_REPORT_ONLY_STUDIES - 1))
    with pytest.raises(ValueError, match="all 4349 report-only studies"):
        full_population_rows(Path("."), {})


def test_every_study_is_labelled_for_training(patched):
    """No held-out split: B50 ran the comparison, B51 is a production run."""
    patched(_train_frame(B51_REPORT_ONLY_STUDIES))
    rows = full_population_rows(Path("."), {})
    assert (rows["split"] == "train").all()


# --- the conversion --------------------------------------------------------


def _b51_payload(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return {
        "experiment": B51_EXPERIMENT,
        "version": "b51_full_population_adapted_hierarchy_v1",
        "adapt_hierarchy": True,
        "hierarchy_lr_scale": 0.05,
        "training_studies": B51_REPORT_ONLY_STUDIES,
        "seed": 2026,
        "base_state": {
            "context.weight": torch.randn(4, 4),
            "encoder.0.weight": torch.randn(2, 3),
        },
        "head_state": {"gate": torch.randn(12), "evidence_weight": torch.randn(12, 8)},
        # Mirrors what the real model's state() emits. The converter needs the
        # geometry keys because the B42 loader would otherwise silently use its
        # own defaults for top_k and temperature, which are not weights.
        "model_state": {
            "version": "b50",
            "grid_size": 6,
            "top_k": 8,
            "temperature": 1.0,
            "encoder_chunk_size": 4,
            "encoder_trainable_stages": 1,
        },
        "history": [{"epoch": 2}],
    }


def test_the_converted_payload_is_what_the_b42_loader_expects():
    converted = convert(_b51_payload())
    assert converted["experiment"] == B42_EXPERIMENT
    assert converted["version"] == B42_VERSION
    assert converted["model_state"]["version"] == B42_VERSION


def test_not_one_weight_changes():
    """The property the submission depends on, asserted bit-for-bit."""
    payload = _b51_payload()
    before = {
        key: {name: tensor.clone() for name, tensor in payload[key].items()}
        for key in WEIGHT_KEYS
    }
    converted = convert(payload)
    for key in WEIGHT_KEYS:
        assert set(converted[key]) == set(before[key])
        for name, tensor in before[key].items():
            assert torch.equal(converted[key][name], tensor)
            assert converted[key][name].dtype == tensor.dtype
    assert weights_are_identical({**payload, **before}, converted)


def test_the_conversion_records_where_it_came_from():
    """A converted file must never be mistaken for a real B42 run."""
    converted = convert(_b51_payload())
    origin = converted["converted_from"]
    assert origin["experiment"] == B51_EXPERIMENT
    assert origin["adapt_hierarchy"] is True
    assert origin["training_studies"] == B51_REPORT_ONLY_STUDIES
    assert "requires_grad" in origin["note"]


def test_a_checkpoint_that_is_not_b51_is_refused():
    payload = _b51_payload()
    payload["experiment"] = B42_EXPERIMENT
    with pytest.raises(ValueError, match="expected a B51"):
        convert(payload)


def test_a_checkpoint_missing_its_weights_is_refused():
    for key in WEIGHT_KEYS:
        payload = _b51_payload()
        del payload[key]
        with pytest.raises(ValueError, match=f"missing its {key}"):
            convert(payload)


def test_a_changed_tensor_is_detected():
    """If this comparison could not fail, the guarantee would be worthless."""
    first, second = _b51_payload(0), _b51_payload(0)
    assert weights_are_identical(first, second)

    second["head_state"]["gate"] = second["head_state"]["gate"] + 1e-8
    assert not weights_are_identical(first, second)

    third = _b51_payload(0)
    third["head_state"]["gate"] = third["head_state"]["gate"].to(torch.float64)
    assert not weights_are_identical(first, third), "a dtype change is a change"

    fourth = _b51_payload(0)
    del fourth["base_state"]["context.weight"]
    assert not weights_are_identical(first, fourth)


# --- writing the file ------------------------------------------------------


def test_the_written_file_round_trips_unchanged(tmp_path):
    source, destination = tmp_path / "b51.pt", tmp_path / "as_b42.pt"
    payload = _b51_payload()
    torch.save(payload, source)

    record = convert_file(source, destination)
    assert record["weights_verified_identical"] is True
    assert record["experiment_out"] == B42_EXPERIMENT
    assert record["tensors_checked"] == 4

    reloaded = torch.load(destination, map_location="cpu", weights_only=False)
    assert reloaded["experiment"] == B42_EXPERIMENT
    assert weights_are_identical(payload, reloaded)

    written = tmp_path / "as_b42.conversion.json"
    assert written.exists(), "the conversion must leave a record beside the file"


def test_an_existing_destination_is_never_overwritten(tmp_path):
    source, destination = tmp_path / "b51.pt", tmp_path / "as_b42.pt"
    torch.save(_b51_payload(), source)
    destination.write_text("already here")
    with pytest.raises(FileExistsError):
        convert_file(source, destination)
