"""The B51 submission reuses B42's path and pins its own artefact.

Two properties carry this file. B42's own fingerprint pin must be exactly as
strict as it was, because loosening it would weaken every B42 submission. And
the B51 launcher must refuse anything that is not a converted B51 checkpoint,
because a launcher that accepted any file with a matching hash would be a way
around the pin rather than a use of it.
"""

from __future__ import annotations

import pytest
import torch

from rsna_knee.b42_constant_area_aspect_sparse_mil import B42_EXPERIMENT, B42_VERSION
from rsna_knee.b42_constant_area_aspect_sparse_submission_dualgpu_fast import (
    B42_FROZEN_CHECKPOINT_SHA256,
    _verify_checkpoint_identity,
)
from rsna_knee.b51_checkpoint_to_b42_format import convert, convert_file
from rsna_knee.b51_full_population_training import (
    B51_EXPERIMENT,
    B51_REPORT_ONLY_STUDIES,
)
from rsna_knee.b51_submission_dualgpu_fast import require_converted_b51


def _b51_payload() -> dict:
    torch.manual_seed(0)
    return {
        "experiment": B51_EXPERIMENT,
        "version": "b51_full_population_adapted_hierarchy_v1",
        "adapt_hierarchy": True,
        "hierarchy_lr_scale": 0.05,
        "training_studies": B51_REPORT_ONLY_STUDIES,
        "seed": 2026,
        "base_state": {"context.weight": torch.randn(4, 4)},
        "head_state": {"gate": torch.randn(12)},
        "model_state": {
            "version": "b50",
            "grid_size": 6,
            "top_k": 8,
            "temperature": 1.0,
            "encoder_chunk_size": 4,
            "encoder_trainable_stages": 1,
        },
    }


def _converted_file(tmp_path, payload=None):
    source, destination = tmp_path / "b51.pt", tmp_path / "as_b42.pt"
    torch.save(payload or _b51_payload(), source)
    record = convert_file(source, destination)
    return destination, record["destination_sha256"]


# --- B42's own pin is untouched --------------------------------------------


def test_b42_still_refuses_a_file_it_did_not_freeze(tmp_path):
    """The default must stay exactly as strict as before B51 existed."""
    other = tmp_path / "not_b42.pt"
    other.write_bytes(b"anything at all")
    with pytest.raises(ValueError, match="declared checkpoint"):
        _verify_checkpoint_identity(other)


def test_b42_default_is_the_frozen_endpoint():
    import inspect  # noqa: PLC0415

    default = inspect.signature(_verify_checkpoint_identity).parameters["expected_sha256"]
    assert default.default == B42_FROZEN_CHECKPOINT_SHA256


def test_a_declared_fingerprint_is_still_checked(tmp_path):
    from rsna_knee.b35_training import sha256_file  # noqa: PLC0415

    target = tmp_path / "declared.pt"
    target.write_bytes(b"some bytes")
    assert _verify_checkpoint_identity(target, sha256_file(target)) == sha256_file(target)

    with pytest.raises(ValueError, match="declared checkpoint"):
        _verify_checkpoint_identity(target, "0" * 64)


# --- the geometry the loader would otherwise guess -------------------------


def test_the_conversion_carries_the_head_geometry_across(tmp_path):
    """top_k and temperature are not weights, so a wrong value is invisible."""
    converted = convert(_b51_payload())
    assert converted["sparse_mil"] == {"grid_size": 6, "top_k": 8, "temperature": 1.0}
    assert converted["encoder_finetune"] == {"encoder_trainable_stages": 1}


def test_a_checkpoint_that_cannot_state_its_geometry_is_refused():
    payload = _b51_payload()
    del payload["model_state"]["top_k"]
    with pytest.raises(ValueError, match="would.*fall back to defaults"):
        convert(payload)


def test_a_geometry_that_contradicts_itself_is_refused():
    payload = _b51_payload()
    payload["sparse_mil"] = {"grid_size": 6, "top_k": 99, "temperature": 1.0}
    with pytest.raises(ValueError, match="disagrees with model_state"):
        convert(payload)


# --- the B51 launcher ------------------------------------------------------


def test_a_converted_b51_checkpoint_is_accepted(tmp_path):
    destination, sha = _converted_file(tmp_path)
    identity = require_converted_b51(destination)
    assert identity["sha256"] == sha
    assert identity["training_studies"] == B51_REPORT_ONLY_STUDIES
    assert identity["sparse_mil"]["top_k"] == 8


def test_a_plain_b42_checkpoint_is_refused(tmp_path):
    """Otherwise this launcher would be a way around B42's own pin."""
    path = tmp_path / "real_b42.pt"
    torch.save(
        {"experiment": B42_EXPERIMENT, "version": B42_VERSION, "model_state": {}}, path
    )
    with pytest.raises(ValueError, match="not produced by b51_checkpoint_to_b42_format"):
        require_converted_b51(path)


def test_an_unconverted_b51_checkpoint_is_refused(tmp_path):
    path = tmp_path / "raw_b51.pt"
    torch.save(_b51_payload(), path)
    with pytest.raises(ValueError, match="not produced by b51_checkpoint_to_b42_format"):
        require_converted_b51(path)


def test_a_run_that_did_not_adapt_the_hierarchy_is_refused(tmp_path):
    payload = _b51_payload()
    payload["adapt_hierarchy"] = False
    destination, _ = _converted_file(tmp_path, payload)
    with pytest.raises(ValueError, match="did not adapt the study hierarchy"):
        require_converted_b51(destination)


def test_a_stripped_geometry_block_is_refused(tmp_path):
    destination, _ = _converted_file(tmp_path)
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    payload["sparse_mil"] = {}
    torch.save(payload, destination)
    with pytest.raises(ValueError, match="fall back to defaults"):
        require_converted_b51(destination)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        require_converted_b51(tmp_path / "nothing.pt")
