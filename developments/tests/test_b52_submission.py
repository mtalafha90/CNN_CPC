"""B52 submits through B42's inference path without pretending to be B42.

Three properties carry this file.

B42's own path must be exactly as strict and exactly as it was, because a
sibling launcher that loosened it would weaken every B42 submission.

The B52 launcher must refuse anything that is not a real B52 checkpoint. A
launcher that ran any file whose hash the operator happened to declare would be
a way around B42's pin rather than a use of it.

And the manifest B52 writes must not inherit B42's claims. `fixed_endpoint`,
`completed_epochs` and B42's training-population counts are assertions about how
a model was trained; B52 satisfies none of them, and a false provenance record
beside a real submission is worse than no record at all.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from rsna_knee.b42_constant_area_aspect_sparse_mil import B42_EXPERIMENT, B42_VERSION
from rsna_knee.b42_constant_area_aspect_sparse_submission_dualgpu_fast import (
    B42_FROZEN_CHECKPOINT_SHA256,
    _b42_endpoint_manifest,
    _load_replica,
    _verify_checkpoint_identity,
    generate_b42_submission_dual_gpu_fast,
)
from rsna_knee.b50_adapted_hierarchy_mil import B50_EXPERIMENT
from rsna_knee.b52_competition_submission_dualgpu_fast import (
    B52_SUBMISSION_EXPERIMENT,
    b52_endpoint_manifest,
    generate_b52_submission_dual_gpu_fast,
    load_b52_checkpoint,
    require_b52_endpoint,
)
from rsna_knee.b52_competition_training import B52_EXPERIMENT, B52_VERSION


def _b52_payload(**overrides) -> dict:
    """A payload shaped exactly as b52_competition_training writes one."""
    torch.manual_seed(0)
    payload = {
        "experiment": B52_EXPERIMENT,
        "version": B52_VERSION,
        "selected_epoch": 5,
        "selection_metric": "macro_auc on unseen_scanner_validation",
        "selection_value": 0.834998,
        "epochs_planned": 6,
        "seed": 2026,
        "encoder_trainable_stages": 5,
        "augmentation_enabled": True,
        "train_splits": ["train", "calibration"],
        "training_studies": 3801,
        "validation_studies": 548,
        "gold_labels_used": False,
        "gold_studies_used_in_gradient": 0,
        "base_checkpoint_sha256": "a" * 64,
        "encoder_sha256_final": "b" * 64,
        "base_state": {"context.weight": torch.randn(4, 4)},
        "head_state": {"gate": torch.randn(12)},
        "model_state": {
            "experiment": B50_EXPERIMENT,
            "version": "b50_adapted_hierarchy_mil_v1",
            "grid_size": 6,
            "top_k": 8,
            "temperature": 1.0,
            "encoder_chunk_size": 4,
            "encoder_trainable_stages": 5,
            "trainable": {"adapt_hierarchy": True, "hierarchy_trainable_parameters": 18_952_716},
        },
    }
    payload.update(overrides)
    return payload


# --- B42's own path is untouched -------------------------------------------


def test_b42_still_refuses_a_file_it_did_not_freeze(tmp_path):
    other = tmp_path / "not_b42.pt"
    other.write_bytes(b"anything at all")
    with pytest.raises(ValueError, match="declared checkpoint"):
        _verify_checkpoint_identity(other)


def test_b42_default_is_still_the_frozen_endpoint():
    default = inspect.signature(_verify_checkpoint_identity).parameters["expected_sha256"]
    assert default.default == B42_FROZEN_CHECKPOINT_SHA256


def test_the_injection_points_default_to_b42s_own():
    """A call that passes neither must behave exactly as it did before they existed."""
    parameters = inspect.signature(generate_b42_submission_dual_gpu_fast).parameters
    assert parameters["load_replica"].default is _load_replica
    assert parameters["endpoint_manifest"].default is _b42_endpoint_manifest


def test_b42s_endpoint_manifest_still_claims_the_frozen_endpoint():
    manifest = _b42_endpoint_manifest({"completed_epochs": 2, "training_studies": 4349})
    assert manifest["experiment"].startswith("B42_frozen")
    assert manifest["fixed_endpoint"] is True
    assert manifest["completed_epochs"] == 2
    assert "Exact frozen B42 fixed-E2 endpoint" in manifest["governance"]


def test_endpoint_identity_lives_in_exactly_one_place():
    """Any B42 claim left in the shared manifest body would leak into B52's.

    This is the failure the split exists to prevent, and it is invisible until a
    hidden run has already written the file.
    """
    body = inspect.getsource(generate_b42_submission_dual_gpu_fast).partition(
        "    manifest = {"
    )[2]
    for leaked in ('"fixed_endpoint"', '"completed_epochs"', '"training_series"',
                   "Exact frozen B42", "frozen B42 combined sparse-MIL"):
        assert leaked not in body, f"{leaked} is still hardcoded in the shared manifest"


# --- what the B52 launcher accepts -----------------------------------------


def test_a_real_b52_checkpoint_is_accepted():
    identity = require_b52_endpoint(_b52_payload())
    assert identity["sparse_mil"] == {"grid_size": 6, "top_k": 8, "temperature": 1.0}
    assert identity["encoder_finetune"] == {"encoder_trainable_stages": 5}
    assert identity["selected_epoch"] == 5
    assert identity["augmentation_enabled"] is True


def test_a_b42_checkpoint_is_refused():
    """Otherwise this launcher would be a way around B42's own pin."""
    with pytest.raises(ValueError, match=f"expected a {B52_EXPERIMENT}"):
        require_b52_endpoint(
            {"experiment": B42_EXPERIMENT, "version": B42_VERSION, "model_state": {}}
        )


def test_a_converted_b51_checkpoint_is_refused():
    """It presents as B42 on purpose, so it must fail B52's check on that."""
    with pytest.raises(ValueError, match=f"expected a {B52_EXPERIMENT}"):
        require_b52_endpoint(
            {
                "experiment": B42_EXPERIMENT,
                "version": B42_VERSION,
                "converted_from": {"experiment": "B51_FULL_POPULATION_ADAPTED_HIERARCHY"},
                "model_state": {},
            }
        )


def test_a_b52_file_of_the_wrong_version_is_refused():
    with pytest.raises(ValueError, match=f"expected {B52_VERSION}"):
        require_b52_endpoint(_b52_payload(version="b52_something_else_v9"))


def test_a_checkpoint_from_the_wrong_model_class_is_refused():
    payload = _b52_payload()
    payload["model_state"]["experiment"] = "B42_CONSTANT_AREA"
    with pytest.raises(ValueError, match="not trained with B50's model class"):
        require_b52_endpoint(payload)


def test_a_run_that_did_not_adapt_the_hierarchy_is_refused():
    payload = _b52_payload()
    payload["model_state"]["trainable"] = {"adapt_hierarchy": False}
    with pytest.raises(ValueError, match="did not adapt the study hierarchy"):
        require_b52_endpoint(payload)


# --- the values a strict state-dict load cannot catch ----------------------


@pytest.mark.parametrize("key", ["grid_size", "top_k", "temperature"])
def test_a_checkpoint_that_cannot_state_its_head_geometry_is_refused(key):
    """top_k and temperature are not weights; a wrong value loads cleanly."""
    payload = _b52_payload()
    del payload["model_state"][key]
    with pytest.raises(ValueError, match="head geometry would be"):
        require_b52_endpoint(payload)


def test_a_different_encoder_chunk_is_refused():
    """The chunk the runtime budget was calibrated against, not a preference."""
    payload = _b52_payload()
    payload["model_state"]["encoder_chunk_size"] = 8
    with pytest.raises(ValueError, match="encoder chunk size 4"):
        require_b52_endpoint(payload)


@pytest.mark.parametrize("key,value", [
    ("base_state", {}), ("head_state", {}), ("model_state", {}),
])
def test_a_checkpoint_missing_its_weights_is_refused(key, value):
    with pytest.raises(ValueError, match=f"missing its {key}"):
        require_b52_endpoint(_b52_payload(**{key: value}))


# --- leakage hygiene, which applies to any endpoint ------------------------


def test_a_checkpoint_that_saw_the_expert_labels_is_refused():
    with pytest.raises(ValueError, match="unexpectedly used expert labels"):
        require_b52_endpoint(_b52_payload(gold_labels_used=True))


def test_a_checkpoint_that_took_expert_gradients_is_refused():
    with pytest.raises(ValueError, match="unexpectedly used expert gradients"):
        require_b52_endpoint(_b52_payload(gold_studies_used_in_gradient=3))


# --- the fingerprint pin ----------------------------------------------------


def test_the_declared_fingerprint_is_checked_before_anything_loads(tmp_path):
    path = tmp_path / "b52.pt"
    torch.save(_b52_payload(), path)
    with pytest.raises(ValueError, match="declared checkpoint"):
        generate_b52_submission_dual_gpu_fast(
            {}, data_root=tmp_path, checkpoint=path,
            base_checkpoint=path, expected_checkpoint_sha256="0" * 64,
        )


def test_a_missing_checkpoint_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint is missing"):
        generate_b52_submission_dual_gpu_fast(
            {}, data_root=tmp_path, checkpoint=tmp_path / "nothing.pt",
            base_checkpoint=tmp_path / "nothing.pt", expected_checkpoint_sha256="0" * 64,
        )


def test_the_base_checkpoint_fingerprint_is_checked(tmp_path):
    """The frozen base these weights were fine-tuned from, not any base."""
    checkpoint = tmp_path / "b52.pt"
    torch.save(_b52_payload(), checkpoint)
    wrong_base = tmp_path / "base.pt"
    wrong_base.write_bytes(b"not the base this run used")
    with pytest.raises(ValueError, match="base checkpoint fingerprint mismatch"):
        load_b52_checkpoint(checkpoint, base_checkpoint=wrong_base, device="cpu")


# --- the manifest tells the truth about B52 --------------------------------


def test_b52s_manifest_does_not_claim_a_frozen_endpoint():
    manifest = b52_endpoint_manifest(_b52_payload())
    assert manifest["experiment"] == B52_SUBMISSION_EXPERIMENT
    assert manifest["fixed_endpoint"] is False
    assert manifest["selected_epoch"] == 5
    assert manifest["training_studies"] == 3801
    assert manifest["validation_studies"] == 548


def test_b52s_manifest_carries_no_b42_claim():
    manifest = b52_endpoint_manifest(_b52_payload())
    rendered = repr(manifest)
    assert "Exact frozen B42" not in rendered
    assert "fixed-E2" not in rendered
    assert "frozen B42 combined" not in rendered


def test_b52s_manifest_answers_every_field_b42s_does():
    """A B42-only key would otherwise be simply absent from a B52 manifest.

    Absent is better than false, but silence about the endpoint is not the point
    of a manifest -- so each of B42's fields must have a B52 answer, even when
    the honest answer is null.
    """
    b42_keys = set(_b42_endpoint_manifest(_b52_payload()))
    b52_keys = set(b52_endpoint_manifest(_b52_payload()))
    missing = b42_keys - b52_keys
    assert not missing, f"B52's manifest says nothing about {sorted(missing)}"


def test_the_counts_b52_never_recorded_are_null_and_named_as_such():
    """Not a plausible-looking number. A reader must be able to tell them apart."""
    manifest = b52_endpoint_manifest(_b52_payload())
    assert manifest["training_series"] is None
    assert manifest["training_supervision_cells"] is None
    assert set(manifest["counts_not_recorded_by_b52"]) == {
        "training_series", "training_supervision_cells"
    }


def test_completed_epochs_comes_from_the_history_not_the_plan():
    """A run that stopped early planned six epochs and completed fewer."""
    assert b52_endpoint_manifest(_b52_payload())["completed_epochs"] is None
    stopped_early = _b52_payload(history=[{"epoch": 1}, {"epoch": 2}, {"epoch": 3}])
    manifest = b52_endpoint_manifest(stopped_early)
    assert manifest["epochs_planned"] == 6
    assert manifest["completed_epochs"] == 3


def test_b52s_manifest_records_that_it_was_selected_not_frozen():
    governance = b52_endpoint_manifest(_b52_payload())["governance"]
    assert "selection statistic" in governance
    assert "held-out" in governance
