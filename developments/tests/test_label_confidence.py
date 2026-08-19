"""Rescaling the label targets must reach the values training actually uses.

A setting that looks applied and changes nothing is the worst outcome here: a
run would burn 90 minutes of GPU and be reported as a different experiment
while being bit-identical to the last one. The targets live in the exported
supervision array, not the config, so these tests work on the array.
"""

from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.label_confidence import (
    EXPORTED_NEGATIVE_TARGET,
    EXPORTED_POSITIVE_TARGET,
    NEGATIVE_TARGET_KEY,
    POSITIVE_TARGET_KEY,
    rescale_label_confidence,
)


def _supervision():
    """Two studies: positive, negative, and an unsupervised cell each."""
    targets = np.array(
        [[EXPORTED_POSITIVE_TARGET, EXPORTED_NEGATIVE_TARGET, 0.0],
         [EXPORTED_NEGATIVE_TARGET, EXPORTED_POSITIVE_TARGET, 0.0]],
        dtype=np.float32,
    )
    weights = np.array([[0.5, 1.0, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32)
    return targets, weights


def test_the_default_config_changes_nothing():
    """An unset target must leave the frozen export exactly as it was."""
    targets, weights = _supervision()
    rescaled, record = rescale_label_confidence(targets, weights, {})
    assert record["changed"] is False
    assert rescaled is targets


def test_a_lower_positive_target_reaches_the_positive_cells():
    targets, weights = _supervision()
    rescaled, record = rescale_label_confidence(
        targets, weights, {POSITIVE_TARGET_KEY: 0.70}
    )
    assert record["changed"] is True
    assert record["positive_cells_rescaled"] == 2
    assert rescaled[0, 0] == pytest.approx(0.70)
    assert rescaled[1, 1] == pytest.approx(0.70)


def test_the_negative_cells_are_left_alone_when_only_positives_move():
    targets, weights = _supervision()
    rescaled, _ = rescale_label_confidence(targets, weights, {POSITIVE_TARGET_KEY: 0.70})
    assert rescaled[0, 1] == pytest.approx(EXPORTED_NEGATIVE_TARGET)
    assert rescaled[1, 0] == pytest.approx(EXPORTED_NEGATIVE_TARGET)


def test_unsupervised_cells_are_never_touched():
    """A zero-weight cell contributes nothing; rewriting it would be noise."""
    targets, weights = _supervision()
    rescaled, _ = rescale_label_confidence(
        targets, weights, {POSITIVE_TARGET_KEY: 0.70, NEGATIVE_TARGET_KEY: 0.20}
    )
    assert rescaled[0, 2] == pytest.approx(0.0)
    assert rescaled[1, 2] == pytest.approx(0.0)


def test_the_original_array_is_not_modified_in_place():
    targets, weights = _supervision()
    before = targets.copy()
    rescale_label_confidence(targets, weights, {POSITIVE_TARGET_KEY: 0.70})
    assert np.array_equal(targets, before)


def test_a_target_crossing_the_boundary_is_refused():
    """Below 0.5 a positive cell would be counted as a negative one."""
    targets, weights = _supervision()
    with pytest.raises(ValueError, match="must stay above"):
        rescale_label_confidence(targets, weights, {POSITIVE_TARGET_KEY: 0.40})
    with pytest.raises(ValueError, match="must stay below"):
        rescale_label_confidence(targets, weights, {NEGATIVE_TARGET_KEY: 0.60})


@pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 85.0])
def test_a_target_outside_zero_to_one_is_refused(value):
    targets, weights = _supervision()
    with pytest.raises(ValueError, match="between 0 and 1"):
        rescale_label_confidence(targets, weights, {POSITIVE_TARGET_KEY: value})


def test_the_record_carries_the_measurement_that_motivated_it():
    targets, weights = _supervision()
    _, record = rescale_label_confidence(targets, weights, {POSITIVE_TARGET_KEY: 0.70})
    assert record["measured_positive_agreement"] == pytest.approx(0.690, abs=1e-3)
    assert "label_audit" in record["measurement_source"]


def test_the_frozen_export_policy_is_left_alone():
    """The first version reused the b7_* keys and the B7-v1 contract said no.

    It was right to. Those keys state what the exported labels contain, and the
    export still contains 0.85; rewriting them would have made the config claim
    something false about the labels on disk. Retargeting is a training-time
    choice and needs keys of its own.
    """
    for key in (POSITIVE_TARGET_KEY, NEGATIVE_TARGET_KEY):
        assert not key.startswith("b7_"), (
            f"{key} collides with the frozen B7 export policy the contract checks"
        )


def test_the_contract_still_passes_with_a_retargeted_run():
    """The guard must keep protecting the export while allowing this."""
    from rsna_knee.b7_weak_supervision import _require_frozen_policy

    config = {
        "b7_positive_target": EXPORTED_POSITIVE_TARGET,
        "b7_negative_target": EXPORTED_NEGATIVE_TARGET,
        POSITIVE_TARGET_KEY: 0.70,
    }
    _require_frozen_policy(config)  # must not raise

    config["b7_positive_target"] = 0.70
    with pytest.raises(ValueError, match="policy is frozen"):
        _require_frozen_policy(config)


def test_a_retargeted_run_records_a_policy_of_its_own():
    """The contract asks for a new name; a run must not pass as B7-v1."""
    from rsna_knee.b7_weak_supervision import B7_VARIANT

    targets, weights = _supervision()
    _, unchanged = rescale_label_confidence(targets, weights, {})
    assert unchanged["supervision_policy"] == B7_VARIANT

    _, retargeted = rescale_label_confidence(
        targets, weights, {POSITIVE_TARGET_KEY: 0.70}
    )
    assert retargeted["supervision_policy"] != B7_VARIANT
    assert "0.7" in retargeted["supervision_policy"]


def test_training_applies_the_rescale_rather_than_only_storing_it():
    """Guard against the config-only version of this, which did nothing."""
    import inspect

    from rsna_knee import phase9_matched_supervision_training as trainer

    source = inspect.getsource(trainer)
    assert "rescale_label_confidence(targets, weights, config)" in source
    applied = source.index("rescale_label_confidence(targets, weights, config)")
    assert applied < source.index("CropFocusedVariableSeriesKneeDataset("), (
        "the rescale must happen before the dataset is built from the targets"
    )
