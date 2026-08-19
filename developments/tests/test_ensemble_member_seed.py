"""A declared ensemble member may vary the seed; an undeclared run may not.

Averaging models that share a seed gains nothing -- they see the same
initialisation and the same data order, so they make the same mistakes. Varying
it is the whole mechanism of an ensemble. But every comparison in this project
so far has relied on the stochastic path being matched, and the B13 contract
freezes the seed to keep it that way.

Both things have to hold at once: a run that says it is an ensemble member may
differ, and a run that drifts off the protocol without saying so must still be
caught. As with the label targets, these tests put the config a real run
produces through the real contract chain rather than reading the source.
"""

from __future__ import annotations

import pytest

from model._implementation import FROZEN_PROTOCOL_SEED, read_config, set_seed


@pytest.fixture(scope="module")
def contract():
    from model.bootstrap import ensure_developments_source

    ensure_developments_source()
    from rsna_knee.b20_crop_focus import require_b20_contract

    return require_b20_contract


def test_a_declared_ensemble_member_clears_the_chain(contract):
    contract(set_seed(read_config("config/current_model.yaml"), 7))


def test_an_undeclared_seed_change_is_still_refused(contract):
    """The drift the contract exists to catch must still be caught."""
    config = read_config("config/current_model.yaml")
    config["seed"] = 7  # changed directly, without declaring anything
    with pytest.raises(ValueError, match="freezes seed"):
        contract(config)


def test_the_protocol_seed_is_not_marked_as_a_member(contract):
    """Passing the frozen seed explicitly is not a deviation."""
    config = set_seed(read_config("config/current_model.yaml"), FROZEN_PROTOCOL_SEED)
    assert "ensemble_member" not in config
    contract(config)


def test_no_seed_given_leaves_the_config_alone():
    original = read_config("config/current_model.yaml")
    assert set_seed(original, None) is original


def test_the_caller_config_is_not_modified():
    original = read_config("config/current_model.yaml")
    before = original["seed"]
    set_seed(original, 7)
    assert original["seed"] == before
    assert "ensemble_member" not in original


def test_the_member_is_recorded_for_later_identification():
    config = set_seed(read_config("config/current_model.yaml"), 7)
    assert config["ensemble_member"] == 7
    assert config["seed"] == 7


def test_declaring_a_member_does_not_unfreeze_anything_else(contract):
    """The escape hatch must cover the seed and nothing more."""
    config = set_seed(read_config("config/current_model.yaml"), 7)
    config["b7_batch_size"] = 4
    with pytest.raises(ValueError, match="freezes b7_batch_size"):
        contract(config)


def test_training_records_whether_the_run_was_a_member():
    import inspect

    from rsna_knee import phase9_matched_supervision_training as trainer

    assert '"ensemble_member"' in inspect.getsource(trainer)
