"""A retargeted run must clear every contract on the way into training.

The first attempt at this failed at `require_b20_contract`, which reaches the
frozen B7 policy check four levels down through B18, B17 and B13. Searching the
training module for that check found nothing, and the wrong conclusion was
drawn from its absence.

So this test does not read source. It builds the config a retargeted run
actually produces and puts it through the real contract chain, which is the
only thing that settles the question.
"""

from __future__ import annotations

import pytest

from model._implementation import read_config, set_label_confidence


@pytest.fixture(scope="module")
def contract():
    from model.bootstrap import ensure_developments_source

    ensure_developments_source()
    from rsna_knee.b20_crop_focus import require_b20_contract

    return require_b20_contract


def test_the_shipped_config_clears_the_chain(contract):
    """Guard the guard: the chain must accept an untouched config."""
    contract(read_config("config/current_model.yaml"))


def test_a_retargeted_config_clears_the_chain(contract):
    config = set_label_confidence(
        read_config("config/current_model.yaml"), positive_target=0.70
    )
    contract(config)


def test_both_targets_together_clear_the_chain(contract):
    config = set_label_confidence(
        read_config("config/current_model.yaml"),
        positive_target=0.70,
        negative_target=0.04,
    )
    contract(config)


def test_the_export_policy_is_still_protected(contract):
    """Retargeting must not have opened a hole in what the contract guards."""
    config = read_config("config/current_model.yaml")
    config["b7_positive_target"] = 0.70
    with pytest.raises(ValueError, match="frozen"):
        contract(config)


def test_the_retargeted_config_leaves_the_export_untouched():
    config = set_label_confidence(
        read_config("config/current_model.yaml"), positive_target=0.70
    )
    assert config["b7_positive_target"] == pytest.approx(0.85)
    assert config["label_confidence_positive_target"] == pytest.approx(0.70)
