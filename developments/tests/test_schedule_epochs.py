"""The learning-rate schedule may be given a horizon that matches the run.

The cosine's `T_max` was five while training stopped at two, so the rate never
came down: the two epochs ran at 100% and 90.5% of peak and the low-rate
refinement phase never happened at all. Setting the horizon to two completes
the cosine at exactly the same cost.

Two things have to hold at once. The horizon has to be settable, and the frozen
epoch count it is easily confused with -- `b18_candidate_epochs`, which names a
different experiment and is contract-checked -- must stay frozen. As with the
label targets and the seed, these tests put the config a real run produces
through the real contract chain rather than reading the source for a guard that
might be four levels down.
"""

from __future__ import annotations

import pytest

from model._implementation import (
    SCHEDULE_EPOCHS_KEY,
    read_config,
    set_schedule_epochs,
)


@pytest.fixture(scope="module")
def contract():
    from model.bootstrap import ensure_developments_source

    ensure_developments_source()
    from rsna_knee.b20_crop_focus import require_b20_contract

    return require_b20_contract


def test_a_shortened_horizon_clears_the_chain(contract):
    contract(set_schedule_epochs(read_config("config/current_model.yaml"), 2))


def test_the_frozen_epoch_count_is_still_frozen(contract):
    """The key this one is easily confused with must not have moved."""
    config = read_config("config/current_model.yaml")
    config["b18_candidate_epochs"] = 2
    with pytest.raises(ValueError, match="b18_candidate_epochs"):
        contract(config)


def test_setting_the_horizon_does_not_unfreeze_anything_else(contract):
    config = set_schedule_epochs(read_config("config/current_model.yaml"), 2)
    config["b7_batch_size"] = 4
    with pytest.raises(ValueError, match="freezes b7_batch_size"):
        contract(config)


def test_no_horizon_given_leaves_the_config_alone():
    original = read_config("config/current_model.yaml")
    assert set_schedule_epochs(original, None) is original


def test_the_caller_config_is_not_modified():
    original = read_config("config/current_model.yaml")
    set_schedule_epochs(original, 2)
    assert SCHEDULE_EPOCHS_KEY not in original


def test_a_horizon_below_one_is_refused():
    config = read_config("config/current_model.yaml")
    for bad in (0, -1):
        with pytest.raises(ValueError, match="at least 1"):
            set_schedule_epochs(config, bad)


def test_the_default_horizon_is_the_one_that_shipped():
    """Omitting the flag must reproduce the historical schedule exactly."""
    from model.bootstrap import ensure_developments_source

    ensure_developments_source()
    from rsna_knee.b18_fisher_selection import B18_CANDIDATE_EPOCHS

    config = read_config("config/current_model.yaml")
    assert config.get(SCHEDULE_EPOCHS_KEY, B18_CANDIDATE_EPOCHS) == B18_CANDIDATE_EPOCHS


def test_the_horizon_reaches_the_scheduler_and_is_recorded():
    import inspect

    from model.bootstrap import ensure_developments_source

    ensure_developments_source()
    from rsna_knee import phase9_matched_supervision_training as trainer

    source = inspect.getsource(trainer)
    assert "T_max=schedule_epochs" in source
    assert f'"{SCHEDULE_EPOCHS_KEY}": schedule_epochs' in source


def test_the_scheduler_actually_anneals_over_the_horizon_given():
    """The point of the flag: at the end of a two-epoch run the rate is at its floor."""
    import torch

    peak, floor = 1e-3, 1e-6
    rates = {}
    for horizon in (5, 2):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=peak)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=horizon, eta_min=floor
        )
        seen = [optimizer.param_groups[0]["lr"]]
        for _ in range(2):  # PHASE9_FIXED_EPOCHS
            scheduler.step()
            seen.append(optimizer.param_groups[0]["lr"])
        rates[horizon] = seen

    # The shipped horizon never comes down: both trained epochs are near peak.
    assert rates[5][0] == pytest.approx(peak)
    assert rates[5][1] / peak > 0.9
    assert rates[5][2] / peak > 0.6

    # The matched horizon finishes the cosine inside the run.
    assert rates[2][2] == pytest.approx(floor, abs=1e-9)
