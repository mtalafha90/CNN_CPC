"""Resuming a run that died thirty hours in.

The tests that matter are the ones about *exactness*. Restoring weights alone
is a warm start, not a resume: Adam's moments and the learning-rate schedule
change what the next step does, so a run continued without them is a different
run. Each of those is pinned by comparing against an uninterrupted run.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from rsna_knee.training_resume import (
    RECOVERY_NAME,
    ResumeState,
    load_checkpoint,
    resume,
    rng_state,
    save_checkpoint,
    set_rng_state,
)

VERSION = "test_experiment_v1"


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _step(model, optimizer, scheduler=None, *, value: float = 1.0) -> float:
    optimizer.zero_grad()
    loss = model(torch.full((3, 4), value)).square().mean()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return float(loss.detach())


# --- what is saved ------------------------------------------------------------


def test_a_missing_checkpoint_is_not_an_error(tmp_path):
    assert load_checkpoint(tmp_path) is None


def test_nothing_to_resume_starts_at_epoch_one(tmp_path):
    state = resume(tmp_path, model=_model(), version=VERSION)
    assert state.restored is False
    assert state.start_epoch == 1
    assert "starting from epoch 1" in state.describe()


def test_the_checkpoint_carries_every_component(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    _step(model, optimizer, scheduler)

    save_checkpoint(
        tmp_path,
        epoch=3,
        model=model,
        version=VERSION,
        optimizer=optimizer,
        scheduler=scheduler,
        history=[{"epoch": 1}],
    )
    payload = load_checkpoint(tmp_path)

    assert payload["epoch"] == 3
    assert payload["version"] == VERSION
    for key in ("model_state", "optimizer_state", "scheduler_state", "rng_state"):
        assert payload[key] is not None, key
    assert payload["history"] == [{"epoch": 1}]


def test_the_write_is_atomic(tmp_path):
    """No temporary file may survive, or the next load could read a stub."""
    save_checkpoint(tmp_path, epoch=1, model=_model(), version=VERSION)

    assert (tmp_path / RECOVERY_NAME).is_file()
    assert not list(tmp_path.glob("*.writing"))


def test_a_second_save_replaces_the_first(tmp_path):
    save_checkpoint(tmp_path, epoch=1, model=_model(), version=VERSION)
    save_checkpoint(tmp_path, epoch=2, model=_model(), version=VERSION)

    assert load_checkpoint(tmp_path)["epoch"] == 2
    assert len(list(tmp_path.glob("recovery_latest*"))) == 1


def test_extra_fields_survive_the_round_trip(tmp_path):
    save_checkpoint(
        tmp_path, epoch=1, model=_model(), version=VERSION, extra={"teacher": "b54"}
    )
    assert resume(tmp_path, model=_model(), version=VERSION).extra == {"teacher": "b54"}


# --- what is restored ---------------------------------------------------------


def test_the_next_epoch_is_the_one_after_the_saved_one(tmp_path):
    save_checkpoint(tmp_path, epoch=7, model=_model(), version=VERSION)
    assert resume(tmp_path, model=_model(), version=VERSION).start_epoch == 8


def test_the_weights_come_back(tmp_path):
    trained = _model()
    optimizer = torch.optim.AdamW(trained.parameters(), lr=0.1)
    _step(trained, optimizer)
    save_checkpoint(tmp_path, epoch=1, model=trained, version=VERSION)

    restored = _model(seed=99)
    resume(tmp_path, model=restored, version=VERSION)

    for a, b in zip(trained.state_dict().values(), restored.state_dict().values()):
        assert torch.allclose(a, b)


def test_the_optimiser_state_comes_back(tmp_path):
    """Adam's moments are half of what the next step does."""
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    _step(model, optimizer)
    save_checkpoint(
        tmp_path, epoch=1, model=model, version=VERSION, optimizer=optimizer
    )

    fresh_model = _model()
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.1)
    state = resume(
        tmp_path, model=fresh_model, version=VERSION, optimizer=fresh_optimizer
    )

    assert "optimizer" in state.parts
    saved = optimizer.state_dict()["state"]
    loaded = fresh_optimizer.state_dict()["state"]
    assert set(saved) == set(loaded)
    for key in saved:
        assert torch.allclose(saved[key]["exp_avg"], loaded[key]["exp_avg"])


def test_the_schedule_does_not_restart_from_step_zero(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    for _ in range(3):
        _step(model, optimizer, scheduler)
    save_checkpoint(
        tmp_path,
        epoch=1,
        model=model,
        version=VERSION,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    fresh_model = _model()
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.1)
    fresh_scheduler = torch.optim.lr_scheduler.StepLR(
        fresh_optimizer, step_size=1, gamma=0.5
    )
    resume(
        tmp_path,
        model=fresh_model,
        version=VERSION,
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
    )

    assert fresh_scheduler.get_last_lr() == scheduler.get_last_lr()
    assert fresh_scheduler.get_last_lr()[0] < 0.1


def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path):
    """The whole point, stated as one comparison."""
    steps = 6
    interrupt = 3

    straight_model = _model()
    straight_optimizer = torch.optim.AdamW(straight_model.parameters(), lr=0.05)
    straight_scheduler = torch.optim.lr_scheduler.StepLR(
        straight_optimizer, step_size=2, gamma=0.5
    )
    torch.manual_seed(1234)
    for _ in range(steps):
        _step(straight_model, straight_optimizer, straight_scheduler)

    crashed_model = _model()
    crashed_optimizer = torch.optim.AdamW(crashed_model.parameters(), lr=0.05)
    crashed_scheduler = torch.optim.lr_scheduler.StepLR(
        crashed_optimizer, step_size=2, gamma=0.5
    )
    torch.manual_seed(1234)
    for _ in range(interrupt):
        _step(crashed_model, crashed_optimizer, crashed_scheduler)
    save_checkpoint(
        tmp_path,
        epoch=interrupt,
        model=crashed_model,
        version=VERSION,
        optimizer=crashed_optimizer,
        scheduler=crashed_scheduler,
    )

    revived_model = _model(seed=7)
    revived_optimizer = torch.optim.AdamW(revived_model.parameters(), lr=0.05)
    revived_scheduler = torch.optim.lr_scheduler.StepLR(
        revived_optimizer, step_size=2, gamma=0.5
    )
    resume(
        tmp_path,
        model=revived_model,
        version=VERSION,
        optimizer=revived_optimizer,
        scheduler=revived_scheduler,
    )
    for _ in range(steps - interrupt):
        _step(revived_model, revived_optimizer, revived_scheduler)

    for a, b in zip(
        straight_model.state_dict().values(), revived_model.state_dict().values()
    ):
        assert torch.allclose(a, b, atol=1e-6)


def test_the_history_comes_back_so_it_is_not_rewritten_from_scratch(tmp_path):
    save_checkpoint(
        tmp_path,
        epoch=2,
        model=_model(),
        version=VERSION,
        history=[{"epoch": 1, "loss": 1.0}, {"epoch": 2, "loss": 0.5}],
    )
    state = resume(tmp_path, model=_model(), version=VERSION)

    assert len(state.history) == 2
    assert state.history[-1]["loss"] == 0.5


# --- what is refused ----------------------------------------------------------


def test_a_checkpoint_from_a_different_experiment_is_refused(tmp_path):
    save_checkpoint(tmp_path, epoch=1, model=_model(), version="b47_something")

    with pytest.raises(ValueError, match="refusing to resume"):
        resume(tmp_path, model=_model(), version="b54_something_else")


def test_the_refusal_names_both_versions(tmp_path):
    save_checkpoint(tmp_path, epoch=1, model=_model(), version="written_by_this")
    with pytest.raises(ValueError) as error:
        resume(tmp_path, model=_model(), version="running_this")

    assert "written_by_this" in str(error.value)
    assert "running_this" in str(error.value)


def test_a_mismatched_architecture_raises_rather_than_partially_loading(tmp_path):
    save_checkpoint(tmp_path, epoch=1, model=_model(), version=VERSION)
    wider = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))

    with pytest.raises(RuntimeError):
        resume(tmp_path, model=wider, version=VERSION)


# --- the random generators ----------------------------------------------------


def test_the_generators_round_trip():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    saved = rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    random.random(), np.random.rand(), torch.rand(1)  # move them on
    set_rng_state(saved)

    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected


def test_restoring_reports_what_it_skipped():
    report = set_rng_state({"python": random.getstate(), "cuda": None})

    assert "python" in report["restored"]
    assert "cuda" in report["skipped"]


def test_restoring_nothing_is_reported_not_raised():
    assert set_rng_state(None)["skipped"] == ["all"]


def test_a_cuda_state_from_a_different_device_count_is_skipped_not_fatal():
    """A run moved between machines must still resume."""
    report = set_rng_state({"cuda": [torch.zeros(16, dtype=torch.uint8)] * 99})
    assert any("cuda" in item for item in report["skipped"])


# --- the description ----------------------------------------------------------


def test_the_description_says_what_happened(tmp_path):
    save_checkpoint(tmp_path, epoch=4, model=_model(), version=VERSION)
    state = resume(tmp_path, model=_model(), version=VERSION)

    assert "resumed after epoch 4" in state.describe()
    assert isinstance(state, ResumeState)
