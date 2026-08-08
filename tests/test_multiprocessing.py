from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset

from rsna_knee.runtime import resolve_runtime


def _collect(seed: int):
    runtime = resolve_runtime(
        {
            "device": "cpu",
            "requested_gpus": 1,
            "num_workers": 2,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "multiprocessing_context": "spawn",
        }
    )
    dataset = TensorDataset(torch.arange(32))
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        **runtime.loader_kwargs(seed=seed),
    )
    return torch.cat([batch[0] for batch in loader])


def test_two_process_loader_runs_and_is_reproducible(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    first = _collect(77)
    second = _collect(77)
    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(32))
