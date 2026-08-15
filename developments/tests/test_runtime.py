from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rsna_knee.runtime import RuntimeConfig, default_workers, make_scaler, resolve_runtime


def test_auto_device_matches_availability(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    runtime = resolve_runtime({})
    assert runtime.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert runtime.world_size == 1 and runtime.rank == 0 and not runtime.distributed


def test_multi_gpu_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="multi-GPU/DDP is disabled"):
        resolve_runtime({"device": "cpu"})


def test_requesting_cuda_without_a_gpu_fails_loudly(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    if torch.cuda.is_available():
        pytest.skip("GPU present")
    with pytest.raises(RuntimeError, match="no CUDA device"):
        resolve_runtime({"device": "cuda"})


def test_cpu_runs_in_fp32(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    runtime = resolve_runtime({"device": "cpu"})
    assert runtime.amp_dtype is None and runtime.use_scaler is False and runtime.pin_memory is False


def test_requested_gpus_must_be_one(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="requested_gpus must be 1"):
        resolve_runtime({"device": "cpu", "requested_gpus": 2})


def test_invalid_device_is_rejected(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="device must"):
        resolve_runtime({"device": "mps"})


def test_worker_count_defaults_to_bounded_range():
    assert 1 <= default_workers(None) <= 6


def test_explicit_worker_count_is_respected():
    assert default_workers(0) == 0
    assert default_workers(3) == 3
    with pytest.raises(ValueError):
        default_workers(-1)


def test_loader_kwargs_omit_multiprocessing_when_single_process(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    kwargs = resolve_runtime({"device": "cpu", "num_workers": 0}).loader_kwargs(seed=7)
    assert kwargs["num_workers"] == 0
    assert "prefetch_factor" not in kwargs
    assert "multiprocessing_context" not in kwargs
    assert kwargs["persistent_workers"] is False
    assert isinstance(kwargs["generator"], torch.Generator)


def test_loader_kwargs_enable_cpu_multiprocessing(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    kwargs = resolve_runtime(
        {"device": "cpu", "num_workers": 2, "prefetch_factor": 2, "multiprocessing_context": "spawn"}
    ).loader_kwargs(seed=11)
    assert kwargs["num_workers"] == 2
    assert kwargs["prefetch_factor"] == 2
    assert kwargs["persistent_workers"] is True
    assert kwargs["multiprocessing_context"] == "spawn"
    assert callable(kwargs["worker_init_fn"])


def test_seeded_loader_generator_is_reproducible(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    runtime = resolve_runtime({"device": "cpu", "num_workers": 0})
    g1 = runtime.loader_kwargs(seed=123)["generator"]
    g2 = runtime.loader_kwargs(seed=123)["generator"]
    assert torch.equal(torch.rand(5, generator=g1), torch.rand(5, generator=g2))


def test_loader_kwargs_are_accepted_by_dataloader(monkeypatch):
    from torch.utils.data import DataLoader, TensorDataset

    monkeypatch.setenv("WORLD_SIZE", "1")
    runtime = resolve_runtime({"device": "cpu", "num_workers": 0})
    loader = DataLoader(
        TensorDataset(torch.zeros(4, 2)), batch_size=2, **runtime.loader_kwargs(seed=1)
    )
    assert len(list(loader)) == 2


def test_scaler_is_disabled_without_fp16(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert make_scaler(resolve_runtime({"device": "cpu"})).is_enabled() is False


def test_describe_mentions_single_gpu_and_workers(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    text = resolve_runtime({"device": "cpu", "num_workers": 5}).describe()
    assert "fp32" in text and "workers=5" in text and "single-gpu" in text


def test_runtime_config_is_constructible_directly():
    runtime = RuntimeConfig(
        device=torch.device("cpu"),
        amp_dtype=None,
        use_scaler=False,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
        visible_gpus=0,
        device_name="cpu",
        multiprocessing_context="spawn",
    )
    assert runtime.is_main and runtime.world_size == 1
