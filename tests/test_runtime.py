from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rsna_knee.runtime import RuntimeConfig, default_workers, make_scaler, resolve_runtime


def test_auto_device_matches_availability():
    runtime = resolve_runtime({})
    assert runtime.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert runtime.world_size == 1 and runtime.rank == 0 and not runtime.distributed


def test_requesting_cuda_without_a_gpu_fails_loudly():
    if torch.cuda.is_available():
        pytest.skip("GPU present")
    with pytest.raises(RuntimeError, match="no CUDA device"):
        resolve_runtime({"device": "cuda"})


def test_cpu_runs_in_fp32():
    runtime = resolve_runtime({"device": "cpu"})
    assert runtime.amp_dtype is None and runtime.use_scaler is False and runtime.pin_memory is False


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="device must"):
        resolve_runtime({"device": "mps"})


def test_worker_count_defaults_to_sensible_range():
    assert 1 <= default_workers(None) <= 16


def test_explicit_worker_count_is_respected():
    assert default_workers(0) == 0
    assert default_workers(3) == 3
    with pytest.raises(ValueError):
        default_workers(-1)


def test_loader_kwargs_omit_prefetch_when_single_process():
    kwargs = resolve_runtime({"device": "cpu", "num_workers": 0}).loader_kwargs()
    assert kwargs["num_workers"] == 0 and "prefetch_factor" not in kwargs and kwargs["persistent_workers"] is False


def test_loader_kwargs_include_prefetch_when_parallel_cpu_loading():
    kwargs = resolve_runtime({"device": "cpu", "num_workers": 4, "prefetch_factor": 2}).loader_kwargs()
    assert kwargs["prefetch_factor"] == 2 and kwargs["persistent_workers"] is True


def test_loader_kwargs_are_accepted_by_dataloader():
    from torch.utils.data import DataLoader, TensorDataset

    runtime = resolve_runtime({"device": "cpu", "num_workers": 0})
    loader = DataLoader(TensorDataset(torch.zeros(4, 2)), batch_size=2, **runtime.loader_kwargs())
    assert len(list(loader)) == 2


def test_scaler_is_disabled_without_fp16():
    assert make_scaler(resolve_runtime({"device": "cpu"})).is_enabled() is False


def test_describe_mentions_distributed_state():
    text = resolve_runtime({"device": "cpu", "num_workers": 5}).describe()
    assert "fp32" in text and "workers=5" in text and "single" in text


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
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
    )
    assert runtime.is_main and runtime.loader_kwargs()["num_workers"] == 2
