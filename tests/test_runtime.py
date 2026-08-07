from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rsna_knee.runtime import (  # noqa: E402
    RuntimeConfig,
    default_workers,
    make_scaler,
    resolve_runtime,
)


def test_auto_device_matches_availability():
    runtime = resolve_runtime({})
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert runtime.device.type == expected


def test_requesting_cuda_without_a_gpu_fails_loudly():
    if torch.cuda.is_available():
        pytest.skip("GPU present")
    with pytest.raises(RuntimeError, match="no CUDA device"):
        resolve_runtime({"device": "cuda"})


def test_cpu_runs_in_fp32():
    runtime = resolve_runtime({"device": "cpu"})
    assert runtime.amp_dtype is None
    assert runtime.use_scaler is False
    assert runtime.pin_memory is False


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="device must"):
        resolve_runtime({"device": "mps"})


def test_worker_count_defaults_to_sensible_range():
    workers = default_workers(None)
    assert 1 <= workers <= 16


def test_explicit_worker_count_is_respected():
    assert default_workers(0) == 0
    assert default_workers(3) == 3
    with pytest.raises(ValueError):
        default_workers(-1)


def test_loader_kwargs_omit_prefetch_when_single_process():
    runtime = resolve_runtime({"device": "cpu", "num_workers": 0})
    kwargs = runtime.loader_kwargs()
    assert kwargs["num_workers"] == 0
    assert "prefetch_factor" not in kwargs
    assert kwargs["persistent_workers"] is False


def test_loader_kwargs_include_prefetch_when_parallel_cpu_loading():
    runtime = resolve_runtime({"device": "cpu", "num_workers": 4, "prefetch_factor": 2})
    kwargs = runtime.loader_kwargs()
    assert kwargs["prefetch_factor"] == 2
    assert kwargs["persistent_workers"] is True


def test_loader_kwargs_are_accepted_by_dataloader():
    from torch.utils.data import DataLoader, TensorDataset

    runtime = resolve_runtime({"device": "cpu", "num_workers": 0})
    loader = DataLoader(TensorDataset(torch.zeros(4, 2)), batch_size=2, **runtime.loader_kwargs())
    assert len(list(loader)) == 2


def test_scaler_is_disabled_without_fp16():
    runtime = resolve_runtime({"device": "cpu"})
    assert make_scaler(runtime).is_enabled() is False


def test_describe_mentions_visible_gpu_count():
    text = resolve_runtime({"device": "cpu", "num_workers": 5}).describe()
    assert "fp32" in text and "workers=5" in text and "visible_gpus=" in text


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
    )
    assert runtime.loader_kwargs()["num_workers"] == 2
