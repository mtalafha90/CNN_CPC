"""Tests for the GPU/parallel runtime resolution.

These run on CPU-only machines too: the point is that the resolver degrades
sensibly rather than assuming a GPU is present.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rsna_knee.runtime import (  # noqa: E402
    RuntimeConfig,
    default_workers,
    make_scaler,
    resolve_runtime,
    unwrap,
    wrap_parallel,
)


def test_auto_device_matches_availability():
    runtime = resolve_runtime({})
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert runtime.device.type == expected


def test_requesting_cuda_without_a_gpu_fails_loudly():
    """Silently training on CPU for hours is worse than an immediate error."""
    if torch.cuda.is_available():
        pytest.skip("a GPU is present, so the error path cannot trigger")
    with pytest.raises(RuntimeError, match="no CUDA device"):
        resolve_runtime({"device": "cuda"})


def test_cpu_runs_in_fp32():
    """Autocast on CPU is slower than plain fp32 for these models."""
    runtime = resolve_runtime({"device": "cpu"})
    assert runtime.amp_dtype is None
    assert runtime.use_scaler is False
    assert runtime.pin_memory is False


def test_explicit_precision_is_honoured():
    runtime = resolve_runtime({"device": "cpu", "precision": "fp32"})
    assert runtime.amp_dtype is None


def test_fp16_requests_a_scaler_and_bf16_does_not():
    """bf16 has fp32's exponent range, so it cannot overflow and needs no scaler."""
    if not torch.cuda.is_available():
        pytest.skip("precision selection only applies on CUDA")
    assert resolve_runtime({"precision": "fp16"}).use_scaler is True
    assert resolve_runtime({"precision": "bf16"}).use_scaler is False


def test_worker_count_defaults_to_the_core_count():
    workers = default_workers(None)
    assert 1 <= workers <= 16


def test_explicit_worker_count_is_respected():
    assert default_workers(0) == 0
    assert default_workers(3) == 3


def test_loader_kwargs_omit_prefetch_when_single_process():
    """prefetch_factor is invalid with num_workers=0 and would raise."""
    runtime = resolve_runtime({"device": "cpu", "num_workers": 0})
    kwargs = runtime.loader_kwargs()

    assert kwargs["num_workers"] == 0
    assert "prefetch_factor" not in kwargs
    assert kwargs["persistent_workers"] is False


def test_loader_kwargs_include_prefetch_when_parallel():
    runtime = resolve_runtime({"device": "cpu", "num_workers": 4})
    kwargs = runtime.loader_kwargs()

    assert kwargs["prefetch_factor"] == 4
    assert kwargs["persistent_workers"] is True


def test_loader_kwargs_are_accepted_by_dataloader():
    """Guards against a kwarg that torch would reject at run time."""
    from torch.utils.data import DataLoader, TensorDataset

    runtime = resolve_runtime({"device": "cpu", "num_workers": 0})
    dataset = TensorDataset(torch.zeros(4, 2))
    loader = DataLoader(dataset, batch_size=2, **runtime.loader_kwargs())

    assert len(list(loader)) == 2


def test_single_gpu_or_cpu_is_not_wrapped():
    model = torch.nn.Linear(2, 2)
    runtime = resolve_runtime({"device": "cpu"})

    assert wrap_parallel(model, runtime) is model


def test_unwrap_returns_the_inner_module():
    """Checkpoints must not carry a `module.` prefix from DataParallel."""
    model = torch.nn.Linear(2, 2)
    wrapped = torch.nn.DataParallel(model)

    assert unwrap(wrapped) is model
    assert unwrap(model) is model
    assert all(not k.startswith("module.") for k in unwrap(wrapped).state_dict())


def test_scaler_is_disabled_without_fp16():
    runtime = resolve_runtime({"device": "cpu"})
    assert make_scaler(runtime).is_enabled() is False


def test_describe_mentions_precision_and_workers():
    text = resolve_runtime({"device": "cpu", "num_workers": 5}).describe()
    assert "fp32" in text and "workers=5" in text


def test_runtime_config_is_constructible_directly():
    runtime = RuntimeConfig(
        device=torch.device("cpu"), amp_dtype=None, use_scaler=False, num_workers=2,
        pin_memory=False, persistent_workers=True, prefetch_factor=2, n_gpus=0,
        device_name="cpu",
    )
    assert runtime.loader_kwargs()["num_workers"] == 2
