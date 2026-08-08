"""Single-GPU runtime with deterministic CPU multiprocessing."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def seed_worker(worker_id: int) -> None:
    """Deterministically seed Python/NumPy inside each DataLoader process."""
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


@dataclass
class RuntimeConfig:
    device: torch.device
    amp_dtype: torch.dtype | None
    use_scaler: bool
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    visible_gpus: int
    device_name: str
    multiprocessing_context: str | None

    @property
    def is_main(self) -> bool:
        return True

    @property
    def distributed(self) -> bool:
        return False

    @property
    def rank(self) -> int:
        return 0

    @property
    def local_rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    def describe(self) -> str:
        precision = {torch.bfloat16: "bf16", torch.float16: "fp16", None: "fp32"}[self.amp_dtype]
        return (
            f"device={self.device_name} | single-gpu | precision={precision} | "
            f"workers={self.num_workers} | visible_gpus={self.visible_gpus}"
        )

    def loader_kwargs(self, *, seed: int | None = None) -> dict:
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers and self.num_workers > 0,
        }
        if self.num_workers > 0:
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
            if self.multiprocessing_context:
                kwargs["multiprocessing_context"] = self.multiprocessing_context
            kwargs["worker_init_fn"] = seed_worker
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            kwargs["generator"] = generator
        return kwargs


def supports_bfloat16(device: torch.device | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device)
        return major >= 8
    except Exception:
        return False


def default_workers(requested: int | None = None) -> int:
    if requested is not None:
        requested = int(requested)
        if requested < 0:
            raise ValueError("num_workers must be >=0 or null")
        return requested
    cores = os.cpu_count() or 2
    if cores <= 2:
        return 1
    return min(6, max(2, cores // 2))


def resolve_runtime(config: dict | None = None) -> RuntimeConfig:
    config = config or {}
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("multi-GPU/DDP is disabled: launch one process on one GPU")
    if int(config.get("requested_gpus", 1)) != 1:
        raise ValueError("requested_gpus must be 1")

    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
    elif requested == "cpu":
        use_cuda = False
        device = torch.device("cpu")
    elif requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but no CUDA device is visible")
        use_cuda = True
        device = torch.device(requested if ":" in requested else "cuda:0")
        index = 0 if device.index is None else int(device.index)
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"requested {device}, but only {torch.cuda.device_count()} GPU(s) are visible")
        torch.cuda.set_device(index)
    else:
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")

    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    name = torch.cuda.get_device_name(device) if use_cuda else "cpu"
    precision = str(config.get("precision", "auto")).lower()
    if not use_cuda:
        amp_dtype, use_scaler = None, False
    elif precision == "auto":
        amp_dtype = torch.bfloat16 if supports_bfloat16(device) else torch.float16
        use_scaler = amp_dtype is torch.float16
    elif precision in {"bf16", "bfloat16"}:
        if not supports_bfloat16(device):
            raise RuntimeError("bf16 requested but unsupported")
        amp_dtype, use_scaler = torch.bfloat16, False
    elif precision in {"fp16", "float16", "half"}:
        amp_dtype, use_scaler = torch.float16, True
    elif precision in {"fp32", "float32", "full"}:
        amp_dtype, use_scaler = None, False
    else:
        raise ValueError("precision must be auto, bf16, fp16, or fp32")

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    prefetch = config.get("prefetch_factor", 2)
    prefetch = None if prefetch is None else int(prefetch)
    if prefetch is not None and prefetch < 1:
        raise ValueError("prefetch_factor must be >=1 or null")
    context = str(config.get("multiprocessing_context", "spawn")) if default_workers(config.get("num_workers")) > 0 else None
    if context not in {None, "spawn", "fork", "forkserver"}:
        raise ValueError("multiprocessing_context must be spawn, fork, forkserver, or null")

    return RuntimeConfig(
        device=device,
        amp_dtype=amp_dtype,
        use_scaler=use_scaler,
        num_workers=default_workers(config.get("num_workers")),
        pin_memory=use_cuda,
        persistent_workers=bool(config.get("persistent_workers", True)),
        prefetch_factor=prefetch,
        visible_gpus=visible,
        device_name=name,
        multiprocessing_context=context,
    )


def make_scaler(runtime: RuntimeConfig):
    try:
        return torch.amp.GradScaler(runtime.device.type, enabled=runtime.use_scaler)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=runtime.use_scaler)


def autocast(runtime: RuntimeConfig):
    return torch.autocast(
        device_type=runtime.device.type,
        dtype=runtime.amp_dtype or torch.float32,
        enabled=runtime.amp_dtype is not None,
    )
