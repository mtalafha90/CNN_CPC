"""Device, precision and data-loading setup for local GPU training.

Everything here is about making one local machine work hard without the caller
having to know the details. Three things dominate throughput on this dataset:

1. **Precision.** bf16 on Ampere and newer (RTX 30xx/40xx/50xx, A100, H100)
   needs no gradient scaler and cannot overflow; fp16 elsewhere. Hard-coding
   fp16 costs stability on new cards and hard-coding bf16 breaks old ones.

2. **Data loading.** Decoding DICOM is the bottleneck, not the GPU. A single
   worker starves an RTX 4090 completely. Workers, prefetching and persistence
   matter more here than any model change.

3. **Multiple GPUs.** `DataParallel` splits each batch across devices in one
   process, which suits a workstation with two cards and keeps the training
   script single-process. It is not as fast as DDP, but it works without a
   launcher — a fair trade until multi-node matters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RuntimeConfig:
    """Resolved hardware settings for a run."""

    device: torch.device
    amp_dtype: torch.dtype | None
    use_scaler: bool
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    n_gpus: int
    device_name: str

    def describe(self) -> str:
        precision = {
            torch.bfloat16: "bf16", torch.float16: "fp16", None: "fp32",
        }[self.amp_dtype]
        gpus = f"{self.n_gpus}x " if self.n_gpus > 1 else ""
        return (
            f"device={gpus}{self.device_name} | precision={precision} | "
            f"workers={self.num_workers}"
        )

    def loader_kwargs(self) -> dict:
        """Keyword arguments for a training DataLoader."""
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers and self.num_workers > 0,
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs


def supports_bfloat16() -> bool:
    """Whether the current GPU handles bf16 natively (compute capability >= 8)."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def default_workers(requested: int | None = None) -> int:
    """Choose a sensible DataLoader worker count.

    DICOM decoding is CPU bound, so more workers help right up to the core
    count. One core is left free for the main process, and the total is capped
    at 16 because beyond that the workers contend for disk rather than helping.
    """
    if requested is not None and requested >= 0:
        return requested
    cores = os.cpu_count() or 4
    return max(1, min(16, cores - 1))


def resolve_runtime(config: dict | None = None) -> RuntimeConfig:
    """Work out how to run on this machine, honouring explicit overrides."""
    config = config or {}

    requested_device = str(config.get("device", "auto")).lower()
    if requested_device == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = requested_device.startswith("cuda")
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError("device=cuda was requested but no CUDA device is visible")

    device = torch.device("cuda" if use_cuda else "cpu")
    n_gpus = torch.cuda.device_count() if use_cuda else 0
    device_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"

    precision = str(config.get("precision", "auto")).lower()
    if not use_cuda:
        # Autocast on CPU is slower than plain fp32 for these small models.
        amp_dtype, use_scaler = None, False
    elif precision == "auto":
        amp_dtype = torch.bfloat16 if supports_bfloat16() else torch.float16
        use_scaler = amp_dtype is torch.float16
    elif precision in {"bf16", "bfloat16"}:
        amp_dtype, use_scaler = torch.bfloat16, False
    elif precision in {"fp16", "float16", "half"}:
        amp_dtype, use_scaler = torch.float16, True
    else:
        amp_dtype, use_scaler = None, False

    if use_cuda:
        # Fixed-shape batches make autotuned convolution algorithms worthwhile.
        torch.backends.cudnn.benchmark = True
        # TF32 costs nothing noticeable in accuracy and speeds up matmuls a lot.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    workers = default_workers(config.get("num_workers"))
    return RuntimeConfig(
        device=device,
        amp_dtype=amp_dtype,
        use_scaler=use_scaler,
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=bool(config.get("persistent_workers", True)),
        prefetch_factor=int(config.get("prefetch_factor", 4)),
        n_gpus=n_gpus,
        device_name=device_name,
    )


def wrap_parallel(model: nn.Module, runtime: RuntimeConfig, enabled: bool = True) -> nn.Module:
    """Spread the batch across every visible GPU, when there is more than one.

    The batch is split along dimension 0, so the effective batch size must be
    at least the GPU count for every card to receive work.
    """
    if enabled and runtime.n_gpus > 1:
        return nn.DataParallel(model)
    return model


def unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module, so checkpoints never carry a `module.` prefix."""
    return model.module if isinstance(model, nn.DataParallel) else model


def make_scaler(runtime: RuntimeConfig):
    """Create a gradient scaler when fp16 needs one, else a disabled scaler."""
    try:
        return torch.amp.GradScaler(runtime.device.type, enabled=runtime.use_scaler)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=runtime.use_scaler)


def autocast(runtime: RuntimeConfig):
    """Autocast context for the resolved precision."""
    return torch.autocast(
        device_type=runtime.device.type,
        dtype=runtime.amp_dtype or torch.float32,
        enabled=runtime.amp_dtype is not None,
    )
