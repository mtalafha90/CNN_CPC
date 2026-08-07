"""Single-device runtime, mixed precision, and DICOM DataLoader settings.

Multi-GPU execution is intentionally not implemented here. The next development
step is proper DistributedDataParallel (DDP), so this clean baseline avoids
keeping the slower and less predictable ``nn.DataParallel`` path around.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


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

    def describe(self) -> str:
        precision = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            None: "fp32",
        }[self.amp_dtype]
        return (
            f"device={self.device_name} | precision={precision} | "
            f"workers={self.num_workers} | visible_gpus={self.visible_gpus}"
        )

    def loader_kwargs(self) -> dict:
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers and self.num_workers > 0,
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs


def supports_bfloat16() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def default_workers(requested: int | None = None) -> int:
    if requested is not None:
        requested = int(requested)
        if requested < 0:
            raise ValueError("num_workers must be >= 0 or null")
        return requested
    cores = os.cpu_count() or 4
    return max(1, min(16, cores - 1))


def resolve_runtime(config: dict | None = None) -> RuntimeConfig:
    config = config or {}
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
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
            )
    else:
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")

    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_name = torch.cuda.get_device_name(device) if use_cuda else "cpu"

    precision = str(config.get("precision", "auto")).lower()
    if not use_cuda:
        amp_dtype, use_scaler = None, False
    elif precision == "auto":
        amp_dtype = torch.bfloat16 if supports_bfloat16() else torch.float16
        use_scaler = amp_dtype is torch.float16
    elif precision in {"bf16", "bfloat16"}:
        if not supports_bfloat16():
            raise RuntimeError("bf16 requested but the selected GPU does not support native bf16")
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

    prefetch = config.get("prefetch_factor", 4)
    prefetch_factor = None if prefetch is None else int(prefetch)
    if prefetch_factor is not None and prefetch_factor < 1:
        raise ValueError("prefetch_factor must be >= 1 or null")

    return RuntimeConfig(
        device=device,
        amp_dtype=amp_dtype,
        use_scaler=use_scaler,
        num_workers=default_workers(config.get("num_workers")),
        pin_memory=use_cuda,
        persistent_workers=bool(config.get("persistent_workers", True)),
        prefetch_factor=prefetch_factor,
        visible_gpus=visible_gpus,
        device_name=device_name,
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
