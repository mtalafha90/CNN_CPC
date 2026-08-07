"""Distributed runtime, mixed precision, and DICOM DataLoader settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


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
    distributed: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def describe(self) -> str:
        precision = {torch.bfloat16:"bf16",torch.float16:"fp16",None:"fp32"}[self.amp_dtype]
        mode = f"ddp {self.rank}/{self.world_size}" if self.distributed else "single"
        return f"device={self.device_name} | {mode} | precision={precision} | workers={self.num_workers} | visible_gpus={self.visible_gpus}"

    def loader_kwargs(self) -> dict:
        kwargs={"num_workers":self.num_workers,"pin_memory":self.pin_memory,"persistent_workers":self.persistent_workers and self.num_workers>0}
        if self.num_workers>0 and self.prefetch_factor is not None: kwargs["prefetch_factor"]=self.prefetch_factor
        return kwargs


def supports_bfloat16(device: torch.device | None = None) -> bool:
    if not torch.cuda.is_available(): return False
    try:
        major,_=torch.cuda.get_device_capability(device)
        return major>=8
    except Exception:
        return False


def default_workers(requested: int | None=None) -> int:
    if requested is not None:
        requested=int(requested)
        if requested<0: raise ValueError("num_workers must be >=0 or null")
        return requested
    cores=os.cpu_count() or 4
    return max(1,min(16,cores-1))


def _distributed_env() -> tuple[int,int,int]:
    world=int(os.environ.get("WORLD_SIZE","1")); rank=int(os.environ.get("RANK","0")); local=int(os.environ.get("LOCAL_RANK","0"))
    return world,rank,local


def resolve_runtime(config: dict | None=None) -> RuntimeConfig:
    config=config or {}; world,rank,local_rank=_distributed_env(); distributed=world>1
    requested=str(config.get("device","auto")).lower()
    if distributed:
        if not torch.cuda.is_available(): raise RuntimeError("DDP currently requires CUDA/NCCL")
        if local_rank>=torch.cuda.device_count(): raise RuntimeError(f"LOCAL_RANK={local_rank} exceeds visible CUDA devices")
        torch.cuda.set_device(local_rank); device=torch.device(f"cuda:{local_rank}"); use_cuda=True
        if not dist.is_initialized(): dist.init_process_group(backend=str(config.get("ddp_backend","nccl")),init_method="env://")
    elif requested=="auto":
        use_cuda=torch.cuda.is_available(); device=torch.device("cuda:0" if use_cuda else "cpu")
    elif requested=="cpu": use_cuda=False; device=torch.device("cpu")
    elif requested.startswith("cuda"):
        if not torch.cuda.is_available(): raise RuntimeError("CUDA device requested but no CUDA device is visible")
        use_cuda=True; device=torch.device(requested if ":" in requested else "cuda:0")
    else: raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")

    visible=torch.cuda.device_count() if torch.cuda.is_available() else 0; name=torch.cuda.get_device_name(device) if use_cuda else "cpu"
    precision=str(config.get("precision","auto")).lower()
    if not use_cuda: amp_dtype,use_scaler=None,False
    elif precision=="auto": amp_dtype=torch.bfloat16 if supports_bfloat16(device) else torch.float16; use_scaler=amp_dtype is torch.float16
    elif precision in {"bf16","bfloat16"}:
        if not supports_bfloat16(device): raise RuntimeError("bf16 requested but unsupported")
        amp_dtype,use_scaler=torch.bfloat16,False
    elif precision in {"fp16","float16","half"}: amp_dtype,use_scaler=torch.float16,True
    elif precision in {"fp32","float32","full"}: amp_dtype,use_scaler=None,False
    else: raise ValueError("precision must be auto, bf16, fp16, or fp32")
    if use_cuda:
        torch.backends.cudnn.benchmark=True; torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
        try: torch.set_float32_matmul_precision("high")
        except AttributeError: pass
    prefetch=config.get("prefetch_factor",4); prefetch=None if prefetch is None else int(prefetch)
    return RuntimeConfig(device,amp_dtype,use_scaler,default_workers(config.get("num_workers")),use_cuda,bool(config.get("persistent_workers",True)),prefetch,visible,name,distributed,rank,local_rank,world)


def barrier(runtime: RuntimeConfig) -> None:
    if runtime.distributed: dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized(): dist.destroy_process_group()


def global_loss_batch(logits: torch.Tensor,target: torch.Tensor,weight: torch.Tensor,runtime: RuntimeConfig):
    """Gather a differentiable global logit batch plus detached labels/weights."""
    if not runtime.distributed: return logits,target,weight
    from torch.distributed.nn.functional import all_gather as differentiable_all_gather
    gathered_logits=differentiable_all_gather(logits)
    logits_global=torch.cat(list(gathered_logits),dim=0)
    targets=[torch.empty_like(target) for _ in range(runtime.world_size)]; weights=[torch.empty_like(weight) for _ in range(runtime.world_size)]
    dist.all_gather(targets,target.detach()); dist.all_gather(weights,weight.detach())
    return logits_global,torch.cat(targets,dim=0),torch.cat(weights,dim=0)


def make_scaler(runtime: RuntimeConfig):
    try: return torch.amp.GradScaler(runtime.device.type,enabled=runtime.use_scaler)
    except (AttributeError,TypeError): return torch.cuda.amp.GradScaler(enabled=runtime.use_scaler)


def autocast(runtime: RuntimeConfig):
    return torch.autocast(device_type=runtime.device.type,dtype=runtime.amp_dtype or torch.float32,enabled=runtime.amp_dtype is not None)
