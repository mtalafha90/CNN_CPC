"""Balanced study sampling for weakly supervised MRI training."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class TwoPoolBatchSampler(Sampler[list[int]]):
    """Mix trusted studies with the general weak-supervision pool.

    ``trusted_fraction`` is enforced in expectation even for tiny local batches
    by accumulating a fractional quota across batches. Pools are independently
    reshuffled/cycled, so the 58 gold studies are seen regularly without forcing
    every batch to contain one and dramatically over-sampling them.

    In DDP each rank uses a different deterministic seed but the same quota
    schedule; therefore the *global* batch has the intended trusted fraction.
    """

    def __init__(
        self,
        trusted_mask: np.ndarray,
        batch_size: int,
        *,
        trusted_fraction: float = 0.30,
        seed: int = 2026,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = True,
    ) -> None:
        mask=np.asarray(trusted_mask,dtype=bool)
        if mask.ndim!=1: raise ValueError("trusted_mask must be one-dimensional")
        if batch_size<1: raise ValueError("batch_size must be positive")
        if not 0.0<=trusted_fraction<=1.0: raise ValueError("trusted_fraction must be in [0,1]")
        self.trusted=np.flatnonzero(mask); self.general=np.flatnonzero(~mask)
        if trusted_fraction>0 and len(self.trusted)==0: raise ValueError("trusted_fraction > 0 but trusted pool is empty")
        if trusted_fraction<1 and len(self.general)==0: raise ValueError("general pool is empty")
        self.n=len(mask); self.batch_size=int(batch_size); self.fraction=float(trusted_fraction); self.seed=int(seed); self.rank=int(rank); self.world_size=int(world_size); self.drop_last=bool(drop_last); self.epoch=0
        global_samples=math.ceil(self.n/max(1,self.world_size)); self.n_batches=(global_samples//self.batch_size if self.drop_last else math.ceil(global_samples/self.batch_size))

    def set_epoch(self,epoch:int)->None: self.epoch=int(epoch)
    def __len__(self)->int: return self.n_batches

    @staticmethod
    def _cycler(pool:np.ndarray,rng:np.random.Generator):
        while True:
            for idx in rng.permutation(pool): yield int(idx)

    def __iter__(self)->Iterator[list[int]]:
        rng=np.random.default_rng(self.seed+10007*self.epoch+97*self.rank)
        trusted_iter=self._cycler(self.trusted,rng) if len(self.trusted) else None; general_iter=self._cycler(self.general,rng) if len(self.general) else None
        previous_quota=0
        for batch_idx in range(self.n_batches):
            cumulative=int(math.floor((batch_idx+1)*self.batch_size*self.fraction+1e-9)); n_trusted=cumulative-previous_quota; previous_quota=cumulative; n_trusted=min(self.batch_size,max(0,n_trusted)); n_general=self.batch_size-n_trusted
            batch=[]
            if trusted_iter is not None: batch.extend(next(trusted_iter) for _ in range(n_trusted))
            if general_iter is not None: batch.extend(next(general_iter) for _ in range(n_general))
            rng.shuffle(batch)
            if batch: yield batch


def trusted_study_mask(gold_rows:np.ndarray,weights:np.ndarray,pseudo_threshold:float=0.60)->np.ndarray:
    """Gold studies plus unusually reliable pseudo-labeled studies form pool A."""
    gold=np.asarray(gold_rows,dtype=bool); weights=np.asarray(weights,float)
    if weights.ndim!=2 or weights.shape[0]!=gold.shape[0]: raise ValueError("weights/gold shape mismatch")
    pseudo_reliable=np.nanmax(weights,axis=1)>=float(pseudo_threshold)
    return gold|pseudo_reliable
