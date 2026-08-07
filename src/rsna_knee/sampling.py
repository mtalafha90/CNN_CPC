"""Balanced study sampling for weakly supervised MRI training."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class TwoPoolBatchSampler(Sampler[list[int]]):
    """Generate one deterministic global two-pool batch, then shard by rank.

    This preserves the requested trusted fraction across the *global* DDP batch
    and prevents different ranks from independently drawing the same study in a
    step when enough unique pool members exist.
    """

    def __init__(self,trusted_mask:np.ndarray,batch_size:int,*,trusted_fraction:float=0.30,seed:int=2026,rank:int=0,world_size:int=1,drop_last:bool=True)->None:
        mask=np.asarray(trusted_mask,dtype=bool)
        if mask.ndim!=1:raise ValueError("trusted_mask must be one-dimensional")
        if batch_size<1 or world_size<1:raise ValueError("batch_size/world_size must be positive")
        if not 0.0<=trusted_fraction<=1.0:raise ValueError("trusted_fraction must be in [0,1]")
        if not 0<=rank<world_size:raise ValueError("rank must be in [0,world_size)")
        self.trusted=np.flatnonzero(mask); self.general=np.flatnonzero(~mask)
        if trusted_fraction>0 and len(self.trusted)==0:raise ValueError("trusted pool is empty")
        if trusted_fraction<1 and len(self.general)==0:raise ValueError("general pool is empty")
        self.n=len(mask); self.batch_size=int(batch_size); self.global_batch_size=self.batch_size*int(world_size); self.fraction=float(trusted_fraction); self.seed=int(seed); self.rank=int(rank); self.world_size=int(world_size); self.drop_last=bool(drop_last); self.epoch=0
        self.n_batches=self.n//self.global_batch_size if self.drop_last else math.ceil(self.n/self.global_batch_size)

    def set_epoch(self,epoch:int)->None:self.epoch=int(epoch)
    def __len__(self)->int:return self.n_batches

    @staticmethod
    def _draw(pool:np.ndarray,count:int,rng:np.random.Generator,state:dict)->list[int]:
        """Draw with reshuffled cycles; no replacement inside a cycle."""
        result=[]
        while len(result)<count:
            order=state.get("order"); pos=int(state.get("pos",0))
            if order is None or pos>=len(order):order=rng.permutation(pool); pos=0
            take=min(count-len(result),len(order)-pos); result.extend(int(x) for x in order[pos:pos+take]); state["order"],state["pos"]=order,pos+take
        return result

    def __iter__(self)->Iterator[list[int]]:
        # Every rank constructs the same global sequence, then receives a
        # disjoint contiguous shard of each global batch.
        rng=np.random.default_rng(self.seed+10007*self.epoch); t_state={}; g_state={}; previous_quota=0
        for batch_idx in range(self.n_batches):
            cumulative=int(math.floor((batch_idx+1)*self.global_batch_size*self.fraction+1e-9)); n_trusted=min(self.global_batch_size,max(0,cumulative-previous_quota)); previous_quota=cumulative; n_general=self.global_batch_size-n_trusted
            global_batch=[]
            if n_trusted:global_batch.extend(self._draw(self.trusted,n_trusted,rng,t_state))
            if n_general:global_batch.extend(self._draw(self.general,n_general,rng,g_state))
            rng.shuffle(global_batch)
            start=self.rank*self.batch_size; local=global_batch[start:start+self.batch_size]
            if len(local)==self.batch_size or (local and not self.drop_last):yield local


def trusted_study_mask(gold_rows:np.ndarray,weights:np.ndarray,pseudo_threshold:float=0.60)->np.ndarray:
    gold=np.asarray(gold_rows,dtype=bool); weights=np.asarray(weights,float)
    if weights.ndim!=2 or weights.shape[0]!=gold.shape[0]:raise ValueError("weights/gold shape mismatch")
    return gold|(np.nanmax(weights,axis=1)>=float(pseudo_threshold))
