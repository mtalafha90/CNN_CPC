"""In-domain self-supervised pretraining on non-gold knee MRI studies.

The encoder learns two complementary signals without using diagnostic labels:
1) different MRI sequences from the same knee share anatomy (contrastive loss);
2) plane and fluid/structural contrast remain identifiable (metadata losses).

Only non-gold studies are used by default so outer gold validation images are
not exposed during representation pretraining.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .constants import DUAL_STREAMS
from .data import backfill_series_metadata, build_series_index, gold_mask, load_series_csv, load_train_csv
from .dataset import DatasetConfig, KneeStudyDataset
from .model import ConvNeXtSliceEncoder
from .runtime import autocast, barrier, make_scaler, resolve_runtime

PLANE_LABELS=torch.tensor([0,0,1,1,2,2],dtype=torch.long)
SEQUENCE_LABELS=torch.tensor([0,1,0,1,0,1],dtype=torch.long)


class MRIRepresentationLearner(nn.Module):
    def __init__(self,*,pretrained:bool=True,normalize_input:bool=True,projection_dim:int=256):
        super().__init__(); self.encoder=ConvNeXtSliceEncoder(3,pretrained_weights=pretrained,normalize_input=normalize_input); d=self.encoder.out_dim
        self.projector=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,projection_dim)); self.plane_head=nn.Linear(d,3); self.sequence_head=nn.Linear(d,2)

    def forward(self,x):
        feat=self.encoder(x); return feat,nn.functional.normalize(self.projector(feat),dim=-1),self.plane_head(feat),self.sequence_head(feat)


def _contrastive_same_study(z:torch.Tensor,study_ids:torch.Tensor,temperature:float=0.15)->torch.Tensor:
    if z.shape[0]<2:return z.sum()*0.0
    logits=(z@z.T)/float(temperature); eye=torch.eye(len(z),dtype=torch.bool,device=z.device); logits=logits.masked_fill(eye,-1e4)
    positives=(study_ids[:,None]==study_ids[None,:])&~eye; valid=positives.any(dim=1)
    if not valid.any():return z.sum()*0.0
    log_prob=logits-torch.logsumexp(logits,dim=1,keepdim=True); mean_pos=(log_prob.masked_fill(~positives,0.0).sum(dim=1)/positives.sum(dim=1).clamp_min(1))[valid]
    return -mean_pos.mean()


def _global_embeddings(z,study_ids,plane,sequence,runtime):
    if not runtime.distributed:return z,study_ids,plane,sequence
    from torch.distributed.nn.functional import all_gather as differentiable_all_gather
    zg=torch.cat(list(differentiable_all_gather(z)),dim=0)
    outs=[]
    for tensor in (study_ids,plane,sequence):
        parts=[torch.empty_like(tensor) for _ in range(runtime.world_size)]; dist.all_gather(parts,tensor); outs.append(torch.cat(parts,dim=0))
    return zg,*outs


def pretrain_ssl(config:dict)->Path:
    runtime=resolve_runtime(config); root=Path(config["data_root"]); train=load_train_csv(root/config.get("train_csv","train.csv")); non_gold=train.loc[~gold_mask(train),"StudyInstanceUID"].tolist()
    if not non_gold:raise ValueError("no non-gold studies available for SSL")
    series=load_series_csv(root/config.get("train_series_csv","train_series.csv")); series,_=backfill_series_metadata(series,root,split="train"); index=build_series_index(series,non_gold,mode="dual")
    ds=KneeStudyDataset(non_gold,index,DatasetConfig(data_root=str(root),split="train",n_slices=int(config.get("ssl_n_slices",5)),image_size=int(config.get("image_size",224)),noise_std=float(config.get("ssl_noise_std",0.01)),slice_dropout=0.0,triplet_gap=int(config.get("triplet_gap",1)),strict_dicom=bool(config.get("strict_dicom",False))),train=True)
    sampler=DistributedSampler(ds,num_replicas=runtime.world_size,rank=runtime.rank,shuffle=True,seed=int(config.get("seed",2026))) if runtime.distributed else None
    loader=DataLoader(ds,batch_size=int(config.get("ssl_batch_size",4)),shuffle=sampler is None,sampler=sampler,**runtime.loader_kwargs())
    model=MRIRepresentationLearner(pretrained=bool(config.get("pretrained",True)),normalize_input=bool(config.get("normalize_input",True)),projection_dim=int(config.get("ssl_projection_dim",256))).to(runtime.device)
    if runtime.distributed:model=DDP(model,device_ids=[runtime.local_rank],output_device=runtime.local_rank,broadcast_buffers=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(config.get("ssl_lr",2e-4)),weight_decay=float(config.get("ssl_weight_decay",1e-4))); scaler=make_scaler(runtime); epochs=int(config.get("ssl_epochs",10)); temperature=float(config.get("ssl_temperature",0.15)); metadata_weight=float(config.get("ssl_metadata_weight",0.25)); outdir=Path(config.get("ssl_output_dir","runs/ssl")); checkpoint=outdir/"ssl_encoder.pt"
    if runtime.is_main:outdir.mkdir(parents=True,exist_ok=True)
    barrier(runtime)
    history=[]
    for epoch in range(epochs):
        if sampler is not None:sampler.set_epoch(epoch)
        model.train(); total=0.0; steps=0
        for batch in loader:
            volumes=batch["volumes"].to(runtime.device,non_blocking=True); present=batch["present"].to(runtime.device,non_blocking=True); b,k,s,c,h,w=volumes.shape; center=s//2; x=volumes[:,:,center].reshape(b*k,c,h,w); active=present.reshape(-1)>0
            if active.sum()<2:continue
            x=x[active]; stream_idx=torch.arange(k,device=runtime.device).repeat(b)[active]; study_ids=(torch.arange(b,device=runtime.device)+runtime.rank*1000000).repeat_interleave(k)[active]; plane=PLANE_LABELS.to(runtime.device)[stream_idx]; sequence=SEQUENCE_LABELS.to(runtime.device)[stream_idx]
            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                feat,z,plane_logits,seq_logits=model(x); zg,study_global,_,_= _global_embeddings(z,study_ids,plane,sequence,runtime); contrast=_contrastive_same_study(zg,study_global,temperature); meta=nn.functional.cross_entropy(plane_logits,plane)+nn.functional.cross_entropy(seq_logits,sequence); loss=contrast+metadata_weight*meta
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total+=float(loss.item()); steps+=1
        row={"epoch":epoch+1,"loss":total/max(steps,1)}
        if runtime.is_main:history.append(row); print(row)
    if runtime.is_main:
        base=model.module if isinstance(model,DDP) else model; torch.save({"encoder":base.encoder.state_dict(),"config":config,"non_gold_studies":len(non_gold)},checkpoint); (outdir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    barrier(runtime); return checkpoint
