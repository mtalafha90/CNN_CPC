from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from .constants import DUAL_STREAMS
from .dicom import find_series_dir, preprocess_triplets, read_dicom_series


@dataclass
class DatasetConfig:
    data_root:str
    split:str="train"
    n_slices:int=16
    image_size:int=224
    noise_std:float=0.02
    slice_dropout:float=0.08
    triplet_gap:int=1
    strict_dicom:bool=False
    train_gap_choices:tuple[int,...]=(1,2)
    center_jitter:int=2
    center_offset:int=0
    rotation_deg:float=5.0
    translate_frac:float=0.03
    scale_jitter:float=0.05
    gamma_jitter:float=0.12
    bias_field_strength:float=0.08

    def __post_init__(self):
        if self.n_slices<1 or self.image_size<1 or self.triplet_gap<1:raise ValueError("n_slices/image_size/triplet_gap must be positive")
        if any(g<1 for g in self.train_gap_choices):raise ValueError("train_gap_choices must be >=1")
        if self.center_jitter<0 or self.noise_std<0:raise ValueError("jitter/noise must be non-negative")
        if not 0<=self.slice_dropout<1:raise ValueError("slice_dropout must be in [0,1)")


class KneeStudyDataset(Dataset):
    def __init__(self,study_uids,series_index,config:DatasetConfig,targets=None,weights=None,train:bool=False):
        self.study_uids=[str(x) for x in study_uids]; self.series_index=series_index; self.config=config; self.targets=targets; self.weights=weights; self.train=bool(train); self.stream_names=list(DUAL_STREAMS)
        n=len(self.study_uids)
        if targets is not None and len(targets)!=n:raise ValueError("targets length mismatch")
        if weights is not None and len(weights)!=n:raise ValueError("weights length mismatch")
    @property
    def in_channels(self):return 3
    def __len__(self):return len(self.study_uids)
    def _zero(self):return torch.zeros(self.config.n_slices,3,self.config.image_size,self.config.image_size,dtype=torch.float32)

    def _augment_mri(self,volume:torch.Tensor)->torch.Tensor:
        """Mild acquisition-like augmentation shared across all triplets in a series."""
        angle=float(torch.empty(1).uniform_(-self.config.rotation_deg,self.config.rotation_deg)); max_shift=int(round(self.config.translate_frac*self.config.image_size)); translate=[int(torch.randint(-max_shift,max_shift+1,(1,)).item()) if max_shift else 0,int(torch.randint(-max_shift,max_shift+1,(1,)).item()) if max_shift else 0]; scale=float(torch.empty(1).uniform_(1-self.config.scale_jitter,1+self.config.scale_jitter))
        volume=TVF.affine(volume,angle=angle,translate=translate,scale=scale,shear=[0.0,0.0],interpolation=InterpolationMode.BILINEAR)
        if self.config.gamma_jitter>0:
            gamma=float(torch.empty(1).uniform_(1-self.config.gamma_jitter,1+self.config.gamma_jitter)); volume=volume.clamp(0,1).pow(gamma)
        if self.config.bias_field_strength>0:
            h,w=volume.shape[-2:]; yy=torch.linspace(-1,1,h,device=volume.device).view(1,1,h,1); xx=torch.linspace(-1,1,w,device=volume.device).view(1,1,1,w); ax=float(torch.empty(1).uniform_(-self.config.bias_field_strength,self.config.bias_field_strength)); ay=float(torch.empty(1).uniform_(-self.config.bias_field_strength,self.config.bias_field_strength)); field=(1+ax*xx+ay*yy).clamp(0.8,1.2); volume=(volume*field).clamp(0,1)
        if self.config.noise_std>0:volume=(volume+torch.randn_like(volume)*self.config.noise_std).clamp(0,1)
        if self.config.slice_dropout>0:
            drop=torch.rand(volume.shape[0])<self.config.slice_dropout; volume[drop]=0
        return volume

    def _load(self,uid,series_uid):
        if not series_uid:return self._zero(),0.0
        path=find_series_dir(self.config.data_root,self.config.split,uid,str(series_uid))
        if path is None:
            if self.config.strict_dicom:raise FileNotFoundError(f"missing series {uid}/{series_uid}")
            return self._zero(),0.0
        try:
            raw=read_dicom_series(path)
            gap=int(np.random.choice(self.config.train_gap_choices)) if self.train else self.config.triplet_gap
            volume=preprocess_triplets(raw,n_slices=self.config.n_slices,image_size=self.config.image_size,gap=gap,center_offset=0 if self.train else self.config.center_offset,jitter=self.config.center_jitter if self.train else 0)
        except Exception:
            if self.config.strict_dicom:raise
            return self._zero(),0.0
        if self.train:volume=self._augment_mri(volume)
        return volume,1.0

    def __getitem__(self,idx):
        uid=self.study_uids[idx]; mapping=self.series_index.get(uid,{}); volumes=[]; present=[]
        for name in self.stream_names:
            volume,flag=self._load(uid,mapping.get(name)); volumes.append(volume); present.append(flag)
        item={"study_uid":uid,"volumes":torch.stack(volumes),"present":torch.tensor(present,dtype=torch.float32)}
        if self.targets is not None:item["target"]=torch.from_numpy(np.asarray(self.targets[idx],dtype=np.float32))
        if self.weights is not None:item["weight"]=torch.from_numpy(np.asarray(self.weights[idx],dtype=np.float32))
        return item
