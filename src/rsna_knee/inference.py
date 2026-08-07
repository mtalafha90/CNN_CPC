from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .constants import DUAL_STREAMS, SUBMISSION_COLUMNS, TARGETS
from .data import backfill_series_metadata, build_series_index, load_series_csv, load_test_csv
from .dataset import DatasetConfig, KneeStudyDataset
from .model import KneeMILNet
from .runtime import resolve_runtime
from .training import predict


def _load_checkpoint_payload(path:str|Path)->dict:
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f"checkpoint not found: {path}")
    payload=torch.load(path,map_location="cpu",weights_only=False); required={"model","model_spec","stream_names"}; missing=sorted(required.difference(payload))
    if missing:raise ValueError(f"checkpoint {path} missing keys: {missing}")
    return payload


def _same_model_spec(a,b)->bool:
    keys={"n_streams","n_slices","in_channels","image_size","triplet_gap","stream_mode","dropout","normalize_input","encoder_batch_size","gradient_checkpointing"}; return all(a.get(k)==b.get(k) for k in keys)


def load_checkpoint(path:str|Path,device:torch.device):
    payload=_load_checkpoint_payload(path); spec=payload["model_spec"]
    if int(spec.get("n_streams",-1))!=len(DUAL_STREAMS) or int(spec.get("in_channels",-1))!=3:raise ValueError("checkpoint violates production stream/channel contract")
    model=KneeMILNet(int(spec["n_streams"]),int(spec["n_slices"]),in_channels=3,pretrained_weights=False,normalize_input=bool(spec.get("normalize_input",True)),dropout=float(spec.get("dropout",0.25)),encoder_batch_size=int(spec.get("encoder_batch_size",24)),gradient_checkpointing=bool(spec.get("gradient_checkpointing",True))); model.load_state_dict(payload["model"],strict=True); return model.to(device),payload


def _dataset(root,test,index,spec,config,offset:int):
    return KneeStudyDataset(test["StudyInstanceUID"].tolist(),index,DatasetConfig(data_root=str(root),split="test",n_slices=int(spec["n_slices"]),image_size=int(spec["image_size"]),noise_std=0.0,slice_dropout=0.0,triplet_gap=int(spec.get("triplet_gap",1)),strict_dicom=bool(config.get("strict_dicom_inference",True)),center_offset=int(offset),center_jitter=0,rotation_deg=0.0,translate_frac=0.0,scale_jitter=0.0,gamma_jitter=0.0,bias_field_strength=0.0),train=False)


def infer_checkpoints(data_root:str|Path,checkpoint_paths,config:dict)->pd.DataFrame:
    paths=[Path(p) for p in checkpoint_paths]
    if not paths:raise ValueError("at least one checkpoint is required")
    payloads=[_load_checkpoint_payload(p) for p in paths]; spec=payloads[0]["model_spec"]
    if list(payloads[0]["stream_names"])!=DUAL_STREAMS:raise ValueError("checkpoint stream order mismatch")
    for path,payload in zip(paths[1:],payloads[1:]):
        if not _same_model_spec(spec,payload["model_spec"]):raise ValueError(f"checkpoint model_spec mismatch: {path}")
    root=Path(data_root); test=load_test_csv(root/config.get("test_csv","test.csv")); series=load_series_csv(root/config.get("test_series_csv","test_series.csv")); series,stats=backfill_series_metadata(series,root,split="test"); print(f"[test metadata] {stats}"); index=build_series_index(series,test["StudyInstanceUID"],mode="dual"); runtime=resolve_runtime(config); offsets=[int(x) for x in config.get("tta_center_offsets",[-1,0,1])]
    if not offsets:offsets=[0]
    all_predictions=[]; reference_uids=None
    for offset in offsets:
        ds=_dataset(root,test,index,spec,config,offset); loader=DataLoader(ds,batch_size=max(1,int(config.get("batch_size",2))),shuffle=False,**runtime.loader_kwargs())
        for path in paths:
            model,_=load_checkpoint(path,runtime.device); uids,p,_=predict(model,loader,runtime.device,runtime)
            if reference_uids is None:reference_uids=uids
            elif uids!=reference_uids:raise ValueError("inference order mismatch")
            all_predictions.append(p)
    probabilities=np.mean(np.stack(all_predictions),axis=0)
    if not np.isfinite(probabilities).all():raise RuntimeError("non-finite probabilities")
    submission=pd.DataFrame(probabilities,columns=TARGETS); submission.insert(0,"StudyInstanceUID",reference_uids); validate_submission(submission); return submission[SUBMISSION_COLUMNS]


def validate_submission(df:pd.DataFrame)->None:
    if list(df.columns)!=SUBMISSION_COLUMNS:raise ValueError(f"submission columns must be exactly {SUBMISSION_COLUMNS}")
    if df["StudyInstanceUID"].astype(str).duplicated().any():raise ValueError("duplicate StudyInstanceUID")
    values=df[TARGETS].to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():raise ValueError("submission probabilities must be finite and in [0,1]")
