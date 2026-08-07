from __future__ import annotations

import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .calibration import fit_calibration
from .constants import DUAL_STREAMS, TARGETS
from .data import add_report_groups, backfill_series_metadata, build_series_index, gold_mask, load_series_csv, load_train_csv, make_balanced_gold_folds
from .dataset import DatasetConfig, KneeStudyDataset
from .evaluation import bootstrap_macro_auc, macro_auc_from_arrays
from .model import KneeMILNet
from .preflight import run_preflight
from .report_labels import combine_gold_and_pseudo, label_dataframe, state_dataframe
from .runtime import autocast, make_scaler, resolve_runtime


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def macro_weighted_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Confidence-weighted BCE with equal contribution from each target.

    Kaggle macro-AUC gives every pathology exactly 1/12 of the metric. We mirror
    that here by normalizing supervision *inside each target* before averaging
    targets. A target with thousands of confident pseudo-label cells can no
    longer dominate one with fewer trusted labels.
    """
    cell = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    numerator = (cell * weight).sum(dim=0)
    denominator = weight.sum(dim=0)
    valid = denominator > 0
    if not valid.any():
        return logits.sum() * 0.0
    per_target = numerator[valid] / denominator[valid].clamp_min(1e-8)
    return per_target.mean()


def confidence_gated_ranking_loss(logits, target, weight, *, pairs_per_target=32, min_confidence=0.35, positive_threshold=0.75, negative_threshold=0.25):
    losses = []
    for j in range(logits.shape[1]):
        trusted = weight[:, j] >= float(min_confidence)
        pos = torch.nonzero(trusted & (target[:, j] >= positive_threshold), as_tuple=False).flatten()
        neg = torch.nonzero(trusted & (target[:, j] <= negative_threshold), as_tuple=False).flatten()
        if pos.numel() == 0 or neg.numel() == 0: continue
        n = min(int(pairs_per_target), int(pos.numel() * neg.numel()))
        pi = pos[torch.randint(pos.numel(), (n,), device=logits.device)]
        ni = neg[torch.randint(neg.numel(), (n,), device=logits.device)]
        pair_weight = torch.minimum(weight[pi, j], weight[ni, j]).clamp(max=1.0)
        losses.append((nn.functional.softplus(-(logits[pi, j] - logits[ni, j])) * pair_weight).mean())
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


@torch.no_grad()
def predict(model, loader, device, runtime=None):
    model.eval(); uids=[]; probs=[]; targets=[]
    for batch in loader:
        with (autocast(runtime) if runtime is not None else nullcontext()):
            logits = model(batch["volumes"].to(device, non_blocking=True), batch["present"].to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy()); uids.extend(list(batch["study_uid"]))
        if "target" in batch: targets.append(batch["target"].numpy())
    return uids, (np.concatenate(probs) if probs else np.empty((0,len(TARGETS)),np.float32)), (np.concatenate(targets) if targets else None)


def _dataset_config(config, root, *, train):
    return DatasetConfig(data_root=str(root), split="train", n_slices=int(config.get("n_slices",16)), image_size=int(config.get("image_size",224)), noise_std=float(config.get("noise_std",0.02)) if train else 0.0, slice_dropout=float(config.get("slice_dropout",0.08)) if train else 0.0, triplet_gap=int(config.get("triplet_gap",1)), strict_dicom=bool(config.get("strict_dicom",False)))


def _model_spec(config):
    return {"n_streams":len(DUAL_STREAMS),"n_slices":int(config.get("n_slices",16)),"in_channels":3,"image_size":int(config.get("image_size",224)),"triplet_gap":int(config.get("triplet_gap",1)),"stream_mode":"dual","dropout":float(config.get("dropout",0.25)),"normalize_input":bool(config.get("normalize_input",True)),"encoder_batch_size":int(config.get("encoder_batch_size",24)),"gradient_checkpointing":bool(config.get("gradient_checkpointing",True))}


def _build_model(spec, config, device):
    return KneeMILNet(spec["n_streams"], spec["n_slices"], in_channels=spec["in_channels"], pretrained_weights=bool(config.get("pretrained",True)), normalize_input=spec["normalize_input"], dropout=spec["dropout"], encoder_batch_size=spec["encoder_batch_size"], gradient_checkpointing=spec["gradient_checkpointing"]).to(device)


def train_fold(config: dict, fold: int) -> Path:
    start_time=time.time(); seed=int(config.get("seed",2026)); n_folds=int(config.get("n_folds",3))
    if n_folds < 3: raise ValueError("nested outer/inner selection requires n_folds >= 3")
    if fold < 0 or fold >= n_folds: raise ValueError(f"fold must be in [0,{n_folds-1}]")
    inner_fold=int(config.get("inner_selection_fold",(fold+1)%n_folds))
    if inner_fold==fold or not 0<=inner_fold<n_folds: raise ValueError("inner selection fold must differ from outer validation fold")
    seed_everything(seed+fold); root=Path(config["data_root"])
    df=load_train_csv(root/config.get("train_csv","train.csv")); series_path=root/config.get("train_series_csv","train_series.csv")
    preflight_payload=None
    if bool(config.get("preflight_before_train",True)):
        result=run_preflight(root,split="train",series_csv=series_path,study_uids=df["StudyInstanceUID"].tolist(),sample_size=int(config.get("preflight_sample_size",24)),stream_mode="dual",seed=seed,max_decode_failure_rate=float(config.get("preflight_max_decode_failure_rate",0.05)),strict=True); preflight_payload=result.to_dict(); print(result.summary())
    series=load_series_csv(series_path); series,metadata_stats=backfill_series_metadata(series,root,split="train"); print(f"[metadata] {metadata_stats}")
    df=add_report_groups(df); df["fold"]=make_balanced_gold_folds(df,n_folds,seed); gold=gold_mask(df)
    outer_mask=df["fold"].eq(fold)&gold; inner_mask=df["fold"].eq(inner_fold)&gold
    if not outer_mask.any() or not inner_mask.any(): raise ValueError("outer and inner selection folds must both contain gold studies")
    heldout_groups=set(df.loc[outer_mask|inner_mask,"report_group"].astype(str)); train_mask=~df["report_group"].astype(str).isin(heldout_groups)
    states=state_dataframe(df); calibration=None; calibration_mask=gold.to_numpy()&train_mask.to_numpy()
    if bool(config.get("calibrate_teacher",True)) and calibration_mask.sum()>=int(config.get("min_calibration_studies",8)):
        calibration=fit_calibration(states[calibration_mask],df.loc[calibration_mask,TARGETS].to_numpy(dtype=np.float64),alpha=float(config.get("calibration_alpha",5.0)))
        pseudo=calibration.apply(states); confidence=calibration.confidence(states,unmentioned_weight=float(config.get("unmentioned_weight",0.0)),uncertain_weight_cap=float(config.get("uncertain_weight_cap",0.10)))
    else:
        pseudo,confidence=label_dataframe(df); confidence[states=="unmentioned"]=float(config.get("unmentioned_weight",0.0))
    train_targets,train_weights=combine_gold_and_pseudo(df,pseudo,confidence,float(config.get("gold_weight",8.0)))
    series_index=build_series_index(series,df["StudyInstanceUID"],mode="dual"); ti=np.flatnonzero(train_mask.to_numpy()); ii=np.flatnonzero(inner_mask.to_numpy()); oi=np.flatnonzero(outer_mask.to_numpy())
    train_ds=KneeStudyDataset(df.iloc[ti]["StudyInstanceUID"].tolist(),series_index,_dataset_config(config,root,train=True),train_targets[ti],train_weights[ti],True)
    inner_ds=KneeStudyDataset(df.iloc[ii]["StudyInstanceUID"].tolist(),series_index,_dataset_config(config,root,train=False),df.iloc[ii][TARGETS].to_numpy(dtype=np.float32),None,False)
    outer_ds=KneeStudyDataset(df.iloc[oi]["StudyInstanceUID"].tolist(),series_index,_dataset_config(config,root,train=False),df.iloc[oi][TARGETS].to_numpy(dtype=np.float32),None,False)
    runtime=resolve_runtime(config); print(f"[fold {fold}] outer={fold} inner={inner_fold} | {runtime.describe()}")
    if runtime.device.type=="cuda": torch.cuda.reset_peak_memory_stats(runtime.device)
    kwargs=runtime.loader_kwargs(); batch_size=int(config.get("batch_size",2))
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,**kwargs); inner_loader=DataLoader(inner_ds,batch_size=batch_size,shuffle=False,**kwargs); outer_loader=DataLoader(outer_ds,batch_size=batch_size,shuffle=False,**kwargs)
    spec=_model_spec(config); model=_build_model(spec,config,runtime.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(config.get("lr",1e-4)),weight_decay=float(config.get("weight_decay",1e-4))); epochs=int(config.get("epochs",20)); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,epochs),eta_min=float(config.get("min_lr",1e-6))); scaler=make_scaler(runtime)
    outdir=Path(config.get("output_dir","runs/model"))/f"fold{fold}"; outdir.mkdir(parents=True,exist_ok=True)
    assignments=df[["StudyInstanceUID","report_group","fold"]].copy(); assignments["role"]="weak_train"; assignments.loc[train_mask&gold,"role"]="gold_train"; assignments.loc[inner_mask,"role"]="inner_selection"; assignments.loc[outer_mask,"role"]="outer_oof"; assignments.to_csv(outdir/"fold_assignments.csv",index=False)
    (outdir/"metadata_repair.json").write_text(json.dumps(metadata_stats,indent=2),encoding="utf-8")
    if preflight_payload is not None:(outdir/"preflight.json").write_text(json.dumps(preflight_payload,indent=2),encoding="utf-8")
    if calibration is not None:(outdir/"calibration.json").write_text(json.dumps(calibration.to_dict(),indent=2),encoding="utf-8")
    best_inner=-np.inf; best_epoch=0; bad_epochs=0; history=[]; rank_weight=float(config.get("rank_loss_weight",0.10)); checkpoint_path=outdir/"best.pt"
    for epoch in range(epochs):
        model.train(); running=0.0; seen=0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True); volumes=batch["volumes"].to(runtime.device,non_blocking=True); present=batch["present"].to(runtime.device,non_blocking=True); target=batch["target"].to(runtime.device,non_blocking=True); weight=batch["weight"].to(runtime.device,non_blocking=True)
            with autocast(runtime):
                logits=model(volumes,present); bce=macro_weighted_bce(logits,target,weight); rank=confidence_gated_ranking_loss(logits,target,weight,pairs_per_target=int(config.get("rank_pairs_per_target",32)),min_confidence=float(config.get("rank_min_confidence",0.35)),positive_threshold=float(config.get("rank_positive_threshold",0.75)),negative_threshold=float(config.get("rank_negative_threshold",0.25))) if rank_weight>0 else logits.new_zeros(()); loss=bce+rank_weight*rank
            scaler.scale(loss).backward(); grad_clip=float(config.get("grad_clip",1.0))
            if grad_clip>0: scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            scaler.step(optimizer); scaler.update(); running+=float(loss.item())*len(target); seen+=len(target)
        scheduler.step(); _,ip,it=predict(model,inner_loader,runtime.device,runtime); score=macro_auc_from_arrays(it,ip)[0]; row={"epoch":epoch+1,"train_loss":running/max(seen,1),"inner_macro_auc":score,"lr":optimizer.param_groups[0]["lr"]}; history.append(row); print(row)
        if np.isfinite(score) and score>best_inner:
            best_inner=float(score); best_epoch=epoch+1; bad_epochs=0; torch.save({"model":model.state_dict(),"model_spec":spec,"config":config,"stream_names":train_ds.stream_names,"fold":fold,"inner_fold":inner_fold,"selected_epoch":best_epoch,"inner_score":best_inner},checkpoint_path)
        else:
            bad_epochs+=1
            if bad_epochs>=int(config.get("patience",5)): break
    if best_epoch==0: raise RuntimeError(f"fold {fold} never produced a finite inner-selection macro AUC")
    payload=torch.load(checkpoint_path,map_location="cpu",weights_only=False); model.load_state_dict(payload["model"],strict=True); ouids,op,ot=predict(model,outer_loader,runtime.device,runtime); oscore=macro_auc_from_arrays(ot,op)[0]
    oof=pd.DataFrame(op,columns=TARGETS); oof.insert(0,"StudyInstanceUID",ouids); oof.to_csv(outdir/"oof.csv",index=False); pd.DataFrame(history).to_csv(outdir/"history.csv",index=False); (outdir/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    selection={"outer_fold":fold,"inner_fold":inner_fold,"selected_epoch":best_epoch,"inner_macro_auc":best_inner,"outer_macro_auc":float(oscore)}; (outdir/"selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    bootstrap=bootstrap_macro_auc(ot,op,n_bootstrap=int(config.get("n_bootstrap",2000)),seed=seed+fold); (outdir/"bootstrap.json").write_text(json.dumps(bootstrap.to_dict(),indent=2),encoding="utf-8")
    runtime_payload={"elapsed_seconds":float(time.time()-start_time),"device":runtime.device_name,"visible_gpus":int(runtime.visible_gpus),"runtime":runtime.describe(),"num_workers":int(runtime.num_workers),"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated(runtime.device)) if runtime.device.type=="cuda" else 0,"git_sha_env":os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT")}; (outdir/"runtime.json").write_text(json.dumps(runtime_payload,indent=2),encoding="utf-8")
    return checkpoint_path
