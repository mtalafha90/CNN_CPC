"""Cross-fitted image/report co-training utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import add_report_groups, load_train_csv
from .report_labels import label_dataframe


def assign_crossfit_folds(df:pd.DataFrame,n_folds:int=3)->pd.Series:
    if n_folds<2:raise ValueError("n_folds must be >=2")
    work=add_report_groups(df) if "report_group" not in df.columns else df
    def fold(group:str)->int:
        digest=hashlib.sha1(str(group).encode("utf-8")).digest(); return int.from_bytes(digest[:8],"big")%n_folds
    return work["report_group"].astype(str).map(fold).astype(int)


def load_image_predictions(paths:list[str|Path],study_uids:pd.Series)->np.ndarray:
    frames=[]
    for path in paths:
        frame=pd.read_csv(path); required={"StudyInstanceUID",*TARGETS}; missing=required.difference(frame.columns)
        if missing:raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frames.append(frame[["StudyInstanceUID",*TARGETS]].copy())
    image=pd.concat(frames,ignore_index=True); image["StudyInstanceUID"]=image["StudyInstanceUID"].astype(str)
    if image["StudyInstanceUID"].duplicated().any():
        dup=image.loc[image["StudyInstanceUID"].duplicated(),"StudyInstanceUID"].iloc[0]; raise ValueError(f"cross-fitted image prediction repeated for study {dup}")
    ordered=pd.DataFrame({"StudyInstanceUID":study_uids.astype(str)}).merge(image,on="StudyInstanceUID",how="left",validate="one_to_one")
    return ordered[TARGETS].to_numpy(np.float32)


def consensus_arrays(report_p:np.ndarray,report_conf:np.ndarray,image_p:np.ndarray,*,positive_threshold:float=0.80,negative_threshold:float=0.20,agreement_weight:float=0.90,disagreement_weight:float=0.05,blend:float=0.50)->tuple[np.ndarray,np.ndarray]:
    """Fuse independent image/report teachers while preserving uncertainty."""
    report_p=np.asarray(report_p,np.float32); report_conf=np.asarray(report_conf,np.float32); image_p=np.asarray(image_p,np.float32)
    if report_p.shape!=report_conf.shape or report_p.shape!=image_p.shape:raise ValueError("teacher arrays must have identical shapes")
    probability=report_p.copy(); confidence=report_conf.copy(); available=np.isfinite(image_p); probability[available]=float(blend)*report_p[available]+(1-float(blend))*image_p[available]
    report_pos=report_p>=positive_threshold; image_pos=image_p>=positive_threshold; report_neg=report_p<=negative_threshold; image_neg=image_p<=negative_threshold; agree=available&((report_pos&image_pos)|(report_neg&image_neg)); disagree=available&((report_pos&image_neg)|(report_neg&image_pos)); confidence[agree]=float(agreement_weight); confidence[disagree]=np.minimum(confidence[disagree],float(disagreement_weight)); return probability,confidence


def build_consensus_labels(train_csv:str|Path,image_oof_paths:list[str|Path],out_csv:str|Path,*,positive_threshold:float=0.80,negative_threshold:float=0.20,agreement_weight:float=0.90,disagreement_weight:float=0.05,blend:float=0.50)->Path:
    """Convenience export using the deterministic rule teacher.

    Production second-stage training instead calls :func:`consensus_arrays`
    after its fold-safe report calibration, retaining the leakage guarantees.
    """
    train=load_train_csv(train_csv); report,report_conf=label_dataframe(train); image=load_image_predictions(image_oof_paths,train["StudyInstanceUID"]); probability,confidence=consensus_arrays(report,report_conf,image,positive_threshold=positive_threshold,negative_threshold=negative_threshold,agreement_weight=agreement_weight,disagreement_weight=disagreement_weight,blend=blend); out=pd.DataFrame({"StudyInstanceUID":train["StudyInstanceUID"].astype(str)})
    for j,target in enumerate(TARGETS):out[target]=probability[:,j]; out[f"{target}__confidence"]=confidence[:,j]
    out_path=Path(out_csv); out.to_csv(out_path,index=False); return out_path
