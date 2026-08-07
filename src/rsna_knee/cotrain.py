"""Cross-fitted image/report co-training utilities.

Stage 1 assigns every non-gold report group to one held-out fold. The model for
that fold must not train on those studies, so its predictions are genuinely
out-of-fold. Stage 2 combines those image predictions with the report teacher:
agreement produces stronger pseudo-labels; disagreement is treated as uncertain
rather than forcing either teacher to be correct.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import add_report_groups, load_train_csv
from .report_labels import label_dataframe


def assign_crossfit_folds(df:pd.DataFrame,n_folds:int=3)->pd.Series:
    """Stable report-group folds for every study, including non-gold rows."""
    if n_folds<2:raise ValueError("n_folds must be >=2")
    work=add_report_groups(df) if "report_group" not in df.columns else df
    def fold(group:str)->int:
        digest=hashlib.sha1(str(group).encode("utf-8")).digest(); return int.from_bytes(digest[:8],"big")%n_folds
    return work["report_group"].astype(str).map(fold).astype(int)


def _load_image_predictions(paths:list[str|Path])->pd.DataFrame:
    frames=[]
    for path in paths:
        frame=pd.read_csv(path); required={"StudyInstanceUID",*TARGETS}; missing=required.difference(frame.columns)
        if missing:raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frames.append(frame[["StudyInstanceUID",*TARGETS]].copy())
    image=pd.concat(frames,ignore_index=True); image["StudyInstanceUID"]=image["StudyInstanceUID"].astype(str)
    if image["StudyInstanceUID"].duplicated().any():
        dup=image.loc[image["StudyInstanceUID"].duplicated(),"StudyInstanceUID"].iloc[0]; raise ValueError(f"cross-fitted image prediction repeated for study {dup}")
    return image


def build_consensus_labels(
    train_csv:str|Path,
    image_oof_paths:list[str|Path],
    out_csv:str|Path,
    *,
    positive_threshold:float=0.80,
    negative_threshold:float=0.20,
    agreement_weight:float=0.90,
    disagreement_weight:float=0.05,
    blend:float=0.50,
)->Path:
    """Create second-generation pseudo labels from independent image/report views."""
    train=load_train_csv(train_csv); report,report_conf=label_dataframe(train); image=_load_image_predictions(image_oof_paths); merged=train[["StudyInstanceUID"]].merge(image,on="StudyInstanceUID",how="left",validate="one_to_one"); image_p=merged[TARGETS].to_numpy(float)
    out=pd.DataFrame({"StudyInstanceUID":train["StudyInstanceUID"].astype(str)}); blend=float(blend)
    for j,target in enumerate(TARGETS):
        r=report[:,j]; i=image_p[:,j]; available=np.isfinite(i); r_pos=r>=positive_threshold; i_pos=i>=positive_threshold; r_neg=r<=negative_threshold; i_neg=i<=negative_threshold; agree=(r_pos&i_pos)|(r_neg&i_neg); disagree=(r_pos&i_neg)|(r_neg&i_pos)
        probability=r.copy(); probability[available]=blend*r[available]+(1-blend)*i[available]
        confidence=report_conf[:,j].copy(); confidence[agree&available]=float(agreement_weight); confidence[disagree&available]=float(disagreement_weight)
        # When only one teacher is decisive, retain the blended target but do
        # not inflate confidence beyond the original report evidence.
        confidence[~available]=0.0
        out[target]=probability.astype(np.float32); out[f"{target}__confidence"]=confidence.astype(np.float32)
    out_path=Path(out_csv); out.to_csv(out_path,index=False); return out_path


def load_consensus_labels(path:str|Path,study_uids:pd.Series)->tuple[np.ndarray,np.ndarray]:
    frame=pd.read_csv(path); frame["StudyInstanceUID"]=frame["StudyInstanceUID"].astype(str); required={"StudyInstanceUID",*TARGETS,*[f"{t}__confidence" for t in TARGETS]}; missing=required.difference(frame.columns)
    if missing:raise ValueError(f"consensus label file missing columns: {sorted(missing)}")
    ordered=pd.DataFrame({"StudyInstanceUID":study_uids.astype(str)}).merge(frame,on="StudyInstanceUID",how="left",validate="one_to_one")
    return ordered[TARGETS].to_numpy(np.float32),ordered[[f"{t}__confidence" for t in TARGETS]].to_numpy(np.float32)
