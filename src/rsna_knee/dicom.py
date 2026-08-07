from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DICOM_SUFFIXES={"",".dcm",".dicom",".ima"}


def _sort_key(ds)->float:
    try:return float(np.dot(np.asarray(ds.ImagePositionPatient,float),np.cross(np.asarray(ds.ImageOrientationPatient,float)[:3],np.asarray(ds.ImageOrientationPatient,float)[3:])))
    except Exception:return float(getattr(ds,"InstanceNumber",0))


def find_series_dir(root:str|Path,split:str,study:str,series:str)->Path|None:
    root=Path(root); study,series=str(study),str(series)
    for p in (root/f"{split}_series"/study/series,root/f"{split}_images"/study/series,root/study/series):
        if p.is_dir():return p
    return None


def _iter_dicom_files(path:Path)->list[Path]:return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES)


def read_dicom_series(path:str|Path,*,return_stats:bool=False):
    import pydicom
    path=Path(path); candidates=_iter_dicom_files(path); items=[]; failed=0
    for p in candidates:
        try:
            ds=pydicom.dcmread(str(p),force=True); arr=np.asarray(ds.pixel_array,dtype=np.float32); arr=arr*float(getattr(ds,"RescaleSlope",1.0))+float(getattr(ds,"RescaleIntercept",0.0))
            if str(getattr(ds,"PhotometricInterpretation","")).upper()=="MONOCHROME1":arr=arr.max()-arr
            base=_sort_key(ds)
            if arr.ndim==2:items.append((base,arr))
            elif arr.ndim==3:items.extend((base+i*1e-4,frame) for i,frame in enumerate(arr))
            else:raise RuntimeError(f"unsupported pixel shape {arr.shape}")
        except Exception:failed+=1
    if not items:raise RuntimeError(f"No readable DICOM pixels in {path} ({len(candidates)} candidates, {failed} failures)")
    items.sort(key=lambda x:x[0]); frames=[f for _,f in items]; shapes={f.shape for f in frames}
    if len(shapes)>1:
        target=max(shapes,key=lambda s:s[0]*s[1]); frames=[_pad_or_crop(f,target) for f in frames]
    volume=np.stack(frames).astype(np.float32,copy=False)
    if return_stats:return volume,{"candidate_files":len(candidates),"decode_failures":failed,"decoded_frames":len(frames)}
    return volume


def _pad_or_crop(image:np.ndarray,target:tuple[int,int])->np.ndarray:
    out=np.zeros(target,dtype=image.dtype); rows=min(image.shape[0],target[0]); cols=min(image.shape[1],target[1]); sr,sc=(image.shape[0]-rows)//2,(image.shape[1]-cols)//2; tr,tc=(target[0]-rows)//2,(target[1]-cols)//2; out[tr:tr+rows,tc:tc+cols]=image[sr:sr+rows,sc:sc+cols]; return out


def _normalise_volume(v:np.ndarray)->np.ndarray:
    v=np.asarray(v,dtype=np.float32)
    if v.ndim!=3 or len(v)==0:raise RuntimeError(f"expected non-empty [S,H,W], got {v.shape}")
    finite=v[np.isfinite(v)]
    if finite.size==0:raise RuntimeError("DICOM volume contains no finite pixels")
    lo,hi=np.percentile(finite,[1,99]); v=np.nan_to_num(v,nan=float(lo),posinf=float(hi),neginf=float(lo)); v=np.clip(v,lo,hi); return ((v-lo)/max(float(hi-lo),1e-6)).astype(np.float32,copy=False)


def _centers(n_frames:int,n_slices:int,gap:int,*,center_offset:int=0,jitter:int=0,rng:np.random.Generator|None=None)->np.ndarray:
    """Distributed center positions, avoiding edge duplication whenever possible."""
    if n_frames<1:raise ValueError("n_frames must be positive")
    lo,hi=(gap,n_frames-1-gap) if n_frames>2*gap else (0,n_frames-1)
    centers=np.round(np.linspace(lo,hi,n_slices)).astype(int)+int(center_offset)
    if jitter>0:
        rng=rng or np.random.default_rng(); centers+=rng.integers(-int(jitter),int(jitter)+1,size=n_slices)
    return np.clip(centers,0,n_frames-1)


def preprocess_volume(v:np.ndarray,n_slices:int=16,image_size:int=224)->torch.Tensor:
    v=_normalise_volume(v); idx=_centers(len(v),n_slices,0); tensor=torch.from_numpy(v[idx]).unsqueeze(1); return F.interpolate(tensor,(image_size,image_size),mode="bilinear",align_corners=False).squeeze(1)


def preprocess_triplets(v:np.ndarray,n_slices:int=16,image_size:int=224,gap:int=1,*,center_offset:int=0,jitter:int=0,rng:np.random.Generator|None=None)->torch.Tensor:
    """Build 2.5D triplets with optional center TTA offset or training jitter."""
    if gap<1:raise ValueError("2.5D gap must be >=1")
    v=_normalise_volume(v); centers=_centers(len(v),n_slices,gap,center_offset=center_offset,jitter=jitter,rng=rng); offsets=np.asarray([-gap,0,gap],int); idx=np.clip(centers[:,None]+offsets[None,:],0,len(v)-1); tensor=torch.from_numpy(v[idx].astype(np.float32,copy=False)); return F.interpolate(tensor,(image_size,image_size),mode="bilinear",align_corners=False)
