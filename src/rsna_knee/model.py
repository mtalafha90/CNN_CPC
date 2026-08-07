from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from .constants import N_TARGETS


class ConvNeXtSliceEncoder(nn.Module):
    def __init__(self,in_channels:int=3,*,pretrained_weights:bool=True,normalize_input:bool=True)->None:
        super().__init__(); weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained_weights else None; net=convnext_tiny(weights=weights); first=net.features[0][0]
        if in_channels!=3:
            replacement=nn.Conv2d(in_channels,first.out_channels,kernel_size=first.kernel_size,stride=first.stride,padding=first.padding,bias=first.bias is not None)
            if pretrained_weights:
                with torch.no_grad():
                    mean_weight=first.weight.mean(dim=1,keepdim=True); replacement.weight.copy_(mean_weight.repeat(1,in_channels,1,1));
                    if first.bias is not None: replacement.bias.copy_(first.bias)
            net.features[0][0]=replacement
        self.features=net.features; self.avgpool=net.avgpool; self.pre_classifier=nn.Sequential(*list(net.classifier.children())[:-1]); self.out_dim=int(net.classifier[-1].in_features); self.normalize_input=bool(normalize_input)
        rgb_mean=torch.tensor([0.485,0.456,0.406],dtype=torch.float32); rgb_std=torch.tensor([0.229,0.224,0.225],dtype=torch.float32); mean=rgb_mean if in_channels==3 else rgb_mean.mean().repeat(in_channels); std=rgb_std if in_channels==3 else rgb_std.mean().repeat(in_channels); self.register_buffer("input_mean",mean.view(1,in_channels,1,1),persistent=False); self.register_buffer("input_std",std.view(1,in_channels,1,1),persistent=False)
    def forward(self,x):
        if self.normalize_input:x=(x-self.input_mean.to(dtype=x.dtype))/self.input_std.to(dtype=x.dtype)
        return self.pre_classifier(self.avgpool(self.features(x)))


class KneeMILNet(nn.Module):
    """ConvNeXt 2.5D tokens + cross-sequence contextual fusion + target MIL."""
    def __init__(self,n_streams:int,n_slices:int,*,in_channels:int=3,pretrained_weights:bool=True,normalize_input:bool=True,dropout:float=0.25,encoder_batch_size:int=24,gradient_checkpointing:bool=True,transformer_layers:int=2,transformer_heads:int=8,transformer_ff_mult:float=2.0)->None:
        super().__init__()
        if n_streams<1 or n_slices<1:raise ValueError("n_streams and n_slices must be positive")
        self.n_streams=int(n_streams); self.n_slices=int(n_slices); self.in_channels=int(in_channels); self.encoder_batch_size=int(encoder_batch_size); self.gradient_checkpointing=bool(gradient_checkpointing)
        self.encoder=ConvNeXtSliceEncoder(in_channels,pretrained_weights=pretrained_weights,normalize_input=normalize_input); d=self.encoder.out_dim
        self.slice_position=nn.Parameter(torch.randn(n_slices,d)*0.02); self.stream_embedding=nn.Parameter(torch.randn(n_streams,d)*0.02)
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=int(transformer_heads),dim_feedforward=int(d*float(transformer_ff_mult)),dropout=float(dropout),activation="gelu",batch_first=True,norm_first=True)
        self.context=nn.TransformerEncoder(layer,num_layers=int(transformer_layers),norm=nn.LayerNorm(d))
        self.slice_key=nn.Linear(d,d,bias=False); self.slice_query=nn.Parameter(torch.randn(N_TARGETS,d)*0.02); self.stream_key=nn.Linear(d,d,bias=False); self.stream_query=nn.Parameter(torch.randn(N_TARGETS,d)*0.02)
        self.norm=nn.LayerNorm(d); self.dropout=nn.Dropout(dropout); self.target_weight=nn.Parameter(torch.empty(N_TARGETS,d)); self.target_bias=nn.Parameter(torch.zeros(N_TARGETS)); nn.init.xavier_uniform_(self.target_weight)
    def _encode_chunk(self,chunk):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():return checkpoint(self.encoder,chunk,use_reentrant=False)
        return self.encoder(chunk)
    def _reshape_streams(self,volumes):
        if volumes.ndim==5:b,k,s,h,w=volumes.shape; stream_volumes=volumes.reshape(b*k,s,1,h,w); channels=1
        elif volumes.ndim==6:b,k,s,channels,h,w=volumes.shape; stream_volumes=volumes.reshape(b*k,s,channels,h,w)
        else:raise ValueError(f"unexpected volume shape {tuple(volumes.shape)}")
        if k!=self.n_streams or s!=self.n_slices or channels!=self.in_channels:raise ValueError("volume tensor does not match model contract")
        return stream_volumes,b,k,s
    def _encode_slices(self,volumes,present):
        stream_volumes,b,k,s=self._reshape_streams(volumes); active_indices=torch.nonzero(present.reshape(-1)>0,as_tuple=False).flatten(); d=self.encoder.out_dim
        if active_indices.numel()==0:return volumes.new_zeros((b,k,s,d))
        active=stream_volumes.index_select(0,active_indices); flat=active.reshape(-1,*active.shape[2:]); encoded=torch.cat([self._encode_chunk(chunk) for chunk in flat.split(self.encoder_batch_size,dim=0)],dim=0).reshape(active.shape[0],s,d); all_features=encoded.new_zeros((b*k,s,d)).index_copy(0,active_indices,encoded); features=all_features.reshape(b,k,s,d); mask=present[:,:,None,None].to(dtype=features.dtype); return (features+self.slice_position[None,None,:,:]+self.stream_embedding[None,:,None,:])*mask
    def _contextualize(self,features,present):
        b,k,s,d=features.shape; tokens=features.reshape(b,k*s,d); padding=(present<=0)[:,:,None].expand(b,k,s).reshape(b,k*s); contextual=self.context(tokens,src_key_padding_mask=padding); contextual=contextual.masked_fill(padding[:,:,None],0.0); return contextual.reshape(b,k,s,d)
    def forward(self,volumes,present):
        if present.ndim!=2 or present.shape[1]!=self.n_streams:raise ValueError("present mask does not match stream contract")
        features=self._contextualize(self._encode_slices(volumes,present),present); d=features.shape[-1]; scale=math.sqrt(d)
        slice_scores=torch.einsum("bksd,td->bkts",self.slice_key(features),self.slice_query)/scale; slice_weights=torch.softmax(slice_scores,dim=-1); series=torch.einsum("bkts,bksd->bktd",slice_weights,features)*present[:,:,None,None].to(dtype=features.dtype)
        stream_scores=torch.einsum("bktd,td->btk",self.stream_key(series),self.stream_query)/scale; stream_scores=stream_scores.masked_fill(present[:,None,:]<=0,-1e4); pooled=torch.einsum("btk,bktd->btd",torch.softmax(stream_scores,dim=-1),series); pooled[present.sum(dim=1)<=0]=0; pooled=self.dropout(self.norm(pooled)); return (pooled*self.target_weight[None,:,:]).sum(dim=-1)+self.target_bias
