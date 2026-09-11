"""Independent B57 model construction and public-only encoder initialization."""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .b57_protocol import ARMS, VERSION, sha256_file, write_json
from .constants import N_TARGETS

DINO_MODEL = "vit_small_patch14_dinov2.lvd142m"
DINO_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
CONVNEXT_URL = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"
# Tensor fingerprints from the above public downloads, verified 2026-09-11
# with torchvision 0.23.0 / timm 1.0.20. Unlike torch.save archive bytes these
# do not depend on the output filename. A relabelled fine-tuned encoder fails.
PUBLIC_STATE_SHA256 = {
    ARMS[0]: "5335ddcf97589274158d7fd1e7b8723a3c20aae1549e414476d2b6817dc7dd1d",
    ARMS[1]: "da01dc066b2a6aea10165a09242353c4c9178032d76f9a5494e880cf5864dd92",
}


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def dino_backbone(*, pretrained=False):
    import timm
    # Pin the implementation in the B57 environment; load the authors' public
    # backbone directly. timm applies its DINOv2 state-dict conversion.
    return timm.create_model(
        DINO_MODEL, pretrained=pretrained, num_classes=0, dynamic_img_size=True,
        pretrained_cfg_overlay={"hf_hub_id": "", "url": DINO_URL},
    )


def public_weights(root, arm):
    if arm not in ARMS:
        raise ValueError(f"unknown B57 arm: {arm}")
    root = Path(root)
    manifest = json.loads((root / f"{arm}.json").read_text())
    expected_url = CONVNEXT_URL if arm == ARMS[0] else DINO_URL
    if (manifest.get("version") != VERSION or manifest.get("arm") != arm
            or manifest.get("source_url") != expected_url
            or manifest.get("competition_training_studies") != []):
        raise ValueError("B57 accepts only its public encoder initialization artifacts")
    path = root / f"{arm}.pt"
    if sha256_file(path) != manifest["sha256"]:
        raise ValueError("B57 public encoder hash mismatch")
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if not weights or not all(isinstance(v, torch.Tensor) for v in weights.values()):
        raise ValueError("public initialization must be a bare encoder state dict")
    if state_digest(weights) != PUBLIC_STATE_SHA256[arm]:
        raise ValueError("encoder tensors differ from the pinned public initialization")
    return weights, manifest


def prepare_public_weights(root):
    from .model import ConvNeXtSliceEncoder
    import timm
    import torchvision
    if timm.__version__ != "1.0.20":
        raise ValueError("B57 v1 requires timm==1.0.20")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        if (root / f"{arm}.json").exists():
            public_weights(root, arm)
            continue
        if (root / f"{arm}.pt").exists():
            raise FileExistsError(f"unmanifested initialization artifact: {root / (arm + '.pt')}")
        encoder = (ConvNeXtSliceEncoder(pretrained_weights=True, normalize_input=True)
                   if arm == ARMS[0] else dino_backbone(pretrained=True))
        fingerprint = state_digest(encoder.state_dict())
        if fingerprint != PUBLIC_STATE_SHA256[arm]:
            raise ValueError("download/cache differs from the pinned public encoder tensors")
        path = root / f"{arm}.pt"
        temporary = path.with_suffix(".writing")
        torch.save(encoder.state_dict(), temporary)
        temporary.replace(path)
        write_json(root / f"{arm}.json", {
            "version": VERSION, "arm": arm, "sha256": sha256_file(path),
            "tensor_sha256": fingerprint,
            "source_url": CONVNEXT_URL if arm == ARMS[0] else DINO_URL,
            "competition_training_studies": [], "fresh_competition_heads": True,
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "timm": timm.__version__,
        })
        del encoder
        gc.collect()


@dataclass
class SliceOutput:
    logits: torch.Tensor


class DinoSliceTransformer(nn.Module):
    """Slice CLS features -> one series Transformer -> 12 study queries.

    This is an MST-inspired multi-series model, not an exact reproduction of
    the single-series meniscus experiment. No additional slice-position
    embedding: the series block contextualizes a set of slices. B42 supplies
    the same ordered-position metadata and 2.5D triplets as the reference.
    """
    def __init__(self, encoder, *, dim=384, heads=6, dropout=.1, chunk_size=2,
                 gradient_checkpointing=True):
        super().__init__()
        if dim % heads or chunk_size < 1:
            raise ValueError("invalid slice transformer dimensions/chunk size")
        self.encoder = encoder
        self.encoder.requires_grad_(True)
        self.chunk_size = int(chunk_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.series_cls = nn.Parameter(torch.randn(1, 1, dim) * .02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 2,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.slice_context = nn.TransformerEncoder(layer, 1, norm=nn.LayerNorm(dim),
                                                   enable_nested_tensor=False)
        self.plane = nn.Embedding(4, dim, padding_idx=0)
        self.fluid = nn.Embedding(3, dim, padding_idx=0)
        self.fat = nn.Embedding(3, dim, padding_idx=0)
        self.queries = nn.Parameter(torch.randn(1, N_TARGETS, dim) * .02)
        self.study_attention = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                                     batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Parameter(torch.randn(N_TARGETS, dim) * .02)
        self.bias = nn.Parameter(torch.zeros(N_TARGETS))
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def encode_chunk(self, x):
        # Preserve the B42 rectangle and area. Pad at most 13 pixels to satisfy
        # ViT's patch-14 grid instead of squeezing MRI rectangles into squares.
        ph, pw = (-x.shape[-2]) % 14, (-x.shape[-1]) % 14
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return self.encoder((x - self.mean.to(x.dtype)) / self.std.to(x.dtype))

    def forward(self, volumes, present, series_meta, slice_position):
        present = present.reshape(-1)
        meta = series_meta.reshape(-1, 3)
        position = slice_position.reshape(len(volumes), -1)
        if len(volumes) != len(present) or len(meta) != len(volumes):
            raise ValueError("B57 series/mask/metadata mismatch")
        tokens = []
        for i, volume in enumerate(volumes):
            if float(present[i]) <= 0:
                continue
            if volume.ndim != 4 or volume.shape[1] != 3 or len(volume) != position.shape[1]:
                raise ValueError("B57 expects [slices,3,H,W] and matching positions")
            chunks = []
            for x in volume.split(self.chunk_size):
                if self.training and self.gradient_checkpointing:
                    z = checkpoint(self.encode_chunk, x, use_reentrant=False, preserve_rng_state=True)
                else:
                    z = self.encode_chunk(x)
                chunks.append(z)
            z = torch.cat(chunks)[torch.argsort(position[i], stable=True)].unsqueeze(0)
            cls = self.series_cls.to(z.dtype)
            token = self.slice_context(torch.cat((cls, z), dim=1))[:, 0]
            m = meta[i]
            token = token + self.plane(m[0]) + self.fluid(m[1]) + self.fat(m[2])
            tokens.append(token)
        if not tokens:
            raise ValueError("B57 study has no readable series")
        series = torch.stack(tokens, dim=1)
        query = self.queries.to(series.dtype)
        attended, _ = self.study_attention(query, series, series, need_weights=False)
        features = self.norm(query + attended)
        logits = (features * self.classifier[None]).sum(-1) + self.bias
        return SliceOutput(logits.float())


def build_model(arm, settings, config, public_root=None):
    """No base-checkpoint argument exists: never inherit competition heads."""
    if arm not in ARMS:
        raise ValueError(f"unknown B57 arm: {arm}")
    weights = public_weights(public_root, arm)[0] if public_root is not None else None
    if arm == ARMS[0]:
        from .b34_training_only_context_scaffold import b34_model_spec, build_b34_model
        from .b50_adapted_hierarchy_mil import B50AdaptedHierarchySparseMILResidual
        base = build_b34_model(b34_model_spec(settings, normalize_input=True),
                               encoder_state=weights, pretrained_weights=False)
        model = B50AdaptedHierarchySparseMILResidual(
            base, grid_size=6, top_k=8, temperature=1., encoder_trainable_stages=5,
            encoder_chunk_size=config["encoder_chunk_size"], adapt_hierarchy=True,
        )
        model.base.encoder.requires_grad_(True)
        model.gradient_checkpointing = config["gradient_checkpointing"]
        return model
    encoder = dino_backbone()
    if weights is not None:
        encoder.load_state_dict(weights, strict=True)
    return DinoSliceTransformer(
        encoder, dim=config["candidate_dim"], heads=config["candidate_heads"],
        dropout=config["candidate_dropout"], chunk_size=config["encoder_chunk_size"],
        gradient_checkpointing=config["gradient_checkpointing"],
    )


def encoder_of(model, arm):
    return model.base.encoder if arm == ARMS[0] else model.encoder


def parameter_groups(model, arm, config):
    encoder = list(encoder_of(model, arm).parameters())
    if not encoder or not all(p.requires_grad for p in encoder):
        raise ValueError("B57 requires full encoder fine-tuning")
    ids = {id(p) for p in encoder}
    heads = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    groups = [{"params": encoder, "lr": config["encoder_lr"], "name": "public_encoder"},
              {"params": heads, "lr": config["head_lr"], "name": "fresh_heads"}]
    grouped = [id(p) for g in groups for p in g["params"]]
    if len(grouped) != len(set(grouped)) or set(grouped) != {
        id(p) for p in model.parameters() if p.requires_grad
    }:
        raise RuntimeError("optimizer does not cover trainable parameters exactly once")
    return groups
