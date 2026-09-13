"""A small two-global-view DINO-style adaptation stage, not full DINOv2 pretraining."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import VERSION
from .protocol import load
from ..b57_models import ARMS, dino_backbone, public_weights, state_digest, PUBLIC_STATE_SHA256
from ..b57_protocol import sha256_file, write_json, require_same_run
from ..b57_training import atomic_torch_save, runtime_for
from ..b7_weak_supervision import seed_everything
from ..runtime import autocast, make_scaler
from ..training_resume import rng_state, set_rng_state


def augmented_view(x, generator, crop_min=.8):
    """The same spatial and intensity transform is applied to all three slices."""
    from torchvision.transforms import functional as TF
    from torchvision.transforms import InterpolationMode
    side = x.shape[-1]
    fraction = crop_min + (1 - crop_min) * float(torch.rand((), generator=generator))
    size = max(2, round(side * math.sqrt(fraction)))
    top, left = [int(torch.randint(side-size+1, (), generator=generator)) for _ in range(2)]
    out = TF.resized_crop(x, top, left, size, size, [side, side], antialias=True)
    angle = (float(torch.rand((), generator=generator)) * 2 - 1) * 7
    out = TF.rotate(out, angle, interpolation=InterpolationMode.BILINEAR)
    gamma = .9 + .2 * float(torch.rand((), generator=generator))
    gain = .9 + .2 * float(torch.rand((), generator=generator))
    noise = torch.randn((1, side, side), generator=generator) * .01
    return (gain * out.clamp(0, 1).pow(gamma) + noise).clamp(0, 1)


class SourceSampler:
    """Stateless source -> group -> series -> slice sampling supports exact step resume."""
    def __init__(self, records, config):
        self.config = config
        self.groups = {s: defaultdict(list) for s in config["sources"]}
        for row in records:
            self.groups[row["source"]][row["group"]].append(row)
        if any(not g for g in self.groups.values()):
            raise ValueError("B58 sampling requires all four sources")
        self.keys = {s: sorted(g) for s, g in self.groups.items()}

    def choices(self, step):
        c = self.config
        rng = np.random.default_rng(c["seed"] + int(step) * 1009)
        result = []
        for i in range(c["ssl_batch_size"]):
            source = c["source_cycle"][(step*c["ssl_batch_size"]+i) % len(c["source_cycle"])]
            key = self.keys[source][int(rng.integers(len(self.keys[source])))]
            records = self.groups[source][key]
            row = records[int(rng.integers(len(records)))]
            result.append((row, int(rng.integers(row["shape"][0]))))
        return result

    def batch(self, step):
        first, second, sources = [], [], []
        generator = torch.Generator().manual_seed(self.config["seed"] + int(step) * 7919)
        for row, centre in self.choices(step):
            # Recheck the bounded cache file before consuming it; changed pixels
            # cannot silently enter after the initial full-cache verification.
            if sha256_file(row["cache"]) != row["cache_sha256"]:
                raise ValueError(f"B58 cached pixels changed: {row['cache']}")
            array = np.load(row["cache"], mmap_mode="r", allow_pickle=False)
            x = torch.from_numpy(np.array(array[centre], dtype=np.float32))
            first.append(augmented_view(x, generator, self.config["ssl_crop_min"]))
            second.append(augmented_view(x, generator, self.config["ssl_crop_min"]))
            sources.append(row["source"])
        return torch.stack(first), torch.stack(second), sources


class Projection(nn.Module):
    def __init__(self, dim, prototypes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, 128))
        self.weight = nn.Parameter(torch.randn(prototypes, 128) * .02)

    def forward(self, x):
        return F.linear(F.normalize(self.mlp(x), dim=-1), F.normalize(self.weight, dim=-1))


class Adaptation(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.student = encoder
        blocks = config["ssl_trainable_blocks"]
        if not hasattr(encoder, "blocks") or not 1 <= blocks <= len(encoder.blocks):
            raise ValueError("unsupported DINO encoder blocks")
        encoder.requires_grad_(False)
        for block in encoder.blocks[-blocks:]:
            block.requires_grad_(True)
        encoder.norm.requires_grad_(True)
        self.student_head = Projection(int(encoder.num_features), config["ssl_prototypes"])
        self.teacher = deepcopy(encoder).requires_grad_(False).eval()
        self.teacher_head = deepcopy(self.student_head).requires_grad_(False).eval()
        if hasattr(encoder, "set_grad_checkpointing"):
            encoder.set_grad_checkpointing(True)
        self.register_buffer("center", torch.zeros(1, config["ssl_prototypes"]))
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])
        self.config = config

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.teacher_head.eval()
        return self

    def loss(self, a, b):
        c = self.config
        a, b = (a-self.mean)/self.std, (b-self.mean)/self.std
        sa, sb = self.student_head(self.student(a)), self.student_head(self.student(b))
        with torch.no_grad():
            ta, tb = self.teacher_head(self.teacher(a)), self.teacher_head(self.teacher(b))
            pa = F.softmax((ta.float()-self.center)/c["ssl_teacher_temperature"], dim=-1)
            pb = F.softmax((tb.float()-self.center)/c["ssl_teacher_temperature"], dim=-1)
        loss = -.5 * ((pa * F.log_softmax(sb.float()/c["ssl_student_temperature"], dim=-1)).sum(-1).mean()
                       + (pb * F.log_softmax(sa.float()/c["ssl_student_temperature"], dim=-1)).sum(-1).mean())
        diagnostics = {"teacher_entropy": float((-(pa*pa.clamp_min(1e-8).log()).sum(-1).mean()).detach()),
                       "teacher_probability_std": float(pa.std(dim=0).mean().detach())}
        return loss, torch.cat((ta, tb)).detach().float().mean(0, keepdim=True), diagnostics

    @torch.no_grad()
    def update_teacher(self, batch_center, step):
        c = self.config
        momentum = 1 - (1-c["ssl_teacher_momentum"]) * (1+math.cos(math.pi*step/c["ssl_steps"]))/2
        for target, source in ((self.teacher, self.student), (self.teacher_head, self.student_head)):
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.mul_(momentum).add_(sp.detach(), alpha=1-momentum)
        self.center.mul_(c["ssl_center_momentum"]).add_(batch_center, alpha=1-c["ssl_center_momentum"])


def contract(root, p, runtime, stage):
    import timm
    import torchvision
    if timm.__version__ != "1.0.20":
        raise ValueError("B58 v1 requires timm==1.0.20")
    return {"version": VERSION, "stage": stage, "protocol_sha256": sha256_file(Path(root)/"protocol.json"),
            "rsna_identity": p["rsna_identity"], "precision": str(runtime.amp_dtype),
            "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__,
            "numpy": np.__version__, "device_type": runtime.device.type}


def expected_exposure(config, steps):
    counts = {source: 0 for source in config["sources"]}
    cycle = config["source_cycle"]
    full, remainder = divmod(steps * config["ssl_batch_size"], len(cycle))
    for i, source in enumerate(cycle):
        counts[source] += full + int(i < remainder)
    return counts


def build(root, p):
    seed_everything(p["config"]["seed"])
    encoder = dino_backbone()
    weights, _ = public_weights(Path(root)/"public_init", ARMS[1])
    encoder.load_state_dict(weights, strict=True)
    return Adaptation(encoder, p["config"])


def check_gradients(model):
    groups = {"encoder": model.student, "projection": model.student_head}
    result = {}
    for name, module in groups.items():
        trainable = [p for p in module.parameters() if p.requires_grad]
        if not trainable or any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
            raise RuntimeError(f"missing/nonfinite B58 gradients in {name}")
        result[name] = sum(int(torch.count_nonzero(p.grad) > 0) for p in trainable)
        if not result[name]:
            raise RuntimeError(f"zero B58 gradients in {name}")
    if any(p.grad is not None for p in model.teacher.parameters()):
        raise RuntimeError("EMA teacher received gradients")
    return result


def preflight(root, *, device="cuda"):
    root = Path(root)
    p = load(root, verify_cache=True)
    runtime = runtime_for(device, 0)
    model = build(root, p).to(runtime.device).train()
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    sampler = SourceSampler(json.loads((root/"cache_manifest.json").read_text()), p["config"])
    a, b, sources = sampler.batch(0)
    with autocast(runtime):
        loss, _, diagnostics = model.loss(a.to(runtime.device), b.to(runtime.device))
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite B58 SSL preflight loss")
    loss.backward()
    result = {"passed": True, "optimizer_steps": 0, "sources": sources,
              "gradients": check_gradients(model), "loss": float(loss.detach()), **diagnostics,
              "contract": contract(root, p, runtime, "pretrain"),
              "peak_cuda_gib": torch.cuda.max_memory_allocated()/2**30 if runtime.device.type == "cuda" else None}
    write_json(root/"ssl_preflight.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def train(root, *, device="cuda"):
    root = Path(root)
    p = load(root, verify_cache=True)
    c, out = p["config"], root/"pretrain"
    runtime = runtime_for(device, 0)
    identity = contract(root, p, runtime, "pretrain")
    check = json.loads((root/"ssl_preflight.json").read_text())
    require_same_run(check["contract"], identity)
    if not check["passed"]:
        raise ValueError("B58 SSL preflight did not pass")
    if (out/"complete.json").exists():
        load_encoder(root, p)
        print("[B58] pretraining already COMPLETE", flush=True)
        return out/"encoder.pt"
    model = build(root, p).to(runtime.device).train()
    groups = [{"params": [v for v in model.student.parameters() if v.requires_grad], "lr": c["ssl_encoder_lr"]},
              {"params": model.student_head.parameters(), "lr": c["ssl_head_lr"]}]
    optimizer = torch.optim.AdamW(groups, weight_decay=c["ssl_weight_decay"])
    scaler = make_scaler(runtime)
    sampler = SourceSampler(json.loads((root/"cache_manifest.json").read_text()), c)
    start, history, seen = 0, [], {s: 0 for s in c["sources"]}
    recovery = out/"recovery_latest.pt"
    if recovery.exists():
        saved = torch.load(recovery, map_location="cpu", weights_only=False)
        require_same_run(saved["contract"], identity)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        set_rng_state(saved["rng"])
        start, history, seen = saved["step"], saved["history"], saved["source_samples"]
        if not 0 <= start <= c["ssl_steps"] or seen != expected_exposure(c, start):
            raise ValueError("B58 recovery step/source exposure mismatch")
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f"B58 pretrain directory has no recovery: {out}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[B58] SSL {runtime.describe()}, starting step {start+1}/{c['ssl_steps']}", flush=True)
    for step in range(start, c["ssl_steps"]):
        factor = min(1., (step+1)/100) * (.01+.99*(1+math.cos(math.pi*step/c["ssl_steps"]))/2)
        for group, lr in zip(optimizer.param_groups, (c["ssl_encoder_lr"], c["ssl_head_lr"])):
            group["lr"] = lr*factor
        a, b, sources = sampler.batch(step)
        optimizer.zero_grad(set_to_none=True)
        with autocast(runtime):
            loss, center, diagnostics = model.loss(a.to(runtime.device), b.to(runtime.device))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"B58 nonfinite SSL loss at step {step+1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1., error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        model.update_teacher(center, step+1)
        for source in sources:
            seen[source] += 1
        if (step+1) % c["ssl_save_every"] == 0 or step+1 == c["ssl_steps"]:
            history.append({"step": step+1, "loss": float(loss.detach()), **diagnostics})
            atomic_torch_save(recovery, {"contract": identity, "step": step+1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
                "history": history, "source_samples": seen})
            write_json(out/"history.json", history)
            print(f"[B58] SSL {step+1}/{c['ssl_steps']} loss={float(loss.detach()):.5f} {diagnostics}", flush=True)
    state = {k: v.detach().cpu() for k, v in model.teacher.state_dict().items()}
    final_sha = state_digest(state)
    if final_sha == PUBLIC_STATE_SHA256[ARMS[1]] or seen != expected_exposure(c, c["ssl_steps"]):
        raise RuntimeError("B58 did not update the encoder or omitted a source")
    atomic_torch_save(out/"encoder.pt", state)
    write_json(out/"complete.json", {"version": VERSION, "contract": identity, "steps": c["ssl_steps"],
        "source_samples": seen, "encoder_sha256": sha256_file(out/"encoder.pt"), "tensor_sha256": final_sha,
        "selection": "EMA teacher at fixed final step", "external_labels_used": False,
        "competition_supervised_ancestor": False, "rsna_training_images_in_ssl": True})
    print("[B58] mixed-source pretraining: COMPLETE", flush=True)
    return out/"encoder.pt"


def load_encoder(root, p):
    root = Path(root)
    meta = json.loads((root/"pretrain/complete.json").read_text())
    if (meta["version"] != VERSION or meta["steps"] != p["config"]["ssl_steps"]
            or meta["contract"]["protocol_sha256"] != sha256_file(root/"protocol.json")
            or meta["contract"]["rsna_identity"] != p["rsna_identity"]
            or meta["contract"]["stage"] != "pretrain"
            or meta["external_labels_used"] is not False
            or meta["competition_supervised_ancestor"] is not False
            or meta["source_samples"] != expected_exposure(p["config"], meta["steps"])):
        raise ValueError("B58 pretraining identity/exposure mismatch")
    path = root/"pretrain/encoder.pt"
    if sha256_file(path) != meta["encoder_sha256"]:
        raise ValueError("B58 adapted encoder file changed")
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if state_digest(weights) != meta["tensor_sha256"] or meta["tensor_sha256"] == PUBLIC_STATE_SHA256[ARMS[1]]:
        raise ValueError("B58 adapted encoder tensors changed or stayed at initialization")
    return weights, meta
