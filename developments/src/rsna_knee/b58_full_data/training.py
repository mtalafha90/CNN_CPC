"""Full-coverage SSL/native-label stages with atomic, exact-step recovery."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import VERSION
from .data import epoch_order, triplets
from .models import NativeHeads, native_loss
from .protocol import load
from ..b58_external_knee.ssl import Adaptation, augmented_view
from ..b57_models import ARMS, dino_backbone, public_weights, state_digest
from ..b57_protocol import digest, require_same_run, sha256_file, write_json
from ..b57_training import atomic_torch_save, runtime_for
from ..b7_weak_supervision import seed_everything
from ..runtime import autocast, make_scaler
from ..training_resume import rng_state, set_rng_state


def contract(root, p, runtime, stage):
    import timm
    import torchvision
    if timm.__version__ != "1.0.20":
        raise ValueError("B58 full run requires timm==1.0.20")
    return {"version": VERSION, "stage": stage, "protocol_sha256": sha256_file(Path(root)/"protocol.json"),
            "rsna_identity": p["rsna_identity"], "precision": str(runtime.amp_dtype),
            "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__,
            "numpy": np.__version__, "device_type": runtime.device.type, "external_labels_used": True}


def load_encoder(root, p, stage="supervised"):
    root = Path(root)
    folder = root / stage
    meta = json.loads((folder / "complete.json").read_text())
    expected_steps = p["ssl_steps"] if stage == "pretrain" else p["supervised_steps"]
    if (meta["version"] != VERSION or meta["steps"] != expected_steps or meta["contract"]["stage"] != stage
            or meta["contract"]["protocol_sha256"] != sha256_file(root / "protocol.json")
            or meta["contract"]["rsna_identity"] != p["rsna_identity"]
            or meta["epochs"] != p["config"]["ssl_epochs" if stage == "pretrain" else "supervised_epochs"]):
        raise ValueError("full-data encoder completion/identity mismatch")
    path = folder / "encoder.pt"
    if sha256_file(path) != meta["encoder_sha256"]:
        raise ValueError("full-data encoder file changed")
    if stage == "supervised" and sha256_file(folder / "native_model.pt") != meta["native_model_sha256"]:
        raise ValueError("full-data native heads file changed")
    manifest = json.loads((root / "manifest.json").read_text())
    items = manifest["records"] if stage == "pretrain" else [r for r in manifest["cases"] if sum(r["weights"]) > 0]
    expected = {source: count * meta["epochs"] for source, count in Counter(r["source"] for r in items).items()}
    if meta["source_exposure"] != expected:
        raise ValueError("full-data completed source exposure mismatch")
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if state_digest(weights) != meta["tensor_sha256"] or meta["tensor_sha256"] == meta["initial_tensor_sha256"]:
        raise ValueError("full-data encoder tensors changed or did not learn")
    return weights, meta


class Job:
    def __init__(self, root, p, manifest, stage, *, adapted=True):
        self.p, self.c, self.stage = p, p["config"], stage
        self.records, self.tasks = manifest["records"], manifest["tasks"]
        seed_everything(self.c["seed"] + int(stage == "supervised"))
        encoder = dino_backbone()
        weights = (load_encoder(root, p, "pretrain")[0] if stage == "supervised" and adapted
                   else public_weights(Path(root) / "public_init", ARMS[1])[0])
        encoder.load_state_dict(weights, strict=True)
        self.initial_digest = state_digest(weights)
        if stage == "pretrain":
            self.items = self.records
            self.epochs = self.c["ssl_epochs"]
            self.model = Adaptation(encoder, self.c | {"ssl_steps": p["ssl_steps"]})
        elif stage == "supervised":
            self.items = [r for r in manifest["cases"] if sum(r["weights"]) > 0]
            self.epochs = self.c["supervised_epochs"]
            self.model = NativeHeads(encoder, self.tasks, self.c["encoder_chunk_size"])
        else:
            raise ValueError("unknown full-data training stage")
        self.source_counts = Counter(r["source"] for r in self.items)

    def parameter_groups(self):
        # Resolve parameters after moving the model to its final device.
        if self.stage == "pretrain":
            return [{"params": [v for v in self.model.student.parameters() if v.requires_grad], "lr": self.c["ssl_encoder_lr"]},
                    {"params": self.model.student_head.parameters(), "lr": self.c["ssl_head_lr"]}]
        return [{"params": self.model.encoder.parameters(), "lr": self.c["encoder_lr"]},
                {"params": self.model.heads.parameters(), "lr": self.c["head_lr"]}]

    def loss(self, item, step, runtime):
        c = self.c
        if self.stage == "pretrain":
            x = triplets(item, c["ssl_batch_size"], c["ssl_side"])
            gen = torch.Generator().manual_seed(c["seed"] + step * 7919)
            a = torch.stack([augmented_view(v, gen, c["ssl_crop_min"]) for v in x]).to(runtime.device)
            b = torch.stack([augmented_view(v, gen, c["ssl_crop_min"]) for v in x]).to(runtime.device)
            with autocast(runtime):
                return self.model.loss(a, b)
        volumes = (triplets(self.records[i], c["slices_per_series"], c["supervised_side"])
                   for i in item["record_indices"])
        with autocast(runtime):
            logits = self.model(item["source"], volumes)
            loss = native_loss(logits, self.tasks[item["source"]], item["values"], item["weights"])
        # Inverse source frequency: equal source contributions over a full pass.
        scale = len(self.items) / (len(self.source_counts) * self.source_counts[item["source"]])
        return loss * scale, None, {"native_loss": float(loss.detach()), "source_scale": scale}

    def update(self, centre, step):
        if self.stage == "pretrain":
            self.model.update_teacher(centre, step)

    def encoder_state(self):
        encoder = self.model.teacher if self.stage == "pretrain" else self.model.encoder
        return {k: v.detach().cpu() for k, v in encoder.state_dict().items()}


def preflight(root, stage, *, device="cuda", adapted=True):
    root = Path(root)
    p, runtime = load(root), runtime_for(device, 0)
    manifest = json.loads((root / "manifest.json").read_text())
    job = Job(root, p, manifest, stage, adapted=adapted)
    job.model.to(runtime.device).train()
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    results = {}
    for source in p["config"]["sources"]:
        # Largest labelled case tests the actual memory demand, not a one-slice toy.
        choices = [r for r in job.items if r["source"] == source]
        item = max(choices, key=lambda r: len(r.get("record_indices", [0])))
        job.model.zero_grad(set_to_none=True)
        loss, _, diagnostics = job.loss(item, 0, runtime)
        if not torch.isfinite(loss):
            raise FloatingPointError("full-data preflight loss is nonfinite")
        loss.backward()
        encoder = job.model.student if stage == "pretrain" else job.model.encoder
        head = job.model.student_head if stage == "pretrain" else job.model.heads[source]
        counts = {}
        for name, module in (("encoder", encoder), ("head", head)):
            grads = [v.grad for v in module.parameters() if v.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise RuntimeError(f"invalid {source}/{name} gradients")
            counts[name] = sum(int(torch.count_nonzero(g) > 0) for g in grads)
            if not counts[name]:
                raise RuntimeError(f"no learning signal: {source}/{name}")
        results[source] = {"loss": float(loss.detach()), "gradients": counts, **diagnostics}
    result = {"passed": True, "optimizer_steps": 0, "adapted": adapted,
              "contract": contract(root, p, runtime, stage), "sources": results,
              "peak_cuda_gib": torch.cuda.max_memory_allocated()/2**30 if runtime.device.type == "cuda" else None}
    name = stage + ("" if adapted else "_architecture") + "_preflight.json"
    write_json(root / name, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def expected_counts(items, seed, epochs, steps):
    if not 0 <= steps <= len(items) * epochs:
        raise ValueError("recovery step outside full-data schedule")
    complete, remainder = divmod(steps, len(items))
    counts = Counter({s: n * complete for s, n in Counter(r["source"] for r in items).items() if complete})
    if remainder:
        order = epoch_order(items, seed, complete)
        counts.update(items[i]["source"] for i in order[:remainder])
    return dict(counts)


def train(root, stage, *, device="cuda"):
    root = Path(root)
    p, runtime = load(root), runtime_for(device, 0)
    c = p["config"]
    identity = contract(root, p, runtime, stage)
    check = json.loads((root / f"{stage}_preflight.json").read_text())
    require_same_run(check["contract"], identity)
    if not check["passed"] or not check["adapted"]:
        raise ValueError("run the matching full-data preflight first")
    out = root / stage
    if (out / "complete.json").exists():
        _, meta = load_encoder(root, p, stage)
        require_same_run(meta["contract"], identity)
        print(f"[B58 full] {stage} already COMPLETE", flush=True)
        return out / "encoder.pt"
    manifest = json.loads((root / "manifest.json").read_text())
    job = Job(root, p, manifest, stage)
    job.model.to(runtime.device).train()
    optimizer = torch.optim.AdamW(job.parameter_groups(), weight_decay=c["ssl_weight_decay" if stage == "pretrain" else "weight_decay"])
    initial_lrs = [group["lr"] for group in optimizer.param_groups]
    scaler = make_scaler(runtime)
    total = len(job.items) * job.epochs
    expected_total = p["ssl_steps" if stage == "pretrain" else "supervised_steps"]
    if total != expected_total:
        raise ValueError("frozen full-data step count changed")
    start, history, seen = 0, [], Counter()
    recovery = out / "recovery_latest.pt"
    if recovery.exists():
        saved = torch.load(recovery, map_location="cpu", weights_only=False)
        require_same_run(saved["contract"], identity)
        start = saved["step"]
        if not 0 <= start <= total or saved["source_exposure"] != expected_counts(job.items, c["seed"], job.epochs, start):
            raise ValueError("full-data recovery step/source exposure mismatch")
        if saved["initial_tensor_sha256"] != job.initial_digest:
            raise ValueError("full-data initial encoder changed")
        job.model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        set_rng_state(saved["rng"])
        history, seen = saved["history"], Counter(saved["source_exposure"])
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f"stage output has no valid recovery: {out}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[B58 full] {stage}: resume {start}/{total}; {runtime.describe()}", flush=True)
    for epoch in range(start // len(job.items), job.epochs):
        order = epoch_order(job.items, c["seed"], epoch)
        for offset, index in enumerate(order):
            step = epoch * len(order) + offset
            if step < start:
                continue
            factor = min(1., (step + 1) / min(100, total)) * (.01 + .99 * (1 + math.cos(math.pi * step / total)) / 2)
            for group, lr in zip(optimizer.param_groups, initial_lrs):
                group["lr"] = lr * factor
            optimizer.zero_grad(set_to_none=True)
            item = job.items[index]
            loss, centre, diagnostics = job.loss(item, step, runtime)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite {stage} loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([v for v in job.model.parameters() if v.requires_grad], 1., error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            job.update(centre, step + 1)
            seen[item["source"]] += 1
            if (step + 1) % c["save_every"] == 0 or offset + 1 == len(order):
                if sum(seen.values()) != step + 1 or (offset + 1 == len(order)
                        and dict(seen) != expected_counts(job.items, c["seed"], job.epochs, step + 1)):
                    raise RuntimeError("full-data schedule omitted or repeated an item")
                history.append({"step": step + 1, "epoch": epoch + 1, "loss": float(loss.detach()), **diagnostics})
                atomic_torch_save(recovery, {"contract": identity, "step": step + 1,
                    "model": job.model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "rng": rng_state(), "history": history, "source_exposure": dict(seen), "initial_tensor_sha256": job.initial_digest})
                write_json(out / "history.json", history)
                print(f"[B58 full] {stage} epoch {epoch+1}/{job.epochs} step {step+1}/{total} loss={float(loss.detach()):.5f}", flush=True)
    state = job.encoder_state()
    fingerprint = state_digest(state)
    if fingerprint == job.initial_digest:
        raise RuntimeError("full-data encoder did not learn")
    atomic_torch_save(out / "encoder.pt", state)
    # Preserve all learned native heads as well as the encoder used by RSNA.
    if stage == "supervised":
        atomic_torch_save(out / "native_model.pt", {"version": VERSION, "contract": identity,
                          "tasks": job.tasks, "model_state": job.model.state_dict()})
    write_json(out / "complete.json", {"version": VERSION, "contract": identity, "steps": total,
        "epochs": job.epochs, "source_exposure": dict(seen), "tensor_sha256": fingerprint,
        "initial_tensor_sha256": job.initial_digest, "encoder_sha256": sha256_file(out / "encoder.pt"),
        "native_model_sha256": sha256_file(out / "native_model.pt") if stage == "supervised" else None,
        "coverage": "each eligible series once per SSL epoch; each labelled case once per supervised epoch",
        "external_labels_used": stage == "supervised", "selection": "fixed final epoch"})
    print(f"[B58 full] {stage}: COMPLETE", flush=True)
    return out / "encoder.pt"
