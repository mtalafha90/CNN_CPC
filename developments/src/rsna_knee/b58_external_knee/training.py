"""Fine-tune the adapted encoder with the unchanged B57 DINOv2 study recipe."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from . import VERSION
from .protocol import load
from .ssl import contract, load_encoder
from ..b57_models import ARMS, build_model, parameter_groups
from ..b57_protocol import digest, load_protocol, require_same_run, sha256_file, write_json
from ..b57_training import (atomic_torch_save, batch_step, loss_for, make_dataset, make_loader,
                            predict, runtime_for)
from ..b7_weak_supervision import seed_everything, target_balance_multipliers
from ..b52_competition_training import macro_auc
from ..runtime import make_scaler
from ..training_resume import load_checkpoint, resume, save_checkpoint


def construct(root, p, rsna, *, adapted=True):
    c = rsna["config"]
    seed_everything(c["seed"])
    model = build_model(ARMS[1], rsna["b42_config"], c, Path(root)/"public_init")
    if adapted:
        weights, _ = load_encoder(root, p)
        model.encoder.load_state_dict(weights, strict=True)
    return model


def fine_contract(root, p, rsna, runtime):
    _, meta = load_encoder(root, p)
    return contract(root, p, runtime, "finetune") | {
        "encoder_initialization_sha256": meta["encoder_sha256"],
        "finetune_config_sha256": digest(rsna["config"]), "epochs": rsna["config"]["epochs"],
        "training_uids_sha256": rsna["split_uid_hashes"]["train"],
        "validation_uids_sha256": rsna["split_uid_hashes"]["validation"],
        "gold_studies_in_gradient": 0, "external_labels_used": False}


def preflight(root, *, device="cuda", workers=0, adapted=False):
    root = Path(root)
    p = load(root)
    rsna = load_protocol(root/"rsna_protocol")
    c, runtime = rsna["config"], runtime_for(device, workers)
    model = construct(root, p, rsna, adapted=adapted).to(runtime.device).train()
    groups = parameter_groups(model, ARMS[1], c)
    data = make_dataset(root/"rsna_protocol", rsna, "train")
    multiplier = torch.from_numpy(target_balance_multipliers(data.weights)).to(runtime.device)
    active = [i for i, weight in enumerate(data.weights) if weight.sum() > 0]
    chosen = sorted(active, key=lambda i: (-len(data.series_records[data.study_uids[i]]), data.study_uids[i]))[:c["batch_size"]]
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    for i in chosen:
        _, loss = loss_for(model, ARMS[1], data[i], runtime, multiplier, c)
        (loss/len(chosen)).backward()
    counts = {}
    for group in groups:
        grads = [v.grad for v in group["params"] if v.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"invalid B58 supervised gradients: {group['name']}")
        counts[group["name"]] = sum(int(torch.count_nonzero(g) > 0) for g in grads)
        if not counts[group["name"]]:
            raise RuntimeError("B58 supervised module received no learning signal")
    identity = fine_contract(root, p, rsna, runtime) if adapted else contract(root, p, runtime, "architecture_preflight")
    result = {"passed": True, "optimizer_steps": 0, "adapted_encoder": adapted,
              "contract": identity, "gradient_tensor_counts": counts,
              "study_uids": [data.study_uids[i] for i in chosen],
              "peak_cuda_gib": torch.cuda.max_memory_allocated()/2**30 if runtime.device.type == "cuda" else None}
    name = "finetune_preflight.json" if adapted else "architecture_preflight.json"
    write_json(root/name, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def export_predictions(path, values, identity, split, checkpoint_sha):
    path = Path(path)
    tmp = path.with_suffix(".writing")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)
    write_json(path.with_suffix(".json"), {"version": VERSION, "contract": identity, "split": split,
               "checkpoint_sha256": checkpoint_sha, "npz_sha256": sha256_file(path),
               "uid_sha256": digest(values["uids"].tolist())})


def train(root, *, device="cuda", workers=0):
    root = Path(root)
    p = load(root)
    rsna = load_protocol(root/"rsna_protocol")
    c, out, runtime = rsna["config"], root/"finetune", runtime_for(device, workers)
    identity = fine_contract(root, p, rsna, runtime)
    check = json.loads((root/"finetune_preflight.json").read_text())
    require_same_run(check["contract"], identity)
    if not check["passed"] or not check["adapted_encoder"]:
        raise ValueError("run B58's adapted-encoder preflight first")
    if (out/"complete.json").exists():
        complete = json.loads((out/"complete.json").read_text())
        require_same_run(complete["contract"], identity)
        if complete["checkpoint_sha256"] != sha256_file(out/"final.pt"):
            raise ValueError("completed B58 checkpoint changed")
        print("[B58] supervised training already COMPLETE", flush=True)
        return out/"final.pt"
    model = construct(root, p, rsna).to(runtime.device).train()
    groups = parameter_groups(model, ARMS[1], c)
    optimizer = torch.optim.AdamW(groups, weight_decay=c["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda e: .01+.99*(1+np.cos(np.pi*min(e, c["epochs"])/c["epochs"]))/2)
    scaler = make_scaler(runtime)
    data = make_dataset(root/"rsna_protocol", rsna, "train")
    valid = make_dataset(root/"rsna_protocol", rsna, "validation")
    multipliers = torch.from_numpy(target_balance_multipliers(data.weights))
    saved = load_checkpoint(out)
    if saved is not None:
        require_same_run(saved["extra"]["contract"], identity)
        if not 0 < saved["epoch"] <= c["epochs"]:
            raise ValueError("invalid B58 recovery epoch")
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f"B58 finetune directory has no recovery: {out}")
    state = resume(out, model=model, version=VERSION, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
    if not state.restored:
        seed_everything(c["seed"]+101)
    history = list(state.history)
    print(f"[B58] finetune {runtime.describe()}; {state.describe()}", flush=True)
    for epoch in range(state.start_epoch, c["epochs"]+1):
        start, total, seen = time.monotonic(), 0., []
        loader = make_loader(data, runtime, c, epoch=epoch)
        model.train()
        for batch, items in enumerate(loader, 1):
            loss, _ = batch_step(model, ARMS[1], items, runtime, optimizer, scaler, multipliers, c)
            total += loss
            seen.extend(item["study_uid"] for item in items)
            if batch % 100 == 0:
                print(f"[B58] E{epoch} {batch}/{len(loader)} loss={total/batch:.5f}", flush=True)
        if len(seen) != len(set(seen)) or set(seen) != set(rsna["splits"]["train"]):
            raise RuntimeError("B58 supervised epoch exposure differs from B57 train membership")
        scheduler.step()
        values = predict(model, make_loader(valid, runtime, c), runtime)
        scores = macro_auc(values["target"], values["weight"], values["prediction"])
        if scores["targets_defined"] != 12 or not np.isfinite(scores["macro_auc"]):
            raise RuntimeError("B58 validation does not define all 12 AUCs")
        history.append({"epoch": epoch, "train_loss": total/batch, "validation": scores,
                        "epoch_minutes": (time.monotonic()-start)/60,
                        "training_uids_sha256": digest(sorted(seen))})
        save_checkpoint(out, epoch=epoch, model=model, version=VERSION, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, history=history, extra={"contract": identity})
        write_json(out/"history.json", history)
        print(f"[B58] E{epoch} macroAUC={scores['macro_auc']:.6f}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(out/"final.pt", {"version": VERSION, "contract": identity,
        "model_state": model.state_dict(), "model_config": c, "b42_config": rsna["b42_config"],
        "completed_epochs": c["epochs"], "selection": "fixed_final_epoch", "history": history})
    checkpoint_sha = sha256_file(out/"final.pt")
    for split, dataset in (("validation", valid), ("expert", make_dataset(root/"rsna_protocol", rsna, "expert"))):
        values = predict(model, make_loader(dataset, runtime, c), runtime)
        export_predictions(out/f"{split}.npz", values, identity, split, checkpoint_sha)
    write_json(out/"complete.json", {"version": VERSION, "contract": identity,
               "completed_epochs": c["epochs"], "checkpoint_sha256": checkpoint_sha})
    print("[B58] supervised training: COMPLETE", flush=True)
    return out/"final.pt"
