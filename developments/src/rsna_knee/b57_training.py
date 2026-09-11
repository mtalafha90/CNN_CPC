"""Prepare, preflight and train the two independent B57 arms on one GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import (make_b7_dataset_config, seed_everything,
                                  target_balance_multipliers, target_balanced_weak_bce)
from .b42_constant_area_aspect_sparse_mil import B42ConstantAreaAspectDataset, collate_b42, require_b42_contract
from .b42_constant_area_aspect_sparse_training import _batch_scales, _move_study
from .b52_competition_training import macro_auc
from .b57_models import (PUBLIC_STATE_SHA256, build_model, encoder_of, parameter_groups,
                         prepare_public_weights, public_weights, state_digest)
from .b57_protocol import (ARMS, VERSION, digest, load_protocol, prepare_protocol,
                           require_same_run, sha256_file, write_json)
from .runtime import autocast, make_scaler, resolve_runtime
from .training_resume import load_checkpoint, resume, save_checkpoint

DEFAULT_ROOT = "runs/093_Experiment_B57_clean_backbone_comparison"


def runtime_for(device, workers):
    return resolve_runtime({"device": device, "precision": "auto", "num_workers": workers,
                            "pin_memory": False, "persistent_workers": False,
                            "prefetch_factor": 1, "multiprocessing_context": "spawn"})


def make_dataset(protocol_root, p, split):
    index = json.loads((Path(protocol_root) / "series_index.json").read_text())
    with np.load(Path(protocol_root) / "labels.npz", allow_pickle=False) as f:
        if split == "expert":
            uids = f["expert_uids"].tolist()
            raw = f["expert_target"].copy()
            weight = np.isfinite(raw).astype(np.float32)
            target = np.nan_to_num(raw, nan=.5)
        else:
            uids = p["splits"][split]
            lookup = {uid: i for i, uid in enumerate(f["uids"].tolist())}
            ix = [lookup[u] for u in uids]
            target, weight = f["target"][ix], f["weight"][ix]
    settings = dict(p["b42_config"])
    settings["strict_dicom"] = True
    # Both arms use exactly the same deterministic B42 tensors. No train=True
    # flag that merely claims to augment while bypassing the pixel transform.
    cfg = make_b7_dataset_config(settings, Path(p["data_root"]), train=False)
    cfg.strict_dicom = True
    cfg.tta_center_offsets = ()
    return B42ConstantAreaAspectDataset(uids, index, cfg,
        crop_focus_policy=require_b42_contract(settings), center_offsets=(0,),
        targets=target, weights=weight)


def make_loader(dataset, runtime, config, *, epoch=None):
    # Reconstruct the generator from epoch, not process age. Resuming after E4
    # gets exactly E5's shuffle, even with a new Python process/worker count.
    seed = config["seed"] + (0 if epoch is None else int(epoch) * 1009)
    return DataLoader(dataset, batch_size=config["batch_size"] if epoch is not None else 1,
        shuffle=epoch is not None, drop_last=False, collate_fn=collate_b42,
        **runtime.loader_kwargs(seed=seed))


def loss_for(model, arm, item, runtime, multipliers, config):
    volumes, position, present, meta, target, weight = _move_study(item, runtime.device)
    with autocast(runtime):
        out = model(volumes, present, meta, position)
        loss = target_balanced_weak_bce(out.logits, target, weight, multipliers)
        if arm == ARMS[0]:
            loss = loss + config["reference_aux_weight"] * target_balanced_weak_bce(
                out.local_logits, target, weight, multipliers)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"nonfinite B57 loss on {item['study_uid']}")
    return out, loss


@torch.no_grad()
def predict(model, loader, runtime):
    was_training = model.training
    uids, predictions, targets, weights = [], [], [], []
    model.eval()
    try:
        for items in loader:
            for item in items:
                volumes, pos, present, meta, _, _ = _move_study(item, runtime.device)
                with autocast(runtime):
                    out = model(volumes, present, meta, pos)
                probability = out.logits.float().sigmoid().cpu().numpy().reshape(-1)
                if not np.isfinite(probability).all():
                    raise FloatingPointError(f"nonfinite prediction on {item['study_uid']}")
                uids.append(item["study_uid"])
                predictions.append(probability)
                targets.append(item["target"].numpy())
                weights.append(item["weight"].numpy())
                del out, volumes
    finally:
        model.train(was_training)
    if not uids or len(uids) != len(set(uids)):
        raise ValueError("empty/duplicate prediction UIDs")
    return {"uids": np.asarray(uids), "prediction": np.stack(predictions),
            "target": np.stack(targets), "weight": np.stack(weights)}


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_predictions(path, prediction, *, contract, split, checkpoint_sha):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **prediction)
    os.replace(temporary, path)
    write_json(path.with_suffix(".json"), {
        "version": VERSION, "contract": contract, "split": split,
        "checkpoint_sha256": checkpoint_sha, "npz_sha256": sha256_file(path),
        "uid_sha256": digest(prediction["uids"].tolist()),
        "metric": "macro ROC AUC; fixed per-target report masks; unweighted AUC",
    })


def run_contract(protocol_root, p, arm, public_root, runtime):
    import timm
    import torchvision
    _, manifest = public_weights(public_root, arm)
    if timm.__version__ != "1.0.20":
        raise ValueError("B57 v1 requires timm==1.0.20")
    return {
        "version": VERSION, "arm": arm,
        "protocol_sha256": sha256_file(Path(protocol_root) / "protocol.json"),
        "source_digest": p["source_digest"], "config_sha256": digest(p["config"]),
        "public_encoder_sha256": manifest["sha256"], "public_source_url": manifest["source_url"],
        "training_uids_sha256": p["split_uid_hashes"]["train"],
        "validation_uids_sha256": p["split_uid_hashes"]["validation"],
        "competition_supervised_ancestor": False, "gold_studies_in_gradient": 0,
        "epochs": p["config"]["epochs"], "precision": str(runtime.amp_dtype),
        "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__,
        "numpy": np.__version__,
    }


def batch_step(model, arm, items, runtime, optimizer, scaler, multiplier_cpu, config):
    optimizer.zero_grad(set_to_none=True)
    scales = _batch_scales(items, multiplier_cpu)
    total = 0.
    for item, scale in zip(items, scales):
        out, loss = loss_for(model, arm, item, runtime, multiplier_cpu.to(runtime.device), config)
        scaler.scale(loss * scale).backward()
        total += float(loss.detach()) * scale
        del out, loss
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"],
                                              error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return total, float(grad_norm)


def preflight(*, run_root, arm, device="auto", workers=0):
    """Real training studies, backward pass, no optimizer step, disposable model."""
    root = Path(run_root)
    p = load_protocol(root / "protocol")
    c = p["config"]
    runtime = runtime_for(device, workers)
    contract = run_contract(root / "protocol", p, arm, root / "public_init", runtime)
    seed_everything(c["seed"])
    model = build_model(arm, p["b42_config"], c, root / "public_init").to(runtime.device).train()
    groups = parameter_groups(model, arm, c)
    dataset = make_dataset(root / "protocol", p, "train")
    multiplier = torch.from_numpy(target_balance_multipliers(dataset.weights)).to(runtime.device)
    active = [i for i, w in enumerate(dataset.weights) if w.sum() > 0]
    # The two studies with the most eligible series exercise accumulation and
    # ragged bags. They do not prove every possible native matrix will fit.
    chosen = sorted(active, key=lambda i: (-len(dataset.series_records[dataset.study_uids[i]]),
                                         dataset.study_uids[i]))[:c["batch_size"]]
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    for i in chosen:
        out, loss = loss_for(model, arm, dataset[i], runtime, multiplier, c)
        (loss / len(chosen)).backward()
        del out, loss
    gradient_counts = {}
    for group in groups:
        grads = [p.grad for p in group["params"] if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"B57 invalid gradients: {group['name']}")
        count = sum(int(torch.count_nonzero(g) > 0) for g in grads)
        if count == 0:
            raise RuntimeError(f"B57 no learning signal in {group['name']}")
        gradient_counts[group["name"]] = count
    encoded = list(encoder_of(model, arm).parameters())
    if any(x.grad is None or not torch.count_nonzero(x.grad) for x in (encoded[0], encoded[-1])):
        raise RuntimeError("B57 gradient did not reach both ends of the encoder")
    result = {"passed": True, "contract": contract, "optimizer_steps": 0,
              "gradient_tensor_counts": gradient_counts,
              "study_uids": [dataset.study_uids[i] for i in chosen],
              "peak_cuda_gib": (torch.cuda.max_memory_allocated(runtime.device) / 2**30
                                if runtime.device.type == "cuda" else None)}
    write_json(root / f"{arm}_preflight.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def train_arm(*, run_root, arm, device="auto", workers=0):
    root, out = Path(run_root), Path(run_root) / arm
    p = load_protocol(root / "protocol")
    c = p["config"]
    runtime = runtime_for(device, workers)
    print(f"[B57/{arm}] {runtime.describe()}", flush=True)
    if runtime.device.type != "cuda" and device != "cpu":
        raise ValueError("no CUDA device; use explicit device=cpu only for small synthetic tests")
    contract = run_contract(root / "protocol", p, arm, root / "public_init", runtime)
    preflight_path = root / f"{arm}_preflight.json"
    if not preflight_path.exists():
        raise ValueError(f"run the published {arm} preflight first")
    check = json.loads(preflight_path.read_text())
    require_same_run(check["contract"], contract)
    if check.get("passed") is not True:
        raise ValueError("B57 preflight did not pass")
    if (out / "complete.json").exists():
        complete = json.loads((out / "complete.json").read_text())
        require_same_run(complete["contract"], contract)
        if complete["checkpoint_sha256"] != sha256_file(out / "final.pt"):
            raise ValueError("completed B57 checkpoint changed")
        print(f"[B57/{arm}] already COMPLETE", flush=True)
        return out / "final.pt"

    seed_everything(c["seed"])
    model = build_model(arm, p["b42_config"], c, root / "public_init").to(runtime.device).train()
    groups = parameter_groups(model, arm, c)
    optimizer = torch.optim.AdamW(groups, weight_decay=c["weight_decay"])
    # Relative cosine rates reach 1% at the fixed endpoint for both groups.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda e: .01 + .99 * (1 + np.cos(np.pi * min(e, c["epochs"]) / c["epochs"])) / 2)
    scaler = make_scaler(runtime)
    train_data = make_dataset(root / "protocol", p, "train")
    valid_data = make_dataset(root / "protocol", p, "validation")
    multiplier = torch.from_numpy(target_balance_multipliers(train_data.weights))
    if out.exists() and any(out.iterdir()) and not (out / "recovery_latest.pt").exists():
        raise FileExistsError(f"B57 refuses nonempty arm directory without recovery: {out}")
    saved = load_checkpoint(out)
    if saved is not None:
        require_same_run(saved["extra"]["contract"], contract)
        if not 0 < saved["epoch"] <= c["epochs"]:
            raise ValueError("invalid recovery epoch")
    state = resume(out, model=model, version=VERSION, optimizer=optimizer,
                   scheduler=scheduler, scaler=scaler)
    history = list(state.history)
    print(f"[B57/{arm}] {state.describe()}", flush=True)
    # Reset after differently sized model construction for the first epoch.
    if not state.restored:
        seed_everything(c["seed"] + 101)
    for epoch in range(state.start_epoch, c["epochs"] + 1):
        start = time.monotonic()
        loader = make_loader(train_data, runtime, c, epoch=epoch)
        model.train()
        loss_sum, seen = 0., []
        for batch, items in enumerate(loader, 1):
            loss, grad = batch_step(model, arm, items, runtime, optimizer, scaler, multiplier, c)
            loss_sum += loss
            seen.extend(x["study_uid"] for x in items)
            if batch % 100 == 0:
                print(f"[B57/{arm}] E{epoch} {batch}/{len(loader)} loss={loss_sum/batch:.5f}", flush=True)
        if len(seen) != len(set(seen)) or set(seen) != set(p["splits"]["train"]):
            raise RuntimeError("B57 epoch did not expose exactly its allowed training UIDs")
        del loader
        scheduler.step()
        predicted = predict(model, make_loader(valid_data, runtime, c), runtime)
        scores = macro_auc(predicted["target"], predicted["weight"], predicted["prediction"])
        if scores["targets_defined"] != 12 or not np.isfinite(scores["macro_auc"]):
            raise RuntimeError("B57 validation must define all 12 target AUCs")
        history.append({"epoch": epoch, "train_loss": loss_sum / batch, "validation": scores,
                        "training_uids_sha256": digest(sorted(seen)),
                        "epoch_minutes": (time.monotonic() - start) / 60,
                        "learning_rates": [float(g["lr"]) for g in groups]})
        save_checkpoint(out, epoch=epoch, model=model, version=VERSION, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, history=history, extra={"contract": contract})
        write_json(out / "history.json", history)
        print(f"[B57/{arm}] E{epoch} macroAUC={scores['macro_auc']:.6f}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    final = out / "final.pt"
    final_encoder_sha = state_digest(encoder_of(model, arm).state_dict())
    if final_encoder_sha == PUBLIC_STATE_SHA256[arm]:
        raise RuntimeError("B57 encoder stayed at public initialization; training did not update it")
    atomic_torch_save(final, {"version": VERSION, "experiment": c["experiment"],
                              "arm": arm, "contract": contract, "completed_epochs": c["epochs"],
                              "selection": "fixed_final_epoch", "model_state": model.state_dict(),
                              "encoder_tensor_sha256_final": final_encoder_sha,
                              "model_config": c, "b42_config": p["b42_config"], "history": history})
    checkpoint_sha = sha256_file(final)
    # Recompute even after recovery at the final epoch; no stale best-epoch
    # predictions can be paired with final weights.
    for split, dataset in (("validation", valid_data),
                           ("expert", make_dataset(root / "protocol", p, "expert"))):
        predicted = predict(model, make_loader(dataset, runtime, c), runtime)
        save_predictions(out / f"{split}.npz", predicted, contract=contract,
                         split=split, checkpoint_sha=checkpoint_sha)
    write_json(out / "complete.json", {"contract": contract, "checkpoint_sha256": checkpoint_sha,
                                       "completed_epochs": c["epochs"], "expert_role": "diagnostic only"})
    print(f"[B57/{arm}] COMPLETE {final}", flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "preflight", "train"))
    parser.add_argument("--run-root", default=DEFAULT_ROOT)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root")
    parser.add_argument("--labels-root")
    parser.add_argument("--domain-split")
    parser.add_argument("--series-policy")
    parser.add_argument("--b42-config", default="config/b42_constant_area_aspect_sparse.yaml")
    parser.add_argument("--b57-config", default="config/b57_clean_backbone_comparison.json")
    args = parser.parse_args()
    if args.stage == "prepare":
        for key in ("data_root", "labels_root", "domain_split", "series_policy"):
            if getattr(args, key) is None:
                parser.error(f"prepare requires --{key.replace('_', '-')}")
        p = prepare_protocol(data_root=args.data_root, labels_root=args.labels_root,
            domain_split=args.domain_split, series_policy=args.series_policy,
            b42_config=args.b42_config, b57_config=args.b57_config,
            out_root=Path(args.run_root) / "protocol")
        print(f"[B57] data boundary: {p['counts']}", flush=True)
        prepare_public_weights(Path(args.run_root) / "public_init")
        print("[B57] protocol and public initialization: COMPLETE", flush=True)
    else:
        if args.arm is None:
            parser.error("preflight/train requires --arm")
        fn = preflight if args.stage == "preflight" else train_arm
        fn(run_root=args.run_root, arm=args.arm, device=args.device, workers=args.num_workers)


if __name__ == "__main__":
    main()
