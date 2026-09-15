"""Importable subprocess entrypoint: notebook kernels never own the GPU model."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json

from .session import LocalRun, exclusive_run


def check_environment():
    import timm
    import torch
    import torchvision
    from pydicom.pixels import get_decoder
    if timm.__version__ != "1.0.20":
        raise RuntimeError("This recipe requires timm==1.0.20 in the notebook kernel.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select your working GPU Python kernel.")
    print(f"GPU: {torch.cuda.get_device_name(0)} | compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"PyTorch: {torch.__version__} | torchvision: {torchvision.__version__} | CUDA build: {torch.version.cuda}")
    # A version string or is_available() alone cannot establish Blackwell support.
    x = torch.randn(128, 128, device="cuda", requires_grad=True)
    (x @ x.T).square().mean().backward()
    torch.cuda.synchronize()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise RuntimeError("CUDA forward/backward check failed.")
    for name, uid in {"JPEG Lossless P14": "1.2.840.10008.1.2.4.57",
                      "JPEG Lossless SV1": "1.2.840.10008.1.2.4.70",
                      "JPEG2000 Lossless": "1.2.840.10008.1.2.4.90",
                      "JPEG2000": "1.2.840.10008.1.2.4.91"}.items():
        decoder = get_decoder(uid)
        if not decoder.is_available:
            raise RuntimeError(f"Missing DICOM decoder for {name}: {decoder.missing_dependencies}")
        print(f"{name}: available ({', '.join(decoder.available_plugins)})")
    print("CUDA arithmetic and DICOM decoder availability: PASS", flush=True)


def prepare_dino_weights(root):
    """Same verified authors' tensors; prepare only the requested encoder."""
    import torch
    import torchvision
    import timm
    from ..b57_models import DINO_URL, PUBLIC_STATE_SHA256, dino_backbone, public_weights, state_digest
    from ..b57_protocol import ARMS, VERSION, sha256_file, write_json
    if timm.__version__ != "1.0.20":
        raise ValueError("This recipe requires timm==1.0.20.")
    arm = ARMS[1]
    root.mkdir(parents=True, exist_ok=True)
    if (root / f"{arm}.json").exists():
        public_weights(root, arm)
        print("Public DINOv2 initialization: verified", flush=True)
        return
    path = root / f"{arm}.pt"
    if path.exists():
        raise FileExistsError(f"Unmanifested encoder file; inspect it before retrying: {path}")
    print("Loading the authors' public DINOv2 weights (first use may download).", flush=True)
    encoder = dino_backbone(pretrained=True)
    fingerprint = state_digest(encoder.state_dict())
    if fingerprint != PUBLIC_STATE_SHA256[arm]:
        raise ValueError("Downloaded tensors differ from the pinned public DINOv2 weights.")
    temporary = path.with_suffix(".writing")
    torch.save(encoder.state_dict(), temporary)
    temporary.replace(path)
    write_json(root / f"{arm}.json", {"version": VERSION, "arm": arm,
        "sha256": sha256_file(path), "tensor_sha256": fingerprint, "source_url": DINO_URL,
        "competition_training_studies": [], "fresh_competition_heads": True,
        "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__})
    public_weights(root, arm)
    print("Public DINOv2 initialization: verified", flush=True)


def run_stage(run, stage):
    from ..dicom import _iter_dicom_files, find_series_dir
    from ..b57_protocol import ARMS, prepare_protocol
    from ..b57_training import preflight, train_arm
    with exclusive_run(run.run_root):
        run.claim()
        if stage == "environment":
            check_environment()
        elif stage == "prepare":
            p = prepare_protocol(data_root=run.data_root, labels_root=run.labels_root,
                domain_split=run.scanner_split_root, series_policy=run.series_policy,
                b42_config=run.geometry_config, b57_config=run.training_config,
                out_root=run.run_root / "protocol")
            index = json.loads((run.run_root / "protocol/series_index.json").read_text())
            # Check every series directory required for training and reporting;
            # a full CSV beside only a small image subset must not pass.
            for uid in p["splits"]["train"] + p["splits"]["validation"] + p["expert_uids"]:
                for record in index[uid]:
                    path = find_series_dir(run.data_root, "train", uid, record["series_uid"])
                    if path is None or not _iter_dicom_files(path):
                        raise FileNotFoundError(f"Required MRI series is missing or empty: {uid}/{record['series_uid']}")
            print(f"Data boundary: {p['counts']}", flush=True)
            prepare_dino_weights(run.run_root / "public_init")
            print("Frozen data, labels, model recipe and initialization: READY", flush=True)
        elif stage == "preflight":
            print("Checking two real training studies and encoder/head gradients; no optimizer update.", flush=True)
            log = run.run_root / "logs/preflight_details.log"
            log.parent.mkdir(exist_ok=True)
            with log.open("a") as stream, redirect_stdout(stream):
                result = preflight(run_root=run.run_root, arm=ARMS[1], device="cuda", workers=run.num_workers)
            print(json.dumps({k: result[k] for k in ("passed", "optimizer_steps", "gradient_tensor_counts", "peak_cuda_gib")}, indent=2), flush=True)
        elif stage == "train":
            train_arm(run_root=run.run_root, arm=ARMS[1], device="cuda", workers=run.num_workers)
        else:
            raise ValueError(f"Unknown notebook stage: {stage}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("environment", "prepare", "preflight", "train"))
    for key in ("project-root", "data-root", "run-root", "labels-root", "scanner-split-root", "series-policy"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    args = vars(parser.parse_args())
    stage = args.pop("stage")
    run_stage(LocalRun(**args), stage)


if __name__ == "__main__":
    main()
