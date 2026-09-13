"""Published B58 preparation, preflight, train and comparison entrypoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from . import DEFAULT_ROOT


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "preflight", "pretrain", "finetune", "run", "compare"))
    p.add_argument("--run-root", default=os.environ.get("B58_RUN_ROOT", DEFAULT_ROOT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-root", default=os.environ.get("B58_DATA_ROOT", "rsna-knee-abnormality-detection"))
    p.add_argument("--labels-root", default=os.environ.get("B58_LABELS_ROOT",
        "runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all"))
    p.add_argument("--series-policy", default=os.environ.get("B58_SERIES_POLICY",
        "runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json"))
    p.add_argument("--selection-root", default=os.environ.get("B58_SELECTION_ROOT"))
    p.add_argument("--mrnet-root", default=os.environ.get("B58_MRNET_ROOT", "data/external/mrnet"))
    p.add_argument("--fastmri-root", default=os.environ.get("B58_FASTMRI_ROOT", "data/external/fastmri"))
    p.add_argument("--oai-root", default=os.environ.get("B58_OAI_ROOT", "data/external/oai"))
    p.add_argument("--reference-root", default=os.environ.get("B58_REFERENCE_ROOT"))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    # `run` starts a new process for each stage, releasing its CUDA allocation.
    def child(stage, extra=()):
        subprocess.run([sys.executable, "-m", "rsna_knee.b58_external_knee", stage,
            "--run-root", str(args.run_root), "--device", args.device,
            "--num-workers", str(args.num_workers), *extra], check=True)

    if args.stage == "prepare":
        from .protocol import prepare
        gate = args.selection_root
        if gate is None:
            gates = sorted({str(p.resolve().parent) for p in Path("runs").rglob("b50_selection_split.json")})
            if len(gates) != 1:
                raise ValueError(f"set B58_SELECTION_ROOT to the existing B50 gate; found {gates}")
            gate = gates[0]
        prepare(run_root=args.run_root, data_root=args.data_root, labels_root=args.labels_root,
            domain_split=gate, series_policy=args.series_policy, mrnet_root=args.mrnet_root,
            fastmri_root=args.fastmri_root, oai_root=args.oai_root)
    elif args.stage == "preflight":
        from . import ssl, training
        ssl.preflight(args.run_root, device=args.device)
        # Return from SSL preflight frees the live models; cached allocator is
        # explicitly released before the much larger full-study backward pass.
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        training.preflight(args.run_root, device=args.device, workers=args.num_workers, adapted=False)
    elif args.stage == "pretrain":
        from .ssl import train
        train(args.run_root, device=args.device)
    elif args.stage == "finetune":
        from . import training
        training.preflight(args.run_root, device=args.device, workers=args.num_workers, adapted=True)
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        training.train(args.run_root, device=args.device, workers=args.num_workers)
    elif args.stage == "run":
        if not (Path(args.run_root)/"protocol.json").exists():
            raise FileNotFoundError("run B58 prepare after obtaining all three external datasets")
        child("preflight")
        child("pretrain")
        child("finetune")
    else:
        if not args.reference_root:
            raise ValueError("compare requires --reference-root pointing to the completed B57 export")
        from .evaluation import evaluate
        evaluate(args.run_root, args.reference_root)


if __name__ == "__main__":
    main()
