"""Full-data B58: inspect OAI joins, prepare, preflight, run, resume and compare."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

from . import DEFAULT_ROOT


def parser():
    from ..b58_external_knee.__main__ import parser as old_parser
    p = old_parser()
    p.description = __doc__
    next(a for a in p._actions if a.dest == "stage").choices = (
        "inventory-oai", "prepare", "preflight", "ssl-preflight", "native-preflight",
        "pretrain", "supervised", "finetune", "run", "compare")
    p.set_defaults(run_root=os.environ.get("B58_FULL_RUN_ROOT", DEFAULT_ROOT))
    for flag, env, default in (
        ("fastmri-annotations", "B58_FASTMRI_ANNOTATIONS", "data/external/fastmri-plus/Annotations/knee.csv"),
        ("fastmri-reviewed", "B58_FASTMRI_REVIEWED", "data/external/fastmri-plus/Annotations/knee_file_list.csv"),
        ("oai-series", "B58_OAI_SERIES", "data/external/oai_labels/series.csv"),
        ("oai-labels", "B58_OAI_LABELS", "data/external/oai_labels/labels.csv"),
        ("oai-tasks", "B58_OAI_TASKS", "data/external/oai_labels/tasks.json")):
        p.add_argument("--" + flag, default=os.environ.get(env, default))
    p.add_argument("--config", default="config/b58_full_data.json")
    return p


def inventory_oai(root, output):
    """Export actual identities to help join official release metadata and labels.

    Unknown visit/side/partition are blank and must be joined explicitly.
    """
    import pydicom
    from ..b58_external_knee.data import index_oai
    rows = index_oai(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as handle:
        fields = ["series_uid", "patient_id", "visit", "side", "partition", "study_uid",
                  "dicom_study_date", "dicom_laterality", "series_description", "example_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            ds = pydicom.dcmread(row["paths"][0], stop_before_pixels=True)
            writer.writerow({"series_uid": row["series"], "patient_id": row["group"], "study_uid": row["study_uid"],
                "visit": "", "side": "", "partition": "", "dicom_study_date": str(getattr(ds, "StudyDate", "")),
                "dicom_laterality": str(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", ""))),
                "series_description": str(getattr(ds, "SeriesDescription", "")), "example_path": row["paths"][0]})
    print(f"[B58 full] OAI identity inventory: {output}; join release visit/side/partition before prepare", flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(args.run_root)

    def child(stage):
        subprocess.run([sys.executable, "-m", "rsna_knee.b58_full_data", stage,
            "--run-root", str(root), "--device", args.device, "--num-workers", str(args.num_workers)], check=True)

    if args.stage == "inventory-oai":
        inventory_oai(args.oai_root, args.oai_series)
    elif args.stage == "prepare":
        from .protocol import prepare
        gate = args.selection_root
        if gate is None:
            gates = sorted({str(p.resolve().parent) for p in Path("runs").rglob("b50_selection_split.json")})
            if len(gates) != 1:
                raise ValueError(f"set B58_SELECTION_ROOT to the frozen B50 gate; found {gates}")
            gate = gates[0]
        prepare(run_root=root, data_root=args.data_root, labels_root=args.labels_root,
                domain_split=gate, series_policy=args.series_policy, mrnet_root=args.mrnet_root,
                fastmri_root=args.fastmri_root, oai_root=args.oai_root,
                fastmri_annotations=args.fastmri_annotations, fastmri_reviewed=args.fastmri_reviewed,
                oai_series=args.oai_series, oai_labels_csv=args.oai_labels, oai_tasks=args.oai_tasks,
                config_path=args.config)
    elif args.stage == "preflight":
        child("ssl-preflight")
        # Architecture + every native label source, using public initialization.
        # A second preflight with the adapted encoder runs before supervised.
        from .training import preflight
        preflight(root, "supervised", device=args.device, adapted=False)
    elif args.stage in ("ssl-preflight", "native-preflight"):
        from .training import preflight
        preflight(root, "pretrain" if args.stage == "ssl-preflight" else "supervised", device=args.device)
    elif args.stage in ("pretrain", "supervised"):
        from .training import train
        train(root, args.stage, device=args.device)
    elif args.stage == "finetune":
        from . import finetune
        finetune.preflight(root, device=args.device, workers=args.num_workers, adapted=True)
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        finetune.train(root, device=args.device, workers=args.num_workers)
    elif args.stage == "run":
        if not (root / "protocol.json").exists():
            raise FileNotFoundError("prepare the full-data protocol and label joins first")
        for stage in ("preflight", "pretrain", "native-preflight", "supervised", "finetune"):
            child(stage)
    else:
        if not args.reference_root:
            raise ValueError("compare requires the completed B57 package-run export")
        from .evaluation import evaluate
        evaluate(root, args.reference_root)


if __name__ == "__main__":
    main()
