"""Build the portable notebook. Extraction happens here, never at notebook runtime."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "developments/src/rsna_knee"
CELLS = []


def markdown(source):
    CELLS.append(("markdown", dedent(source).strip(), []))


def code(source, tag="implementation"):
    CELLS.append(("code", dedent(source).strip(), [tag]))


def extract(module, names, replacements=None):
    """Inline selected definitions as ordinary visible Python, without imports."""
    tree = ast.parse((SOURCE / f"{module}.py").read_text())
    selected = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    class RemoveLocalImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None if node.level else node
    definitions = []
    for name in names:
        node = RemoveLocalImports().visit(selected[name])
        docstrings = {
            "partition": "Exclude held scanner profiles from every report-only training row.",
            "verify_b50_selection_split": "Verify the frozen scanner assignments and their parent boundary.",
            "b35_centers": "Return 32 centers: the fixed 16-center grid followed by 16 additional centers.",
            "DinoSliceTransformer": "Encode triplets, pool slices within each series, and attend across series.",
            "prepare_all_report_only_supervision": "Build soft targets and weights, retaining zero-weight studies.",
        }
        if name in docstrings and ast.get_docstring(node):
            node.body[0] = ast.Expr(value=ast.Constant(value=docstrings[name]))
        text = ast.unparse(ast.fix_missing_locations(node))
        for old, new in (replacements or {}).items():
            text = text.replace(old, new)
        text = re.sub(r"\bB\d+(?:\.\d+)?(?:-v1)?\b", "established recipe", text)
        text = text.replace("Phase 9", "Report-label")
        definitions.append(text)
    return "\n\n\n".join(definitions)


markdown(r'''
# Knee MRI diagnosis with DINOv2

**Standalone notebook · full original dataset · one NVIDIA RTX 5090**

Everything specific to this training pipeline is defined in the cells below:
input checks, DICOM reading, preprocessing, the dataset, the complete study
model, loss, optimization, checkpoint recovery, evaluation and plots. You can
copy this single notebook elsewhere and run it without the project package,
its Python files, shell scripts or configuration files.

The notebook uses standard scientific Python libraries. The public DINOv2
ViT-S/14 image backbone comes from **timm**, just as a convolution layer comes
from PyTorch; all knee-specific model classes and functions are visible below.
The first use downloads the authors' pretrained weights and verifies their
tensor fingerprint. These weights are general-image pretraining, not a knee
MRI model.

You still need your original MRI files, report-label export, frozen scanner
split and series-policy JSON. These are data inputs, not executable code. The
notebook preserves their existing train/validation boundary and trains all
permitted studies. Validation and expert studies never produce gradients.

Read the explanation, run each definition cell, then run the workflow cells at
the end. This starts a separate twelve-epoch run in its own output directory.
''')
markdown(r'''
## 1. Environment and paths

Use the Python environment whose PyTorch and torchvision already work on the
5090. If needed, install the following packages in that environment **before**
starting Jupyter (there is no editable project installation):

    python -m pip install "timm==1.0.20" "numpy>=1.26" "pandas>=2" "scikit-learn>=1.3" "matplotlib>=3.8" "pydicom>=3" "python-gdcm>=3.0.10" "jupyterlab>=4,<5" "tornado==6.5.8" "ipykernel>=6"

The temporary Tornado pin supports the local Jupyter static-file handler.
Launch Jupyter bound to 127.0.0.1 and keep its normal authentication enabled.
The GPU check below performs real arithmetic and backpropagation.

**WORK_ROOT** is just a directory containing your data/artifacts and output;
it does not need to contain any project source. The optional paths can be set
explicitly. With the defaults, discovery looks only inside WORK_ROOT/runs,
first for input paths recorded in an existing frozen data protocol, then for
matching data files. Ambiguous matches stop and ask for an explicit path.

For portable Jupyter execution, the DataLoader uses **zero subprocess
workers**. Notebook-defined classes therefore need no exported worker module.
This can reduce data-loading throughput; it does not change the sampling,
model, loss, batch size, or split. GPU micro-batches remain two triplets and
optimizer batches remain two studies.
''')
code(r'''
from pathlib import Path

WORK_ROOT = Path("/media/talafha/Disk_1/CNN_CPC")
DATA_ROOT = WORK_ROOT / "rsna-knee-abnormality-detection"
RUN_ROOT = WORK_ROOT / "runs/dinov2_knee_mri_standalone"

# Set these to existing data artifacts if automatic discovery is ambiguous.
LABELS_ROOT = None
SCANNER_SPLIT_ROOT = None
SERIES_POLICY = None
DEVICE = "cuda:0"
''', "settings")
code(r'''
from __future__ import annotations

import ast
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
import gc
import hashlib
import inspect
import json
import math
import os
import random
import sys
import time
from textwrap import dedent
from typing import Any, Iterable, Sequence
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import pydicom
from sklearn.metrics import roc_auc_score
import timm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
import torchvision
from IPython.display import display

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
N_TARGETS = len(TARGETS)
FORMAT_VERSION = "standalone_dinov2_knee_v1"
DINO_MODEL = "vit_small_patch14_dinov2.lvd142m"
DINO_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
PUBLIC_TENSOR_SHA256 = "da01dc066b2a6aea10165a09242353c4c9178032d76f9a5494e880cf5864dd92"
SERIES_SIGNATURE = "5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376"

RECIPE = dict(seed=2026, epochs=12, batch_size=2, encoder_lr=1e-5, head_lr=1e-4,
              weight_decay=1e-4, grad_clip=1.0, encoder_chunk_size=2,
              gradient_checkpointing=True, dim=384, heads=6, dropout=0.1,
              slices_per_series=32, base_slices=16, triplet_gap=1, crop_fraction=0.90,
              reference_area=448**2, alignment=32, augmentation=False,
              slice_layers=1, slice_position_embedding=False, auxiliary_loss=0.0,
              selection="fixed_final_epoch", tta=False)
LABEL_RULE = dict(min_confidence=0.75, positive_target=0.85, negative_target=0.05,
                  positive_weight=0.50, negative_weight=1.00)
EXPECTED_POPULATION = dict(report_only=4349, usable_cells=34010, train=3603,
                           validation=548, excluded_profile_overlap=198, expert=58)
VALIDATION_NAME = "validation_unseen_scanners"
''')
markdown(r'''
## 2. File integrity, run ownership and environment checks

A run records input hashes, the actual notebook function definitions, the
recipe, library versions and public initialization. Resume requires the same
contract. A changed definition or input stops resume instead of mixing runs.
Checkpoint writes are atomic. On Linux, an exclusive file lock prevents two
notebooks from writing the same run simultaneously.

An interrupted epoch is repeated from the last completed epoch. The notebook
uses its own checkpoint format; choose its dedicated output folder rather
than an older training folder. Ordinary kernel interruption releases the GPU
model in a finally block. After a forced kernel shutdown, restart the kernel
and rerun the cells with the same settings.
''')
code(extract("b57_protocol", ["sha256_file", "digest", "write_json", "coverage", "require_same_run"]))
code(r'''
def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def notebook_code_digest():
    """Hash executed definitions, independent of notebook name/cell line numbers."""
    definitions = {}
    for name in IMPLEMENTATION_NAMES:
        obj = globals()[name]
        if inspect.isclass(obj):
            methods = {}
            for key, member in vars(obj).items():
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if inspect.isfunction(member):
                    member = inspect.unwrap(member)
                    # Dataclass-generated methods have no user-written source.
                    if member.__code__.co_filename == "<string>":
                        continue
                    methods[key] = ast.dump(ast.parse(dedent(inspect.getsource(member))))
            definitions[name] = {"methods": methods,
                "annotations": {k: str(v) for k, v in getattr(obj, "__annotations__", {}).items()}}
        else:
            definitions[name] = ast.dump(ast.parse(dedent(inspect.getsource(inspect.unwrap(obj)))))
    return digest(definitions)


def implementation_contract():
    expected = dict(slices_per_series=32, base_slices=16, triplet_gap=1, crop_fraction=0.90,
                    reference_area=448**2, alignment=32, augmentation=False,
                    slice_layers=1, slice_position_embedding=False, auxiliary_loss=0.0,
                    selection="fixed_final_epoch", tta=False)
    if any(RECIPE.get(k) != v for k, v in expected.items()):
        raise ValueError("The declared geometry/context must match the implemented fixed recipe.")
    return {"version": FORMAT_VERSION, "code_sha256": notebook_code_digest(),
            "recipe": RECIPE, "labels": LABEL_RULE, "targets": TARGETS,
            "expected_population": EXPECTED_POPULATION,
            "public_tensor_sha256": PUBLIC_TENSOR_SHA256,
            "series_signature": SERIES_SIGNATURE,
            "dicom_suffixes": sorted(DICOM_SUFFIXES), "planes": list(PLANES),
            "plane_ids": PLANE_TO_ID, "true_flags": sorted(TRUE_TOKENS),
            "false_flags": sorted(FALSE_TOKENS), "allowed_splits": sorted(ALLOWED_SPLITS),
            "validation_assignment": VALIDATION_NAME, "backbone": DINO_MODEL, "public_url": DINO_URL}


@contextmanager
def exclusive_run(root):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if os.name != "posix":
        raise RuntimeError("This local training notebook uses Linux/POSIX file locking.")
    import fcntl
    with (root / ".run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is writing this run folder.") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def claim_run(root):
    root = Path(root)
    manifest = root / "notebook_run.json"
    contract = implementation_contract()
    if manifest.exists():
        require_same_run(json.loads(manifest.read_text()), contract)
    else:
        occupied = [p for p in root.iterdir() if p.name != ".run.lock"]
        if occupied:
            raise FileExistsError("Choose an empty output directory; this one contains another run.")
        write_json(manifest, contract)


def runtime_for(device):
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device. Select the environment with working 5090 PyTorch.")
        torch.cuda.set_device(device)
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    elif device.type == "cpu":
        amp_dtype = None  # Only practical for the small synthetic verification suite.
    else:
        raise ValueError("Use a CUDA device for full-data training.")
    return device, amp_dtype


def autocast(device, amp_dtype):
    return torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32,
                          enabled=amp_dtype is not None)


def environment_check(device):
    if timm.__version__ != "1.0.20":
        raise RuntimeError("Install timm==1.0.20 before running this recipe.")
    device, amp_dtype = runtime_for(device)
    with torch.enable_grad():
        x = torch.ones((32, 32), device=device, requires_grad=True)
        with autocast(device, amp_dtype):
            loss = (x @ x).square().mean()
        loss.backward()
        if x.grad is None or not torch.isfinite(x.grad).all():
            raise RuntimeError("GPU arithmetic/backpropagation failed.")
    del x, loss
    from pydicom.pixels import get_decoder
    for name, uid in {"JPEG Lossless": "1.2.840.10008.1.2.4.57",
                      "JPEG Lossless SV1": "1.2.840.10008.1.2.4.70",
                      "JPEG2000 Lossless": "1.2.840.10008.1.2.4.90",
                      "JPEG2000": "1.2.840.10008.1.2.4.91"}.items():
        decoder = get_decoder(uid)
        if not decoder.is_available:
            raise RuntimeError(f"Missing {name} decoder: {decoder.missing_dependencies}")
    print("Python:", sys.executable)
    print("Device:", torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
    print("Precision:", amp_dtype or torch.float32)
    print("CUDA arithmetic and DICOM decoder checks: PASS")
''')
markdown(r'''
## 3. DICOM loading and scan metadata

Each study supplies all eligible sagittal, coronal and axial series. Known
metadata comes from the series CSV. Missing values are repaired from DICOM
orientation and acquisition timing using the existing deterministic rules.

Pixels are ordered by physical slice position when available, with an
instance-number fallback. Rescale slope/intercept and MONOCHROME1 inversion
are applied. For compatibility with the original loader, individual unreadable
files may be skipped; an entirely unreadable or missing series raises an error.
This is not a file-by-file completeness audit of every scan.
''')
code(r'''
DICOM_SUFFIXES = {"", ".dcm", ".dicom", ".ima"}
PLANES = ("Sagittal", "Coronal", "Axial")
PLANE_TO_ID = {"Sagittal": 1, "Coronal": 2, "Axial": 3}
TRUE_TOKENS = {"true", "t", "yes", "y", "1", "1.0"}
FALSE_TOKENS = {"false", "f", "no", "n", "0", "0.0"}
''')
code(extract("dicom", ["_sort_key", "find_series_dir", "_iter_dicom_files", "_pixel_spacing",
                       "_pad_or_crop", "read_dicom_series", "_normalise_volume", "_centers"]))
code(extract("dicom_meta", ["plane_from_orientation", "weighting_from_parameters",
                            "is_fluid_sensitive", "_multiframe_orientation", "read_series_metadata"]))
code(extract("data", ["_require_columns", "load_train_csv", "gold_mask", "coerce_bool",
                      "normalise_plane", "load_series_csv", "backfill_series_metadata"]))
code(extract("b12_variable_series", ["_flag_id", "build_variable_series_index"]))
markdown(r'''
## 4. Labels and the frozen scanner boundary

Report states with confidence at least **0.75** supervise training. Positive
states use target **0.85** and weight **0.5**; negated states use target **0.05**
and weight **1.0**. Missing or uncertain cells have weight zero. These soft
targets describe the established training recipe, not calibrated probabilities.

The original report-only population is 4,349 studies with 34,010 supervised
cells. The scanner split yields 3,603 training studies, 548 validation studies
and 198 additional excluded studies whose scanner profiles overlap validation.
The 58 expert studies are a diagnostic evaluation set only.

Scanner separation is enforced across the complete training population.
Patient identities are unavailable in this contract, so this is not a claim
of verified patient separation. The reused validation surface is a development
measure, not an untouched test or competition score.
''')
code(extract("phase9_supervision", ["load_fill_merged_export", "prepare_all_report_only_supervision"], {
    "REPORT_ONLY_STUDIES": "EXPECTED_POPULATION['report_only']",
    "B7_MIN_CONFIDENCE": "LABEL_RULE['min_confidence']",
    "B7_POSITIVE_TARGET": "LABEL_RULE['positive_target']",
    "B7_NEGATIVE_TARGET": "LABEL_RULE['negative_target']",
    "B7_POSITIVE_WEIGHT": "LABEL_RULE['positive_weight']",
    "B7_NEGATIVE_WEIGHT": "LABEL_RULE['negative_weight']",
}))
code(extract("b50_ordered_slice_selection_split", ["verify_b50_selection_split"], {
    "verify_b50_selection_split": "verify_selection_split", "b50_split": "selection_split",
    "parent_b48_split": "parent_split", "B50_ALLOWED_SPLITS": "ALLOWED_SPLITS",
    "B50_PARENT_TRAIN_SPLIT": "'train'", "B50_SPLIT_EXCLUDED": "'excluded_prior_surface'",
    "B50_SPLIT_TRAIN": "'train'", "B50_SPLIT_SEEN": "'validation_seen_scanners'",
    "B50_SPLIT_UNSEEN": "VALIDATION_NAME", "b50 =": "assignment =",
    "b50.loc": "assignment.loc", "b50.eq": "assignment.eq", "B48/B49": "earlier",
}))
code(extract("b57_protocol", ["partition"], {"b50_split": "selection_split", "VALID)": "VALIDATION_NAME)"}))
code(r'''
ALLOWED_SPLITS = {"train", "validation_seen_scanners", "validation_unseen_scanners", "excluded_prior_surface"}


def only_match(paths, label):
    paths = sorted({Path(p).expanduser().resolve() for p in paths})
    if len(paths) != 1:
        choices = "\n".join(str(p) for p in paths[:10])
        raise FileNotFoundError(f"Set {label} explicitly: found {len(paths)} matching inputs.\n{choices}")
    return paths[0]


def load_json(path):
    return json.loads(Path(path).read_text())


def discover_inputs(work_root, labels_root=None, scanner_split_root=None, series_policy=None):
    search = Path(work_root) / "runs"
    recorded = {"training_targets": set(), "gate_json": set(), "series_policy": set()}
    for path in search.glob("**/protocol.json"):
        try:
            inputs = load_json(path)["inputs"]
            for key in recorded:
                item = inputs.get(key, {})
                candidate = Path(item.get("path", ""))
                if candidate.is_file() and sha256_file(candidate) == item.get("sha256"):
                    recorded[key].add(candidate.resolve())
        except (KeyError, OSError, ValueError):
            continue
    if labels_root is None:
        candidates = recorded["training_targets"]
        if not candidates:
            candidates = []
            for path in search.glob("**/training_targets.csv"):
                try:
                    audit = load_json(path.parent / "audit.json")
                    if (audit.get("base_cells_overridden") == 0
                            and audit.get("gold_rows_in_training_targets") == 0):
                        candidates.append(path)
                except (OSError, ValueError):
                    continue
        labels_root = only_match(candidates, "LABELS_ROOT").parent
    if scanner_split_root is None:
        candidates = recorded["gate_json"] or list(search.glob("**/*selection_split.json"))
        scanner_split_root = only_match(candidates, "SCANNER_SPLIT_ROOT").parent
    if series_policy is None:
        candidates = recorded["series_policy"]
        if not candidates:
            candidates = []
            for path in search.glob("**/series_policy.json"):
                try:
                    if load_json(path).get("series_summary", {}).get("series_signature_sha256") == SERIES_SIGNATURE:
                        candidates.append(path)
                except (OSError, ValueError):
                    continue
        series_policy = only_match(candidates, "SERIES_POLICY")
    return Path(labels_root).resolve(), Path(scanner_split_root).resolve(), Path(series_policy).resolve()


def canonical_split_rows(rows):
    """Accept historical column names as data, without hard-coding experiment labels."""
    rows = rows.copy()
    assignments = [c for c in rows if c.endswith("split") and not c.startswith("parent_")]
    parents = [c for c in rows if c.startswith("parent_") and c.endswith("split")]
    if len(assignments) != 1 or len(parents) != 1:
        raise ValueError("Expected one selection assignment and one parent assignment column.")
    rows = rows.rename(columns={assignments[0]: "selection_split", parents[0]: "parent_split"})
    needed = ["StudyInstanceUID", "scanner_profile", "selection_split", "parent_split"]
    _require_columns(rows, set(needed), "frozen split")
    if rows[needed].isna().any().any():
        raise ValueError("The frozen split contains missing assignments or identifiers.")
    verify_selection_split(rows)
    return rows


def validate_series_policy(path):
    policy = load_json(path)
    if (policy.get("policy") != "all_repaired_anatomical_series_v1"
            or policy.get("uses_gold_labels") is not False
            or policy.get("viability_passed") is not True
            or policy.get("series_summary", {}).get("series_signature_sha256") != SERIES_SIGNATURE):
        raise ValueError("The original label-free all-series policy is required.")
    for suffix, value in (("_active_studies", 3120), ("_usable_cells", 14123)):
        matches = [v for k, v in policy.items() if k.endswith(suffix)]
        if matches != [value]:
            raise ValueError("The series-policy population does not match the established input.")
    return policy


def prepare_data(*, work_root, data_root, run_root, labels_root=None,
                 scanner_split_root=None, series_policy=None):
    data_root, run_root = Path(data_root).resolve(), Path(run_root).resolve()
    if run_root in {data_root, Path(work_root).resolve(), Path(work_root).resolve() / "runs"}:
        raise ValueError("Use a dedicated run directory.")
    labels_root, gate, policy_path = discover_inputs(work_root, labels_root, scanner_split_root, series_policy)
    if gate.is_file():
        gate = gate.parent
    gate_json = only_match(gate.glob("*selection_split.json"), "SCANNER_SPLIT_ROOT")
    stem = gate_json.stem
    paths = dict(train_csv=data_root / "train.csv", train_series_csv=data_root / "train_series.csv",
                 training_targets=labels_root / "training_targets.csv", label_policy=labels_root / "policy.json",
                 label_audit=labels_root / "audit.json", series_policy=policy_path, gate_json=gate_json,
                 gate_rows=gate / f"{stem}_by_study.csv", gate_hash=gate / f"{stem}.sha256")
    inputs = {k: dict(path=str(p), sha256=sha256_file(p)) for k, p in paths.items()}
    out = run_root / "protocol"
    with exclusive_run(run_root):
        claim_run(run_root)
        if (out / "protocol.json").exists():
            protocol = load_protocol(run_root)
            if protocol["inputs"] != inputs or protocol["data_root"] != str(data_root):
                raise ValueError("Frozen data inputs changed; use the original inputs for resume.")
            return protocol
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Incomplete protocol exists. Inspect it before choosing a new run folder.")
        gate_payload = load_json(gate_json)
        if paths["gate_hash"].read_text().split()[0] != inputs["gate_json"]["sha256"]:
            raise ValueError("Scanner gate JSON/hash mismatch.")
        for source_key, input_key in (("source_train_csv_sha256", "train_csv"),
                                      ("source_training_targets_sha256", "training_targets")):
            if gate_payload.get(source_key) != inputs[input_key]["sha256"]:
                raise ValueError(f"Original gate/input mismatch: {source_key}")
        rows = canonical_split_rows(pd.read_csv(paths["gate_rows"], dtype={"StudyInstanceUID": str}))
        actual = set(rows.loc[rows.selection_split.eq(VALIDATION_NAME), "scanner_profile"].astype(str))
        if actual != set(gate_payload["unseen_scanner_profiles"]):
            raise ValueError("Gate JSON and CSV validation scanner profiles disagree.")
        train = load_train_csv(paths["train_csv"])
        frame, _, _ = load_fill_merged_export(labels_root)
        uids, target, weight, summary = prepare_all_report_only_supervision(train, frame)
        if summary["usable_cells"] != EXPECTED_POPULATION["usable_cells"]:
            raise ValueError("The original report-label export is required; supervised-cell count changed.")
        experts = train.loc[gold_mask(train)].copy()
        expert_uids = experts.StudyInstanceUID.tolist()
        indices, profiles = partition(uids, rows, expert_uids)
        splits = {name: [uids[int(i)] for i in ix] for name, ix in indices.items()}
        counts = {k: len(v) for k, v in splits.items()}
        if any(counts[k] != EXPECTED_POPULATION[k] for k in counts) or len(experts) != EXPECTED_POPULATION["expert"]:
            raise ValueError(f"Original full-data population changed: {counts}, expert={len(experts)}")
        audits = {name: coverage(target[ix], weight[ix]) for name, ix in indices.items()}
        for split in ("train", "validation"):
            if any(min(c["positive"], c["negative"]) == 0 for c in audits[split].values()):
                raise ValueError(f"Cannot measure all twelve targets in {split}.")
        validate_series_policy(policy_path)
        series = load_series_csv(paths["train_series_csv"])
        series, repair = backfill_series_metadata(series, data_root)
        index = build_variable_series_index(series, uids + expert_uids)
        for uid in uids + expert_uids:
            if not index[uid]:
                raise ValueError(f"No eligible MRI series for {uid}")
        for uid in splits["train"] + splits["validation"] + expert_uids:
            for record in index[uid]:
                directory = find_series_dir(data_root, "train", uid, record["series_uid"])
                if directory is None or not _iter_dicom_files(directory):
                    raise FileNotFoundError(f"Missing DICOM input: {uid}/{record['series_uid']}")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "series_index.json", index)
        atomic_npz(out / "labels.npz", uids=np.asarray(uids), target=target, weight=weight,
                   expert_uids=np.asarray(expert_uids), expert_target=experts[TARGETS].to_numpy(np.float32))
        protocol = dict(implementation=implementation_contract(), data_root=str(data_root),
                        inputs=inputs, splits=splits, counts=counts, expert_uids=expert_uids,
                        split_uid_hashes={k: digest(v) for k, v in splits.items()},
                        scanner_profiles=dict(zip(uids, profiles.tolist())), coverage=audits,
                        metadata_repair=repair, competition_supervised_ancestor=False, gold_studies_in_gradient=0,
                        validation_role="reused development surface", expert_role="diagnostic only",
                        labels_sha256=sha256_file(out / "labels.npz"),
                        series_index_sha256=sha256_file(out / "series_index.json"))
        write_json(out / "protocol.json", protocol)
        (out / "protocol.sha256").write_text(sha256_file(out / "protocol.json") + "\n")
    return protocol


def load_protocol(run_root):
    root = Path(run_root) / "protocol"
    if sha256_file(root / "protocol.json") != (root / "protocol.sha256").read_text().strip():
        raise ValueError("Protocol hash mismatch.")
    p = load_json(root / "protocol.json")
    require_same_run(p["implementation"], implementation_contract())
    for name, key in (("labels.npz", "labels_sha256"), ("series_index.json", "series_index_sha256")):
        if sha256_file(root / name) != p[key]:
            raise ValueError(f"Frozen artifact changed: {name}")
    for key, item in p["inputs"].items():
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError(f"Source input changed: {key}")
    return p
''')
markdown(r'''
## 5. From a volume to 32 triplets

Intensity is normalized using the whole volume's 1st and 99th percentiles.
Sixteen base centers and sixteen additional centers sample the series.
Short volumes may repeat centers. Each input contains the preceding, center
and following slice; boundary indices are clamped.

A central **90%** crop is resized with one isotropic scale to approximately
**448² pixels**, then minimally reflection-padded to stride 32. Rectangular
series keep their aspect ratio. Each series becomes **[32, 3, H, W]**. There
are no random pixel augmentations and no test-time augmentation in this recipe.
''')
code(extract("b35_target_spatial_residual", ["_extra_centers", "b35_centers"], {
    "b35_centers": "dense_centers", "B35_BASE_SLICES": "16", "B35_DENSE_SLICES": "32",
}))
code(extract("b37_highres_sparse_mil", ["_native_center_crop"]))
code(extract("b42_constant_area_aspect_sparse_mil", ["constant_area_shape", "resize_triplets_constant_area", "preprocess_dense_triplets_b42"], {
    "B42_REFERENCE_AREA": "448**2", "B42_STRIDE_ALIGNMENT": "32", "B42_PADDING_MODE": "'reflect'",
    "preprocess_dense_triplets_b42": "preprocess_triplets", "B37_CROP_FRACTION": "0.90",
    "b35_centers": "dense_centers",
}))
code(r'''
class KneeMRIDataset(Dataset):
    """One item is a complete study, with a variable number of rectangular series."""
    def __init__(self, uids, series_records, data_root, targets, weights):
        self.study_uids = list(uids)
        self.series_records = series_records
        self.data_root = Path(data_root)
        self.targets = np.asarray(targets, np.float32)
        self.weights = np.asarray(weights, np.float32)
        if self.targets.shape != (len(uids), N_TARGETS) or self.weights.shape != self.targets.shape:
            raise ValueError("Study supervision dimensions disagree.")
        if any(not series_records.get(uid) for uid in uids):
            raise ValueError("A study has no eligible series.")

    def __len__(self):
        return len(self.study_uids)

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        volumes, positions, meta, geometry = [], [], [], []
        for record in self.series_records[uid]:
            directory = find_series_dir(self.data_root, "train", uid, record["series_uid"])
            if directory is None:
                raise FileNotFoundError(f"Missing series {uid}/{record['series_uid']}")
            raw = read_dicom_series(directory)
            image, position = preprocess_triplets(raw)
            volumes.append(image)
            positions.append(torch.from_numpy(position))
            meta.append([record["plane_id"], record["fluid_id"], record["fat_id"]])
            geometry.append(dict(series_uid=record["series_uid"], height=image.shape[-2],
                                 width=image.shape[-1], present=True))
        return dict(study_uid=uid, volumes=volumes, slice_position=torch.stack(positions),
                    present=torch.ones(len(volumes)), series_meta=torch.tensor(meta, dtype=torch.long),
                    geometry=geometry, target=torch.from_numpy(self.targets[idx]),
                    weight=torch.from_numpy(self.weights[idx]))


def make_dataset(run_root, p, split):
    root = Path(run_root) / "protocol"
    index = load_json(root / "series_index.json")
    with np.load(root / "labels.npz", allow_pickle=False) as f:
        if split == "expert":
            uids, raw = f["expert_uids"].tolist(), f["expert_target"].copy()
            target, weight = np.nan_to_num(raw, nan=0.5), np.isfinite(raw).astype(np.float32)
        else:
            uids = p["splits"][split]
            lookup = {uid: i for i, uid in enumerate(f["uids"].tolist())}
            ix = [lookup[uid] for uid in uids]
            target, weight = f["target"][ix], f["weight"][ix]
    return KneeMRIDataset(uids, index, p["data_root"], target, weight)


def collate_studies(items):
    return list(items)


def make_loader(dataset, *, epoch=None):
    seed = RECIPE["seed"] + (0 if epoch is None else int(epoch) * 1009)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=RECIPE["batch_size"] if epoch is not None else 1,
                      shuffle=epoch is not None, drop_last=False, collate_fn=collate_studies,
                      num_workers=0, pin_memory=False, generator=generator)
''')
markdown(r'''
## 6. DINOv2 and the complete study model

The public ViT-S/14 encoder turns each triplet into a **384-value vector**.
384 is the feature width, not the image size or the number of slices. Inputs
receive at most thirteen pixels of right/bottom reflection padding to fit
14×14 patches, followed by ImageNet channel normalization. The three channels
are neighboring grayscale MRI slices, not color channels.

A learned summary token and the slice vectors enter one transformer layer
with six attention heads, a 768-unit feed-forward block and dropout 0.1.
There is **no additional slice-position or physical-spacing embedding**.
Although vectors are sorted by their supplied position, this block provides
set context rather than explicit distance/direction reasoning.

Plane, fluid-sensitivity and fat-suppression embeddings describe each series.
Twelve learned finding queries attend over all series. A residual connection,
layer normalization and twelve finding-specific linear heads produce logits.
Every encoder layer and every new head is trainable. Gradient checkpointing
recomputes encoder activations during backward to save GPU memory.
''')
code(extract("b57_models", ["state_digest", "dino_backbone", "SliceOutput", "DinoSliceTransformer"]))
code(r'''
def prepare_public_weights(run_root):
    root = Path(run_root) / "public_init"
    root.mkdir(parents=True, exist_ok=True)
    path, manifest_path = root / "encoder.pt", root / "encoder.json"
    if manifest_path.exists():
        return read_public_weights(run_root)[1]
    if path.exists():
        raise FileExistsError("Public weights have no manifest; inspect this incomplete initialization.")
    encoder = dino_backbone(pretrained=True)
    try:
        fingerprint = state_digest(encoder.state_dict())
        if fingerprint != PUBLIC_TENSOR_SHA256:
            raise ValueError("Downloaded weights differ from the pinned authors' public initialization.")
        atomic_torch_save(path, encoder.state_dict())
    finally:
        del encoder
        gc.collect()
    manifest = dict(source_url=DINO_URL, tensor_sha256=fingerprint,
                    sha256=sha256_file(path), competition_training_studies=[])
    write_json(manifest_path, manifest)
    return manifest


def read_public_weights(run_root):
    root = Path(run_root) / "public_init"
    manifest = load_json(root / "encoder.json")
    if (manifest["source_url"] != DINO_URL or manifest["tensor_sha256"] != PUBLIC_TENSOR_SHA256
            or manifest["competition_training_studies"] != []
            or manifest["sha256"] != sha256_file(root / "encoder.pt")):
        raise ValueError("Public initialization manifest/hash mismatch.")
    weights = torch.load(root / "encoder.pt", map_location="cpu", weights_only=True)
    if state_digest(weights) != PUBLIC_TENSOR_SHA256:
        raise ValueError("Public initialization tensors changed.")
    return weights, manifest


def build_model(run_root=None):
    encoder = dino_backbone(pretrained=False)
    if run_root is not None:
        encoder.load_state_dict(read_public_weights(run_root)[0], strict=True)
    return DinoSliceTransformer(encoder, dim=RECIPE["dim"], heads=RECIPE["heads"],
                               dropout=RECIPE["dropout"], chunk_size=RECIPE["encoder_chunk_size"],
                               gradient_checkpointing=RECIPE["gradient_checkpointing"])


def parameter_groups(model):
    encoder = list(model.encoder.parameters())
    if not encoder or not all(p.requires_grad for p in encoder):
        raise ValueError("Every encoder layer must be trainable.")
    ids = {id(p) for p in encoder}
    heads = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    groups = [dict(params=encoder, lr=RECIPE["encoder_lr"], name="public_encoder"),
              dict(params=heads, lr=RECIPE["head_lr"], name="fresh_heads")]
    grouped = [id(p) for g in groups for p in g["params"]]
    if len(grouped) != len(set(grouped)) or set(grouped) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError("Optimizer must cover trainable parameters exactly once.")
    return groups
''')
markdown(r'''
## 7. Probabilities, loss and AUC

Each finding has its own logit z. Its probability is sigmoid(z) = 1/(1+exp(-z)).
For example, logits −2, 0 and 2 become about 0.119, 0.5 and 0.881. Findings
are independent outputs: probabilities do not need to add to one. A sigmoid
output alone does not establish clinical calibration.

Training uses numerically stable binary cross-entropy with logits. Label
weights mask uncertain cells; inverse target weight-mass balances the twelve
findings. Two studies form one optimizer update, processed sequentially and
scaled by their supervised weight mass. This matches the weighted loss of the
combined batch. Even zero-weight studies remain in MRI exposure.

Evaluation computes each finding's ROC AUC on cells with weight greater than
zero, with report targets above 0.5 treated as positive. AUC itself is
unweighted. Macro AUC averages defined findings; all twelve must be measurable
on scanner validation. An undefined expert AUC is reported as missing.
''')
code(extract("b7_weak_supervision", ["seed_everything", "target_balance_multipliers", "target_balanced_weak_bce"]))
code(extract("b42_constant_area_aspect_sparse_training", ["_move_study", "_study_mass", "_batch_scales"]))
code(r'''
def macro_auc(target, weight, prediction):
    target, weight, prediction = map(np.asarray, (target, weight, prediction))
    if target.shape != weight.shape or target.shape != prediction.shape or target.shape[1] != N_TARGETS:
        raise ValueError("AUC inputs must have matching [studies,12] shapes.")
    if not all(np.isfinite(a).all() for a in (target, weight, prediction)):
        raise ValueError("Nonfinite AUC input.")
    scores = {}
    for j, name in enumerate(TARGETS):
        active = weight[:, j] > 0
        truth = (target[active, j] > 0.5).astype(int)
        scores[name] = float(roc_auc_score(truth, prediction[active, j])) if len(np.unique(truth)) == 2 else float("nan")
    defined = [v for v in scores.values() if np.isfinite(v)]
    return dict(macro_auc=float(np.mean(defined)) if defined else float("nan"),
                per_target_auc=scores, targets_defined=len(defined))


def loss_for(model, item, device, amp_dtype, multiplier):
    volumes, position, present, meta, target, weight = _move_study(item, device)
    with autocast(device, amp_dtype):
        output = model(volumes, present, meta, position)
        loss = target_balanced_weak_bce(output.logits, target, weight, multiplier)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite loss: {item['study_uid']}")
    return output, loss


def batch_step(model, items, device, amp_dtype, optimizer, scaler, multiplier):
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for item, scale in zip(items, _batch_scales(items, multiplier)):
        output, loss = loss_for(model, item, device, amp_dtype, multiplier.to(device))
        scaler.scale(loss * scale).backward()
        total += float(loss.detach()) * scale
        del output, loss
    scaler.unscale_(optimizer)
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), RECIPE["grad_clip"], error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return total, float(grad_norm)


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    was_training = model.training
    uids, probabilities, targets, weights = [], [], [], []
    model.eval()
    try:
        for items in loader:
            for item in items:
                volumes, position, present, meta, _, _ = _move_study(item, device)
                with autocast(device, amp_dtype):
                    output = model(volumes, present, meta, position)
                probability = output.logits.float().sigmoid().cpu().numpy().reshape(-1)
                if not np.isfinite(probability).all():
                    raise FloatingPointError(f"Nonfinite prediction: {item['study_uid']}")
                uids.append(item["study_uid"])
                probabilities.append(probability)
                targets.append(item["target"].numpy())
                weights.append(item["weight"].numpy())
                del output, volumes
    finally:
        model.train(was_training)
    if not uids or len(uids) != len(set(uids)):
        raise ValueError("Empty or duplicate prediction UIDs.")
    return dict(uids=np.asarray(uids), prediction=np.stack(probabilities),
                target=np.stack(targets), weight=np.stack(weights))
''')
markdown(r'''
## 8. Preflight, training and recovery

The real-data preflight selects the two training studies with the most eligible
series, runs backward, and verifies finite nonzero gradients in the encoder
and new heads. It performs **zero optimizer updates** and discards that model.
It estimates peak GPU allocation for those studies, not every possible scan.

Training uses AdamW with encoder learning rate 0.00001, head learning rate
0.0001, weight decay 0.0001 and gradient clipping at 1.0. A cosine schedule
reaches one percent of each initial learning rate after twelve epochs.
The seed is 2026 and shuffling is reconstructed deterministically per epoch.
On a compatible 5090, automatic mixed precision uses bfloat16.

The checkpoint includes optimizer, scheduler, gradient scaler, random-number
states and complete history. Evaluation uses the **fixed final epoch**, not
the epoch with the highest observed validation score. Training runs directly
in this kernel; keep Jupyter alive. Ordinary interruption cleans up the model.
''')
code(extract("training_resume", ["rng_state", "set_rng_state"]))
code(r'''
def run_contract(run_root, device, amp_dtype):
    p = load_protocol(run_root)
    _, manifest = read_public_weights(run_root)
    return dict(implementation=p["implementation"],
                protocol_sha256=sha256_file(Path(run_root) / "protocol/protocol.json"),
                public_encoder_sha256=manifest["sha256"], public_source_url=DINO_URL,
                training_uids_sha256=p["split_uid_hashes"]["train"],
                validation_uids_sha256=p["split_uid_hashes"]["validation"],
                competition_supervised_ancestor=False, gold_studies_in_gradient=0,
                epochs=RECIPE["epochs"], precision=str(amp_dtype), device_type=device.type,
                torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                timm=timm.__version__, numpy=np.__version__)


def release_accelerator():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def preflight(run_root, device="cuda:0"):
    root = Path(run_root)
    model = groups = output = loss = None
    with exclusive_run(root):
        claim_run(root)
        p = load_protocol(root)
        device, amp_dtype = runtime_for(device)
        contract = run_contract(root, device, amp_dtype)
        seed_everything(RECIPE["seed"])
        try:
            model = build_model(root).to(device).train()
            groups = parameter_groups(model)
            dataset = make_dataset(root, p, "train")
            multiplier = torch.from_numpy(target_balance_multipliers(dataset.weights)).to(device)
            active = [i for i, w in enumerate(dataset.weights) if w.sum() > 0]
            chosen = sorted(active, key=lambda i: (-len(dataset.series_records[dataset.study_uids[i]]),
                                                  dataset.study_uids[i]))[:RECIPE["batch_size"]]
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for i in chosen:
                output, loss = loss_for(model, dataset[i], device, amp_dtype, multiplier)
                (loss / len(chosen)).backward()
                output = loss = None
            counts = {}
            for group in groups:
                grads = [p.grad for p in group["params"] if p.grad is not None]
                if not grads or any(not torch.isfinite(g).all() for g in grads):
                    raise RuntimeError(f"Invalid gradients: {group['name']}")
                count = sum(int(torch.count_nonzero(g) > 0) for g in grads)
                if not count:
                    raise RuntimeError(f"No learning signal: {group['name']}")
                counts[group["name"]] = count
            encoded = list(model.encoder.parameters())
            if any(x.grad is None or not torch.count_nonzero(x.grad) for x in (encoded[0], encoded[-1])):
                raise RuntimeError("Gradient did not reach both ends of the image encoder.")
            result = dict(passed=True, contract=contract, optimizer_steps=0, gradient_tensor_counts=counts,
                          study_uids=[dataset.study_uids[i] for i in chosen],
                          peak_cuda_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else None)
            write_json(root / "preflight.json", result)
            print("Model preflight: PASS; optimizer updates: 0; peak CUDA GiB:", result["peak_cuda_gib"])
            return result
        except BaseException as error:
            import traceback
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            model = groups = output = loss = None
            # The local parameter/gradient lists also own GPU storage.
            encoded = grads = group = multiplier = None
            release_accelerator()


def save_recovery(path, epoch, model, optimizer, scheduler, scaler, history, contract):
    atomic_torch_save(path, dict(contract=contract, epoch=epoch, model_state=model.state_dict(),
                                optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                                scaler_state=scaler.state_dict(), rng_state=rng_state(), history=history))


def save_predictions(path, prediction, contract, split, checkpoint_sha):
    path = Path(path)
    atomic_npz(path, **prediction)
    write_json(path.with_suffix(".json"), dict(contract=contract, split=split,
               checkpoint_sha256=checkpoint_sha, npz_sha256=sha256_file(path),
               uid_sha256=digest(prediction["uids"].tolist())))


def train_model(run_root, device="cuda:0"):
    root, out = Path(run_root), Path(run_root) / "model"
    model = optimizer = scheduler = scaler = groups = saved = None
    with exclusive_run(root):
        claim_run(root)
        p = load_protocol(root)
        device, amp_dtype = runtime_for(device)
        contract = run_contract(root, device, amp_dtype)
        check = load_json(root / "preflight.json")
        require_same_run(check["contract"], contract)
        if check.get("passed") is not True or check.get("optimizer_steps") != 0:
            raise ValueError("Run and pass the real-data preflight first.")
        if (out / "complete.json").exists():
            require_same_run(load_json(out / "complete.json")["contract"], contract)
            evaluate_results(root)
            print("Training is already complete; final artifacts verified.")
            return out / "final.pt"
        recovery = out / "recovery_latest.pt"
        if out.exists() and any(p.name != "recovery_latest.pt.writing" for p in out.iterdir()) and not recovery.exists():
            raise FileExistsError("Model directory is occupied without a recovery checkpoint.")
        seed_everything(RECIPE["seed"])
        try:
            model = build_model(root).to(device).train()
            groups = parameter_groups(model)
            optimizer = torch.optim.AdamW(groups, weight_decay=RECIPE["weight_decay"])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                lambda e: .01 + .99 * (1 + np.cos(np.pi * min(e, RECIPE["epochs"]) / RECIPE["epochs"])) / 2)
            scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is torch.float16)
            training = make_dataset(root, p, "train")
            validation = make_dataset(root, p, "validation")
            multiplier = torch.from_numpy(target_balance_multipliers(training.weights))
            history, start_epoch = [], 1
            if recovery.exists():
                # Trusted local checkpoint written by this notebook, includes Python/NumPy RNG state.
                saved = torch.load(recovery, map_location="cpu", weights_only=False)
                require_same_run(saved["contract"], contract)
                if not 0 < saved["epoch"] <= RECIPE["epochs"]:
                    raise ValueError("Invalid recovery epoch.")
                model.load_state_dict(saved["model_state"], strict=True)
                optimizer.load_state_dict(saved["optimizer_state"])
                scheduler.load_state_dict(saved["scheduler_state"])
                scaler.load_state_dict(saved["scaler_state"])
                set_rng_state(saved["rng_state"])
                history, start_epoch = list(saved["history"]), saved["epoch"] + 1
                if [row["epoch"] for row in history] != list(range(1, start_epoch)):
                    raise ValueError("Recovery history does not match completed epochs.")
                saved = None
            else:
                seed_everything(RECIPE["seed"] + 101)
            print(f"Training on {len(training)} studies; starting epoch {start_epoch} of {RECIPE['epochs']}.")
            for epoch in range(start_epoch, RECIPE["epochs"] + 1):
                start = time.monotonic()
                loader = make_loader(training, epoch=epoch)
                model.train()
                loss_sum, seen = 0.0, []
                for batch, items in enumerate(loader, 1):
                    loss, grad = batch_step(model, items, device, amp_dtype, optimizer, scaler, multiplier)
                    loss_sum += loss
                    seen.extend(item["study_uid"] for item in items)
                    if batch % 100 == 0:
                        print(f"Epoch {epoch}: {batch}/{len(loader)} batches; loss={loss_sum / batch:.5f}", flush=True)
                if len(seen) != len(set(seen)) or set(seen) != set(p["splits"]["train"]):
                    raise RuntimeError("Epoch did not expose exactly the permitted training studies.")
                del loader
                scheduler.step()
                predictions = predict(model, make_loader(validation), device, amp_dtype)
                scores = macro_auc(predictions["target"], predictions["weight"], predictions["prediction"])
                if scores["targets_defined"] != N_TARGETS or not np.isfinite(scores["macro_auc"]):
                    raise RuntimeError("Validation must define twelve AUCs.")
                history.append(dict(epoch=epoch, train_loss=loss_sum / batch, validation=scores,
                                    training_uids_sha256=digest(sorted(seen)), epoch_minutes=(time.monotonic()-start)/60,
                                    learning_rates=[float(g["lr"]) for g in optimizer.param_groups]))
                save_recovery(recovery, epoch, model, optimizer, scheduler, scaler, history, contract)
                write_json(out / "history.json", history)
                print(f"Epoch {epoch} complete: macro AUC={scores['macro_auc']:.6f}", flush=True)
            encoder_sha = state_digest(model.encoder.state_dict())
            if encoder_sha == PUBLIC_TENSOR_SHA256:
                raise RuntimeError("The public encoder was not updated during training.")
            final = out / "final.pt"
            atomic_torch_save(final, dict(contract=contract, completed_epochs=RECIPE["epochs"],
                                         selection="fixed_final_epoch", recipe=RECIPE,
                                         model_state=model.state_dict(), history=history,
                                         encoder_tensor_sha256_final=encoder_sha))
            # A crash after the last recovery can leave history.json behind by one epoch.
            write_json(out / "history.json", history)
            checkpoint_sha = sha256_file(final)
            for split, dataset in (("validation", validation), ("expert", make_dataset(root, p, "expert"))):
                predictions = predict(model, make_loader(dataset), device, amp_dtype)
                save_predictions(out / f"{split}.npz", predictions, contract, split, checkpoint_sha)
            write_json(out / "complete.json", dict(contract=contract, checkpoint_sha256=checkpoint_sha,
                                                   completed_epochs=RECIPE["epochs"], expert_role="diagnostic only"))
            print("Training and fixed-final-epoch predictions: COMPLETE", flush=True)
            return final
        except BaseException as error:
            import traceback
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            model = optimizer = scheduler = scaler = groups = saved = None
            release_accelerator()
''')
markdown(r'''
## 9. Verified evaluation and reusable inference

Before exporting scores, evaluation checks the checkpoint hash, prediction
hashes, exact study lists, labels and confidence masks. The final model can
also be reconstructed entirely from the notebook definitions and its saved
state dictionary. It needs no repository checkpoint loader and performs no
public-weight download when loading a trained checkpoint.
''')
code(r'''
def evaluate_results(run_root):
    root, out = Path(run_root), Path(run_root) / "model"
    p = load_protocol(root)
    complete = load_json(out / "complete.json")
    contract = complete["contract"]
    if (contract["implementation"] != p["implementation"]
            or contract["protocol_sha256"] != sha256_file(root / "protocol/protocol.json")
            or complete["completed_epochs"] != RECIPE["epochs"]
            or contract["epochs"] != RECIPE["epochs"]
            or contract["competition_supervised_ancestor"] is not False
            or contract["gold_studies_in_gradient"] != 0):
        raise ValueError("Final model/protocol contract mismatch.")
    checkpoint_sha = sha256_file(out / "final.pt")
    if checkpoint_sha != complete["checkpoint_sha256"]:
        raise ValueError("Final checkpoint changed.")
    metrics, tables, exports = {}, {}, {}
    for split in ("validation", "expert"):
        path = out / f"{split}.npz"
        meta = load_json(path.with_suffix(".json"))
        if (meta["contract"] != contract or meta["split"] != split
                or meta["checkpoint_sha256"] != checkpoint_sha or meta["npz_sha256"] != sha256_file(path)):
            raise ValueError(f"Prediction provenance mismatch: {split}")
        with np.load(path, allow_pickle=False) as f:
            data = {k: f[k].copy() for k in ("uids", "target", "weight", "prediction")}
        uids = data["uids"].tolist()
        expected = make_dataset(root, p, split)
        if len(uids) != len(set(uids)) or set(uids) != set(expected.study_uids) or digest(uids) != meta["uid_sha256"]:
            raise ValueError(f"Wrong prediction studies: {split}")
        lookup = {uid: i for i, uid in enumerate(uids)}
        ix = [lookup[uid] for uid in expected.study_uids]
        data = {k: v[ix] for k, v in data.items()}
        for key in ("target", "weight", "prediction"):
            if data[key].shape != expected.targets.shape or not np.isfinite(data[key]).all():
                raise ValueError(f"Invalid prediction array: {split}/{key}")
        if not np.array_equal(data["target"], expected.targets) or not np.array_equal(data["weight"], expected.weights):
            raise ValueError(f"Prediction labels or masks changed: {split}")
        if ((data["prediction"] < 0) | (data["prediction"] > 1)).any():
            raise ValueError("Probabilities must be in [0,1].")
        scores = macro_auc(data["target"], data["weight"], data["prediction"])
        if split == "validation" and scores["targets_defined"] != N_TARGETS:
            raise ValueError("Validation must define all twelve AUCs.")
        table = pd.DataFrame.from_dict(coverage(data["target"], data["weight"]), orient="index")
        table["AUC"] = pd.Series(scores["per_target_auc"])
        metrics[split], tables[split] = finite_json(scores), table
        exports[split] = pd.DataFrame(data["prediction"], columns=TARGETS).assign(StudyInstanceUID=data["uids"])[["StudyInstanceUID", *TARGETS]]
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    write_json(reports / "metrics.json", metrics)
    for split in tables:
        tables[split].to_csv(reports / f"{split}_by_finding.csv", index_label="Finding")
        exports[split].to_csv(reports / f"{split}_probabilities.csv", index=False)
    return metrics, tables


def load_trained_model(checkpoint_path, device="cuda:0"):
    """Load a trusted checkpoint created by this standalone notebook."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload["recipe"] != RECIPE or payload["completed_epochs"] != RECIPE["epochs"]:
        raise ValueError("Checkpoint recipe/endpoint mismatch.")
    require_same_run(payload["contract"]["implementation"], implementation_contract())
    model = build_model().to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval()
''')
markdown(r'''
## 10. Plotting functions

The architecture and sigmoid figures are explanatory. Slice previews use
actual training images; learning curves and per-finding plots use saved
results from this run. PNG and SVG versions are saved alongside the run.
''')
code(extract("local_dinov2/plots", ["architecture", "slice_example", "sigmoid", "training_history", "target_auc", "save_figure"]))

# Names for hashing the actual executed definitions, including all training helpers.
_names = []
for kind, text, tags in CELLS:
    if kind == "code" and tags == ["implementation"]:
        _names.extend(n.name for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
code("IMPLEMENTATION_NAMES = " + repr(_names))
markdown(r'''
## 11. Run the workflow

All definitions are now loaded. The following cells do the work. Review the
population table and preview, then let the preflight finish before starting
training. A repeated preparation verifies the same inputs. Repeating training
after an interruption resumes this notebook's last completed epoch.
''')
code(r'''
environment_check(DEVICE)
protocol = prepare_data(work_root=WORK_ROOT, data_root=DATA_ROOT, run_root=RUN_ROOT,
                        labels_root=LABELS_ROOT, scanner_split_root=SCANNER_SPLIT_ROOT,
                        series_policy=SERIES_POLICY)
with exclusive_run(RUN_ROOT):
    claim_run(RUN_ROOT)
    public_manifest = prepare_public_weights(RUN_ROOT)
population = {**protocol["counts"], "expert": len(protocol["expert_uids"])}
display(pd.DataFrame.from_dict(population, orient="index", columns=["Studies"]))
display(pd.DataFrame.from_dict(protocol["coverage"]["train"], orient="index"))
''', "prepare")
code(r'''
example = make_dataset(RUN_ROOT, protocol, "train")[0]
display(pd.DataFrame(example["geometry"])[["height", "width", "present"]].rename_axis("Series"))
fig = slice_example(example)
save_figure(fig, RUN_ROOT, "slice_inputs")
plt.show()
plt.close(fig)
del example

for name, draw in (("architecture", architecture), ("sigmoid", sigmoid)):
    fig = draw()
    save_figure(fig, RUN_ROOT, name)
    plt.show()
    plt.close(fig)
''', "preview")
code(r'''
preflight_result = preflight(RUN_ROOT, DEVICE)
''', "preflight")
code(r'''
final_checkpoint = train_model(RUN_ROOT, DEVICE)
print("Final model:", final_checkpoint)
''', "train")
markdown(r'''
## 12. Learning curves and final results

The loss curve shows the average weighted training loss. The AUC curve shows
scanner validation after each completed epoch. The final reports always use
the model after epoch twelve, even if an earlier point is higher.

The prediction CSVs contain one row per held-out study and twelve sigmoid
probabilities. Missing expert AUCs indicate insufficient class coverage; they
are not zero scores. These are evaluation exports, not competition submissions.
''')
code(r'''
history = load_json(RUN_ROOT / "model/history.json")
fig = training_history(history)
save_figure(fig, RUN_ROOT, "training_history")
plt.show()
plt.close(fig)

metrics, tables = evaluate_results(RUN_ROOT)
display(pd.DataFrame({name: {"Macro AUC": row["macro_auc"], "Defined findings": row["targets_defined"]}
                      for name, row in metrics.items()}).T)
for split, table in tables.items():
    print(split)
    display(table)
fig = target_auc(tables)
save_figure(fig, RUN_ROOT, "auc_by_finding")
plt.show()
plt.close(fig)
print("Saved checkpoint, probabilities, metrics and figures:", RUN_ROOT)
''', "results")
markdown(r'''
## Outputs and recovery

| Location under RUN_ROOT | Contents |
|---|---|
| notebook_run.json | Executed implementation fingerprint and recipe |
| protocol/ | Input hashes, frozen labels, study lists and series metadata |
| public_init/ | Verified public image-encoder initialization |
| preflight.json | Actual backward-pass checks and peak GPU allocation |
| model/recovery_latest.pt | Latest completed epoch, optimizer and random states |
| model/final.pt | Fixed final model and architecture recipe |
| model/history.json | Training loss and validation AUC by epoch |
| reports/ | Metrics, per-finding coverage and per-study probabilities |
| figures/ | Architecture, real MRI preview and result plots |

To resume, restart the kernel, retain the same code, settings and inputs, and
run the cells in order. A partial epoch is repeated; a completed run is
verified and returned. New code or a new recipe requires a new run directory.
The data and public initialization may be copied to another machine, but a
resume contract still requires matching file paths, software and precision.

To use the saved model, call load_trained_model with model/final.pt. Build
unseen-study items with KneeMRIDataset using the same preprocessing and series
metadata, then pass their DataLoader through predict. The helper returns the
independent sigmoid probabilities for the twelve findings.

References: [DINOv2 paper](https://arxiv.org/abs/2304.07193),
[authors' implementation](https://github.com/facebookresearch/dinov2),
[PyTorch binary cross-entropy with logits](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html),
[ROC AUC definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
''')


def notebook():
    cells = []
    for i, (kind, text, tags) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"knee-{i:02d}", "metadata": {"tags": tags} if tags else {},
                "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Knee MRI · DINOv2",
            "language": "python", "name": "dinov2-local"}, "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    path = Path(__file__).with_name("dinov2_knee_mri_local.ipynb")
    path.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n")
    print(path)
