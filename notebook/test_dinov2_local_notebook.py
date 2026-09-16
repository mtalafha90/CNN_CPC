"""Test the actual standalone definitions, numerical parity and isolated execution."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebook/dinov2_knee_mri_local.ipynb"


def definitions(nb=None):
    nb = nb or nbformat.read(NOTEBOOK, as_version=4)
    return "\n\n".join(c.source for c in nb.cells
                       if c.cell_type == "code" and c.metadata.get("tags") == ["implementation"])


@pytest.fixture
def standalone(tmp_path):
    source = definitions()
    path = tmp_path / "standalone_implementation.py"
    path.write_text(source)
    name = f"standalone_{id(tmp_path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(name, None)


def test_notebook_is_generated_clean_valid_and_contains_complete_code():
    spec = importlib.util.spec_from_file_location("standalone_builder", NOTEBOOK.with_name("build_dinov2_local_notebook.py"))
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert json.loads(NOTEBOOK.read_text()) == builder.notebook()
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    all_source = "\n".join(c.source for c in nb.cells)
    assert not re.search(r"\b[Bb]\d+\b|[Bb]\d+_|Experiment[_ -]", all_source)
    assert not any(x in all_source for x in ("rsna_knee", "sys.path", "%run", "import_module"))
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and cell.outputs == []
            tree = ast.parse(cell.source)
            assert not any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(tree))
            assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                           and n.func.id in {"exec", "eval", "__import__"} for n in ast.walk(tree))
            assert not any(isinstance(n, ast.Import) and any(a.name == "subprocess" for a in n.names)
                           for n in ast.walk(tree))
    names = {n.name for n in ast.parse(definitions()).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    assert {"KneeMRIDataset", "DinoSliceTransformer", "read_dicom_series", "preprocess_triplets",
            "prepare_data", "preflight", "train_model", "evaluate_results", "load_trained_model"} <= names
    assert nb.metadata.kernelspec.name == "dinov2-local"


@pytest.mark.parametrize("shape", [(1, 29, 57), (3, 80, 40), (17, 31, 49), (43, 67, 35)])
def test_pixels_and_slice_positions_match_original_exactly(standalone, shape):
    from rsna_knee.b42_constant_area_aspect_sparse_mil import preprocess_dense_triplets_b42
    rng = np.random.default_rng(19)
    raw = rng.normal(size=shape).astype(np.float32)
    raw.ravel()[:3] = [np.nan, np.inf, -np.inf]
    original, original_pos = preprocess_dense_triplets_b42(raw)
    actual, actual_pos = standalone.preprocess_triplets(raw)
    torch.testing.assert_close(actual, original, atol=0, rtol=0)
    np.testing.assert_array_equal(actual_pos, original_pos)


def test_real_dino_model_has_identical_state_logits_and_gradients(standalone):
    from rsna_knee.b57_models import DinoSliceTransformer, dino_backbone
    torch.manual_seed(23)
    expected = DinoSliceTransformer(dino_backbone(), dropout=0.0, chunk_size=2).train()
    torch.manual_seed(23)
    actual = standalone.DinoSliceTransformer(standalone.dino_backbone(), dropout=0.0, chunk_size=2).train()
    assert set(actual.state_dict()) == set(expected.state_dict())
    for key, tensor in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], tensor, atol=0, rtol=0)
    volumes = [torch.randn(2, 3, 28, 42), torch.randn(2, 3, 42, 28)]
    present = torch.ones(2)
    meta = torch.tensor([[1, 2, 1], [2, 1, 2]])
    position = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    a = actual(volumes, present, meta, position).logits
    b = expected(volumes, present, meta, position).logits
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.square().mean().backward()
    b.square().mean().backward()
    for (name, p), (other_name, q) in zip(actual.named_parameters(), expected.named_parameters()):
        assert name == other_name
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_supervision_balance_and_auc_match_original(standalone):
    from rsna_knee.b7_weak_supervision import target_balance_multipliers, target_balanced_weak_bce
    from rsna_knee.b42_constant_area_aspect_sparse_training import _batch_scales
    from rsna_knee.b52_competition_training import macro_auc
    rng = np.random.default_rng(13)
    target = rng.choice([.05, .85], (20, 12)).astype(np.float32)
    weight = rng.choice([0., .5, 1.], (20, 12)).astype(np.float32)
    prediction = rng.random((20, 12)).astype(np.float32)
    multiplier = standalone.target_balance_multipliers(weight)
    np.testing.assert_array_equal(multiplier, target_balance_multipliers(weight))
    args = [torch.from_numpy(v) for v in (prediction[:2], target[:2], weight[:2], multiplier)]
    torch.testing.assert_close(standalone.target_balanced_weak_bce(*args), target_balanced_weak_bce(*args), atol=0, rtol=0)
    items = [{"weight": torch.from_numpy(w)} for w in weight[:2]]
    assert standalone._batch_scales(items, args[-1]) == _batch_scales(items, args[-1])
    a, b = standalone.macro_auc(target, weight, prediction), macro_auc(target, weight, prediction)
    assert a["targets_defined"] == b["targets_defined"] == 12
    assert a["macro_auc"] == pytest.approx(b["macro_auc"], abs=1e-15)


# Deliberately small CPU fixtures. These are not MRI performance evidence.
# They replace only scale, public backbone and fixture counts, while exercising
# the notebook's actual DICOM loader, data boundary, model heads and train loop.
FIXTURE_OVERRIDES = '''
print("SYNTHETIC CPU VERIFICATION: no clinical images or reported experiment score")
DEVICE = "cpu"
RECIPE = {**RECIPE, "epochs": 2, "dim": 24}
EXPECTED_POPULATION = dict(report_only=8, usable_cells=96, train=5, validation=2,
                           excluded_profile_overlap=1, expert=2)

class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 24)
    def forward(self, x):
        return self.projection(x.mean((-2, -1)))

with torch.random.fork_rng():
    torch.manual_seed(71)
    SYNTHETIC_PUBLIC = TinyEncoder().state_dict()
PUBLIC_TENSOR_SHA256 = state_digest(SYNTHETIC_PUBLIC)

def dino_backbone(*, pretrained=False):
    model = TinyEncoder()
    if pretrained:
        model.load_state_dict(SYNTHETIC_PUBLIC)
    return model

def resize_triplets_constant_area(triplets, *, reference_area=448**2, alignment=32):
    return F.interpolate(torch.from_numpy(np.ascontiguousarray(triplets)),
                         size=(28, 42), mode="bilinear", align_corners=False, antialias=True)

def environment_check(device):
    assert device == "cpu"
    print("Synthetic CPU fixture; production CUDA/decoder preflight is not claimed.")
'''


def apply_fixture_overrides(standalone, tmp_path):
    path = tmp_path / "synthetic_overrides.py"
    path.write_text(FIXTURE_OVERRIDES)
    exec(compile(FIXTURE_OVERRIDES, str(path), "exec"), vars(standalone))


def create_input_fixture(base, s):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage
    data, labels, gate = [base / name for name in ("data", "labels", "gate")]
    for folder in (data, labels, gate):
        folder.mkdir(parents=True, exist_ok=True)
    uids = [f"1.2.826.0.1.3680043.10.99.{i}" for i in range(10)]
    train = pd.DataFrame({"StudyInstanceUID": uids, "Report": ["synthetic fixture"]*10})
    for target in s.TARGETS:
        train[target] = [np.nan]*8 + [0., 1.]
    train.loc[8:, s.TARGETS[-1]] = np.nan  # Undefined expert AUC must remain missing.
    train.to_csv(data / "train.csv", index=False)
    label = pd.DataFrame({"StudyInstanceUID": uids[:8]})
    for target in s.TARGETS:
        label[f"{target}__state"] = ["negated", "positive"]*4
        label[f"{target}__confidence"] = 1.0
    label.to_csv(labels / "training_targets.csv", index=False)
    s.write_json(labels / "policy.json", {"fixture": True})
    s.write_json(labels / "audit.json", {"base_cells_overridden": 0, "gold_rows_in_training_targets": 0})
    rows = pd.DataFrame({"StudyInstanceUID": uids[:8],
                        "scanner_profile": ["A", "A", "A", "A", "B", "B", "B", "C"],
                        "parent_split": ["train"]*6 + ["holdout_unseen_scanners"]*2,
                        "selection_split": ["train"]*2 + ["validation_seen_scanners"]*2
                           + ["validation_unseen_scanners"]*2 + ["excluded_prior_surface"]*2})
    rows.to_csv(gate / "selection_split_by_study.csv", index=False)
    s.write_json(gate / "selection_split.json", {
        "source_train_csv_sha256": s.sha256_file(data / "train.csv"),
        "source_training_targets_sha256": s.sha256_file(labels / "training_targets.csv"),
        "unseen_scanner_profiles": ["B"]})
    (gate / "selection_split.sha256").write_text(s.sha256_file(gate / "selection_split.json") + "\n")
    series = []
    for i, uid in enumerate(uids):
        series_uid = f"{uid}.1"
        directory = data / "train_series" / uid / series_uid
        directory.mkdir(parents=True)
        for j in range(3):
            path = directory / f"{2-j}.dcm"
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = f"{series_uid}.{j+1}"
            ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0"*128)
            ds.Rows, ds.Columns = 29, 43
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit, ds.PixelRepresentation, ds.SamplesPerPixel = 15, 0, 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.ImageOrientationPatient = [0., 1., 0., 0., 0., 1.]
            ds.ImagePositionPatient = [float(j), 0., 0.]
            ds.InstanceNumber = 3-j
            ds.EchoTime, ds.RepetitionTime = 40., 2000.
            ds.ScanOptions = "FS"
            pixels = (np.arange(29*43).reshape(29,43)*(i+1) + j*17).astype(np.uint16)
            ds.PixelData = pixels.tobytes()
            ds.save_as(str(path), enforce_file_format=True)
        series.append(dict(StudyInstanceUID=uid, SeriesInstanceUID=series_uid,
                           Anatomical_Plane="" if i == 0 else "Sagittal",
                           Fluid_Sensitive=np.nan if i == 0 else True,
                           Fat_Suppression=np.nan if i == 0 else True))
    pd.DataFrame(series).to_csv(data / "train_series.csv", index=False)
    policy = base / "series_policy.json"
    s.write_json(policy, {"policy": "all_repaired_anatomical_series_v1", "uses_gold_labels": False,
                         "viability_passed": True, "retained_active_studies": 3120,
                         "retained_usable_cells": 14123,
                         "series_summary": {"series_signature_sha256": s.SERIES_SIGNATURE}})
    return dict(work_root=base, data_root=data, run_root=base / "runs/standalone",
                labels_root=labels, scanner_split_root=gate, series_policy=policy)


def prepared_fixture(s, tmp_path):
    apply_fixture_overrides(s, tmp_path)
    inputs = create_input_fixture(tmp_path / "fixture", s)
    p = s.prepare_data(**inputs)
    s.prepare_public_weights(inputs["run_root"])
    return inputs, p


def test_raw_inputs_metadata_and_scanner_partition_match_original(standalone, tmp_path):
    from rsna_knee.b57_protocol import partition
    from rsna_knee.data import load_series_csv, backfill_series_metadata
    from rsna_knee.b12_variable_series import build_variable_series_index
    inputs, p = prepared_fixture(standalone, tmp_path)
    s = standalone
    rows = pd.read_csv(inputs["scanner_split_root"] / "selection_split_by_study.csv")
    uids = pd.read_csv(inputs["labels_root"] / "training_targets.csv").StudyInstanceUID.tolist()
    indices, _ = partition(uids, rows.rename(columns={"selection_split": "b50_split"}), p["expert_uids"])
    assert p["splits"] == {k: [uids[i] for i in ix] for k, ix in indices.items()}
    series, stats = backfill_series_metadata(load_series_csv(inputs["data_root"] / "train_series.csv"), inputs["data_root"])
    assert stats == p["metadata_repair"]
    expected = build_variable_series_index(series, uids + p["expert_uids"])
    assert s.load_json(inputs["run_root"] / "protocol/series_index.json") == expected
    before = (inputs["run_root"] / "protocol/protocol.json").read_bytes()
    s.prepare_data(**inputs)
    assert (inputs["run_root"] / "protocol/protocol.json").read_bytes() == before
    with s.exclusive_run(inputs["run_root"]), pytest.raises(RuntimeError, match="Another process"):
        with s.exclusive_run(inputs["run_root"]):
            pass


def test_train_interrupt_resume_matches_uninterrupted_and_exports_verified(standalone, tmp_path, monkeypatch):
    s = standalone
    inputs, _ = prepared_fixture(s, tmp_path)
    root = inputs["run_root"]
    s.preflight(root, "cpu")
    s.train_model(root, "cpu")
    expected = torch.load(root / "model/final.pt", weights_only=False, map_location="cpu")
    metrics, tables = s.evaluate_results(root)
    assert metrics["validation"]["targets_defined"] == 12
    assert metrics["expert"]["targets_defined"] == 11
    assert np.isnan(tables["expert"].loc[s.TARGETS[-1], "AUC"])
    saved_model = s.load_trained_model(root / "model/final.pt", "cpu")
    for key, tensor in saved_model.state_dict().items():
        torch.testing.assert_close(tensor, expected["model_state"][key], atol=0, rtol=0)
    second = root.with_name("interrupted")
    s.prepare_data(**{**inputs, "run_root": second})
    s.prepare_public_weights(second)
    s.preflight(second, "cpu")
    original_step = torch.optim.AdamW.step
    calls = []
    def interrupt_after_epoch_one(optimizer, *args, **kwargs):
        calls.append(1)
        if len(calls) == 4:
            raise KeyboardInterrupt("synthetic interruption during the second epoch")
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, "step", interrupt_after_epoch_one)
    with pytest.raises(KeyboardInterrupt):
        s.train_model(second, "cpu")
    saved = torch.load(second / "model/recovery_latest.pt", weights_only=False, map_location="cpu")
    assert saved["epoch"] == 1
    monkeypatch.setattr(torch.optim.AdamW, "step", original_step)
    s.train_model(second, "cpu")
    actual = torch.load(second / "model/final.pt", weights_only=False, map_location="cpu")
    for key in expected["model_state"]:
        torch.testing.assert_close(actual["model_state"][key], expected["model_state"][key], atol=0, rtol=0)
    for a, b in zip(actual["history"], expected["history"]):
        assert {k: v for k, v in a.items() if k != "epoch_minutes"} == {k: v for k, v in b.items() if k != "epoch_minutes"}
    before = s.sha256_file(second / "model/final.pt")
    s.train_model(second, "cpu")
    assert s.sha256_file(second / "model/final.pt") == before
    # Predictions cannot be reused after byte corruption.
    with (second / "model/validation.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="provenance"):
        s.evaluate_results(second)


def test_input_code_changes_and_foreign_outputs_are_rejected(standalone, tmp_path):
    s = standalone
    inputs, p = prepared_fixture(s, tmp_path)
    label = inputs["labels_root"] / "training_targets.csv"
    original = label.read_bytes()
    label.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Source input changed"):
        s.load_protocol(inputs["run_root"])
    label.write_bytes(original)
    s.RECIPE["encoder_lr"] *= 2
    with pytest.raises(ValueError, match="contract mismatch"):
        s.load_protocol(inputs["run_root"])
    s.RECIPE["encoder_lr"] /= 2
    original_function = s.macro_auc
    s.macro_auc = s.coverage
    with pytest.raises(ValueError, match="contract mismatch"):
        s.load_protocol(inputs["run_root"])
    s.macro_auc = original_function
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "old.pt").write_bytes(b"keep")
    with s.exclusive_run(foreign), pytest.raises(FileExistsError, match="another run"):
        s.claim_run(foreign)
    assert (foreign / "old.pt").read_bytes() == b"keep"


def execution_fixture(tmp_path, standalone):
    inputs = create_input_fixture(tmp_path / "portable-data", standalone)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    settings = next(c for c in nb.cells if c.metadata.get("tags") == ["settings"])
    settings.source = "from pathlib import Path\n" + "\n".join(f"{name} = Path({str(inputs[key])!r})" for name, key in {
        "WORK_ROOT": "work_root", "DATA_ROOT": "data_root", "RUN_ROOT": "run_root", "LABELS_ROOT": "labels_root",
        "SCANNER_SPLIT_ROOT": "scanner_split_root", "SERIES_POLICY": "series_policy"}.items()) + '\nDEVICE = "cpu"'
    preparation = next(i for i, c in enumerate(nb.cells) if c.metadata.get("tags") == ["prepare"])
    nb.cells.insert(preparation, nbformat.v4.new_code_cell(FIXTURE_OVERRIDES))
    return nb, inputs


def assert_execution_outputs(root):
    assert (root / "model/final.pt").is_file()
    assert (root / "model/recovery_latest.pt").is_file()
    assert (root / "reports/metrics.json").is_file()
    assert len(list((root / "figures").glob("*.svg"))) == 5


def test_every_cell_executes_outside_repository_with_repository_imports_blocked(tmp_path, standalone):
    nb, inputs = execution_fixture(tmp_path, standalone)
    detached = tmp_path / "detached"
    detached.mkdir()
    notebook_path = detached / "portable.ipynb"
    nbformat.write(nb, notebook_path)
    script = detached / "execute_notebook.py"
    script.write_text('''
import importlib.abc
import json
import sys
from pathlib import Path
class NoRepository(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "rsna_knee" or fullname.startswith("rsna_knee."):
            raise ModuleNotFoundError("Repository imports are forbidden in this test")
sys.meta_path.insert(0, NoRepository())
from IPython.core.interactiveshell import InteractiveShell
shell = InteractiveShell()
nb = json.loads(Path("portable.ipynb").read_text())
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        shell.run_cell("".join(cell["source"]), store_history=True).raise_error()
assert not any(name.startswith("rsna_knee") for name in sys.modules)
print("ALL STANDALONE CELLS: PASS")
''')
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                                      if p and Path(p).is_absolute() and not str(Path(p).resolve()).startswith(str(REPO)))
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLBACKEND="Agg")
    completed = subprocess.run([sys.executable, str(script)], cwd=detached, env=env,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    assert completed.returncode == 0, completed.stdout
    assert "ALL STANDALONE CELLS: PASS" in completed.stdout
    assert_execution_outputs(inputs["run_root"])


def test_every_cell_executes_in_real_jupyter_kernel(tmp_path, standalone):
    import socket
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
    probe = tmp_path / "socket-probe"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.bind(str(probe))
    except PermissionError:
        pytest.skip("Local kernel sockets are prohibited here; isolated IPython execution is tested separately.")
    finally:
        probe.unlink(missing_ok=True)
    nb, inputs = execution_fixture(tmp_path, standalone)
    class TestSpecs(KernelSpecManager):
        def get_kernel_spec(self, name):
            return KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                              display_name="Synthetic test", language="python")
    class TestKernelManager(KernelManager):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, kernel_spec_manager=TestSpecs(),
                             transport="ipc", ip=str(tmp_path.parent / f"standalone-kernel-{os.getpid()}"))
    NotebookClient(nb, kernel_name="synthetic", kernel_manager_class=TestKernelManager,
                   timeout=180, resources={"metadata": {"path": str(tmp_path)}}).execute()
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type == "code")
    assert_execution_outputs(inputs["run_root"])
