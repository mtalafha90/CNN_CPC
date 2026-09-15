"""Exercise the notebook, real process control and frozen single-arm exports."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

import nbformat
import numpy as np
import pytest

from rsna_knee.b57_protocol import ARMS, VERSION, coverage, digest, sha256_file, source_digest, write_json
from rsna_knee.b57_training import save_predictions
from rsna_knee.local_dinov2.session import LocalRun, exclusive_run, stream_process

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebook/dinov2_knee_mri_local.ipynb"


def make_run(tmp_path, *, workers=2):
    project = tmp_path / "project"
    data, labels, gate = (project / "data", project / "runs/labels", project / "runs/gate")
    for path in (data, labels, gate, project / "config"):
        path.mkdir(parents=True)
    for name in ("b42_constant_area_aspect_sparse.yaml", "b57_clean_backbone_comparison.json"):
        shutil.copyfile(REPO / "config" / name, project / "config" / name)
    for name in ("train.csv", "train_series.csv"):
        (data / name).write_text("StudyInstanceUID\nsynthetic\n")
    for name in ("training_targets.csv", "policy.json", "audit.json"):
        (labels / name).write_text("{}")
    for name in ("b50_selection_split.json", "b50_selection_split_by_study.csv", "b50_selection_split.sha256"):
        (gate / name).write_text("{}")
    policy = project / "runs/series_policy.json"
    policy.write_text("{}")
    return LocalRun(project_root=project, data_root=data, run_root=project / "runs/dinov2_local",
        labels_root=labels, scanner_split_root=gate, series_policy=policy, num_workers=workers)


def complete_synthetic_run(run):
    """Real hashed output files with synthetic labels/scores; no claimed MRI run."""
    import yaml
    with exclusive_run(run.run_root):
        run.claim()
    uids = [f"t{i}" for i in range(4)] + [f"v{i}" for i in range(4)]
    experts = [f"e{i}" for i in range(4)]
    y = np.tile(np.arange(8) % 2, (12, 1)).T.astype(np.float32)
    w = np.ones_like(y)
    expert_y = y[:4].copy()
    expert_y[:, -1] = np.nan  # Undefined target AUC must be reported, not fabricated.
    out = run.run_root / "protocol"
    out.mkdir()
    np.savez_compressed(out / "labels.npz", uids=np.asarray(uids), target=y, weight=w,
                        expert_uids=np.asarray(experts), expert_target=expert_y)
    write_json(out / "series_index.json", {u: [{"series_uid": "s"}] for u in uids + experts})
    splits = {"train": uids[:4], "validation": uids[4:], "excluded_profile_overlap": []}
    config = json.loads(run.training_config.read_text())
    p = {"version": VERSION, "config": config, "b42_config": yaml.safe_load(run.geometry_config.read_text()),
        "source_digest": source_digest(), "competition_supervised_ancestor": False,
        "gold_studies_in_gradient": 0, "splits": splits, "expert_uids": experts,
        "counts": {k: len(v) for k, v in splits.items()},
        "split_uid_hashes": {k: digest(v) for k, v in splits.items()},
        "coverage": {"validation": coverage(y[4:], w[4:])}, "data_root": str(run.data_root),
        "inputs": {"training_config": {"path": str(run.training_config), "sha256": sha256_file(run.training_config)}},
        "labels_sha256": sha256_file(out / "labels.npz"), "series_index_sha256": sha256_file(out / "series_index.json")}
    write_json(out / "protocol.json", p)
    (out / "protocol.sha256").write_text(sha256_file(out / "protocol.json") + "\n")
    arm = run.run_root / ARMS[1]
    arm.mkdir()
    (arm / "final.pt").write_bytes(b"synthetic fixture: no trained model")
    contract = {"arm": ARMS[1], "protocol_sha256": sha256_file(out / "protocol.json"),
                "competition_supervised_ancestor": False, "gold_studies_in_gradient": 0,
                "source_digest": p["source_digest"], "epochs": config["epochs"]}
    checkpoint_sha = sha256_file(arm / "final.pt")
    for split, names, target, weight in (("validation", uids[4:], y[4:], w[4:]),
            ("expert", experts, np.nan_to_num(expert_y, nan=.5), np.isfinite(expert_y).astype(np.float32))):
        pred = {"uids": np.asarray(names), "target": target, "weight": weight, "prediction": .1 + .8 * target}
        # Intentionally reverse export order: the reporter must align by UID.
        save_predictions(arm / f"{split}.npz", {k: v[::-1] for k, v in pred.items()},
                         contract=contract, split=split, checkpoint_sha=checkpoint_sha)
    write_json(arm / "complete.json", {"contract": contract, "checkpoint_sha256": checkpoint_sha,
                                      "completed_epochs": config["epochs"]})
    write_json(arm / "history.json", [{"epoch": i, "train_loss": .8 / i, "epoch_minutes": 1.,
                                      "validation": {"macro_auc": .5 + .04 * i}} for i in range(1, 13)])
    return p


def test_notebook_is_generated_clean_valid_and_has_no_duplicate_model():
    spec = importlib.util.spec_from_file_location("local_builder", NOTEBOOK.with_name("build_dinov2_local_notebook.py"))
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert json.loads(NOTEBOOK.read_text()) == builder.notebook()
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    all_source = "\n".join(c.source for c in nb.cells)
    assert not re.search(r"\b[Bb]\d{2,}\b|[Bb]\d{2,}_|Experiment[_ -]", all_source)
    for c in nb.cells:
        if c.cell_type == "code":
            assert c.execution_count is None and c.outputs == []
            tree = ast.parse(c.source)
            assert not any(isinstance(n, (ast.ClassDef, ast.FunctionDef)) for n in ast.walk(tree))
    assert nb.metadata.kernelspec.name == "dinov2-local"


def test_output_ownership_resume_and_lock(tmp_path):
    run = make_run(tmp_path)
    with exclusive_run(run.run_root):
        run.claim()
        with pytest.raises(RuntimeError, match="already using"):
            with exclusive_run(run.run_root):
                pass
    before = (run.run_root / "notebook_session.json").read_bytes()
    run.num_workers = 0
    with exclusive_run(run.run_root):
        run.claim()
    assert (run.run_root / "notebook_session.json").read_bytes() == before
    run.data_root = run.data_root / "different"
    with pytest.raises(ValueError, match="inputs or adapter"):
        run.claim()
    foreign = run.project_root / "runs/foreign"
    foreign.mkdir()
    (foreign / "final.pt").write_bytes(b"preserve existing training output")
    run.run_root = foreign
    with exclusive_run(foreign), pytest.raises(FileExistsError, match="another run"):
        run.claim()
    assert (foreign / "final.pt").read_bytes() == b"preserve existing training output"


def test_constructor_refuses_missing_data_bad_workers_and_root_output(tmp_path):
    run = make_run(tmp_path)
    kwargs = dict(project_root=run.project_root, data_root=run.data_root, run_root=run.run_root,
                  labels_root=run.labels_root, scanner_split_root=run.scanner_split_root, series_policy=run.series_policy)
    with pytest.raises(ValueError, match="nonnegative"):
        LocalRun(**kwargs, num_workers=-1)
    with pytest.raises(ValueError, match="dedicated folder"):
        LocalRun(**{**kwargs, "run_root": run.project_root / "runs"})
    (run.data_root / "train.csv").unlink()
    with pytest.raises(FileNotFoundError, match="original train.csv"):
        LocalRun(**kwargs)


def test_previous_protocol_supplies_exact_existing_inputs_without_changing_it(tmp_path):
    run = make_run(tmp_path)
    original = run.project_root / "runs/093_Experiment_B57_clean_backbone_comparison/protocol/protocol.json"
    write_json(original, {"inputs": {"training_targets": {"path": str(run.labels_root / "training_targets.csv")},
              "gate_json": {"path": str(run.scanner_split_root / "b50_selection_split.json")},
              "series_policy": {"path": str(run.series_policy)}}})
    before = original.read_bytes()
    discovered = LocalRun(project_root=run.project_root, data_root=run.data_root, run_root=run.run_root)
    assert discovered.labels_root == run.labels_root
    assert discovered.scanner_split_root == run.scanner_split_root
    assert discovered.series_policy == run.series_policy
    assert original.read_bytes() == before


@pytest.mark.parametrize("layout", ["train_images", "train_series", ""])
def test_prepare_uses_actual_series_resolver_and_rejects_partial_images(tmp_path, monkeypatch, layout):
    from rsna_knee import b57_protocol as protocol
    from rsna_knee.local_dinov2 import __main__ as runner
    run = make_run(tmp_path)
    calls = []
    p = {"splits": {"train": ["t"], "validation": ["v"]}, "expert_uids": ["e"],
         "counts": {"train": 1, "validation": 1}}
    def prepare(**kwargs):
        assert kwargs["b42_config"] == run.geometry_config
        assert kwargs["b57_config"] == run.training_config
        assert kwargs["domain_split"] == run.scanner_split_root
        write_json(kwargs["out_root"] / "series_index.json", {u: [{"series_uid": "s"}] for u in ("t", "v", "e")})
        return p
    monkeypatch.setattr(protocol, "prepare_protocol", prepare)
    monkeypatch.setattr(runner, "prepare_dino_weights", lambda root: calls.append(root))
    for uid in ("t", "v", "e"):
        path = run.data_root / layout / uid / "s"
        path.mkdir(parents=True)
        (path / "image.IMA").write_bytes(b"directory-completeness fixture, not a real DICOM")
    runner.run_stage(run, "prepare")
    assert calls == [run.run_root / "public_init"]
    (run.data_root / layout / "e/s/image.IMA").unlink()
    with pytest.raises(FileNotFoundError, match="e/s"):
        runner.run_stage(run, "prepare")
    assert len(calls) == 1


def test_only_candidate_is_initialized_and_existing_public_hashes_are_checked(tmp_path, monkeypatch):
    import torch
    from rsna_knee import b57_models as models
    from rsna_knee.local_dinov2.__main__ import prepare_dino_weights
    encoder = torch.nn.Linear(3, 4)
    seen = []
    def build(*, pretrained):
        seen.append(pretrained)
        return encoder
    monkeypatch.setattr(models, "dino_backbone", build)
    monkeypatch.setitem(models.PUBLIC_STATE_SHA256, ARMS[1], models.state_digest(encoder.state_dict()))
    root = tmp_path / "public_init"
    prepare_dino_weights(root)
    prepare_dino_weights(root)
    assert seen == [True]
    assert {p.name for p in root.iterdir()} == {f"{ARMS[1]}.pt", f"{ARMS[1]}.json"}
    with (root / f"{ARMS[1]}.pt").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_dino_weights(root)


def test_preflight_and_training_delegate_only_the_same_candidate(tmp_path, monkeypatch, capsys):
    from rsna_knee import b57_training as training
    from rsna_knee.local_dinov2.__main__ import run_stage
    run = make_run(tmp_path, workers=3)
    calls = []
    def preflight(**kwargs):
        calls.append(("preflight", kwargs))
        print("internal full protocol detail")
        return {"passed": True, "optimizer_steps": 0, "gradient_tensor_counts": {"encoder": 1}, "peak_cuda_gib": 1.}
    monkeypatch.setattr(training, "preflight", preflight)
    monkeypatch.setattr(training, "train_arm", lambda **k: calls.append(("train", k)))
    run_stage(run, "preflight")
    run_stage(run, "train")
    assert calls == [(stage, dict(run_root=run.run_root, arm=ARMS[1], device="cuda", workers=3))
                     for stage in ("preflight", "train")]
    assert "internal full protocol detail" not in capsys.readouterr().out
    assert "internal full protocol detail" in (run.run_root / "logs/preflight_details.log").read_text()


def test_streams_process_output_preserves_raw_log_and_propagates_failure(tmp_path, capsys):
    log = tmp_path / "stage.log"
    script = "import sys; print('[B57/dinov2_slice_candidate] E1'); print('failure detail', file=sys.stderr); sys.exit(7)"
    with pytest.raises(RuntimeError, match="exit 7"):
        stream_process([sys.executable, "-u", "-c", script], cwd=REPO, log_path=log)
    assert "[DINOv2] E1" in capsys.readouterr().out
    assert "[B57/dinov2_slice_candidate] E1" in log.read_text()
    assert "failure detail" in log.read_text()


def test_interrupt_stops_real_trainer_and_worker_group(tmp_path, monkeypatch):
    import rsna_knee.local_dinov2.session as session
    marker = tmp_path / "worker_stopped"
    worker = "\n".join(["import signal, pathlib, sys",
        f"marker = pathlib.Path({str(marker)!r})",
        "def stop(*args):", "    marker.write_text('stopped')", "    sys.exit(0)",
        "signal.signal(signal.SIGINT, stop)", "signal.signal(signal.SIGTERM, stop)",
        "print('worker ready', flush=True)", "signal.pause()"])
    parent = "\n".join(["import signal, subprocess, sys",
        f"worker = subprocess.Popen([sys.executable, '-u', '-c', {worker!r}], stdout=subprocess.PIPE, text=True)",
        "def stop(*args):", "    worker.wait(timeout=3)", "    sys.exit(0)",
        "signal.signal(signal.SIGINT, stop)", "signal.signal(signal.SIGTERM, stop)",
        "worker.stdout.readline()", "print('READY', flush=True)", "signal.pause()"])
    def interrupt(line, **kwargs):
        if "READY" in line:
            raise KeyboardInterrupt
    monkeypatch.setattr(session, "print", interrupt, raising=False)
    with pytest.raises(KeyboardInterrupt):
        stream_process([sys.executable, "-u", "-c", parent], cwd=REPO, log_path=tmp_path / "interrupt.log")
    assert marker.read_text() == "stopped"


def test_report_aligns_full_labels_and_exposes_undefined_expert_targets(tmp_path):
    run = make_run(tmp_path)
    complete_synthetic_run(run)
    report, tables = run.results()
    assert report["scores"]["validation"]["macro_auc"] == 1.
    assert report["scores"]["validation"]["targets_defined"] == 12
    assert report["scores"]["expert"]["targets_defined"] == 11
    assert np.isnan(tables["expert"].loc["Fracture", "AUC"])
    assert (run.run_root / "reports/validation_probabilities.csv").read_text().splitlines()[1].startswith("v0,")
    assert json.loads((run.run_root / "reports/metrics.json").read_text())["scores"]["expert"]["per_target_auc"]["Fracture"] is None


@pytest.mark.parametrize("damage", ["checkpoint", "prediction_bytes", "wrong_labels", "wrong_uids"])
def test_report_refuses_changed_final_artifacts(tmp_path, damage):
    run = make_run(tmp_path)
    complete_synthetic_run(run)
    arm = run.run_root / ARMS[1]
    if damage == "checkpoint":
        (arm / "final.pt").write_bytes(b"changed model")
    elif damage == "prediction_bytes":
        with (arm / "validation.npz").open("ab") as f:
            f.write(b"changed")
    else:
        path = arm / "validation.npz"
        with np.load(path) as f:
            data = {k: f[k].copy() for k in f.files}
        if damage == "wrong_labels":
            data["weight"][0, 0] = 0
        else:
            data["uids"][0] = "xx"
        meta = json.loads(path.with_suffix(".json").read_text())
        save_predictions(path, data, contract=meta["contract"], split="validation", checkpoint_sha=meta["checkpoint_sha256"])
    with pytest.raises(ValueError):
        run.results()
    assert not (run.run_root / "reports").exists()


def execution_fixture(tmp_path):
    """Substitute only unavailable data/GPU operations in a copy of the notebook."""
    run = make_run(tmp_path)
    complete_synthetic_run(run)
    (run.project_root / "developments").mkdir()
    (run.project_root / "developments/src").symlink_to(REPO / "developments/src", target_is_directory=True)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    settings = next(c for c in nb.cells if c.cell_type == "code")
    settings.source = settings.source.replace('Path("/media/talafha/Disk_1/CNN_CPC")', f"Path({str(run.project_root)!r})")
    settings.source += "\n" + "\n".join(f"{key} = Path({str(value)!r})" for key, value in {
        "DATA_ROOT": run.data_root, "RUN_ROOT": run.run_root, "LABELS_ROOT": run.labels_root,
        "SCANNER_SPLIT_ROOT": run.scanner_split_root, "SERIES_POLICY": run.series_policy}.items())
    imports = next(c for c in nb.cells if c.cell_type == "code" and "run = LocalRun" in c.source)
    synthetic = """
import torch
def synthetic_stage(self, name):
    print('CPU-only notebook test: synthetic ' + name)
def synthetic_example(self, study_index=0):
    return {'volumes': [torch.linspace(0, 1, 32*3*32*64).reshape(32,3,32,64)],
            'slice_position': torch.linspace(0,1,32)[None],
            'geometry': [{'height': 32, 'width': 64, 'present': True}]}
LocalRun._stage = synthetic_stage
LocalRun.training_example = synthetic_example
"""
    imports.source = imports.source.replace("run = LocalRun", synthetic + "\nrun = LocalRun")
    return nb, run


def check_execution_outputs(run):
    assert len(list((run.run_root / "figures").glob("*.svg"))) == 5
    assert (run.run_root / "reports/metrics.json").is_file()


def test_all_notebook_cells_execute_in_ipython_with_synthetic_data(tmp_path):
    """All cells execute even on machines where local kernel sockets are blocked."""
    from IPython.core.interactiveshell import InteractiveShell
    nb, run = execution_fixture(tmp_path)
    shell = InteractiveShell()
    original_stage, original_example = LocalRun._stage, LocalRun.training_example
    original_path = sys.path.copy()
    try:
        for cell in nb.cells:
            if cell.cell_type == "code":
                result = shell.run_cell(cell.source)
                result.raise_error()
        check_execution_outputs(run)
    finally:
        LocalRun._stage, LocalRun.training_example = original_stage, original_example
        sys.path[:] = original_path


def test_all_notebook_cells_execute_in_a_real_kernel_with_synthetic_data(tmp_path):
    """CI exercises Jupyter messaging as well as executing the same user cells."""
    import socket
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
    probe = tmp_path / "socket-probe"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.bind(str(probe))
    except PermissionError:
        pytest.skip("This sandbox prohibits local kernel sockets; all cells are tested through IPython separately.")
    finally:
        probe.unlink(missing_ok=True)
    nb, run = execution_fixture(tmp_path)
    # Use this exact test interpreter, not a system kernelspec pointing at a
    # Python without the test dependencies. Production uses its installed kernel.
    class TestSpecs(KernelSpecManager):
        def get_kernel_spec(self, name):
            return KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                              display_name="Synthetic test", language="python")
    # Local IPC also works in test sandboxes that prohibit TCP listeners.
    class TestKernelManager(KernelManager):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, kernel_spec_manager=TestSpecs(),
                             transport="ipc", ip=str(tmp_path.parent / f"kernel-{os.getpid()}"))
    client = NotebookClient(nb, kernel_name="synthetic", kernel_manager_class=TestKernelManager,
                            timeout=120, resources={"metadata": {"path": str(REPO)}})
    client.execute()
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    check_execution_outputs(run)
