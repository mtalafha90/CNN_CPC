"""Resolve local inputs, launch isolated training processes and read real results."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def _read(path):
    return json.loads(Path(path).read_text())


def _file(path, description):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _choose_gate(project):
    roots = sorted({p.resolve().parent for p in (project / "runs").rglob("b50_selection_split.json")})
    if len(roots) != 1:
        raise ValueError("Set SCANNER_SPLIT_ROOT to your existing frozen scanner split. "
                         f"Found {len(roots)} distinct copies: {roots}. No new split is generated.")
    return roots[0]


def _stop_group(process):
    """Stop the trainer AND its spawned data workers on notebook interruption."""
    for sig, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        # A child can exit before its DataLoader workers. Terminate any workers
        # that still belong to its group rather than leaving an orphan job.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        break
    process.wait(timeout=5)


def stream_process(command, *, cwd, log_path):
    """Live output plus a verbatim disk log; never launch through a shell."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(Path(cwd) / "developments/src") + os.pathsep + env.get("PYTHONPATH", "")
    with log_path.open("a", buffering=1) as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line)
                print(line.replace("[B57/dinov2_slice_candidate]", "[DINOv2]"), end="", flush=True)
            code = process.wait()
            if code:
                raise RuntimeError(f"Stage failed (exit {code}). Full log: {log_path}")
        except BaseException:
            _stop_group(process)
            raise
        finally:
            process.stdout.close()


@contextmanager
def exclusive_run(root):
    """The OS releases this Linux lock on exit, including a crashed kernel."""
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".notebook.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another notebook process is already using this output folder.") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class LocalRun:
    """One DINOv2-only run under the primary checkout's runs/ directory.

    Execution settings can change; the original frozen scientific settings
    cannot. Inputs default to the existing full-data run when it is present.
    """
    def __init__(self, *, project_root, data_root, run_root, labels_root=None,
                 scanner_split_root=None, series_policy=None, num_workers=2):
        self.project_root = Path(project_root).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        self.run_root = Path(run_root).expanduser().resolve()
        self.num_workers = num_workers
        if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
            raise ValueError("NUM_WORKERS must be a nonnegative integer.")
        runs = (self.project_root / "runs").resolve()
        if self.run_root == runs or not self.run_root.is_relative_to(runs):
            raise ValueError("RUN_ROOT must be a dedicated folder inside PROJECT_ROOT/runs.")
        self.geometry_config = _file(self.project_root / "config/b42_constant_area_aspect_sparse.yaml", "geometry config")
        self.training_config = _file(self.project_root / "config/b57_clean_backbone_comparison.json", "training config")
        _file(self.data_root / "train.csv", "original train.csv")
        _file(self.data_root / "train_series.csv", "original train_series.csv")

        previous = self.project_root / "runs/093_Experiment_B57_clean_backbone_comparison/protocol/protocol.json"
        inputs = _read(previous).get("inputs", {}) if previous.is_file() else {}
        def original(name, fallback):
            return Path(inputs[name]["path"]) if name in inputs else fallback
        default_labels = self.project_root / "runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all"
        default_policy = self.project_root / "runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json"
        self.labels_root = Path(labels_root or original("training_targets", default_labels / "training_targets.csv").parent).expanduser().resolve()
        self.series_policy = _file(series_policy or original("series_policy", default_policy), "series policy; set SERIES_POLICY if moved")
        gate = scanner_split_root or (Path(inputs["gate_json"]["path"]).parent if "gate_json" in inputs else _choose_gate(self.project_root))
        self.scanner_split_root = Path(gate).expanduser().resolve()
        for name in ("training_targets.csv", "policy.json", "audit.json"):
            _file(self.labels_root / name, "original label artifact; set LABELS_ROOT if moved")
        for name in ("b50_selection_split.json", "b50_selection_split_by_study.csv", "b50_selection_split.sha256"):
            _file(self.scanner_split_root / name, "frozen scanner split")
        for protected in (self.data_root, self.labels_root, self.scanner_split_root):
            if self.run_root.is_relative_to(protected) or protected.is_relative_to(self.run_root):
                raise ValueError("Output and input artifact folders must be separate.")

    def _identity(self):
        # These adapters live outside the historical flat-module source hash.
        # Record their own identity so a resumed notebook has the same adapter.
        files = (Path(__file__), Path(__file__).with_name("__main__.py"))
        return {"format": "local_dinov2_notebook_v1", "paths": {
            key: str(getattr(self, key)) for key in ("project_root", "data_root", "run_root",
                "labels_root", "scanner_split_root", "series_policy")},
            "adapter_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}

    def claim(self):
        from ..b57_protocol import write_json
        marker = self.run_root / "notebook_session.json"
        expected = self._identity()
        if marker.exists():
            if _read(marker) != expected:
                raise ValueError("Notebook inputs or adapter code changed. Restore the original run settings/code to resume.")
        else:
            occupied = [p for p in self.run_root.iterdir() if p.name not in (".notebook.lock", "logs")]
            if occupied:
                raise FileExistsError("This output folder already contains another run. Choose a new RUN_ROOT.")
            write_json(marker, expected)

    def _stage(self, name):
        command = [sys.executable, "-u", "-m", "rsna_knee.local_dinov2", name]
        for key in ("project_root", "data_root", "run_root", "labels_root", "scanner_split_root", "series_policy", "num_workers"):
            command.extend(("--" + key.replace("_", "-"), str(getattr(self, key))))
        # Claim before the log is opened, so a wrong RUN_ROOT remains untouched.
        with exclusive_run(self.run_root):
            self.claim()
        stream_process(command, cwd=self.project_root, log_path=self.run_root / "logs" / f"{name}.log")

    def check_environment(self):
        self._stage("environment")

    def prepare(self):
        self._stage("prepare")

    def preflight(self):
        self._stage("preflight")

    def train(self):
        self._stage("train")

    def protocol(self):
        from ..b57_protocol import load_protocol
        return load_protocol(self.run_root / "protocol")

    def population(self):
        import pandas as pd
        p = self.protocol()
        labels = {"train": "Training", "validation": "Scanner validation",
                  "excluded_profile_overlap": "Excluded: scanner overlap"}
        rows = [{"Population": labels[k], "Studies": v,
                 "Used for gradients": k == "train"} for k, v in p["counts"].items()]
        rows.append({"Population": "Expert diagnostic", "Studies": len(p["expert_uids"]), "Used for gradients": False})
        return pd.DataFrame(rows).set_index("Population")

    def label_coverage(self):
        import pandas as pd
        return pd.DataFrame.from_dict(self.protocol()["coverage"]["validation"], orient="index").rename(
            columns={"positive": "Positive", "negative": "Negative", "unlabelled": "Unlabelled"})

    def recipe(self):
        import pandas as pd
        c = self.protocol()["config"]
        rows = {"Encoder": "DINOv2 ViT-S/14; all layers trainable", "Seed": c["seed"],
            "Epochs": c["epochs"], "Studies per optimizer step": c["batch_size"],
            "Encoder learning rate": c["encoder_lr"], "Head learning rate": c["head_lr"],
            "Weight decay": c["weight_decay"], "Gradient clip": c["grad_clip"],
            "Encoder images per chunk": c["encoder_chunk_size"],
            "Gradient checkpointing": c["gradient_checkpointing"],
            "Slice feature dimensions": c["candidate_dim"], "Slice transformer layers": c["candidate_slice_layers"],
            "Attention heads": c["candidate_heads"], "Dropout": c["candidate_dropout"],
            "Pixel augmentation": c["augmentation"], "Checkpoint selection": "Fixed final epoch",
            "Data workers": self.num_workers, "Prefetch batches per worker": 1}
        return pd.DataFrame({"Setting": rows.keys(), "Value": [str(v) for v in rows.values()]}).set_index("Setting")

    def training_example(self, study_index=0):
        from ..b57_training import make_dataset
        return make_dataset(self.run_root / "protocol", self.protocol(), "train")[study_index]

    def history(self):
        from ..b57_protocol import ARMS
        path = self.run_root / ARMS[1] / "history.json"
        if not path.exists():
            raise FileNotFoundError("No completed epoch yet. Train through an epoch before plotting history.")
        return _read(path)

    def results(self):
        """Validate single-arm final exports, then export readable diagnostic tables."""
        import numpy as np
        import pandas as pd
        from ..b52_competition_training import macro_auc
        from ..b57_evaluation import align_pair, finite_json, read_predictions
        from ..b57_protocol import ARMS, coverage, sha256_file, write_json
        from ..constants import TARGETS

        p = self.protocol()
        root = self.run_root
        protocol_sha = sha256_file(root / "protocol/protocol.json")
        results, tables, exports = {}, {}, {}
        for split in ("validation", "expert"):
            data, contract = read_predictions(root, ARMS[1], split, protocol_sha)
            uids = p["splits"][split] if split == "validation" else p["expert_uids"]
            data = align_pair(data, data, uids)[0]
            with np.load(root / "protocol/labels.npz", allow_pickle=False) as f:
                if split == "validation":
                    lookup = {u: i for i, u in enumerate(f["uids"].tolist())}
                    ix = [lookup[u] for u in uids]
                    target, weight = f["target"][ix], f["weight"][ix]
                else:
                    lookup = {u: i for i, u in enumerate(f["expert_uids"].tolist())}
                    raw = f["expert_target"][[lookup[u] for u in uids]]
                    target, weight = np.nan_to_num(raw, nan=.5), np.isfinite(raw).astype(np.float32)
            if not np.array_equal(data["target"], target) or not np.array_equal(data["weight"], weight):
                raise ValueError("Predictions disagree with the frozen labels or confidence masks.")
            if contract["epochs"] != p["config"]["epochs"] or contract["source_digest"] != p["source_digest"]:
                raise ValueError("Final checkpoint does not match the frozen training recipe.")
            score = macro_auc(target, weight, data["prediction"])
            if split == "validation" and score["targets_defined"] != len(TARGETS):
                raise ValueError("Scanner validation must define all twelve AUCs.")
            results[split] = {**score, "studies": len(uids), "coverage": coverage(target, weight)}
            tables[split] = pd.DataFrame.from_dict(results[split]["coverage"], orient="index")
            tables[split].insert(0, "AUC", pd.Series(score["per_target_auc"]))
            exports[split] = pd.DataFrame(data["prediction"], columns=TARGETS)
            exports[split].insert(0, "StudyInstanceUID", uids)
        # Publish reports only after BOTH surfaces passed provenance checks.
        out = root / "reports"
        out.mkdir(exist_ok=True)
        report = finite_json({"model": "DINOv2 knee MRI", "protocol_sha256": protocol_sha,
            "epochs": p["config"]["epochs"], "selection": "fixed final epoch", "scores": results,
            "validation_role": "reused scanner development surface", "expert_role": "diagnostic only"})
        write_json(out / "metrics.json", report)
        for split in tables:
            tables[split].to_csv(out / f"{split}_per_target.csv", index_label="Finding")
            exports[split].to_csv(out / f"{split}_probabilities.csv", index=False)
        return report, tables

    def artifacts(self):
        from ..b57_protocol import ARMS
        return {"Model checkpoint": self.run_root / ARMS[1] / "final.pt",
                "Recovery checkpoint": self.run_root / ARMS[1] / "recovery_latest.pt",
                "History": self.run_root / ARMS[1] / "history.json",
                "Metrics": self.run_root / "reports/metrics.json",
                "Figures": self.run_root / "figures", "Logs": self.run_root / "logs"}
