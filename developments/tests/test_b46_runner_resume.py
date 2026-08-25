"""The five-fold B46 runner can be re-invoked after an interruption.

Five folds train one after another on one GPU, and the machine they run on has
a session limit shorter than the whole sequence is likely to need. If the run
dies part-way, the operator has to be able to start it again and have it carry
on -- without retraining or overwriting any fold that already finished.

These tests run the real script with a stand-in trainer, so they check what the
script does rather than what it says.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

RUNNER = Path("developments/scripts/run_b46_all_folds.sh").resolve()

# The script refuses to start unless the manifest hashes to this. It is the
# frozen B46 fold identity, so the fixture below must reproduce it byte for byte
# rather than inventing its own manifest.
FROZEN_MANIFEST_SHA = "054c4ce9ab808af714cd4b86f159ef02a2b7e67de0c80e5c930d29fa5fb22e03"


def _frozen_manifest_sha_from_script() -> str:
    for line in RUNNER.read_text().splitlines():
        if line.startswith("EXPECTED_MANIFEST_SHA="):
            return line.split("=", 1)[1].strip().strip('"')
    raise AssertionError("the runner no longer declares a manifest SHA")


@dataclass
class Workspace:
    """The pieces of a B46 run, with the trainer replaced by a stand-in."""

    root: Path
    _invoke: object
    _folds_trained: object

    def invoke(self):
        return self._invoke()

    def folds_trained(self):
        return self._folds_trained()


@pytest.fixture
def workspace(tmp_path):
    """A run root, a manifest that hashes correctly, and a fake trainer."""
    if shutil.which("bash") is None:  # pragma: no cover - bash is expected
        pytest.skip("bash is not available")

    manifest = tmp_path / "gold_folds.json"
    # Search for content hashing to the frozen SHA is impossible, so the test
    # rewrites the expectation instead: it copies the script and substitutes the
    # SHA of the manifest it actually built. Every other line is the real one.
    manifest.write_text(json.dumps({"n_folds": 5, "note": "test fixture"}))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    script = tmp_path / "run_b46_all_folds.sh"
    script.write_text(
        RUNNER.read_text().replace(_frozen_manifest_sha_from_script(), digest)
    )
    script.chmod(0o755)

    # Stands in for `python -m rsna_knee.b46_gold_crossfit_training`: writes the
    # checkpoint the fold is expected to produce, and records that it ran.
    trainer = tmp_path / "fake_trainer.py"
    trainer.write_text(
        "import sys, pathlib\n"
        "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "fold = args['--fold']\n"
        "root = pathlib.Path(args['--out-root'])\n"
        "(root / 'trained.log').open('a').write(fold + '\\n')\n"
        "if pathlib.Path(root / 'fail_fold').exists():\n"
        "    if (root / 'fail_fold').read_text().strip() == fold:\n"
        "        sys.exit(9)\n"
        "d = root / f'fold_{fold}'\n"
        "d.mkdir(parents=True, exist_ok=True)\n"
        "(d / f'b46_fold{fold}_model.pt').write_text('checkpoint ' + fold)\n"
    )

    python_bin = tmp_path / "python_bin"
    python_bin.write_text(
        "#!/usr/bin/env bash\n"
        'shift 2  # drop "-m rsna_knee.b46_gold_crossfit_training"\n'
        f'exec "{sys.executable}" "{trainer}" "$@"\n'
    )
    python_bin.chmod(0o755)

    run_root = tmp_path / "b46"
    run_root.mkdir()

    def invoke():
        environment = dict(os.environ)
        environment.update(
            DATA_ROOT=str(tmp_path / "data"),
            LABELS_ROOT=str(tmp_path / "labels"),
            SERIES_POLICY=str(tmp_path / "policy.json"),
            BASE_CHECKPOINT=str(tmp_path / "base.pt"),
            B46_ROOT=str(run_root),
            B46_MANIFEST=str(manifest),
            PYTHON_BIN=str(python_bin),
        )
        return subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            env=environment,
            cwd=str(Path.cwd()),
        )

    def folds_trained():
        log = run_root / "trained.log"
        return log.read_text().split() if log.exists() else []

    return Workspace(root=run_root, _invoke=invoke, _folds_trained=folds_trained)


def test_the_frozen_manifest_sha_is_still_the_one_that_was_recorded():
    """The fixture rewrites this value, so the real one is checked separately."""
    assert _frozen_manifest_sha_from_script() == FROZEN_MANIFEST_SHA


def test_a_clean_run_trains_all_five_folds(workspace):
    result = workspace.invoke()
    assert result.returncode == 0, result.stderr
    assert workspace.folds_trained() == ["0", "1", "2", "3", "4"]
    assert "ALL B46 FIXED-E2 FOLDS: COMPLETE" in result.stdout


def test_an_interrupted_run_carries_on_from_where_it_stopped(workspace):
    """The behaviour the session limit makes necessary."""
    (workspace.root / "fail_fold").write_text("2")
    first = workspace.invoke()
    assert first.returncode == 9
    assert workspace.folds_trained() == ["0", "1", "2"]

    (workspace.root / "fail_fold").unlink()
    second = workspace.invoke()
    assert second.returncode == 0, second.stderr

    # Folds 0 and 1 were finished already and must not have been trained again.
    assert workspace.folds_trained() == ["0", "1", "2", "2", "3", "4"]
    assert "FOLD 0 ALREADY COMPLETE" in second.stdout
    assert "FOLD 1 ALREADY COMPLETE" in second.stdout
    assert "trained on this invocation: 2 3 4" in second.stdout
    assert "already complete beforehand: 0 1" in second.stdout


def test_a_finished_checkpoint_is_never_rewritten(workspace):
    assert workspace.invoke().returncode == 0
    before = {
        path: path.read_bytes() for path in workspace.root.rglob("b46_fold*_model.pt")
    }
    assert len(before) == 5

    second = workspace.invoke()
    assert second.returncode == 0, second.stderr
    assert workspace.folds_trained() == ["0", "1", "2", "3", "4"], "nothing retrained"
    for path, content in before.items():
        assert path.read_bytes() == content


def test_an_empty_checkpoint_stops_the_run_instead_of_being_guessed_about(workspace):
    """A save cut off part-way is ambiguous, and deleting it is not ours to do."""
    fold_root = workspace.root / "fold_0"
    fold_root.mkdir()
    (fold_root / "b46_fold0_model.pt").write_text("")

    result = workspace.invoke()
    assert result.returncode == 4
    assert "exists but is empty" in result.stderr
    assert workspace.folds_trained() == [], "no fold ran"


def test_the_log_of_a_failed_attempt_survives_the_retry(workspace):
    (workspace.root / "fail_fold").write_text("0")
    assert workspace.invoke().returncode == 9
    (workspace.root / "fail_fold").unlink()
    assert workspace.invoke().returncode == 0

    log = (workspace.root / "fold_0" / "training.log").read_text()
    assert log.count("attempt started") == 2


def test_a_manifest_that_does_not_match_is_still_refused(workspace):
    """Resuming must not have loosened the identity check."""
    result = subprocess.run(
        ["bash", str(RUNNER)],  # the real script, against the fixture manifest
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DATA_ROOT": "d",
            "LABELS_ROOT": "l",
            "SERIES_POLICY": "p",
            "BASE_CHECKPOINT": "b",
            "B46_ROOT": str(workspace.root),
            "B46_MANIFEST": str(workspace.root.parent / "gold_folds.json"),
        },
    )
    assert result.returncode == 3
    assert "manifest SHA mismatch" in result.stderr
