"""The B48 two-arm runner must preserve its matched-pair and B46 guards."""
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

RUNNER = Path("developments/scripts/run_b48_domain_pair.sh").resolve()


@dataclass
class Workspace:
    root: Path
    env: dict[str, str]

    def invoke(self):
        return subprocess.run(
            ["bash", str(RUNNER)],
            capture_output=True,
            text=True,
            env=self.env,
            cwd=str(Path.cwd()),
        )

    def trained(self) -> list[str]:
        path = self.root / "b48" / "seed_2026" / "trained.log"
        return path.read_text().split() if path.exists() else []

    def preflighted(self) -> list[str]:
        path = self.root / "b48" / "seed_2026" / "preflight.log"
        return path.read_text().split() if path.exists() else []


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    if shutil.which("bash") is None:  # pragma: no cover - expected in CI
        pytest.skip("bash is not available")
    b46 = tmp_path / "b46"
    for fold in range(5):
        path = b46 / f"fold_{fold}"
        path.mkdir(parents=True)
        (path / f"b46_fold{fold}_model.pt").write_text("completed")

    domain = tmp_path / "domain"
    domain.mkdir()
    payload = domain / "domain_split.json"
    payload.write_text(json.dumps({"version": "official_scanner_domain_split_v1"}))
    (domain / "domain_split_by_study.csv").write_text(
        "StudyInstanceUID,scanner_profile,holdout,split\n"
    )
    (domain / "domain_split.sha256").write_text(
        hashlib.sha256(payload.read_bytes()).hexdigest() + "\n"
    )

    fake = tmp_path / "fake_trainer.py"
    fake.write_text(
        "import pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "def value(name):\n"
        "  return args[args.index(name)+1]\n"
        "arm=value('--arm'); root=pathlib.Path(value('--out-root'))\n"
        "pair=root.parent\n"
        "pair.mkdir(parents=True, exist_ok=True)\n"
        "if '--preflight-only' in args:\n"
        "  (pair/'preflight.log').open('a').write(arm+'\\n')\n"
        "else:\n"
        "  (pair/'trained.log').open('a').write(arm+'\\n')\n"
        "  root.mkdir(parents=True, exist_ok=True)\n"
        "  (root/f'b48_{arm}_model.pt').write_text('checkpoint '+arm)\n"
    )
    python_bin = tmp_path / "python_bin"
    python_bin.write_text(
        "#!/usr/bin/env bash\n"
        "shift 2\n"
        f'exec "{sys.executable}" "{fake}" "$@"\n'
    )
    python_bin.chmod(0o755)
    root = tmp_path / "root"
    env = {
        **os.environ,
        "DATA_ROOT": str(tmp_path / "data"),
        "LABELS_ROOT": str(tmp_path / "labels"),
        "SERIES_POLICY": str(tmp_path / "series.json"),
        "BASE_CHECKPOINT": str(tmp_path / "base.pt"),
        "B46_ROOT": str(b46),
        "DOMAIN_SPLIT_ROOT": str(domain),
        "B48_ROOT": str(root / "b48"),
        "PYTHON_BIN": str(python_bin),
    }
    return Workspace(root=root, env=env)


def test_clean_runner_preflights_both_arms_then_trains_the_matched_pair(workspace):
    result = workspace.invoke()
    assert result.returncode == 0, result.stderr
    assert workspace.preflighted() == [
        "static_prior_control",
        "post_cross_attention_candidate",
    ]
    assert workspace.trained() == [
        "static_prior_control",
        "post_cross_attention_candidate",
    ]
    assert "matched pair: COMPLETE" in result.stdout


def test_completed_pair_is_not_preflighted_or_overwritten_again(workspace):
    assert workspace.invoke().returncode == 0
    before = {
        path: path.read_bytes()
        for path in (workspace.root / "b48").rglob("b48_*_model.pt")
    }
    second = workspace.invoke()
    assert second.returncode == 0, second.stderr
    assert workspace.trained() == ["static_prior_control", "post_cross_attention_candidate"]
    assert workspace.preflighted() == ["static_prior_control", "post_cross_attention_candidate"]
    assert "already complete" in second.stdout
    assert {path: path.read_bytes() for path in before} == before


def test_runner_refuses_to_begin_until_every_b46_fold_checkpoint_exists(workspace):
    (Path(workspace.env["B46_ROOT"]) / "fold_4" / "b46_fold4_model.pt").unlink()
    result = workspace.invoke()
    assert result.returncode == 4
    assert "B46 fold 4 is not complete" in result.stderr
    assert workspace.preflighted() == []
    assert workspace.trained() == []


def test_runner_refuses_a_tampered_domain_split_before_preflight(workspace):
    payload = Path(workspace.env["DOMAIN_SPLIT_ROOT"]) / "domain_split.json"
    payload.write_text("tampered")
    result = workspace.invoke()
    assert result.returncode == 3
    assert "domain_split.sha256" in result.stderr
    assert workspace.preflighted() == []
