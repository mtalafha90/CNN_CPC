"""Kaggle full training-data audit template.

Run this separately before expensive model training. It uses CPU multiprocessing
and writes teacher/fold/stream statistics plus a full selected-series decode
report. The audit itself obeys the same <9 h budget guard.
"""
from pathlib import Path
import subprocess

import yaml

CODE_ROOT = Path("/kaggle/input/<your-code-dataset>/CNN_CPC")
DATA_ROOT = "/kaggle/input/rsna-knee-abnormality-detection"
CONFIG_RUN = Path("/kaggle/working/train.yaml")

config = yaml.safe_load((CODE_ROOT / "configs/train.yaml").read_text())
config["data_root"] = DATA_ROOT
config["competition_mode"] = True
config["requested_gpus"] = 1
config["runtime_budget_hours"] = 8.5
config["pretrained"] = False
config["allow_external_pretrained"] = False
CONFIG_RUN.write_text(yaml.safe_dump(config, sort_keys=False))

subprocess.run(
    [
        "rsna-knee",
        "audit",
        "--config",
        str(CONFIG_RUN),
        "--out-dir",
        "/kaggle/working/runs/audit",
    ],
    check=True,
)
print("Audit: /kaggle/working/runs/audit/audit.json")
