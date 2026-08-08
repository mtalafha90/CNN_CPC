"""Kaggle one-GPU in-domain SSL template.

Run this as its own committed notebook execution. The attached repository is
executed directly via PYTHONPATH; no pip/network installation is required.
"""
from pathlib import Path
import os
import subprocess
import sys

import yaml

CODE_ROOT = Path("/kaggle/input/<your-code-dataset>/CNN_CPC")
DATA_ROOT = "/kaggle/input/rsna-knee-abnormality-detection"
CONFIG_RUN = Path("/kaggle/working/train.yaml")

config = yaml.safe_load((CODE_ROOT / "configs/train.yaml").read_text())
config["data_root"] = DATA_ROOT
config["ssl_output_dir"] = "/kaggle/working/runs/ssl"
config["competition_mode"] = True
config["requested_gpus"] = 1
config["runtime_budget_hours"] = 8.5
config["pretrained"] = False
config["allow_external_pretrained"] = False
CONFIG_RUN.write_text(yaml.safe_dump(config, sort_keys=False))

env = os.environ.copy()
env["PYTHONPATH"] = str(CODE_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
subprocess.run(
    [sys.executable, "-m", "rsna_knee.cli", "pretrain", "--config", str(CONFIG_RUN)],
    check=True,
    env=env,
)
print("SSL checkpoint: /kaggle/working/runs/ssl/ssl_encoder.pt")
