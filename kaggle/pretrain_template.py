"""Kaggle one-GPU in-domain SSL template.

Run this as its own committed notebook execution. The production budget guard
keeps the job below 9 h and saves ``ssl_encoder.pt`` after the last completed
epoch.
"""
from pathlib import Path
import subprocess

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

subprocess.run(["rsna-knee", "pretrain", "--config", str(CONFIG_RUN)], check=True)
print("SSL checkpoint: /kaggle/working/runs/ssl/ssl_encoder.pt")
