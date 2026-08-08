"""Kaggle one-GPU training template.

Train exactly ONE fold per committed notebook run. The attached repository is
executed directly via PYTHONPATH, so the committed run does not need pip/network
installation.
"""
from pathlib import Path
import os
import subprocess
import sys

import yaml

CODE_ROOT = Path("/kaggle/input/<your-code-dataset>/CNN_CPC")
DATA_ROOT = "/kaggle/input/rsna-knee-abnormality-detection"
CONFIG_SRC = CODE_ROOT / "configs/train.yaml"
CONFIG_RUN = Path("/kaggle/working/train.yaml")

FOLD = 0  # edit to 0, 1, or 2 in separate notebook copies
STAGE1_MODEL_ROOT = None  # Stage 2: "/kaggle/input/<stage1-model-dataset>/runs/model"
SSL_CHECKPOINT = None     # optional checkpoint produced by pretrain_template.py
SMOKE = False

stage_name = "cotrain" if STAGE1_MODEL_ROOT else "model"
output_root = f"/kaggle/working/runs/{stage_name}"

config = yaml.safe_load(CONFIG_SRC.read_text())
config["data_root"] = DATA_ROOT
config["output_dir"] = output_root
config["competition_mode"] = True
config["requested_gpus"] = 1
config["runtime_budget_hours"] = 8.5
config["pretrained"] = False
config["allow_external_pretrained"] = False
config["cotrain_stage1_root"] = STAGE1_MODEL_ROOT
config["ssl_encoder_checkpoint"] = SSL_CHECKPOINT
config["ssl_checkpoint_source"] = "competition_training_data" if SSL_CHECKPOINT else None
CONFIG_RUN.write_text(yaml.safe_dump(config, sort_keys=False))

env = os.environ.copy()
env["PYTHONPATH"] = str(CODE_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
command = [sys.executable, "-m", "rsna_knee.cli", "train", "--config", str(CONFIG_RUN), "--fold", str(FOLD)]
if SMOKE:
    command.append("--smoke")
subprocess.run(command, check=True, env=env)

print(f"Completed {stage_name} fold {FOLD}: {output_root}/fold{FOLD}")
