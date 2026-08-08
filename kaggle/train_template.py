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
# Stage 2 option A: one already-frozen Stage-1 method.
STAGE1_MODEL_ROOT = None
# Stage 2 option B (recommended for random-vs-SSL): attach both Stage-1 datasets.
# Each outer fold is chosen using only that fold's inner_macro_auc.
STAGE1_CANDIDATE_ROOTS = None
# Example:
# STAGE1_CANDIDATE_ROOTS = [
#     "/kaggle/input/stage1-random/runs/stage1_random",
#     "/kaggle/input/stage1-ssl/runs/stage1_ssl",
# ]
SSL_CHECKPOINT = None     # optional checkpoint produced by pretrain_template.py
SMOKE = False

if STAGE1_MODEL_ROOT and STAGE1_CANDIDATE_ROOTS:
    raise ValueError("use STAGE1_MODEL_ROOT or STAGE1_CANDIDATE_ROOTS, not both")
stage2 = bool(STAGE1_MODEL_ROOT or STAGE1_CANDIDATE_ROOTS)
stage_name = "stage2" if stage2 else "stage1"
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
config["cotrain_stage1_candidates"] = STAGE1_CANDIDATE_ROOTS
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
