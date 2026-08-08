"""Kaggle one-GPU training template.

Train exactly ONE fold per committed notebook run. Use separate runs for folds
0, 1 and 2 so each execution stays safely below the competition runtime ceiling.

For Stage 1 leave STAGE1_MODEL_ROOT=None. For Stage 2 attach the completed
Stage-1 model dataset and point STAGE1_MODEL_ROOT at its ``runs/model`` folder.
"""
from pathlib import Path
import subprocess

import yaml

CODE_ROOT = Path("/kaggle/input/<your-code-dataset>/CNN_CPC")
DATA_ROOT = "/kaggle/input/rsna-knee-abnormality-detection"
CONFIG_SRC = CODE_ROOT / "configs/train.yaml"
CONFIG_RUN = Path("/kaggle/working/train.yaml")

# EDIT ONE VALUE PER NOTEBOOK COPY: 0, 1, or 2.
FOLD = 0
# Stage 2 example: "/kaggle/input/<stage1-model-dataset>/runs/model"
STAGE1_MODEL_ROOT = None
# Optional attached competition-data SSL checkpoint.
SSL_CHECKPOINT = None
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
CONFIG_RUN.write_text(yaml.safe_dump(config, sort_keys=False))

command = ["rsna-knee", "train", "--config", str(CONFIG_RUN), "--fold", str(FOLD)]
if SMOKE:
    command.append("--smoke")
subprocess.run(command, check=True)

print(f"Completed {stage_name} fold {FOLD}: {output_root}/fold{FOLD}")
