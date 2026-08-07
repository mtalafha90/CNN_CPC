"""Kaggle image-only submission template for the production CNN_CPC model.

Attach the code repository and a Kaggle Dataset containing `fold*/best.pt`.
Checkpoint metadata defines the MRI preprocessing/model contract, so the runtime
config only supplies the competition data location and hardware settings.
"""
from pathlib import Path

import yaml

from rsna_knee.inference import infer_checkpoints

CODE_ROOT = Path("/kaggle/input/<your-code-dataset>/CNN_CPC")
MODEL_ROOT = Path("/kaggle/input/<your-model-dataset>/runs/model")

config = yaml.safe_load((CODE_ROOT / "configs/train.yaml").read_text())
config["data_root"] = "/kaggle/input/rsna-knee-abnormality-detection"

checkpoints = sorted(MODEL_ROOT.glob("fold*/best.pt"))
if not checkpoints:
    raise FileNotFoundError(f"no fold checkpoints found under {MODEL_ROOT}")

submission = infer_checkpoints(config["data_root"], checkpoints, config)
submission.to_csv("/kaggle/working/submission.csv", index=False)
print(submission.head())
