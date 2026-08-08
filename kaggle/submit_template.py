"""Kaggle one-GPU submission template.

The attached repository is imported directly from its ``src`` directory, so the
Internet-disabled committed run requires no pip/network installation.
"""
from pathlib import Path
import sys

import yaml

CODE_ROOT = Path("/kaggle/input/<your-code-dataset>/CNN_CPC")
MODEL_ROOT = Path("/kaggle/input/<your-model-dataset>/runs/model")
sys.path.insert(0, str(CODE_ROOT / "src"))

from rsna_knee.inference import infer_checkpoints  # noqa: E402

config = yaml.safe_load((CODE_ROOT / "configs/train.yaml").read_text())
config["data_root"] = "/kaggle/input/rsna-knee-abnormality-detection"
config["competition_mode"] = True
config["requested_gpus"] = 1
config["runtime_budget_hours"] = 8.5
config["submission_filename"] = "submission.csv"
config["pretrained"] = False
config["allow_external_pretrained"] = False

checkpoints = sorted(MODEL_ROOT.glob("fold*/best.pt"))
if len(checkpoints) != int(config.get("n_folds", 3)):
    raise FileNotFoundError(
        f"expected {config.get('n_folds', 3)} fold checkpoints under {MODEL_ROOT}, found {len(checkpoints)}"
    )

submission = infer_checkpoints(config["data_root"], checkpoints, config)
submission.to_csv("/kaggle/working/submission.csv", index=False)
print(submission.head())
print("/kaggle/working/submission.csv")
