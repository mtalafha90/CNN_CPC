"""Kaggle Code Competition inference template.

Attach this code repository plus a Kaggle Dataset containing fold checkpoints.
The notebook should run with Internet OFF and write /kaggle/working/submission.csv.
"""
from pathlib import Path
import yaml
from rsna_knee.inference import infer_checkpoints, validate_submission

CODE_ROOT = Path("/kaggle/input/<your-code-dataset>/CNN_CPC")
MODEL_ROOT = Path("/kaggle/input/<your-model-dataset>/runs")
config = yaml.safe_load((CODE_ROOT / "configs/baseline.yaml").read_text())
config["data_root"] = "/kaggle/input/rsna-knee-abnormality-detection"
checkpoints = sorted(MODEL_ROOT.glob("fold*/best.pt"))
sub = infer_checkpoints(config["data_root"], checkpoints, config, fusion_alpha=0.70)
validate_submission(sub)
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(sub.head())
