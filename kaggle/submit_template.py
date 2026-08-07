"""Kaggle Code Competition image-only inference template.

Attach the code repository plus a Kaggle Dataset containing the checkpoints for
ONE frozen experiment. The notebook should run with Internet OFF and write
`/kaggle/working/submission.csv`. Report text is not required at inference.
"""
from pathlib import Path
import yaml
from rsna_knee.inference import infer_checkpoints, validate_submission

CODE_ROOT = Path("/kaggle/input/<your-code-dataset>/CNN_CPC")
MODEL_ROOT = Path("/kaggle/input/<your-model-dataset>/runs/e01_baseline")
EXPERIMENT = "e01_baseline"

config = yaml.safe_load((CODE_ROOT / "configs" / f"{EXPERIMENT}.yaml").read_text())
config["data_root"] = "/kaggle/input/rsna-knee-abnormality-detection"
config["allow_test_report_fusion"] = False
checkpoints = sorted(MODEL_ROOT.glob("fold*/best.pt"))
if not checkpoints:
    raise FileNotFoundError(f"no fold checkpoints found under {MODEL_ROOT}")

sub = infer_checkpoints(
    config["data_root"], checkpoints, config, fusion_alpha=1.0
)
validate_submission(sub)
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(sub.head())
