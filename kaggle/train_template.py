"""Minimal Kaggle training launcher.

Attach this repository as a Kaggle Dataset (or copy it into the notebook), install it
with --no-deps if dependencies are already present, then train fold 0..2.
"""
import subprocess

CONFIG = "/kaggle/input/<your-code-dataset>/CNN_CPC/configs/baseline.yaml"
for fold in range(3):
    subprocess.run(["rsna-knee", "train", "--config", CONFIG, "--fold", str(fold)], check=True)
print("Training finished. Save /kaggle/working/runs as a Kaggle Dataset for inference.")
