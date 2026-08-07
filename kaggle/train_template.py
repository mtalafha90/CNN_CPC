"""Kaggle training launcher for the single production CNN_CPC pipeline.

Attach this repository, ensure permitted pretrained ConvNeXt weights are available
before Internet is disabled, then train the three leakage-safe folds.
"""
from pathlib import Path
import subprocess

import yaml

CODE_ROOT = Path("/kaggle/input/<your-code-dataset>/CNN_CPC")
CONFIG_SRC = CODE_ROOT / "configs/train.yaml"
CONFIG_RUN = Path("/kaggle/working/train.yaml")

config = yaml.safe_load(CONFIG_SRC.read_text())
config["data_root"] = "/kaggle/input/rsna-knee-abnormality-detection"
config["output_dir"] = "/kaggle/working/runs/model"
CONFIG_RUN.write_text(yaml.safe_dump(config, sort_keys=False))

subprocess.run(
    [
        "rsna-knee",
        "preflight",
        "--data-root",
        config["data_root"],
        "--split",
        "train",
        "--sample-size",
        str(config.get("preflight_sample_size", 24)),
        "--max-decode-failure-rate",
        str(config.get("preflight_max_decode_failure_rate", 0.05)),
    ],
    check=True,
)

for fold in range(int(config.get("n_folds", 3))):
    subprocess.run(
        ["rsna-knee", "train", "--config", str(CONFIG_RUN), "--fold", str(fold)],
        check=True,
    )

print("Training finished: /kaggle/working/runs/model")
