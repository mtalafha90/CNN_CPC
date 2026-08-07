"""Kaggle training launcher for one frozen CNN_CPC experiment.

Attach this repository as a Kaggle Dataset (or copy it into the notebook). Choose
one experiment config, override data_root/output_dir for Kaggle, save the resolved
config, then train the three frozen folds. Internet may be OFF only when every
required pretrained weight/package is already available in the notebook image or
attached as a Kaggle Dataset.
"""
from pathlib import Path
import subprocess
import yaml

CODE_ROOT = Path("/kaggle/input/<your-code-dataset>/CNN_CPC")
EXPERIMENT = "e01_baseline"  # change deliberately: e02_..., e03_..., etc.
CONFIG_SRC = CODE_ROOT / "configs" / f"{EXPERIMENT}.yaml"
CONFIG_RUN = Path("/kaggle/working") / f"{EXPERIMENT}.yaml"

config = yaml.safe_load(CONFIG_SRC.read_text())
config["data_root"] = "/kaggle/input/rsna-knee-abnormality-detection"
config["output_dir"] = f"/kaggle/working/runs/{EXPERIMENT}"
CONFIG_RUN.write_text(yaml.safe_dump(config, sort_keys=False))

# Explicit preflight before consuming GPU time; training repeats the same gate.
subprocess.run([
    "rsna-knee", "preflight",
    "--data-root", config["data_root"],
    "--split", "train",
    "--sample-size", str(config.get("preflight_sample_size", 24)),
    "--stream-mode", config.get("stream_mode", "best"),
    "--max-failure-rate", str(config.get("preflight_max_failure_rate", 0.05)),
], check=True)

for fold in range(int(config.get("n_folds", 3))):
    subprocess.run([
        "rsna-knee", "train", "--config", str(CONFIG_RUN), "--fold", str(fold)
    ], check=True)

print(f"Training finished: /kaggle/working/runs/{EXPERIMENT}")
