# Local Real-Data Training Runbook

This is the command sequence to use from a local Linux terminal when you are ready to train `CNN_CPC` on the real RSNA Knee Abnormality Detection data.

The production pipeline is intentionally configured for:

- one GPU only;
- CPU multiprocessing for DICOM loading/preprocessing;
- no DDP / no `torchrun`;
- no external pretrained weights by default;
- an 8.5-hour runtime budget per long GPU run;
- leakage-safe Stage-1 and Stage-2 validation.

Run the steps below in order.

---

## 0. Define your paths

Open a terminal and define the repository and real-data locations.

```bash
export REPO="$HOME/CNN_CPC"
export DATA_ROOT="/path/to/rsna-knee-abnormality-detection"
export CUDA_VISIBLE_DEVICES=0

cd "$REPO"
```

Replace `/path/to/rsna-knee-abnormality-detection` with the real extracted competition-data directory.

The directory should contain at least:

```text
train.csv
train_series.csv
test.csv
test_series.csv
train_images/ or train_series/
test_images/ or test_series/
```

Check the important files:

```bash
ls -lh \
  "$DATA_ROOT/train.csv" \
  "$DATA_ROOT/train_series.csv" \
  "$DATA_ROOT/test.csv" \
  "$DATA_ROOT/test_series.csv"
```

Do not continue if those files are missing.

---

## 1. Update the local repository

```bash
cd "$REPO"
git checkout main
git pull origin main

git rev-parse --short HEAD
```

You should be on the current production `main` branch.

---

## 2. Create and activate the Python environment

If you do not already have a project environment:

```bash
cd "$REPO"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install pytest
```

For later terminals, only run:

```bash
cd "$REPO"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
```

Check Python and PyTorch:

```bash
python --version
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

For GPU training you want:

```text
CUDA available: True
CUDA device count: >= 1
```

Only GPU `0` will be exposed because of `CUDA_VISIBLE_DEVICES=0`.

---

## 3. Run the software tests before touching the real data

```bash
cd "$REPO"
source .venv/bin/activate
pytest -q
```

Also check that the CLI imports correctly:

```bash
python -m rsna_knee.cli --help
```

If tests fail because of a software/import problem, fix that before starting a long real-data run.

---

## 4. Create a local configuration file

Do not edit the production template every time. Create a machine-specific copy:

```bash
cd "$REPO"
cp configs/train.yaml configs/train_local.yaml
```

Patch the real data path and choose the local output directories:

```bash
export DATA_ROOT="/path/to/rsna-knee-abnormality-detection"

python - <<'PY'
import os
from pathlib import Path
import yaml

path = Path("configs/train_local.yaml")
config = yaml.safe_load(path.read_text())
config["data_root"] = os.environ["DATA_ROOT"]
config["output_dir"] = "runs/stage1_random"
config["ssl_output_dir"] = "runs/ssl"
config["requested_gpus"] = 1
config["competition_mode"] = True
config["runtime_budget_hours"] = 8.5
config["pretrained"] = False
config["allow_external_pretrained"] = False
config["ssl_encoder_checkpoint"] = None
config["ssl_checkpoint_source"] = None
config["cotrain_stage1_root"] = None
path.write_text(yaml.safe_dump(config, sort_keys=False))
print(path)
PY
```

Inspect it:

```bash
cat configs/train_local.yaml
```

Most importantly verify:

```text
data_root: <your real data path>
requested_gpus: 1
runtime_budget_hours: 8.5
pretrained: false
allow_external_pretrained: false
cotrain_stage1_root: null
```

---

## 5. Check the runtime configuration

```bash
python -m rsna_knee.cli runtime --config configs/train_local.yaml
```

Then check the GPU separately:

```bash
nvidia-smi
```

During training it is useful to open a second terminal and run:

```bash
watch -n 2 nvidia-smi
```

The model must use one GPU only. CPU worker processes will feed that GPU in parallel.

---

## 6. Inspect the competition CSVs

```bash
python -m rsna_knee.cli inspect --data-root "$DATA_ROOT"
```

Check that the study count, gold count, report count, and series count are sensible before continuing.

---

## 7. Run a small DICOM preflight

```bash
python -m rsna_knee.cli preflight \
  --data-root "$DATA_ROOT" \
  --split train \
  --sample-size 24 \
  --max-decode-failure-rate 0.05 \
  --max-file-decode-failure-rate 0.05 \
  --out runs/preflight_train.json
```

Do not start long GPU training if this command fails.

Also preflight the test set if it is already available locally:

```bash
python -m rsna_knee.cli preflight \
  --data-root "$DATA_ROOT" \
  --split test \
  --sample-size 24 \
  --max-decode-failure-rate 0.05 \
  --max-file-decode-failure-rate 0.05 \
  --out runs/preflight_test.json
```

---

## 8. Run the full real-data audit

This should be done before expensive GPU experiments.

```bash
mkdir -p runs/audit

python -m rsna_knee.cli audit \
  --config configs/train_local.yaml \
  --out-dir runs/audit
```

Important outputs:

```text
runs/audit/audit.json
runs/audit/series_decode_audit.csv
```

Inspect the summary:

```bash
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("runs/audit/audit.json").read_text())
print(json.dumps(p["decode_audit"], indent=2))
print("\nSelected streams:")
print(json.dumps(p["selected_stream_counts"], indent=2))
print("\nMissing streams:")
print(json.dumps(p["missing_stream_counts"], indent=2))
PY
```

Do **not** move to long GPU training if the full audit fails. Fix DICOM/path/codec problems first.

---

# STAGE 1

## 9. Run three short Stage-1 smoke tests

These are the first real end-to-end GPU runs.

Run fold 0:

```bash
python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 0 \
  --smoke
```

Run fold 1:

```bash
python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 1 \
  --smoke
```

Run fold 2:

```bash
python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 2 \
  --smoke
```

Smoke outputs will be below:

```text
runs/stage1_random/smoke/fold0/
runs/stage1_random/smoke/fold1/
runs/stage1_random/smoke/fold2/
```

Verify each fold produced at least:

```text
best.pt
oof.csv
weak_oof.csv
history.csv
selection.json
ranking_pairs.json
```

Example:

```bash
find runs/stage1_random/smoke -maxdepth 2 -type f | sort
```

If a smoke run crashes, do not start production training yet.

---

## 10. Start Stage-1 production training — random initialization

The production configuration uses no external/ImageNet pretrained weights.

Run one fold at a time.

### Fold 0

```bash
python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 0
```

### Fold 1

```bash
python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 1
```

### Fold 2

```bash
python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 2
```

Do not launch the three folds simultaneously on the same GPU.

Expected directories:

```text
runs/stage1_random/fold0/
runs/stage1_random/fold1/
runs/stage1_random/fold2/
```

Check the files:

```bash
find runs/stage1_random -maxdepth 2 -type f | sort
```

---

## 11. Evaluate Stage-1 random-initialization OOF

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof \
    runs/stage1_random/fold0/oof.csv \
    runs/stage1_random/fold1/oof.csv \
    runs/stage1_random/fold2/oof.csv \
  --n-bootstrap 2000 \
  --out runs/stage1_random/evaluation.json
```

Inspect it:

```bash
cat runs/stage1_random/evaluation.json
```

Also inspect ranking-pair usage:

```bash
for f in 0 1 2; do
  echo "===== FOLD $f ====="
  cat "runs/stage1_random/fold${f}/ranking_pairs.json"
done
```

This tells you whether rare pathologies actually receive useful ranking pairs.

---

# OPTIONAL IN-DOMAIN SSL

## 12. Train the competition-data SSL encoder

Run this as a separate GPU job:

```bash
python -m rsna_knee.cli pretrain \
  --config configs/train_local.yaml
```

Expected output:

```text
runs/ssl/ssl_encoder.pt
runs/ssl/history.json
```

Check it:

```bash
ls -lh runs/ssl/ssl_encoder.pt runs/ssl/history.json
```

---

## 13. Create the SSL Stage-1 configuration

```bash
cp configs/train_local.yaml configs/train_local_ssl.yaml

python - <<'PY'
from pathlib import Path
import yaml

path = Path("configs/train_local_ssl.yaml")
config = yaml.safe_load(path.read_text())
config["output_dir"] = "runs/stage1_ssl"
config["ssl_encoder_checkpoint"] = str(Path("runs/ssl/ssl_encoder.pt").resolve())
config["ssl_checkpoint_source"] = "competition_training_data"
config["cotrain_stage1_root"] = None
path.write_text(yaml.safe_dump(config, sort_keys=False))
print(path)
PY
```

Check:

```bash
grep -E 'output_dir|ssl_encoder_checkpoint|ssl_checkpoint_source|cotrain_stage1_root' \
  configs/train_local_ssl.yaml
```

---

## 14. Train Stage-1 with the SSL initialization

Run folds sequentially:

```bash
python -m rsna_knee.cli train --config configs/train_local_ssl.yaml --fold 0
python -m rsna_knee.cli train --config configs/train_local_ssl.yaml --fold 1
python -m rsna_knee.cli train --config configs/train_local_ssl.yaml --fold 2
```

Expected output:

```text
runs/stage1_ssl/fold0/
runs/stage1_ssl/fold1/
runs/stage1_ssl/fold2/
```

---

## 15. Compare SSL Stage-1 against random Stage-1

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof \
    runs/stage1_ssl/fold0/oof.csv \
    runs/stage1_ssl/fold1/oof.csv \
    runs/stage1_ssl/fold2/oof.csv \
  --compare-oof \
    runs/stage1_random/fold0/oof.csv \
    runs/stage1_random/fold1/oof.csv \
    runs/stage1_random/fold2/oof.csv \
  --n-bootstrap 2000 \
  --out runs/stage1_ssl/evaluation_vs_random.json
```

Inspect:

```bash
cat runs/stage1_ssl/evaluation_vs_random.json
```

Choose the Stage-1 initialization using the **paired OOF result**, not intuition.

If random initialization is better, use:

```bash
export BEST_STAGE1="$(pwd)/runs/stage1_random"
export BEST_STAGE1_CONFIG="$(pwd)/configs/train_local.yaml"
```

If SSL is better, use:

```bash
export BEST_STAGE1="$(pwd)/runs/stage1_ssl"
export BEST_STAGE1_CONFIG="$(pwd)/configs/train_local_ssl.yaml"
```

Check:

```bash
echo "$BEST_STAGE1"
echo "$BEST_STAGE1_CONFIG"
```

---

# STAGE 2

## 16. Create the leakage-safe Stage-2 configuration

Make sure `BEST_STAGE1` and `BEST_STAGE1_CONFIG` point to the Stage-1 method you chose.

```bash
cp "$BEST_STAGE1_CONFIG" configs/train_local_stage2.yaml

python - <<'PY'
import os
from pathlib import Path
import yaml

path = Path("configs/train_local_stage2.yaml")
config = yaml.safe_load(path.read_text())
config["output_dir"] = "runs/stage2"
config["cotrain_stage1_root"] = os.environ["BEST_STAGE1"]
path.write_text(yaml.safe_dump(config, sort_keys=False))
print(path)
PY
```

Verify:

```bash
grep -E 'output_dir|cotrain_stage1_root|ssl_encoder_checkpoint|ssl_checkpoint_source' \
  configs/train_local_stage2.yaml
```

The important rule is:

```text
Stage-2 outer fold k may use only Stage-1 fold{k}/weak_oof.csv.
```

The code enforces this automatically.

---

## 17. Run Stage-2 folds sequentially

### Fold 0

```bash
python -m rsna_knee.cli train \
  --config configs/train_local_stage2.yaml \
  --fold 0
```

### Fold 1

```bash
python -m rsna_knee.cli train \
  --config configs/train_local_stage2.yaml \
  --fold 1
```

### Fold 2

```bash
python -m rsna_knee.cli train \
  --config configs/train_local_stage2.yaml \
  --fold 2
```

Expected outputs:

```text
runs/stage2/fold0/
runs/stage2/fold1/
runs/stage2/fold2/
```

Stage 2 intentionally does **not** create another `weak_oof.csv` for chaining another co-training stage.

---

## 18. Compare Stage 2 against the chosen Stage 1

If `BEST_STAGE1` is still defined:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof \
    runs/stage2/fold0/oof.csv \
    runs/stage2/fold1/oof.csv \
    runs/stage2/fold2/oof.csv \
  --compare-oof \
    "$BEST_STAGE1/fold0/oof.csv" \
    "$BEST_STAGE1/fold1/oof.csv" \
    "$BEST_STAGE1/fold2/oof.csv" \
  --n-bootstrap 2000 \
  --out runs/stage2/evaluation_vs_stage1.json
```

Inspect:

```bash
cat runs/stage2/evaluation_vs_stage1.json
```

Keep Stage 2 only if the paired comparison supports it.

---

# FINAL MODEL CHOICE AND TEST INFERENCE

## 19. Choose the final checkpoint set

If Stage 2 is better:

```bash
export FINAL_MODEL_ROOT="$(pwd)/runs/stage2"
export FINAL_CONFIG="$(pwd)/configs/train_local_stage2.yaml"
```

If Stage 1 is better:

```bash
export FINAL_MODEL_ROOT="$BEST_STAGE1"
export FINAL_CONFIG="$BEST_STAGE1_CONFIG"
```

Check:

```bash
echo "$FINAL_MODEL_ROOT"
echo "$FINAL_CONFIG"
ls -lh \
  "$FINAL_MODEL_ROOT/fold0/best.pt" \
  "$FINAL_MODEL_ROOT/fold1/best.pt" \
  "$FINAL_MODEL_ROOT/fold2/best.pt"
```

---

## 20. Run local test-set inference

Run from the repository root:

```bash
rm -f submission.csv

python -m rsna_knee.cli infer \
  --config "$FINAL_CONFIG" \
  --checkpoints \
    "$FINAL_MODEL_ROOT/fold0/best.pt" \
    "$FINAL_MODEL_ROOT/fold1/best.pt" \
    "$FINAL_MODEL_ROOT/fold2/best.pt" \
  --out submission.csv
```

The output must be exactly:

```text
submission.csv
```

Check it:

```bash
ls -lh submission.csv
head -5 submission.csv
```

Verify the row/column shape:

```bash
python - <<'PY'
import pandas as pd
p = pd.read_csv("submission.csv")
print("shape:", p.shape)
print("columns:")
print(p.columns.tolist())
print("duplicate studies:", p["StudyInstanceUID"].duplicated().sum())
print("probability min:", p.iloc[:, 1:].min().min())
print("probability max:", p.iloc[:, 1:].max().max())
PY
```

---

# Useful monitoring commands

## GPU

```bash
watch -n 2 nvidia-smi
```

## CPU and RAM

```bash
htop
```

## Disk usage

```bash
df -h

du -sh runs/* 2>/dev/null
```

## Follow a saved terminal log

You can save a run while still seeing the output with `tee`:

```bash
mkdir -p logs

python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 0 \
  2>&1 | tee logs/stage1_random_fold0.log
```

Then from another terminal:

```bash
tail -f logs/stage1_random_fold0.log
```

---

# Recommended first real run

When you first receive/mount the complete real dataset, do **not** jump directly to the full model.

Use this exact order:

```bash
cd "$REPO"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export DATA_ROOT="/path/to/rsna-knee-abnormality-detection"

python -m rsna_knee.cli runtime --config configs/train_local.yaml
python -m rsna_knee.cli inspect --data-root "$DATA_ROOT"

python -m rsna_knee.cli preflight \
  --data-root "$DATA_ROOT" \
  --split train \
  --sample-size 24 \
  --max-decode-failure-rate 0.05 \
  --max-file-decode-failure-rate 0.05 \
  --out runs/preflight_train.json

python -m rsna_knee.cli audit \
  --config configs/train_local.yaml \
  --out-dir runs/audit

python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 0 \
  --smoke
```

Only after the audit and fold-0 smoke run succeed should you continue with smoke folds 1 and 2, followed by production Stage-1.

---

# Short command checklist

```text
1. git pull
2. activate .venv
3. export CUDA_VISIBLE_DEVICES=0
4. set DATA_ROOT
5. create/update configs/train_local.yaml
6. pytest -q
7. runtime check
8. inspect CSVs
9. preflight
10. full audit
11. Stage-1 smoke fold 0
12. Stage-1 smoke fold 1
13. Stage-1 smoke fold 2
14. Stage-1 production fold 0
15. Stage-1 production fold 1
16. Stage-1 production fold 2
17. evaluate Stage 1
18. optional competition-data SSL
19. optional 3-fold Stage-1 SSL comparison
20. choose best Stage 1
21. Stage-2 fold 0
22. Stage-2 fold 1
23. Stage-2 fold 2
24. paired Stage-2 vs Stage-1 evaluation
25. choose final checkpoints
26. local test inference
27. inspect submission.csv
```
