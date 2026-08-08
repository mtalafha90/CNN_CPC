# Local Real-Data Training Runbook

Use this exact sequence from a local Linux terminal when training `CNN_CPC` on the real RSNA Knee Abnormality Detection data.

Production assumptions:

- one GPU only;
- CPU multiprocessing for DICOM/data work;
- no DDP / no `torchrun`;
- competition-data-only SSL by default;
- every long GPU job has an 8.5 h software budget, below the 9 h ceiling;
- validation TTA is identical to planned submission TTA;
- random-vs-SSL Stage-1 selection is nested: **each outer fold is chosen from inner AUC only**.

Do not skip the audit or smoke steps.

---

## 0. Define paths

```bash
export REPO="$HOME/CNN_CPC"
export DATA_ROOT="/path/to/rsna-knee-abnormality-detection"
export CUDA_VISIBLE_DEVICES=0
cd "$REPO"
```

Check the data surface:

```bash
ls -lh \
  "$DATA_ROOT/train.csv" \
  "$DATA_ROOT/train_series.csv" \
  "$DATA_ROOT/test.csv" \
  "$DATA_ROOT/test_series.csv"
```

The image tree must also contain `train_images/` or `train_series/` and, when available locally, the corresponding test tree.

---

## 1. Update the repository

```bash
cd "$REPO"
git checkout main
git pull origin main
git rev-parse --short HEAD
```

---

## 2. Create/activate the environment

First time:

```bash
cd "$REPO"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install pytest
```

Later terminals:

```bash
cd "$REPO"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
```

Check CUDA:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

## 3. Run the complete software test suite

```bash
pytest -q
python -m rsna_knee.cli --help
```

Do not start real GPU training if these fail.

---

## 4. Create the local production config

```bash
cp configs/train.yaml configs/train_local.yaml
```

Patch only machine/output paths and keep the production safety contract:

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml

path = Path("configs/train_local.yaml")
config = yaml.safe_load(path.read_text())
config["data_root"] = os.environ["DATA_ROOT"]
config["output_dir"] = "runs/stage1_random"
config["ssl_output_dir"] = "runs/ssl"
config["competition_mode"] = True
config["requested_gpus"] = 1
config["runtime_budget_hours"] = 8.5
config["pretrained"] = False
config["allow_external_pretrained"] = False
config["ssl_encoder_checkpoint"] = None
config["ssl_checkpoint_source"] = None
config["cotrain_stage1_root"] = None
config["cotrain_stage1_candidates"] = None
config["expected_checkpoint_stage"] = None
path.write_text(yaml.safe_dump(config, sort_keys=False))
print(path)
PY
```

Verify the important lines:

```bash
grep -E 'data_root|requested_gpus|runtime_budget_hours|pretrained|allow_external|tta_center_offsets|validation_tta_offsets|cotrain_stage1' configs/train_local.yaml
```

`tta_center_offsets` and `validation_tta_offsets` must match.

---

## 5. Verify runtime

```bash
python -m rsna_knee.cli runtime --config configs/train_local.yaml
nvidia-smi
```

Optional second terminal during GPU runs:

```bash
watch -n 2 nvidia-smi
```

---

## 6. Inspect CSVs

```bash
python -m rsna_knee.cli inspect --data-root "$DATA_ROOT"
```

Confirm study/gold/report/series counts are sensible.

---

## 7. DICOM preflight

Train:

```bash
mkdir -p runs
python -m rsna_knee.cli preflight \
  --data-root "$DATA_ROOT" \
  --split train \
  --sample-size 24 \
  --max-decode-failure-rate 0.05 \
  --max-file-decode-failure-rate 0.05 \
  --out runs/preflight_train.json
```

Test, when present locally:

```bash
python -m rsna_knee.cli preflight \
  --data-root "$DATA_ROOT" \
  --split test \
  --sample-size 24 \
  --max-decode-failure-rate 0.05 \
  --max-file-decode-failure-rate 0.05 \
  --out runs/preflight_test.json
```

Do not continue after a preflight failure.

---

## 8. Full CPU multiprocessing audit

```bash
python -m rsna_knee.cli audit \
  --config configs/train_local.yaml \
  --out-dir runs/audit
```

Inspect:

```bash
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("runs/audit/audit.json").read_text())
print(json.dumps(p["decode_audit"], indent=2))
print("\nSelected streams:", json.dumps(p["selected_stream_counts"], indent=2))
print("\nMissing streams:", json.dumps(p["missing_stream_counts"], indent=2))
print("\nTeacher confidence:", json.dumps(p["teacher_confidence_counts"], indent=2))
PY
```

Do not start production training unless the audit completes successfully.

---

# Stage 1A — Random initialization

## 9. First real GPU smoke: fold 0 only

```bash
python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 0 \
  --smoke
```

Inspect:

```bash
find runs/stage1_random/smoke/fold0 -maxdepth 1 -type f -printf '%f\n' | sort
cat runs/stage1_random/smoke/fold0/selection.json
cat runs/stage1_random/smoke/fold0/runtime.json
cat runs/stage1_random/smoke/fold0/training_diagnostics.json
```

Expected core artifacts include:

```text
best.pt
oof.csv                 # primary TTA OOF, same policy as submission
oof_center.csv          # diagnostic center-only OOF
weak_oof.csv             # Stage-1 only
selection.json
history.csv
training_diagnostics.json
supervision_plan.json
runtime.json
bootstrap.json
```

Only after fold 0 succeeds, smoke folds 1 and 2:

```bash
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 1 --smoke
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 2 --smoke
```

---

## 10. Stage-1 random production folds

Run sequentially, never simultaneously on the same GPU:

```bash
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 0
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 1
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 2
```

Each command is independently budgeted below 9 h and reserves time for outer OOF, weak OOF, bootstrap, and checkpoint writing.

---

## 11. Evaluate random Stage-1 OOF

Primary TTA OOF:

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

Diagnostic TTA-vs-center comparison:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof \
    runs/stage1_random/fold0/oof_center.csv \
    runs/stage1_random/fold1/oof_center.csv \
    runs/stage1_random/fold2/oof_center.csv \
  --compare-oof \
    runs/stage1_random/fold0/oof.csv \
    runs/stage1_random/fold1/oof.csv \
    runs/stage1_random/fold2/oof.csv \
  --n-bootstrap 2000 \
  --out runs/stage1_random/tta_vs_center.json
```

Do **not** change submission TTA after looking at outer OOF. TTA is already part of the predeclared inner-selection policy; this comparison is diagnostic only.

Inspect target supervision/ranking utilization:

```bash
for f in 0 1 2; do
  echo "===== RANDOM FOLD $f ====="
  cat "runs/stage1_random/fold${f}/training_diagnostics.json"
done
```

---

# Stage 1B — Competition-data SSL candidate

## 12. Train SSL separately

```bash
python -m rsna_knee.cli pretrain --config configs/train_local.yaml
```

Expected:

```text
runs/ssl/ssl_encoder.pt
runs/ssl/history.json
```

---

## 13. Build the SSL Stage-1 config

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
config["cotrain_stage1_candidates"] = None
path.write_text(yaml.safe_dump(config, sort_keys=False))
PY
```

---

## 14. Train all three SSL Stage-1 folds

```bash
python -m rsna_knee.cli train --config configs/train_local_ssl.yaml --fold 0
python -m rsna_knee.cli train --config configs/train_local_ssl.yaml --fold 1
python -m rsna_knee.cli train --config configs/train_local_ssl.yaml --fold 2
```

You may evaluate random vs SSL OOF for research diagnostics, but **do not use total outer OOF to decide which candidate supplies a given outer fold**.

---

## 15. Perform leakage-safe per-fold Stage-1 selection

This is the only supported random-vs-SSL selection for downstream Stage 2:

```bash
python -m rsna_knee.cli select-stage1 \
  --candidate-root "$(pwd)/runs/stage1_random" \
  --candidate-root "$(pwd)/runs/stage1_ssl" \
  --n-folds 3 \
  --out runs/stage1_selection.json
```

Inspect:

```bash
cat runs/stage1_selection.json
```

For each outer fold `k`, the selector uses **only `inner_macro_auc` for fold `k`**. It ignores `outer_macro_auc` even if one candidate has a much better outer result.

---

# Stage 2 — Fold-local image/report co-training

## 16. Create the Stage-2 config

Use the base random-init config for Stage-2 initialization. Candidate selection controls the fold-local image teacher; it does not use outer OOF to choose Stage-2 initialization.

```bash
cp configs/train_local.yaml configs/train_local_stage2.yaml
python - <<'PY'
from pathlib import Path
import yaml

path = Path("configs/train_local_stage2.yaml")
config = yaml.safe_load(path.read_text())
config["output_dir"] = "runs/stage2"
config["ssl_encoder_checkpoint"] = None
config["ssl_checkpoint_source"] = None
config["cotrain_stage1_root"] = None
config["cotrain_stage1_candidates"] = [
    str(Path("runs/stage1_random").resolve()),
    str(Path("runs/stage1_ssl").resolve()),
]
path.write_text(yaml.safe_dump(config, sort_keys=False))
PY
```

Verify:

```bash
grep -E 'output_dir|ssl_encoder|cotrain_stage1' configs/train_local_stage2.yaml
```

---

## 17. Run Stage-2 folds sequentially

```bash
python -m rsna_knee.cli train --config configs/train_local_stage2.yaml --fold 0
python -m rsna_knee.cli train --config configs/train_local_stage2.yaml --fold 1
python -m rsna_knee.cli train --config configs/train_local_stage2.yaml --fold 2
```

For each fold inspect how much supervision Stage 2 actually added:

```bash
for f in 0 1 2; do
  echo "===== STAGE2 FOLD $f ====="
  cat "runs/stage2/fold${f}/stage2_supervision.json"
  cat "runs/stage2/fold${f}/training_diagnostics.json"
done
```

`zero_to_nonzero_weight` is especially important: it measures report-silent cells for which a very confident cross-fitted image teacher added modest BCE supervision.

Stage 2 intentionally does **not** produce another `weak_oof.csv`.

---

## 18. Evaluate Stage 2

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof \
    runs/stage2/fold0/oof.csv \
    runs/stage2/fold1/oof.csv \
    runs/stage2/fold2/oof.csv \
  --n-bootstrap 2000 \
  --out runs/stage2/evaluation.json
```

For a paired comparison against the *nested-selected* Stage-1 folds, extract their OOF paths from the selection manifest:

```bash
mapfile -t SELECTED_STAGE1_OOF < <(python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("runs/stage1_selection.json").read_text())
for f in range(3):
    root = Path(p["folds"][str(f)]["selected_root"])
    print(root / f"fold{f}" / "oof.csv")
PY
)

python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof \
    "${SELECTED_STAGE1_OOF[0]}" \
    "${SELECTED_STAGE1_OOF[1]}" \
    "${SELECTED_STAGE1_OOF[2]}" \
  --compare-oof \
    runs/stage2/fold0/oof.csv \
    runs/stage2/fold1/oof.csv \
    runs/stage2/fold2/oof.csv \
  --n-bootstrap 2000 \
  --out runs/stage2/vs_nested_stage1.json
```

This Stage-2-vs-Stage-1 outer comparison is useful for competition model choice, but once you use it to choose the final method it is **model-selection CV**, not a pristine unbiased generalization estimate.

---

# Final inference

## 19. Choose final checkpoint stage

If Stage 2 is your frozen final method:

```bash
cp configs/train_local_stage2.yaml configs/final_infer.yaml
python - <<'PY'
from pathlib import Path
import yaml
p = Path("configs/final_infer.yaml")
c = yaml.safe_load(p.read_text())
c["expected_checkpoint_stage"] = "stage2"
p.write_text(yaml.safe_dump(c, sort_keys=False))
PY
```

Run:

```bash
python -m rsna_knee.cli infer \
  --config configs/final_infer.yaml \
  --checkpoints \
    runs/stage2/fold0/best.pt \
    runs/stage2/fold1/best.pt \
    runs/stage2/fold2/best.pt \
  --out submission.csv
```

The inference code verifies:

- exactly three checkpoints;
- folds exactly `{0,1,2}`;
- all checkpoints from one stage;
- matching architecture/stream order;
- checkpoint validation TTA exactly matching submission TTA;
- finite probabilities and exact submission columns.

If final Stage 1 is preferred instead, use `runs/stage1_selection.json` to supply exactly the selected fold checkpoint paths and set `expected_checkpoint_stage: stage1`.

---

## 20. Final file checks

```bash
ls -lh submission.csv
head -3 submission.csv
python - <<'PY'
import pandas as pd
x = pd.read_csv("submission.csv")
print(x.shape)
print(x.columns.tolist())
print(x.isna().sum().sum(), "NaNs")
PY
```

---

# What to inspect after every long fold

```bash
cat runs/<stage>/fold0/selection.json
cat runs/<stage>/fold0/runtime.json
cat runs/<stage>/fold0/training_diagnostics.json
cat runs/<stage>/fold0/supervision_plan.json
```

For Stage 2 also inspect:

```bash
cat runs/stage2/fold0/stage2_supervision.json
```

Key warning signs:

- `budget_limited_selection: true` very early;
- actual batches far below planned batches;
- near-zero ranking pairs for most targets;
- very small nonzero supervision counts for rare pathologies;
- zero Stage-2 `zero_to_nonzero_weight` counts;
- high DICOM decode failure rates;
- TTA fallback during final inference.

If any of these occur, evaluate the diagnostic before spending more GPU time.
