# Local Real-Data Training Runbook

This is the concise production runbook for the current local RSNA knee workflow.

The repository has already passed real-data CSV inspection, nested-fold generation, train/test DICOM preflight, full selected-series audit, OA weak-label verification, and paired-sampler fold-0 smoke training.

## Current verified environment snapshot

Observed on 2026-08-08:

```text
Conda environment: rsna-knee
GPU: NVIDIA RTX A4500 Laptop GPU
precision: bf16
one visible GPU
```

The current local repository/data layout used during verification was:

```text
repo:
/media/talafha/Disk_1/CNN_CPC

data root:
/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection
```

Do not hard-code these paths on another machine; set `REPO` and `DATA_ROOT` appropriately.

## 1. Start a terminal

```bash
export REPO="/media/talafha/Disk_1/CNN_CPC"
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export CUDA_VISIBLE_DEVICES=0

cd "$REPO"
conda activate rsna-knee
```

Check:

```bash
which python
python --version
python -m rsna_knee.cli --help
```

## 2. Pull repository updates before starting a new production stage

```bash
git checkout main
git pull --ff-only origin main
git log -5 --oneline
```

Do not pull into the middle of an already running training process. A running process uses the code/config it loaded at startup.

## 3. Verify the local config

The production local config should be:

```text
configs/train_local.yaml
```

Check the critical values:

```bash
python - <<'PY'
import yaml
from pathlib import Path

c = yaml.safe_load(Path("configs/train_local.yaml").read_text())
for key in [
    "data_root", "output_dir", "requested_gpus", "runtime_budget_hours",
    "pretrained", "allow_external_pretrained", "batch_size",
    "trusted_fraction", "trusted_pseudo_threshold", "rank_min_confidence",
    "tta_center_offsets", "validation_tta_offsets", "weak_oof_tta_offsets",
]:
    print(f"{key}: {c.get(key)}")
print("TTA match:", c["tta_center_offsets"] == c["validation_tta_offsets"])
PY
```

Expected core production values:

```yaml
output_dir: runs/stage1_random
requested_gpus: 1
runtime_budget_hours: 8.5
pretrained: false
allow_external_pretrained: false
batch_size: 2
trusted_fraction: 0.30
trusted_pseudo_threshold: 0.60
rank_min_confidence: 0.35
tta_center_offsets: [-1, 0, 1]
validation_tta_offsets: [-1, 0, 1]
weak_oof_tta_offsets: [0]
```

## 4. Data checks already completed

The real CSV inspection returned:

```text
studies=4407
gold=58
unlabeled=4349
reports_present=4407
series=24371
```

Nested validation manifests:

```text
fold 0: gold_train 20, inner 20, outer 18
fold 1: gold_train 18, inner 20, outer 20
fold 2: gold_train 20, inner 18, outer 20
```

Train preflight:

```text
24 studies
121/121 selected streams decoded
4045/4045 DICOM files decoded
0 failures
```

Complete local test preflight:

```text
3 studies
14/14 selected streams decoded
533/533 DICOM files decoded
0 failures
```

Full audit:

```text
21,886/21,886 selected series decoded
732,554/732,556 DICOM files decoded
2 partial one-file failures
0 selected series failed
```

These steps do not need to be repeated before every fold unless the dataset, DICOM code, routing code or local files change.

## 5. OA report supervision already verified

After the compartment-aware parser update:

```text
Medial OA:  492 positive, 339 negated, 3576 unmentioned
Lateral OA: 409 positive, 387 negated, 3611 unmentioned
PF OA:      695 positive, 379 negated, 3333 unmentioned
```

No confidence threshold was lowered to obtain these labels.

## 6. Fold-0 smoke already passed

The current paired-sampler smoke produced:

```text
selected epoch:        2
inner macro AUC:       0.5513549264
outer TTA macro AUC:   0.5139555403
outer center AUC:      0.5228523149
budget limited:        false
```

Ranking diagnostics:

```text
selection pairs: 63
retrain pairs:   61
all 12 targets: nonzero
```

This proves the end-to-end training path and ranking branch are active. It is not a production result.

## 7. Current next step — Stage-1 random production fold 0

```bash
python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 0
```

There is no `--smoke` flag here.

Expected console rows resemble:

```text
{'phase': 'selection',
 'epoch': 1,
 'train_loss': ...,
 'inner_macro_auc': ...,
 'inner_center_macro_auc': ...,
 'lr': ...,
 'epoch_seconds': ...,
 'train_batches': ...}
```

The model selects the best epoch from **inner** macro-AUC, not outer macro-AUC.

## 8. Harmless warnings

PyTorch may print:

```text
enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
```

This is an optimization warning, not a training failure.

Python may also emit a multiprocessing `resource_tracker` semaphore warning at interpreter shutdown. If it appears **after `best.pt` and all artifacts are written**, treat it as a worker-cleanup warning rather than a failed training run. Investigate only if artifacts are missing, the process exits early, or workers fail during training.

## 9. Inspect a completed production fold

After fold 0 finishes:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("runs/stage1_random/fold0")
s = json.loads((root / "selection.json").read_text())
r = json.loads((root / "runtime.json").read_text())
d = json.loads((root / "training_diagnostics.json").read_text())

print("selected_epoch      :", s["selected_epoch"])
print("inner_macro_auc     :", s["inner_macro_auc"])
print("outer_tta_macro_auc :", s["outer_macro_auc"])
print("outer_center_auc    :", s["outer_center_macro_auc"])
print("budget_limited      :", s["budget_limited_selection"])
print("runtime_hours       :", r["elapsed_seconds"] / 3600)
print("device              :", r["device"])
print("peak_gpu_GB         :", r["peak_gpu_memory_bytes"] / 1024**3)
print("selection rank pairs:", sum(d["selection"]["rank_pairs"].values()))
print("retrain rank pairs  :", sum(d["retrain"]["rank_pairs"].values()))
PY
```

Also inspect the training curve:

```bash
column -s, -t < runs/stage1_random/fold0/history.csv | less -S
```

## 10. Required Stage-1 artifacts

A successful Stage-1 production fold should contain:

```text
best.pt
bootstrap.json
calibration.json
calibration_selection.json
config.json
fold_assignments.csv
history.csv
metadata_repair.json
oof.csv
oof_center.csv
preflight.json
runtime.json
sampling.json
selection.json
supervision_plan.json
training_diagnostics.json
weak_oof.csv
```

## 11. Run folds 1 and 2 unchanged

After fold 0 is confirmed computationally healthy, do not tune from its outer score. Run:

```bash
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 1
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 2
```

## 12. Evaluate the three-fold random baseline

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

Diagnostic TTA versus center-only:

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

Do not change TTA retroactively from this diagnostic.

## 13. Optional competition-data SSL

```bash
python -m rsna_knee.cli pretrain --config configs/train_local.yaml
```

Then create `configs/train_local_ssl.yaml` with:

```yaml
output_dir: runs/stage1_ssl
ssl_encoder_checkpoint: /absolute/path/to/runs/ssl/ssl_encoder.pt
ssl_checkpoint_source: competition_training_data
cotrain_stage1_root: null
cotrain_stage1_candidates: null
```

Train folds 0/1/2 with the same validation policy.

## 14. Leakage-safe Stage-1 candidate selection

```bash
python -m rsna_knee.cli select-stage1 \
  --candidate-root "$(pwd)/runs/stage1_random" \
  --candidate-root "$(pwd)/runs/stage1_ssl" \
  --n-folds 3 \
  --out runs/stage1_selection.json
```

The selector uses only fold-local **inner AUC**.

## 15. Stage 2

Create a Stage-2 config with both candidate roots, then run folds sequentially. Each fold should produce `stage2_supervision.json`, but not `weak_oof.csv`.

Inspect especially:

```text
zero_to_nonzero_weight
stage2_high_confidence
probability_changed_gt_0.05
```

## 16. Final inference

After freezing the final stage, set:

```yaml
expected_checkpoint_stage: stage1
```

or

```yaml
expected_checkpoint_stage: stage2
```

as appropriate, then call `infer` with exactly folds 0, 1 and 2.

## Rule for interpreting results

Do not report:

- smoke AUC as production AUC;
- one outer fold as the final CV score;
- OOF as pristine independent validation after using it to choose the final method;
- a leaderboard score until an actual Kaggle submission has been evaluated.

The production baseline is complete only after all three non-smoke Stage-1 folds have been combined.