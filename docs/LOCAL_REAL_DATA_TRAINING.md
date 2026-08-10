# Local Real-Data Training Runbook

> **Current stage — 2026-08-10:** package `0.14.0`. **B7.1 remains the current leader at macro AUC `0.5644802945`. B8 is rejected at `0.5300962807`. B9 strict semantic routing is implemented and ready for testing/training.** See [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Environment

```text
Conda environment: rsna-knee
GPU: NVIDIA RTX A4500 Laptop GPU
precision: bf16
one visible GPU
```

Verified paths:

```text
repo:      /media/talafha/Disk_1/CNN_CPC
data root: /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection
```

Start a terminal:

```bash
export REPO="/media/talafha/Disk_1/CNN_CPC"
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export CUDA_VISIBLE_DEVICES=0
cd "$REPO"
conda activate rsna-knee
```

## Pull/install between stages

```bash
git checkout main
git pull --ff-only origin main
python -m pip install -e .

python - <<'PY'
import rsna_knee
print(rsna_knee.__version__)
PY
```

Expected version:

```text
0.14.0
```

## Current measured ladder

```text
B0 random                         0.4762536432
B1 strong SSL                    0.5030284974
B4 frozen SSL + classical        0.5137567459
B5 image-report SSL              0.5243650851
B7-v1 weak supervision           0.5397724412
B7.1 full coverage               0.5644802945   CURRENT LEADER
B5+B7.1 fixed rank ensemble      0.5540141184   REJECTED
B8 spatial anatomy               0.5300962807   REJECTED
B9 strict semantic routing       pending        ACTIVE
```

## Why B9

The historical six-stream selector sometimes places a same-class acquisition into the opposite semantic slot when a plane has multiple series but lacks one contrast class.

Full training metadata audit:

```text
historical selected streams   21886
strict selected streams       21334
wrong-slot substitutions        552
wrong-slot fraction             2.52%
strict semantic mismatches         0
```

Per stream:

```text
sagittal_fluid       251 removed
sagittal_structural   28 removed
coronal_fluid          2 removed
coronal_structural    34 removed
axial_fluid            0 removed
axial_structural     237 removed
```

Provided test metadata:

```text
historical selected streams 14
strict selected streams     13
wrong-slot substitutions     1
```

B9 uses exact semantics:

```text
*_fluid       -> Fluid_Sensitive == True only
*_structural  -> Fluid_Sensitive == False only
missing class -> None / masked
```

## B9 scientific contract

Only routing differs from B7.1. These remain unchanged:

```text
B5 initialization
B6 v1.2.1 weak labels
KneeMILNet architecture
16 slices/stream
batch size 2
4 epochs
1560 batches/epoch
encoder LR 1e-5
head LR 1e-4
same augmentation
TTA [-1,0,1]
5000 bootstrap replicates
no gold gradients
no gold early stopping
```

## Test B9 before training

```bash
pytest -q \
  tests/test_b6_report_labels.py \
  tests/test_b6_gold_audit.py \
  tests/test_b7_weak_supervision.py \
  tests/test_b9_strict_routing.py
```

Also verify the commands are installed:

```bash
which rsna-knee-b9
which rsna-knee-b9-eval
```

## Train B9-v1

```bash
rsna-knee-b9 \
  --config configs/b9_strict_routing.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b9_strict_routing
```

Expected outputs:

```text
runs/b9_strict_routing/
├── b9_model.pt
├── history.json
├── policy.json
├── routing_audit.json
└── supervision_plan.json
```

The model checkpoint is refreshed after each completed epoch.

## Inspect before gold evaluation

```bash
cat runs/b9_strict_routing/routing_audit.json
cat runs/b9_strict_routing/history.json
cat runs/b9_strict_routing/supervision_plan.json
```

Mandatory routing checks:

```text
routing_policy             fluid_sensitive_exact_v1
strict_semantic_mismatches 0
```

For each complete full epoch, expect approximately the B7.1 supervision contract:

```text
batches                         1560
study draws                     3120
active supervision cells       14123
positive cells                  6871
negative cells                  7252
```

If strict routing unexpectedly leaves a weak-training study with no selected MRI stream at all, `supervision_plan.json` will report that explicitly. Do not conceal or patch such filtering after the fact.

## B9 gold evaluation

Only after inspecting the artifacts:

```bash
python - <<'PY'
import yaml
with open('configs/b9_strict_routing.yaml') as f:
    c=yaml.safe_load(f)
c['num_workers']=0
c['persistent_workers']=False
with open('/tmp/b9_eval.yaml','w') as f:
    yaml.safe_dump(c,f,sort_keys=False)
print('/tmp/b9_eval.yaml')
PY

rsna-knee-b9-eval \
  --config /tmp/b9_eval.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b9_strict_routing/b9_model.pt \
  --out-root runs/b9_strict_routing/gold_eval
```

Primary benchmark:

```text
B7.1 = 0.5644802945
```

Then compare B7.1 -> B9:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b7_1_full_coverage/gold_eval/gold_predictions.csv \
  --compare-oof runs/b9_strict_routing/gold_eval/gold_predictions.csv \
  --n-bootstrap 5000 \
  --out runs/b9_strict_routing/gold_eval/b71_vs_b9.json
```

For this orientation, positive `median_difference` favors B9 and `probability_b_better` is `P(B9 > B7.1)`.

## Preserve these artifacts

```text
runs/b5_report_ssl/
runs/b6_report_labels_v121/
runs/b7_weak_supervision/
runs/b7_1_full_coverage/
runs/b8_spatial_anatomy/
runs/b9_strict_routing/
```

Do not delete completed prediction/evaluation files; they are the experiment audit trail.

## Interpretation rule

The 58 gold studies are now a repeated development/model-selection set. Do not tune B9 target-specific routing, restore individual substituted streams, change weak-label weights, select target-specific model winners, or optimize ensemble weights from the first B9 gold result and then call that result independent validation.
