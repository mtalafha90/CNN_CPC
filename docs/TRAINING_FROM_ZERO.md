# CNN_CPC Training From Zero

This is the clean end-to-end guide for setting up `CNN_CPC` on a fresh Linux machine and reproducing the current experiment path.

> **Repository snapshot — 2026-08-10:** package `0.14.0`. **B7.1 is the current best standalone development model at macro AUC `0.5644802945`; B8 is rejected at `0.5300962807`; B9 strict semantic routing is the current predeclared experiment.** Exact results are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Current reproducible ladder

```text
clone/update
-> Conda environment
-> install + tests
-> verify one GPU
-> place competition data
-> inspect CSVs
-> DICOM preflight + full audit
-> strong competition-only MRI SSL
-> B5 image-report representation learning
-> B6 frozen structured multilingual report labels
-> B7-v1 direct weak supervision
-> B7.1 full-corpus weak supervision [current leader]
-> fixed B5+B7.1 rank ensemble [rejected]
-> B8 spatial anatomy [rejected]
-> B9 strict semantic routing [current]
-> inspect B9 routing/training artifacts
-> one frozen B9 gold development evaluation
-> paired B7.1 -> B9 bootstrap
```

## 1. Clone/update

```bash
export REPO="/path/to/CNN_CPC"
cd "$(dirname "$REPO")"
git clone https://github.com/mtalafha90/CNN_CPC.git  # first time only
cd "$REPO"
git checkout main
git pull --ff-only origin main
```

## 2. Environment

```bash
conda create -n rsna-knee python=3.12 -y
conda activate rsna-knee
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install pytest pillow kaggle
python -m pip check
```

Verify:

```bash
python - <<'PY'
import rsna_knee
print('version:', rsna_knee.__version__)
print('package:', rsna_knee.__file__)
PY

pytest -q
python -m compileall -q src tests kaggle scripts
```

Expected package version:

```text
0.14.0
```

## 3. GPU

```bash
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
```

The conservative production path uses one GPU and no `torchrun`.

## 4. Competition data

```bash
export DATA_ROOT="/path/to/rsna-knee-abnormality-detection"
```

Expected files:

```text
train.csv
train_series.csv
test.csv
test_series.csv
sample_submission.csv
```

Verified release:

```text
training studies  4407
gold studies        58
report-only        4349
training series   24371
```

## 5. Inspect/preflight

```bash
python -m rsna_knee.cli inspect --data-root "$DATA_ROOT"

python -m rsna_knee.cli preflight \
  --data-root "$DATA_ROOT" \
  --split train \
  --sample-size 24 \
  --max-decode-failure-rate 0.05 \
  --max-file-decode-failure-rate 0.05 \
  --out runs/preflight_train.json
```

Historical full DICOM audit:

```text
21,886 historically selected series checked
21,886 decoded
732,554 / 732,556 candidate DICOM files decoded
2 partial one-file failures
0 selected series failed
```

## 6. Current measured ladder

```text
B0 random                         0.4762536432
B1 strong SSL                    0.5030284974
B4 frozen SSL + classical        0.5137567459
B5 image-report SSL              0.5243650851
B7-v1 direct weak supervision    0.5397724412
B7.1 full coverage               0.5644802945  [current leader]
B5+B7.1 fixed rank ensemble      0.5540141184  [rejected]
B8 spatial anatomy               0.5300962807  [rejected]
B9 strict routing                pending       [current]
```

## 7. Required retained artifacts

B5 initialization:

```text
runs/b5_report_ssl/b5_encoder.pt
```

B6 frozen weak labels:

```text
runs/b6_report_labels_v121/
├── training_targets.csv
├── policy.json
└── audit.json
```

B7.1 benchmark:

```text
runs/b7_1_full_coverage/b7_model.pt
runs/b7_1_full_coverage/gold_eval/gold_predictions.csv
```

Frozen B6 scope:

```text
active studies  3120
usable cells   14123
positive        6871
negative        7252
```

## 8. Why B9 exists

A label-free audit of `train_series.csv` found that the historical dual-stream selector can populate a missing contrast slot with a same-plane acquisition from the opposite contrast class.

```text
historical selected streams  21886
strict selected streams      21334
wrong-slot substitutions       552
wrong-slot fraction            2.52%
strict semantic mismatches        0
```

The three-study test metadata contain one analogous false sagittal-fluid assignment.

B9 exact rule:

```text
*_fluid       -> Fluid_Sensitive == True only
*_structural  -> Fluid_Sensitive == False only
missing class -> None / presence mask False
```

This is the only scientific change versus B7.1.

## 9. Test B9 implementation

```bash
pytest -q \
  tests/test_b6_report_labels.py \
  tests/test_b6_gold_audit.py \
  tests/test_b7_weak_supervision.py \
  tests/test_b9_strict_routing.py

which rsna-knee-b9
which rsna-knee-b9-eval
```

## 10. Train B9

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

## 11. Inspect B9 before gold evaluation

```bash
cat runs/b9_strict_routing/routing_audit.json
cat runs/b9_strict_routing/history.json
cat runs/b9_strict_routing/supervision_plan.json
```

Mandatory routing condition:

```text
strict_semantic_mismatches = 0
routing_policy = fluid_sensitive_exact_v1
```

Every complete epoch should retain the B7.1 supervision contract:

```text
batches                1560
study draws            3120
active cells          14123
positive cells         6871
negative cells         7252
```

## 12. B9 gold development evaluation

Use a runtime-only worker-safe config if desired:

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
```

Then:

```bash
rsna-knee-b9-eval \
  --config /tmp/b9_eval.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b9_strict_routing/b9_model.pt \
  --out-root runs/b9_strict_routing/gold_eval
```

Primary benchmark:

```text
B7.1 macro AUC = 0.5644802945
```

Paired comparison:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b7_1_full_coverage/gold_eval/gold_predictions.csv \
  --compare-oof runs/b9_strict_routing/gold_eval/gold_predictions.csv \
  --n-bootstrap 5000 \
  --out runs/b9_strict_routing/gold_eval/b71_vs_b9.json
```

Positive `median_difference` favors B9.

## 13. Reporting discipline

The 58 gold studies are a repeated development/model-selection set. Do not tune target-specific B9 routing, restore individual substituted streams, change weak-label weights, select per-target winners, or optimize ensemble weights after seeing B9 gold results and then describe the result as independent validation.
