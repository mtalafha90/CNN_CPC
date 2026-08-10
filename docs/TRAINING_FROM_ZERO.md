# CNN_CPC Training From Zero

This is the clean end-to-end guide for setting up `CNN_CPC` on a fresh Linux machine and reproducing the current experiment path.

> **Repository snapshot — 2026-08-10:** package `0.13.0`. B7.1 full-corpus weak supervision is the current best standalone development model at macro AUC `0.5644802945`. The fixed B5+B7.1 rank ensemble is rejected. **B8 spatial-anatomy learning is the current training experiment and has no gold score yet.** Exact results are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

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
-> B0-B4 controlled baselines/ablations
-> B5 image-report representation learning
-> B6 frozen structured multilingual report labels
-> B7-v1 direct weak supervision
-> B7.1 full-corpus weak supervision [current leader]
-> fixed B5+B7.1 rank ensemble [rejected]
-> B8 spatial anatomy learning [current training experiment]
-> inspect B8 training artifacts
-> one frozen B8 gold development evaluation
-> paired B7.1 -> B8 bootstrap
```

The earlier Stage-2/co-training and historical B4 selector code remain in the repository for reproducibility, but they are not the current active development path.

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

Expected current package version:

```text
0.13.0
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

After accepting the competition rules, place the data locally and set:

```bash
export DATA_ROOT="/path/to/rsna-knee-abnormality-detection"
```

Expected metadata files:

```text
train.csv
train_series.csv
test.csv
test_series.csv
sample_submission.csv
```

Do not commit competition images, reports, credentials or machine-local paths.

## 5. Inspect and preflight

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

Verified release:

```text
studies=4407
gold=58
report-only=4349
series=24371
```

Full audit reference:

```text
21,886 selected series checked
21,886 selected series decoded
732,554 / 732,556 candidate DICOM files decoded
2 partial one-file failures
0 selected series failed
```

## 6. Historical controlled ladder

Reference development scores:

```text
B0 random                         0.4762536432
B1 strong SSL                    0.5030284974
B2 lower encoder LR              0.4993244663
B3 pathology-aware MIL           0.4944652486
B4 frozen SSL + classical        0.5137567459
B4.1 shared policy               0.4847792672
B4.2 grouped policies            0.4901328905
B4.3 two-way CV selector         0.4966083942
B1+B4 fixed rank                 0.5167
B5 image-report SSL              0.5243650851
B7-v1 direct weak supervision    0.5397724412
B7.1 full coverage               0.5644802945
B5+B7.1 fixed rank ensemble      0.5540141184  [rejected]
B8 spatial anatomy               pending        [training]
```

Do not reopen B4 selector searches or ensemble-weight searches on the same 58 development labels.

## 7. Strong SSL reference

Checkpoint:

```text
runs/ssl_strong/ssl_encoder.pt
```

Reference coverage:

```text
8 epochs
8,000 batches
24,000 study draws
~5.52 corpus passes
238,274 active 2.5D examples
```

## 8. B5 reference

Completed representation checkpoint:

```text
runs/b5_report_ssl/b5_encoder.pt
```

B5 used only the 4,349 report-only competition studies and excluded all 58 gold studies from representation training.

Frozen unchanged B4 probe:

```text
macro AUC = 0.5243650851
95% CI   = [0.4728108406, 0.5761619105]
```

## 9. B6 frozen structured report labels

Expected artifact root:

```text
runs/b6_report_labels_v121/
├── training_targets.csv
├── policy.json
└── audit.json
```

Frozen training supervision:

```text
report-only rows                4349
active studies                  3120
usable cells                   14123
positive cells                  6871
negative cells                  7252
```

B6 v1.2.1 is frozen. Do not patch parser behavior from later B7/B8 gold outcomes.

## 10. B7-v1 reference

Checkpoint:

```text
runs/b7_weak_supervision/b7_model.pt
```

Result:

```text
macro AUC = 0.5397724412
```

B7-v1 used only 500 batches/epoch and therefore about 1.28 nominal corpus passes over four epochs.

## 11. B7.1 full-corpus reference — current leader

Configuration:

```text
configs/b7_1_full_coverage.yaml
```

Checkpoint:

```text
runs/b7_1_full_coverage/b7_model.pt
```

Training contract:

```text
active studies       3120
usable cells        14123
batch size              2
batches/epoch        1560
study draws/epoch    3120
epochs                  4
```

Result:

```text
macro AUC = 0.5644802945
95% CI   = [0.5052432984, 0.6229422178]
```

This is the current benchmark B8 must beat.

## 12. Fixed B5+B7.1 rank ensemble — closed

The one predeclared 50:50 percentile-rank blend scored:

```text
0.5540141184
```

below B7.1. Do not search other weights or target-specific combinations.

## 13. B8 install/test

After pulling B8 code:

```bash
cd "$REPO"
git pull --ff-only origin main
python -m pip install -e .

pytest -q \
  tests/test_b6_report_labels.py \
  tests/test_b6_gold_audit.py \
  tests/test_b7_weak_supervision.py \
  tests/test_b8_anatomy_spatial.py
```

B8 changes MRI memory from 96 globally pooled slice tokens to 384 coarse spatial tokens while retaining B7.1 initialization and B6 supervision.

## 14. Train B8 — current step

```bash
rsna-knee-b8 \
  --config configs/b8_spatial_anatomy.yaml \
  --data-root "$DATA_ROOT" \
  --b71-checkpoint runs/b7_1_full_coverage/b7_model.pt \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b8_spatial_anatomy
```

Expected outputs:

```text
runs/b8_spatial_anatomy/
├── b8_model.pt
├── history.json
├── policy.json
└── supervision_plan.json
```

The checkpoint is refreshed after every completed epoch.

## 15. Inspect B8 before gold evaluation

```bash
cat runs/b8_spatial_anatomy/history.json
cat runs/b8_spatial_anatomy/supervision_plan.json
```

Verify:

```text
4 completed epochs
1560 batches/epoch unless runtime budget stopped a later epoch
3120 study draws for every complete epoch
14123 active supervision cells for every complete full pass
6871 positive / 7252 negative cells for every complete full pass
finite monotonic/reasonable loss trajectory
no unexpected supervision/MRI filtering changes
```

Do not run the gold evaluation until the training artifacts have been inspected.

## 16. First B8 gold development evaluation

After the frozen run is accepted:

```bash
rsna-knee-b8-eval \
  --config configs/b8_spatial_anatomy.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b8_spatial_anatomy/b8_model.pt \
  --out-root runs/b8_spatial_anatomy/gold_eval
```

A runtime-only `num_workers: 0` copy of the config may be used for evaluation if DataLoader teardown is noisy; this does not alter the model.

Primary benchmark:

```text
B7.1 macro AUC = 0.5644802945
```

Primary paired comparison is B7.1 -> B8 using 5,000 study-level bootstrap replicates.

## 17. Reporting discipline

Keep these terms distinct:

- preflight/audit;
- training run;
- gold development score;
- paired comparison;
- model-selection CV;
- leaderboard score.

Because many method decisions have been informed by the same 58 gold studies, the campaign-level table is model-selection CV. Do not claim it as a pristine independent hidden-test estimate.

Do not tune B8 grid size, anatomy-prior strength, epochs, target-specific priors or ensemble weights from the first B8 gold result and then reuse the same 58 studies as if untouched.
