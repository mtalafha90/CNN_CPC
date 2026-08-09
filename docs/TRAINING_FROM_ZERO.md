# CNN_CPC Training From Zero

This is the clean end-to-end guide for setting up `CNN_CPC` on a fresh Linux machine and reproducing the current experiment path.

> **Repository experiment snapshot — 2026-08-09:** B0-B4.3 and fixed B1/B4 ensembles are complete. B5 image-report representation training completed all four predefined epochs cleanly; the unchanged B4 frozen probe on the B5 encoder is now pending. Measured scores are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Current reproducible ladder

```text
clone/update
-> Conda environment
-> install + tests
-> verify one GPU
-> place competition data
-> create local config
-> inspect CSVs
-> DICOM preflight
-> full selected-series audit
-> verify report parsing
-> B0 random baseline
-> strong competition-only MRI SSL
-> B1 strong-SSL Stage-1
-> B2/B3 controlled neural alternatives
-> B4 frozen representation probe
-> B4.1/B4.2/B4.3 selector diagnostics
-> fixed B1+B4 ensemble check
-> B5 competition-only image-report representation learning [complete]
-> unchanged B4 probe on B5 encoder [current]
-> paired B4-vs-B5 evaluation
```

The earlier Stage-2/co-training code remains in the repository, but the current active experiment is the B5 frozen representation probe because the completed B0-B4 evidence points to representation quality / small-gold variance as the more useful lever.

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

Current package version for B5 is `0.10.0`.

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

## 4. External technical fixture

```bash
pytest -q tests/test_external_fixture.py

mkdir -p runs
python -m rsna_knee.cli preflight \
  --data-root fixtures/external_validation \
  --split test \
  --sample-size 4 \
  --max-decode-failure-rate 0 \
  --max-file-decode-failure-rate 0 \
  --out runs/external_test_preflight.json
```

This fixture validates software plumbing only. Never use it for scientific model selection.

## 5. Competition data

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

## 6. Create local config

```bash
cp configs/train.yaml configs/train_local.yaml
```

Patch at least:

```yaml
data_root: /path/to/rsna-knee-abnormality-detection
output_dir: runs/stage1_random
competition_mode: true
requested_gpus: 1
runtime_budget_hours: 8.5
pretrained: false
allow_external_pretrained: false
```

Keep the validation/submission TTA contract unchanged unless a new experiment is explicitly declared.

## 7. Inspect data

```bash
python -m rsna_knee.cli inspect --data-root "$DATA_ROOT"
```

Verified reference release:

```text
studies=4407
gold=58
unlabeled=4349
reports_present=4407
series=24371
```

If a later release differs, document the difference before training.

## 8. Preflight

Train:

```bash
python -m rsna_knee.cli preflight \
  --data-root "$DATA_ROOT" \
  --split train \
  --sample-size 24 \
  --max-decode-failure-rate 0.05 \
  --max-file-decode-failure-rate 0.05 \
  --out runs/preflight_train.json
```

Test:

```bash
python -m rsna_knee.cli preflight \
  --data-root "$DATA_ROOT" \
  --split test \
  --sample-size 24 \
  --max-decode-failure-rate 0.05 \
  --max-file-decode-failure-rate 0.05 \
  --out runs/preflight_test.json
```

Verified reference:

```text
train: 121/121 selected streams, 4045/4045 files decoded
test:   14/14 selected streams, 533/533 files decoded
```

## 9. Full audit

```bash
python -m rsna_knee.cli audit \
  --config configs/train_local.yaml \
  --out-dir runs/audit
```

Verified reference:

```text
21,886 selected series checked
21,886 selected series decoded
732,554 / 732,556 candidate DICOM files decoded
2 partial one-file failures
0 selected series failed
```

Repeat the full audit when the data, DICOM decoder or routing code changes—not after every documentation change.

## 10. B0 random baseline

Train three folds:

```bash
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 0
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 1
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 2
```

Reference result:

```text
B0 macro AUC = 0.4762536432
```

## 11. Strong competition-only MRI SSL

Reference checkpoint:

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

## 12. B1-B3 references

```text
B1 strong SSL       = 0.5030284974
B2 lower encoder LR = 0.4993244663
B3 pathology MIL    = 0.4944652486
```

B2 and B3 were rejected as replacements for B1.

## 13. B4 frozen representation probe

Reference result:

```text
B4 macro AUC = 0.5137567459
95% CI      = [0.4619827141, 0.5642366629]
```

The deterministic frozen feature cache has shape `[58, 6, 2304]`.

## 14. B4.1-B4.3 diagnostics

```text
B4.1 shared policy       = 0.4847792672
B4.2 grouped policies    = 0.4901328905
B4.3 two-way CV selector = 0.4966083942
```

All were rejected. Do not create further B4 selector variants from the same outer labels.

## 15. Fixed ensemble check

The fixed B1+B4 50:50 rank average reached `0.5167`, but paired bootstrap versus B4 gave only `P=0.5544`. Keep it as a fixed candidate; do not tune weights.

## 16. B5 image-report representation learning — complete

B5 used only the 4,349 report-only competition studies and excluded all 58 gold studies. No external language model or image weights were used.

Checkpoint:

```text
runs/b5_report_ssl/b5_encoder.pt
```

Verified text/coverage contract:

```text
reports used                 4349
gold studies excluded          58
TF-IDF features             20000
SVD dimension                 256
SVD explained variance      0.58477
unique report groups          4198
duplicate report rows          151
empty reports                    0
study draws                  16000
approx corpus passes        3.6790
batches                       4000
active 2.5D examples        158886
report queue                   256
```

Training history:

```text
total loss    5.5204 -> 4.7049
image contrast 3.0068 -> 2.8937
metadata       0.4472 -> 0.3684
report NCE     4.6031 -> 3.2901
report cosine  0.8015 -> 0.5924
budget limited false for all epochs
```

## 17. B5 frozen feature extraction — complete

The B5 encoder has already been frozen and applied to all 58 gold studies with the unchanged B4 extractor.

Verified artifact:

```text
runs/b5_frozen_probe/gold_features.npz
```

Contract:

```text
studies                         58
feature shape        [58, 6, 2304]
encoder frozen                  true
encoder trainable parameters       0
checkpoint source competition_training_data
checkpoint epochs                 4
external pretrained            false
n_slices                          16
image size                       224
triplet gap                        1
metadata repair needed             0
```

## 18. Current task: unchanged B4 nested probe on B5 features

```bash
rsna-knee-b4 nested \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b5_frozen_probe/gold_features.npz \
  --out-root runs/b5_frozen_probe \
  --n-bootstrap 5000

cat runs/b5_frozen_probe/evaluation.json
```

Do not switch to B4.1/B4.2/B4.3. The first B5 test must use the original B4 protocol to isolate the representation change.

## 19. B4 versus B5

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b4_frozen_ssl/oof.csv \
  --compare-oof runs/b5_frozen_probe/oof.csv \
  --n-bootstrap 5000 \
  --out runs/b4_vs_b5.json

cat runs/b4_vs_b5.json
```

Orientation:

```text
A = B4 image-only strong SSL representation
B = B5 image-report representation
```

Positive `median_difference` and `probability_b_better > 0.5` favor B5.

## 20. Reporting discipline

Keep these terms distinct:

- preflight/audit;
- smoke;
- individual OOF result;
- paired comparison;
- model-selection CV;
- leaderboard score.

Because many method decisions have now been informed by the same 58 gold studies, the campaign-level OOF table is model-selection CV. Do not claim it as a pristine independent hidden-test estimate.

Do not enter a B5 score until the unchanged B4 nested probe and paired B4-vs-B5 evaluation have completed.
