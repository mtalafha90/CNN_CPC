# CNN_CPC Training From Zero

This is the clean end-to-end guide for setting up `CNN_CPC` on a fresh Linux machine and reproducing the current experiment path.

> **Repository experiment snapshot — 2026-08-09:** B0-B4.3 and fixed B1/B4 ensembles are complete; B5 image-report representation learning is running. Measured scores are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

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
-> B5 competition-only image-report representation learning
-> unchanged B4 probe on B5 encoder
-> paired B4-vs-B5 evaluation
```

The earlier Stage-2/co-training code remains in the repository, but the current active experiment branch is B5 representation learning because the completed B0-B4 evidence points to representation quality / small-gold variance as the more useful next lever.

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

Evaluate:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof \
    runs/stage1_random/fold0/oof.csv \
    runs/stage1_random/fold1/oof.csv \
    runs/stage1_random/fold2/oof.csv \
  --n-bootstrap 5000 \
  --out runs/stage1_random/evaluation.json
```

Reference result:

```text
B0 macro AUC = 0.4762536432
```

## 11. Strong competition-only MRI SSL

Create a strong SSL config from the verified local config. The completed reference schedule used:

```yaml
ssl_output_dir: runs/ssl_strong
ssl_epochs: 8
ssl_max_batches_per_epoch: 1000
ssl_batch_size: 3
ssl_n_slices: 9
ssl_positions_per_stream: 2
ssl_projection_dim: 256
ssl_temperature: 0.15
ssl_metadata_weight: 0.25
ssl_lr: 0.0002
ssl_min_lr: 0.000001
ssl_weight_decay: 0.0001
pretrained: false
allow_external_pretrained: false
```

Run:

```bash
python -m rsna_knee.cli pretrain \
  --config configs/train_local_ssl_pretrain.yaml
```

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

## 12. B1 strong-SSL Stage-1

Create `configs/train_local_ssl_strong.yaml` pointing to the strong checkpoint with source `competition_training_data` and train the same three folds.

Reference result:

```text
B1 macro AUC = 0.5030284974
95% CI      = [0.4474281231, 0.5566718294]
```

## 13. B2/B3 controlled alternatives

These are already implemented as separate commands:

```bash
rsna-knee-b2 --config configs/train_local_ssl_b2.yaml --fold <0|1|2>
rsna-knee-b3 --config configs/train_local_ssl_b3.yaml --fold <0|1|2>
```

Completed reference results:

```text
B2 = 0.4993244663  -> rejected
B3 = 0.4944652486  -> rejected globally
```

See the dedicated experiment docs for exact policies.

## 14. B4 frozen representation probe

Extract deterministic gold features:

```bash
rsna-knee-b4 extract \
  --config configs/train_local_ssl_strong.yaml \
  --split train \
  --scope gold \
  --out runs/b4_frozen_ssl/gold_features.npz
```

Expected:

```text
features = [58, 6, 2304]
finite   = true
```

Run original B4 nested probe:

```bash
rsna-knee-b4 nested \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b4_frozen_ssl/gold_features.npz \
  --out-root runs/b4_frozen_ssl \
  --n-bootstrap 5000
```

Reference result:

```text
B4 macro AUC = 0.5137567459
95% CI      = [0.4619827141, 0.5642366629]
```

## 15. B4.1-B4.3 diagnostics

Commands:

```bash
rsna-knee-b4-shared   --config configs/train_local_ssl_strong.yaml --features runs/b4_frozen_ssl/gold_features.npz --out-root runs/b4_1_shared_ssl --n-bootstrap 5000
rsna-knee-b4-grouped  --config configs/train_local_ssl_strong.yaml --features runs/b4_frozen_ssl/gold_features.npz --out-root runs/b4_2_grouped_ssl --n-bootstrap 5000
rsna-knee-b4-crossval --config configs/train_local_ssl_strong.yaml --features runs/b4_frozen_ssl/gold_features.npz --out-root runs/b4_3_crossval_ssl --n-bootstrap 5000
```

Completed reference results:

```text
B4.1 = 0.4847792672
B4.2 = 0.4901328905
B4.3 = 0.4966083942
```

All were rejected. Do not create further B4 selector variants from the same outer labels.

## 16. Fixed ensemble check

The fixed B1+B4 50:50 rank average reached `0.5167`, but paired bootstrap versus B4 gave only `P=0.5544`. Keep it as a fixed candidate; do not tune weights.

## 17. B5 image-report representation learning

B5 is the current active stage.

Run:

```bash
rsna-knee-b5 \
  --config configs/train_local_ssl_strong.yaml \
  --checkpoint runs/ssl_strong/ssl_encoder.pt \
  --out-root runs/b5_report_ssl
```

B5 uses only the 4,349 report-only competition studies and excludes all 58 gold studies. No external language model or image weights are used.

## 18. B5 frozen probe

After B5 finishes:

```bash
mkdir -p runs/b5_frozen_probe

rsna-knee-b4 extract \
  --config configs/train_local_ssl_strong.yaml \
  --checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --split train \
  --scope gold \
  --out runs/b5_frozen_probe/gold_features.npz

rsna-knee-b4 nested \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b5_frozen_probe/gold_features.npz \
  --out-root runs/b5_frozen_probe \
  --n-bootstrap 5000
```

Do not change the B4 probe for this first B5 comparison.

## 19. B4 versus B5

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b4_frozen_ssl/oof.csv \
  --compare-oof runs/b5_frozen_probe/oof.csv \
  --n-bootstrap 5000 \
  --out runs/b4_vs_b5.json
```

This is the primary B5 representation test.

## 20. Reporting discipline

Keep these terms distinct:

- preflight/audit;
- smoke;
- individual OOF result;
- paired comparison;
- model-selection CV;
- leaderboard score.

Because many method decisions have now been informed by the same 58 gold studies, the campaign-level OOF table is model-selection CV. Do not claim it as a pristine independent hidden-test estimate.

Do not enter a B5 score until its training, frozen feature extraction, unchanged B4 probe and paired B4-vs-B5 evaluation have completed.
