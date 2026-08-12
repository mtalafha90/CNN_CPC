# CNN_CPC Training From Zero

This is the clean setup/reproduction guide for the current `CNN_CPC` state.

> **Repository snapshot — 2026-08-12:** package `0.24.1`. **B13 is the reused-gold development champion at `0.6293565948`. B15 is completed: weak-v2 gate passed, one-look gold `0.6209002783`, no global improvement over B13.** Exact results are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

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
0.24.1
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

The production path uses one GPU and no DDP/`torchrun`.

## 4. Competition data

```bash
export DATA_ROOT="/path/to/rsna-knee-abnormality-detection"
```

Expected release:

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
B7.1 full coverage               0.5644802945
B8 spatial anatomy               0.5300962807
B9 strict routing                0.5334962669
B10 physical scale               0.5523982721
B11.1 quantile pseudo labels     0.5506902702
B12 all real series              0.5660915179
B13 ImageNet hierarchy           0.6293565948  CHAMPION
B14 ImageNet full tokens         0.6197914249  REJECTED
B15 ImageNet->MRI SSL hierarchy  0.6209002783  NO GLOBAL GOLD IMPROVEMENT
```

B11-v1 failed viability; B12.1 was implemented but skipped.

## 7. Historical build path

For full historical reproduction, follow the experiment-specific documents from strong SSL through B5/B6/B7, B8-B12, B13/B14 and B15. The current canonical ledger is [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

Core retained historical artifacts include:

```text
runs/ssl_strong/ssl_encoder.pt
runs/b5_report_ssl/b5_encoder.pt
runs/b6_report_labels_v121/
runs/b7_1_full_coverage/b7_model.pt
runs/b12_variable_series/audit/series_policy.json
runs/b13_imagenet/b13_model.pt
```

## 8. Frozen weak-v2 surface

The B15-era nested validation surface is:

```text
surface                 weak_b6_holdout_v2
active studies          3120
weak-train studies      2497
holdout studies          623
holdout cells           2875
positive / negative  1407 / 1468
report-group overlap       0
manifest SHA
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

It measures B6 teacher agreement only. Do not regenerate it based on model performance.

## 9. Reproduce completed B15 training path

Set:

```bash
export B6_ROOT="runs/b6_report_labels_v121"
export SERIES_POLICY="runs/b12_variable_series/audit/series_policy.json"
export WEAK_V2="runs/weak_holdout_v2"
```

B15 SSL:

```bash
rsna-knee-b15-ssl \
  --config configs/b15_mri_ssl.yaml \
  --data-root "$DATA_ROOT" \
  --weak-holdout-root "$WEAK_V2" \
  --out-root runs/b15_mri_ssl
```

Matched B13-v2 control:

```bash
rsna-knee-b13-v2 \
  --config configs/b15_mri_ssl.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --series-policy "$SERIES_POLICY" \
  --weak-holdout-root "$WEAK_V2" \
  --out-root runs/b13_v2_control
```

B15 downstream:

```bash
rsna-knee-b15 \
  --config configs/b15_mri_ssl.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --series-policy "$SERIES_POLICY" \
  --weak-holdout-root "$WEAK_V2" \
  --ssl-checkpoint runs/b15_mri_ssl/b15_ssl_encoder.pt \
  --out-root runs/b15_mri_ssl/downstream
```

These commands reproduce the frozen experiment; **do not use them to tune B15 after its gold result**.

## 10. Reproduce B15 validation

Control weak-v2:

```bash
rsna-knee-b15-weak-eval \
  --config configs/b15_mri_ssl.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b13_v2_control/b13_v2_control.pt \
  --b6-root "$B6_ROOT" \
  --weak-holdout-root "$WEAK_V2" \
  --mode control \
  --out-root runs/b13_v2_control/weak_eval
```

B15 weak-v2:

```bash
rsna-knee-b15-weak-eval \
  --config configs/b15_mri_ssl.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b15_mri_ssl/downstream/b15_model.pt \
  --b6-root "$B6_ROOT" \
  --weak-holdout-root "$WEAK_V2" \
  --mode b15 \
  --out-root runs/b15_mri_ssl/weak_eval
```

Paired gate:

```bash
rsna-knee-b15-compare \
  --config configs/b15_mri_ssl.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --weak-holdout-root "$WEAK_V2" \
  --control-predictions runs/b13_v2_control/weak_eval/weak_predictions.csv \
  --b15-predictions runs/b15_mri_ssl/weak_eval/weak_predictions.csv \
  --out runs/b15_mri_ssl/weak_eval/b13_v2_vs_b15.json
```

Observed gate:

```text
control 0.5652498118
B15    0.7319060415
paired median +0.1675245839
95% CI [+0.1124433208,+0.2165156305]
P=1.0000
PASS
```

B15 earned one gold look:

```bash
rsna-knee-b15-gold-eval \
  --config configs/b15_mri_ssl.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b15_mri_ssl/downstream/b15_model.pt \
  --gate-json runs/b15_mri_ssl/weak_eval/b13_v2_vs_b15.json \
  --out-root runs/b15_mri_ssl/gold_confirmation
```

Observed gold:

```text
B15 0.6209002783
B13 0.6293565948
```

## 11. Current next step

Do not rerun/tune B15 from this result. First audit B6 `positive`, `negated`, `uncertain`, and `unmentioned` states against expert truth. Any new supervision policy must be separately named and frozen.

## 12. Reporting discipline

The 58 gold studies are repeated development/model-selection data. Weak-v2 is teacher agreement only. Do not select target-specific winners, optimize blend weights, map unmentioned to negative by assumption, regenerate weak-v2, or describe local AUC as a hidden-test guarantee.

The hidden Kaggle evaluation remains the next independent performance signal.