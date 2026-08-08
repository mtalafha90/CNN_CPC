# Test and validation workflow

The repository now has two deliberately separate resources. Do not mix them.

## 1. Four-image external test set

`fixtures/external_validation/` contains four openly licensed knee MRI examples downloaded from Wikimedia Commons and converted into the competition-like test contract:

```text
test.csv
test_series.csv
test_images/<StudyInstanceUID>/<SeriesInstanceUID>/image.dcm
```

Purpose:

- DICOM decoding;
- series routing;
- 2.5D preprocessing;
- missing-stream masking;
- CPU multiprocessing/preflight;
- model/inference plumbing once trained checkpoints exist.

Run a strict four-study preflight:

```bash
python -m rsna_knee.cli preflight \
  --data-root fixtures/external_validation \
  --split test \
  --sample-size 4 \
  --max-decode-failure-rate 0 \
  --max-file-decode-failure-rate 0 \
  --out runs/external_test_preflight.json
```

A parallel `validation.csv` / `validation_series.csv` / `validation_images/` copy provides **sparse source-supported sanity labels**. It is too small and incompletely labeled for meaningful macro ROC-AUC. Targets not explicitly documented by the online source remain `NaN`; caption silence is never a negative label.

## 2. Real competition validation set

The real evaluation set remains the leakage-safe nested gold folds built from the official competition `train.csv`.

For outer fold `k`:

- `outer_validation`: untouched official gold studies used only for final OOF evaluation;
- `inner_selection`: official gold studies used to choose Phase-A training duration;
- `gold_train`: remaining official gold studies available to training.

Export the exact validation manifest for every fold before training:

```bash
mkdir -p runs/validation
for f in 0 1 2; do
  python -m rsna_knee.cli validation-manifest \
    --config configs/train_local.yaml \
    --fold "$f" \
    --out "runs/validation/fold${f}.csv"
done
```

Each manifest contains `StudyInstanceUID`, fold, role, outer fold, inner fold, and the official target cells. These are the validation assignments that should be used when interpreting OOF performance.

## Important separation

Never append the four external Wikimedia studies to competition training. Never use their predictions to choose final competition hyperparameters. The external four-study set tests the pipeline; the official nested gold folds estimate competition performance.
