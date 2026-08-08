# Validation workflow

There are two different things called "validation" in this repository. They must not be mixed.

## 1. Four-image external smoke validation

`fixtures/external_validation/` contains four openly licensed knee MRI examples downloaded from Wikimedia Commons and converted into competition-like DICOM/CSV structure.

Purpose:

- DICOM decoding;
- plane/sequence routing;
- 2.5D preprocessing;
- missing-stream masking;
- CPU multiprocessing/preflight;
- later model-forward sanity checks.

It is **not** used for competition model selection and is too small/partially labeled for a meaningful macro ROC-AUC.

Materialize the fixture manually if needed:

```bash
source .venv/bin/activate
pip install pillow
PYTHONPATH=src python scripts/materialize_external_validation.py \
  --output fixtures/external_validation \
  --overwrite
```

Check all four studies:

```bash
python -m rsna_knee.cli preflight \
  --data-root fixtures/external_validation \
  --split validation \
  --sample-size 4 \
  --max-decode-failure-rate 0 \
  --max-file-decode-failure-rate 0 \
  --out runs/external_validation_preflight.json
```

The source-supported positive cells in `validation.csv` are deliberately sparse. Targets not explicitly documented by the source remain `NaN`; report/caption silence is never treated as a negative label.

## 2. Real competition validation set

The real evaluation remains the leakage-safe nested gold folds built from the official competition `train.csv`.

For outer fold `k`:

- `outer_validation`: untouched gold studies used only for final OOF evaluation;
- `inner_selection`: gold studies used to choose the training duration in Phase A;
- `gold_train`: remaining gold studies available to training.

Export the exact real validation manifest before training.

Fold 0:

```bash
python -m rsna_knee.cli validation-manifest \
  --config configs/train_local.yaml \
  --fold 0 \
  --out runs/validation/fold0.csv
```

Fold 1 and fold 2:

```bash
python -m rsna_knee.cli validation-manifest \
  --config configs/train_local.yaml \
  --fold 1 \
  --out runs/validation/fold1.csv

python -m rsna_knee.cli validation-manifest \
  --config configs/train_local.yaml \
  --fold 2 \
  --out runs/validation/fold2.csv
```

These files contain the official target cells and exact fold/role assignments. This is the validation data that should be used when interpreting OOF performance.

## Important separation

Never append the four Wikimedia fixture studies to the competition training data. Never use their predictions to choose final competition hyperparameters. The fixture tests plumbing; the official gold folds estimate model performance.
