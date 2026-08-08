# External four-image knee MRI test fixture

This directory contains **four online knee MRI studies for technical testing**. It is not the competition validation set and must not be used as a leaderboard/scientific macro-AUC benchmark.

The materializer downloads four openly licensed Wikimedia Commons knee MRI images, keeps the source JPEGs under `source_jpgs/`, and wraps each published slice into a seven-frame synthetic DICOM. The same four studies are exposed in two layouts:

```text
fixtures/external_validation/
  test.csv
  test_series.csv
  test_images/
    EXTVAL_ACL_001/...
    EXTVAL_MEDMEN_001/...
    EXTVAL_BAKER_001/...
    EXTVAL_REFERENCE_001/...

  validation.csv
  validation_series.csv
  validation_images/
    ...same four studies...

  source_jpgs/
  sources.csv
  materialization.json
```

Use `test.csv` / `test_series.csv` / `test_images/` when you want the production code to consume these four studies exactly like a test set. `validation.csv` exists only as a sparse sanity-label file: it contains source-supported positive cells and leaves every unspecified target as `NaN`.

The seven repeated DICOM frames are **not an original MRI volume**. They are only for decoding, routing, resizing, 2.5D construction, missing-stream masking, and model/inference plumbing.

## Four sources

1. `EXTVAL_ACL_001` — anterior cruciate ligament rupture, sagittal PD-weighted MRI. Hellerhoff, CC BY-SA 3.0.
2. `EXTVAL_MEDMEN_001` — grade 2 medial meniscal tear, coronal proton-density MRI. Nicolas Lefevre et al., CC BY 4.0.
3. `EXTVAL_BAKER_001` — Baker cyst in a patient with ACL rupture. Hellerhoff, CC BY-SA 3.0.
4. `EXTVAL_REFERENCE_001` — sagittal PD TSE FS knee MRI; the source supplies no pathology label. Ptrump16, CC BY-SA 4.0.

Full source URLs, attribution, licenses, findings, and downloaded SHA-256 hashes are recorded in `sources.csv`.

## Test all four DICOM studies

```bash
python -m rsna_knee.cli preflight \
  --data-root fixtures/external_validation \
  --split test \
  --sample-size 4 \
  --max-decode-failure-rate 0 \
  --max-file-decode-failure-rate 0 \
  --out runs/external_test_preflight.json
```

You can also check the sparse labeled copy:

```bash
python -m rsna_knee.cli preflight \
  --data-root fixtures/external_validation \
  --split validation \
  --sample-size 4 \
  --max-decode-failure-rate 0 \
  --max-file-decode-failure-rate 0 \
  --out runs/external_validation_preflight.json
```

## Re-materialize manually

```bash
source .venv/bin/activate
pip install pillow
PYTHONPATH=src python scripts/materialize_external_validation.py \
  --output fixtures/external_validation \
  --overwrite
```

Never append these four studies to the competition training data and never use their predictions to select final competition hyperparameters.
