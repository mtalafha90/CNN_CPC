# External four-study knee MRI technical fixture

This directory contains **four openly licensed online knee MRI examples for technical pipeline testing**. It is not the competition validation set and must never be used as a leaderboard/scientific macro-AUC benchmark.

The materializer keeps the source JPEGs under `source_jpgs/` and wraps each published image into a seven-frame synthetic DICOM so that the production code can exercise DICOM discovery, decoding, routing, resizing, 2.5D construction and inference plumbing.

## Layout

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

Use `test.csv`, `test_series.csv`, and `test_images/` when testing the production image-only test path.

`validation.csv` is a sparse sanity-label copy. It includes only pathology cells directly supported by the source material; all unspecified targets remain `NaN`. Caption/report silence is not treated as a negative.

## Important limitation

The seven repeated frames are **not an original MRI volume**. These synthetic DICOMs are unsuitable for scientific performance evaluation.

They are valid for testing:

- file/path discovery;
- DICOM pixel decoding;
- basic metadata handling;
- series routing;
- tensor shape construction;
- missing-stream masking;
- 2.5D sampling code;
- model forward/inference contracts;
- strict preflight behavior.

## Four source cases

1. `EXTVAL_ACL_001` — anterior cruciate ligament rupture, sagittal PD-weighted MRI; Hellerhoff, CC BY-SA 3.0.
2. `EXTVAL_MEDMEN_001` — grade 2 medial meniscal tear, coronal proton-density MRI; Nicolas Lefevre et al., CC BY 4.0.
3. `EXTVAL_BAKER_001` — Baker cyst in a patient with ACL rupture; Hellerhoff, CC BY-SA 3.0.
4. `EXTVAL_REFERENCE_001` — sagittal PD TSE FS knee MRI with no pathology claim used by this fixture; Ptrump16, CC BY-SA 4.0.

Full source URLs, attribution, licenses, source findings and downloaded SHA-256 hashes are recorded in `sources.csv`.

## Expected DICOM contract

Each fixture study contains one selected synthetic series. The materializer creates seven frames from one source image.

Expected technical shape per synthetic series:

```text
(7, 384, 384)
```

The high missing-stream rate is intentional because each study has one series rather than six semantic streams.

## Run the strict test preflight

```bash
python -m rsna_knee.cli preflight \
  --data-root fixtures/external_validation \
  --split test \
  --sample-size 4 \
  --max-decode-failure-rate 0 \
  --max-file-decode-failure-rate 0 \
  --out runs/external_test_preflight.json
```

Expected committed-fixture result:

```text
studies_sampled          4
streams_possible        24
streams_selected         4
streams_missing          20
directories_found        4
streams_decoded          4
candidate_files          4
file_decode_failures     0
decoded_frames          28
decode_failure_rate    0.0
file_decode_failure_rate 0.0
```

The `missing_stream_rate` is approximately `0.8333` by construction and is not a failure.

## Run the sparse labeled copy

```bash
python -m rsna_knee.cli preflight \
  --data-root fixtures/external_validation \
  --split validation \
  --sample-size 4 \
  --max-decode-failure-rate 0 \
  --max-file-decode-failure-rate 0 \
  --out runs/external_validation_preflight.json
```

Sparse source-supported labels currently include:

- ACL positive for the ACL case;
- Medial Meniscus positive for the medial meniscus case;
- ACL and Baker's positive for the Baker-cyst case;
- no asserted pathology for the reference case.

Unspecified cells remain `NaN`.

## Re-materialize manually

The committed fixture should normally be used directly. To rebuild it from sources:

```bash
conda activate rsna-knee
python -m pip install pillow
PYTHONPATH=src python scripts/materialize_external_validation.py \
  --output fixtures/external_validation \
  --overwrite
```

Re-materialization requires source network access and may be affected if a remote source changes. The committed files and their hashes provide the reproducible technical fixture.

## Separation from real competition validation

Never:

- append these four studies to competition training;
- include them in the 58-study gold folds;
- use their predictions to tune TTA, thresholds, architecture or Stage-1/Stage-2 selection;
- report their sparse labels as a macro-AUC benchmark.

Real model validation is documented in `../../docs/VALIDATION.md` and uses only the official competition gold labels.