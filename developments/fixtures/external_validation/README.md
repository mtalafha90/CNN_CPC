# External four-study knee MRI technical fixture

> **Current campaign note — 2026-08-12:** this fixture remains a **technical-only** resource. It was not used for B0-B15 scientific model selection or B15 MRI-domain SSL. Current experiment results are in [`../../docs/EXPERIMENT_STATUS.md`](../../docs/EXPERIMENT_STATUS.md): B13 remains the reused-gold development champion at `0.6293565948`; B15 passed weak-v2 but did not improve the reused-gold global metric.

This directory contains **four openly licensed online knee MRI examples for technical pipeline testing**. It is not the competition validation set and must never be used as a leaderboard/scientific macro-AUC benchmark.

The materializer keeps source JPEGs under `source_jpgs/` and wraps each published image into a seven-frame synthetic DICOM so production code can exercise discovery, decoding, routing, resizing, 2.5D construction and inference plumbing.

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

Use `test.csv`, `test_series.csv`, and `test_images/` for the production image-only test path.

`validation.csv` is a sparse sanity-label copy. It includes only pathology cells directly supported by the source material; unspecified targets remain `NaN`. Caption/report silence is not a negative.

## Important limitation

The seven repeated frames are **not an original MRI volume** and are unsuitable for scientific performance evaluation.

Valid uses:

- file/path discovery;
- DICOM pixel decoding;
- metadata handling;
- series routing;
- tensor construction;
- missing-stream masking;
- 2.5D sampling;
- model forward/inference contracts;
- strict preflight behavior.

## Four source cases

1. `EXTVAL_ACL_001` — ACL rupture, sagittal PD-weighted MRI; Hellerhoff, CC BY-SA 3.0.
2. `EXTVAL_MEDMEN_001` — grade 2 medial meniscal tear, coronal proton-density MRI; Nicolas Lefevre et al., CC BY 4.0.
3. `EXTVAL_BAKER_001` — Baker cyst in a patient with ACL rupture; Hellerhoff, CC BY-SA 3.0.
4. `EXTVAL_REFERENCE_001` — sagittal PD TSE FS knee MRI with no pathology claim used by this fixture; Ptrump16, CC BY-SA 4.0.

Full source URLs, attribution, licenses, source findings and downloaded SHA-256 hashes are recorded in `sources.csv`.

## Expected synthetic contract

Each fixture study contains one selected synthetic series with seven frames generated from one source image.

```text
expected series shape: (7, 384, 384)
```

The high missing-stream rate is intentional because each study has one series rather than six semantic streams.

## Strict test preflight

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
studies_sampled           4
streams_possible         24
streams_selected          4
streams_missing          20
directories_found         4
streams_decoded           4
candidate_files           4
file_decode_failures      0
decoded_frames           28
decode_failure_rate     0.0
file_decode_failure_rate 0.0
```

The missing-stream rate is approximately `0.8333` by construction and is not a failure.

## Sparse labelled copy

```bash
python -m rsna_knee.cli preflight \
  --data-root fixtures/external_validation \
  --split validation \
  --sample-size 4 \
  --max-decode-failure-rate 0 \
  --max-file-decode-failure-rate 0 \
  --out runs/external_validation_preflight.json
```

Source-supported sparse positives include ACL for the ACL case, Medial Meniscus for the meniscus case, and ACL plus Baker's for the Baker-cyst case. Unspecified cells remain `NaN`.

## Re-materialize

```bash
conda activate rsna-knee
python -m pip install pillow
PYTHONPATH=src python scripts/materialize_external_validation.py \
  --output fixtures/external_validation \
  --overwrite
```

Re-materialization requires source network access. The committed files/hashes are the reproducible technical fixture.

## Separation from competition validation and B15

Never:

- append these four studies to competition training;
- include them in the 58-study gold development surface;
- include them in strong SSL, B5 representation training, or B15 knee-MRI SSL;
- include them in frozen weak-v2 construction;
- use their predictions to tune TTA, thresholds, architecture, supervision rules or classifier selection;
- report sparse fixture labels as macro AUC.

B15 used only competition MRI, excluding all 58 gold studies and all 623 weak-v2 holdout studies from its SSL pool. This fixture did not enter that 3,726-study SSL pool.

Scientific validation governance is documented in [`../../docs/VALIDATION.md`](../../docs/VALIDATION.md).