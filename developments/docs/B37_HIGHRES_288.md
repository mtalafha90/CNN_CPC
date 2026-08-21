# B37-288 — superseded unrun design

## Status

**SUPERSEDED BEFORE TRAINING / NEVER RUN.**

This 288-resolution B34-only design was written before the complete native-DICOM
geometry audit.  It was not trained and produced no expert or hidden result.

The subsequent audit of all 24,371 training series showed:

- median matrix 512 x 512;
- median PixelSpacing 0.3125 mm/pixel;
- median physical FOV about 160 mm;
- median 90% native-crop maximum dimension 461 pixels;
- substantial matrix/PixelSpacing heterogeneity, making padding-only impractical.

Those measurements made 448x448 a better prospectively justified high-resolution
endpoint than 288: a 90% crop of the median 160 mm FOV mapped to 448 corresponds
to about 0.321 mm/pixel, close to the observed native median sampling.

At the same time, completed B36 had already demonstrated genuine pathology-specific
sparse localization at 224 without meaningful expert-AUC improvement.  The next
experiment was therefore frozen as a *joint* high-resolution + sparse-MIL test
rather than spending a run on this earlier B34-only 288 proposal.

The active prospective B37 protocol is:

[`B37_HIGHRES_SPARSE_MIL.md`](B37_HIGHRES_SPARSE_MIL.md)

Active files:

```text
developments/src/rsna_knee/b37_highres_sparse_mil.py
developments/src/rsna_knee/b37_highres_sparse_training.py
developments/src/rsna_knee/b37_highres_sparse_eval.py
developments/tests/test_b37_highres_sparse_mil.py
config/b37_highres_sparse_448.yaml
```

The old 288 implementation files are retained only as an auditable record of the
superseded design.  Do not run them as B37.
