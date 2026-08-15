# B20 crop-order implementation audit

> **Status — 2026-08-14:** VERIFIED / FOLLOW-UP COMPLETE. Historical B20 is unchanged. B21 tested the corrected pre-resize ordering and was not promoted; B22 showed that longer training did not rescue it.

The B20 dataset calls the parent `VariableSeriesKneeDataset.__getitem__()` first and then applies `apply_crop_focus()` to the returned tensor. The parent path has already run `preprocess_triplets()`, which bilinearly resizes the sampled MRI triplets to `224 x 224`.

Therefore historical B20 actually executes:

```text
native DICOM volume
  -> normalize / sample 2.5D triplets
  -> bilinear resize to 224 x 224
  -> centered 90% crop of the 224 image (about 202 x 202)
  -> bilinear resize back to 224 x 224
```

It does **not** execute a native-resolution crop before the first resize.

This finding does not invalidate or modify any stored B20 result. B20 remains the active working model and its checkpoint remains:

```text
runs/b20_crop_focus/b20_model.pt
```

## Controlled follow-up

B21 tested the corrected ordering:

```text
native DICOM volume
  -> centered 90% crop at native resolution
  -> percentile normalization / sample 2.5D triplets
  -> single resize to 224 x 224
```

Because percentile normalization is computed after the raw crop, B21 also changes the pixel support from which the normalization window is derived. That support difference was declared as part of the B21-v1 intervention.

A leakage-safe weak-v2 comparison favored B21:

```text
B20-v2 control macro AUC        0.7298727911
B21-v1 macro AUC                0.7410090411
paired raw delta               +0.0111362500
paired 95% CI        [+0.0001624070,+0.0226346590]
```

However, the frozen full-data expert acceptance comparison did not:

```text
B20 replay macro AUC            0.6674066371
B21 macro AUC                   0.6573196516
B21 - B20 replay               -0.0100869854
paired 95% CI        [-0.0328814731,+0.0117052345]
```

B21 therefore failed promotion.

B22 then retrained the same corrected crop pipeline for five epochs:

```text
E1  0.6135270850
E2  0.6574269018  <- best
E3  0.6387456622
E4  0.6136783995
E5  0.6282683534
```

Later epochs did not recover the lost expert performance, even though the weak-training loss continued to decrease. Thus the crop-order defect is real as an implementation issue, but **correcting it did not improve the current model under the tested frozen-encoder/B6 supervision regime**.

## Current conclusion

```text
historical B20 preprocessing       preserved
pre-resize crop correction         tested
weak-v2 development result         positive
expert acceptance result           negative
longer-training rescue             not supported
working model                      B20
```

The next optimization priority is the label/development-selection problem rather than another crop-order or duration experiment.

See:
- [`B21_PRERESIZE_CROP.md`](B21_PRERESIZE_CROP.md)
- [`B21_FULL_ACCEPTANCE.md`](B21_FULL_ACCEPTANCE.md)
- [`B22_DURATION_AUDIT.md`](B22_DURATION_AUDIT.md)
- [`WORKING_MODEL.md`](WORKING_MODEL.md)
