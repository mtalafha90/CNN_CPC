# B37 — 288 high-resolution single-resize representation test

## Status

Prospective experiment. Protocol frozen before the first B37 training/evaluation result.

## Why B37 exists

B35 and B36 tested whether retaining/aggregating local ConvNeXt spatial features
could recover the weak focal-pathology performance of the current model. B35's
softmax attention stayed nearly uniform. B36 fixed that failure with direct
pathology-specific sparse top-k MIL and a direct local auxiliary loss; the sparse
selector did localize, but the reused expert-58 result still improved macro AUC
by only about +0.0018 and did not improve the focal-six mean.

That makes the upstream image representation the next clean target.

### Important prior result: B21/B22 already tested crop order at 224

B37 must not be described as the first native-crop experiment. Historical B21
changed the B20 order from

```text
normalize native volume
-> resize 224
-> center crop 90%
-> resize 224
```

to a raw native pre-resize crop at 224. B21 improved the leakage-safe weak-v2
surface but failed the full expert acceptance comparison; B22 showed that longer
training did not rescue it.

B37 therefore tests a different regime: **preserve the historical full-volume
normalization support, keep more native in-plane information, and adapt the last
ConvNeXt stage at 288 resolution under the current full LLM-fill supervision.**

## Representative DICOM geometry audit

A representative de-identified training slice supplied before B37 was frozen had:

```text
matrix                     512 x 512
pixel spacing               0.330078125 mm x 0.330078125 mm
full in-plane FOV           about 169.0 mm x 169.0 mm
sequence description        DP SPIR CS_SAG
field strength              1.5 T
scanner                     Philips Ingenia
```

For that geometry, the fixed 90% native crop is `461 x 461` pixels and spans
about `152.17 mm`. Mapping that crop once to `288 x 288` corresponds to about
`0.528 mm` per output pixel over the retained FOV.

By contrast, the first historical B20 resize maps the full 169 mm FOV to 224,
about `0.754 mm` per pixel before its later crop-and-resize. This example is an
illustration of the information-budget difference only; it is not evidence that
B37 improves AUC, and no parameter was selected from this one image.

## Frozen B37 deterministic preprocessing

```text
native DICOM series
    |
    | RescaleSlope / RescaleIntercept, MONOCHROME handling
    v
full native [S,H,W] float volume
    |
    | 1st/99th percentile normalization on the FULL volume
    v
normalized native volume
    |
    | same 16 centers, same 2.5D [-gap,0,+gap] triplets
    v
sampled native-resolution triplets
    |
    | centered 90% crop, NO interpolation
    v
native cropped triplets
    |
    | ONE bilinear resize
    v
16 x 3 x 288 x 288
    |
    | unchanged MRI augmentation during training
    v
B34 / ConvNeXt pipeline
```

The phrase "one resize" refers to deterministic anatomical preprocessing.
Training-time affine augmentation remains unchanged and can itself interpolate,
as it did in the matched base run.

## What is deliberately preserved

```text
architecture                    B34 training-only context / eval bypass
initial encoder                 completed B16 report-aligned encoder
supervision                     full LLM-fill export
report-only studies             4,349
eligible MRI series             24,035
training supervision cells      34,010 (subject to the frozen export audit)
seed                            2026
n_slices                        16
triplet gap policy              [1,2] train / 1 eval
center jitter                   2 train / 0 eval
TTA offsets                     [-1,0,+1]
batch size                      2
head LR                         1e-4
encoder trainable stages        1 (last stage only)
encoder LR scale                0.05
weight decay                    1e-4
fixed endpoint                  epoch 2
expert labels in gradient       0
```

## What changes

```text
historical full-fill B34:
  full normalization -> resize 224 -> 90% crop -> resize 224

B37:
  full normalization -> native 90% crop -> one resize 288
```

The normalization support is intentionally kept on the full native volume so
B37 does not repeat B21's crop-before-normalization intervention.

## Predeclared expert-58 interpretation

The 58 expert studies are repeatedly reused development diagnostics, not
independent validation and not a final promotion surface.

Strong GO:

```text
macro delta >= +0.02
OR
focal-six delta >= +0.03 with macro delta >= 0
```

Main-resolution mechanism rejected:

```text
macro delta <= +0.005
AND
focal-six delta < +0.01
```

The focal six are fixed before evaluation:

```text
ACL
MCL
Medial Meniscus
Lateral Meniscus
Contusion
Fracture
```

No target-wise post-hoc hybrid, crop-fraction sweep, resolution sweep, favorable
seed rerun, or endpoint change is allowed after viewing expert-58.

A B37 candidate that passes the development diagnostic still requires hidden
competition evidence before promotion.

## Files

```text
developments/src/rsna_knee/b37_highres.py
developments/src/rsna_knee/b37_training.py
developments/src/rsna_knee/b37_eval.py
developments/tests/test_b37_highres.py
config/b37_highres_288.yaml
```
