# Dataset and DICOM handling

## CSV contracts

`train.csv` contains `StudyInstanceUID`, `Report`, and the 12 target columns. `test.csv` requires `StudyInstanceUID`; reports are not required at inference.

Series CSVs require `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, and `Anatomical_Plane`. Duplicate study/series rows and missing UIDs are rejected.

## Nullable sequence metadata

Missing `Fluid_Sensitive` and `Fat_Suppression` values remain **unknown** when the CSV is loaded. They are not converted to `False`. DICOM backfill independently repairs three fields:

- anatomical plane from image orientation;
- fluid sensitivity from sequence timing/weighting;
- fat suppression from acquisition metadata.

A populated CSV field remains authoritative. If a field is still unknown after repair, routing uses a conservative `False` fallback only at the final scoring step.

## Six-stream routing

The production study contract is:

```text
sagittal_fluid       sagittal_structural
coronal_fluid        coronal_structural
axial_fluid          axial_structural
```

Routing is built with one grouped pass over the series table. Fluid/fat-suppressed candidates rank toward the fluid slot; non-fluid/non-fat-suppressed candidates rank toward the structural slot. When alternatives exist, the same MRI series is not placed in both slots. Missing streams remain absent and are masked before ConvNeXt, so they consume no backbone compute.

## DICOM decoding

The reader supports `.dcm`, `.dicom`, `.ima`, suffix-less, mixed-suffix, and enhanced multi-frame layouts. It applies rescale slope/intercept and `MONOCHROME1` inversion, normalizes mixed in-plane dimensions, and sorts slices using physical orientation/position with deterministic fallback.

## 2.5D sampling

A series is globally normalized with finite 1st/99th percentiles and mapped to `[0,1]`. A triplet is:

```text
[z-gap, z, z+gap]
```

Centers are distributed across the valid depth range. Training defaults to gaps `{1,2}` and center jitter of ±2 slices. Inference is deterministic and averages center offsets `[-1,0,+1]`.

## MRI-specific augmentation

Training applies mild series-consistent perturbations rather than generic natural-image augmentation:

- small rotation;
- small translation and scale;
- gamma variation;
- low-frequency multiplicative bias field;
- Gaussian noise;
- slice dropout.

Validation and inference disable stochastic augmentation.

## Preflight

```bash
rsna-knee preflight --data-root DATA_ROOT --split train --sample-size 24
```

Preflight executes real DICOM decoding and the production 2.5D transform. It distinguishes legitimate missing semantic streams from selected-stream path/decode failures. It also reports missing versus repaired metadata fields independently.

## Gold, inner, outer, and weak cross-fit roles

For fold `k`, studies can have these roles:

- `outer_oof`: official gold studies used only for final fold evaluation;
- `inner_selection`: gold studies used only to choose training duration in phase A;
- `gold_train_selection`: trusted gold used in phase-A training;
- `weak_oof`: non-gold report groups withheld from the fold so image predictions are cross-fitted;
- `weak_train`: report-supervised training rows.

After epoch selection, phase B retrains a fresh model using all non-outer gold studies while continuing to exclude the fold's `weak_oof` report groups.

## Self-supervised data scope

`rsna-knee pretrain` uses non-gold studies only by default. This learns same-knee cross-sequence anatomy representations without exposing the outer gold images to SSL.

## Training versus inference

Reports, report calibration, and co-training consensus are training-only. Final submission inference uses MRI images plus self-describing checkpoints and deterministic TTA.
