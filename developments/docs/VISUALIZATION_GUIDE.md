# Visualization command guide

This document separates the three visualization concepts used by B18--B20.

## 1. Preprocessing preview

Purpose: inspect only the image transform before/after preprocessing.

No classifier is evaluated. No Grad-CAM is computed. No lesion truth is inferred.

### B19 crop + cosine

```bash
rsna-knee-b19-preview \
  --config configs/b19_joint_focus.yaml \
  --data-root "$DATA_ROOT" \
  --out runs/b19_joint_focus/joint_focus_preview.png
```

B19 preview preserves the original frozen historical contract and therefore uses
one of the 58 expert-labelled studies. An explicit `--uid` must also identify an
expert-labelled study.

### B20 crop only

```bash
rsna-knee-b20-preview \
  --config configs/b20_crop_focus.yaml \
  --data-root "$DATA_ROOT" \
  --out runs/b20_crop_focus/crop_focus_preview.png
```

B20 defaults to an expert-labelled study for a stable reference, but an explicit
`--uid` may identify any training study. The preview is intentionally lightweight:
it reads one deterministic representative DICOM from at most one sagittal,
coronal and axial series rather than instantiating the full B12 study dataset.

## 2. Single-model Grad-CAM visualization

Purpose: inspect where a selected classifier obtains positive evidence.

The current B18/B19/B20 `*-visualize` commands operate on the 58-study
expert-labelled surface so a requested case can be checked as an expert-positive
true positive. Grad-CAM is model localization, **not** ground-truth lesion
segmentation.

B19 example:

```bash
rsna-knee-b19-visualize \
  --config configs/b19_joint_focus.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b19_joint_focus/b19_model.pt \
  --target effusion \
  --cam-layer 28x28 \
  --cam-threshold 0.65
```

B20 example:

```bash
rsna-knee-b20-visualize \
  --config configs/b20_crop_focus.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b20_crop_focus/b20_model.pt \
  --target effusion \
  --cam-layer 28x28 \
  --cam-threshold 0.65
```

A report-only/non-gold Grad-CAM mode has **not** been added to the frozen CLI.
If such a diagnostic is implemented later, it must clearly distinguish weak
report labels from expert truth and must not claim a true-positive lesion without
expert confirmation.

## 3. Same-source B18/B19/B20 comparison

Purpose: compare model evidence while holding the source image fixed.

```bash
rsna-knee-focus-compare \
  --data-root "$DATA_ROOT" \
  --uid 1.2.826.0.1.3680043.8.498.12801308844398614687904447633432197492 \
  --target effusion \
  --b18-checkpoint runs/b18_fisher_selection/b18_model.pt \
  --b19-checkpoint runs/b19_joint_focus/b19_model.pt \
  --b20-checkpoint runs/b20_crop_focus/b20_model.pt \
  --reference-model b18 \
  --cam-layer 28x28 \
  --cam-threshold 0.65 \
  --out runs/focus_comparison/b18_b19_b20_same_source_effusion.png
```

If `--view-offset`, `--series-index`, and `--slice-index` are not forced, the
reference model chooses them once and the same source is then used for every
included model. The comparison runs probability passes in deterministic
`eval()` mode and checks direct-view versus Grad-CAM-forward probability
consistency.

## Current interpretation

The canonical corrected effusion comparison supports:

```text
B19: rejected spatial formulation; cosine vignette created an artificial border shortcut
B20: removes the synthetic vignette boundary
B18 vs B20: localization remains unresolved from the single audited case
```

All expert-set visualizations are post-selection diagnostics. They are not
independent validation and should not be used to retune B18--B20.
