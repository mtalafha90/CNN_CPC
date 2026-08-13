# B19 — Joint-focused MRI input

> **Status — 2026-08-13:** IMPLEMENTED / PREDECLARED / NOT YET RUN.

B19 is the next controlled experiment after B18. It changes exactly one modeling
choice: the spatial field of view presented to the MRI classifier.

## Motivation

B18 Grad-CAM inspection showed that a correctly detected effusion case had
substantial activation not only around plausible joint-fluid regions but also in
peripheral image areas. B19 tests whether suppressing peripheral field-of-view
context reduces shortcut learning and makes the model rely more strongly on the
knee joint itself.

B18 remains frozen and unchanged. B19 is a separate experiment.

## Frozen joint-focus transform

Every sampled 2.5D MRI triplet is transformed after the ordinary B18
preprocessing/augmentation and before it enters ConvNeXt:

```text
1. centered crop to 90% of image height and width
2. resize crop back to 224 x 224
3. central 72% of each axis retains full weight
4. smooth cosine taper toward the image border
5. extreme border is forced to zero from normalized |x| or |y| >= 0.90
```

Because the crop itself removes 5% from each side and the final window also
forces the outer portion of the resized crop to zero, the model receives
negligible/zero information from the original image periphery while retaining a
conservative field of view around the joint.

Frozen policy:

```text
version                 joint_focus_center_crop_cosine_v1
crop_fraction           0.90
full_weight_fraction    0.72
outer_zero_fraction     0.90
```

These values are frozen before B19 outcomes are inspected. Changing them requires
a new experiment ID rather than retuning B19-v1 on the expert-selection set.

## Why the transform is conservative

A much tighter center crop could remove clinically relevant peripheral findings,
including suprapatellar effusion, posterior Baker's cyst, collateral-ligament
abnormality, or peripheral bone/soft-tissue findings. B19-v1 therefore suppresses
the image edges without attempting a hard anatomical segmentation.

The transform is a spatial prior, not a lesion mask and not a learned knee
segmentation network.

## Everything else is unchanged from B18

```text
training studies                    3120
usable B6 cells                    14123
positive / negative                6871 / 7252
eligible MRI series               17475
initializer                        completed B16 report-aligned encoder
encoder                            frozen
encoder LR                         0
head LR                            1e-4
candidate epochs                   5
resolution after focus             224 x 224
sampled positions / series         16
TTA                                [-1,0,1]
selection metric                   global 12-target expert macro AUC
selection tie break                earliest epoch
expert labels in gradients         no
```

B19 retains B18's global expert checkpoint-selection rule so the spatial
preprocessing is the only scientific change relative to B18.

As with B18, the selected 58-study expert score is a checkpoint-selection
statistic only and is **not independent validation evidence**.

## Preview before training

Pull and reinstall:

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
git pull --ff-only origin main
python -m pip install -e .

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
```

Generate a sagittal/coronal/axial before-versus-after panel:

```bash
rsna-knee-b19-preview \
  --config configs/b19_joint_focus.yaml \
  --data-root "$DATA_ROOT" \
  --out runs/b19_joint_focus/joint_focus_preview.png
```

A specific expert-labelled study can be inspected with:

```bash
rsna-knee-b19-preview \
  --config configs/b19_joint_focus.yaml \
  --data-root "$DATA_ROOT" \
  --uid <StudyInstanceUID> \
  --out runs/b19_joint_focus/joint_focus_preview_<StudyInstanceUID>.png
```

Do not start B19 training until representative sagittal, coronal and axial
previews show that the knee remains fully represented and important recesses are
not clipped.

## Train B19

```bash
rsna-knee-b19 \
  --config configs/b19_joint_focus.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --report-ssl-checkpoint runs/b16_full_report/report_ssl/b16_report_encoder.pt \
  --out-root runs/b19_joint_focus
```

Expected selected checkpoint:

```text
runs/b19_joint_focus/b19_model.pt
```

## Local submission smoke test after training

```bash
rsna-knee-b19-submit \
  --config configs/b19_joint_focus.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b19_joint_focus/b19_model.pt \
  --out runs/b19_joint_focus/submission_smoke.csv
```

The local three-study output remains a schema/inference smoke test only. A
meaningful comparison of B19 versus B18 requires an independent competition
evaluation.

## Interpretation rule

If B19's reused 58-study selected statistic is higher than B18's, that is useful
for checkpoint selection/development but is not evidence of true superiority.
The key comparison is the independent hidden competition result and the
post-hoc localization behavior.
