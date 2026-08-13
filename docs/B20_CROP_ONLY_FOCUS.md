# B20 — Crop-only knee focus

> **Status — 2026-08-13:** IMPLEMENTED / PREDECLARED / NOT YET RUN.

B20 follows the completed B19 joint-focus experiment. B19 retained strong
effusion classification but Grad-CAM showed that its deterministic cosine
vignette created a new artificial border shortcut: activation concentrated on
the top/bottom taper boundaries rather than only on plausible joint-fluid
regions.

B20 tests the narrowest corrective change.

## Single change versus B19

B19:

```text
90% centered crop -> resize 224x224 -> cosine/vignette mask
```

B20:

```text
90% centered crop -> resize 224x224
```

There is **no multiplicative spatial mask, no black border, no cosine taper and
no additional crop jitter** in B20-v1.

Frozen policy:

```text
version        joint_focus_center_crop_only_v1
crop_fraction  0.90
```

Changing the crop fraction after looking at B20 expert-selection outcomes
requires a new experiment ID.

## Everything else remains B18/B19

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
resolution after crop              224 x 224
sampled positions / series         16
TTA                                [-1,0,1]
selection metric                   global 12-target expert macro AUC
selection tie break                earliest epoch
expert labels in gradients         no
```

The selected 58-study score remains a checkpoint-selection statistic only and
is not independent validation evidence.

## Preview before training

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
git pull --ff-only origin main
python -m pip install -e .

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b20-preview \
  --config configs/b20_crop_focus.yaml \
  --data-root "$DATA_ROOT" \
  --out runs/b20_crop_focus/crop_focus_preview.png
```

The expected preview should show a modest zoom into the knee with ordinary MRI
signal continuing to every output edge. There should be no synthetic dark
frame.

## Train

```bash
rsna-knee-b20 \
  --config configs/b20_crop_focus.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --report-ssl-checkpoint runs/b16_full_report/report_ssl/b16_report_encoder.pt \
  --out-root runs/b20_crop_focus
```

Expected selected checkpoint:

```text
runs/b20_crop_focus/b20_model.pt
```

## Submission smoke test after training

```bash
rsna-knee-b20-submit \
  --config configs/b20_crop_focus.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b20_crop_focus/b20_model.pt \
  --out runs/b20_crop_focus/submission_smoke.csv
```

## Same-source Grad-CAM comparison

The comparison tool fixes the TTA offset, series index and sampled slice across
models. Before B20 is trained, compare B18 and B19 on the same source image:

```bash
rsna-knee-focus-compare \
  --data-root "$DATA_ROOT" \
  --uid 1.2.826.0.1.3680043.8.498.12801308844398614687904447633432197492 \
  --target effusion \
  --b18-checkpoint runs/b18_fisher_selection/b18_model.pt \
  --b19-checkpoint runs/b19_joint_focus/b19_model.pt \
  --reference-model b18 \
  --cam-layer 28x28 \
  --cam-threshold 0.65 \
  --out runs/focus_comparison/b18_b19_same_source_effusion.png
```

After B20 completes, add:

```text
--b20-checkpoint runs/b20_crop_focus/b20_model.pt
```

The tool then shows B18 full-FOV, B19 crop+cosine, and B20 crop-only for the
**same TTA view, same MRI series and same sampled slice**.

## Interpretation rule

B20 succeeds conceptually if it preserves strong abnormality classification
while reducing both classes of shortcut seen so far:

1. B18 peripheral/full-FOV context activation;
2. B19 artificial vignette-boundary activation.

The decisive performance comparison still requires independent hidden
competition evaluation.
