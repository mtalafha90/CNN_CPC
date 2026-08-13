# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, including 58 fully expert-labelled studies and 4,349 report-only/non-gold studies, with multiple MRI series per knee and 12 study-level targets evaluated by macro ROC AUC.

## Current snapshot — 2026-08-13

The current controlled spatial-ablation sequence is complete:

| Model | Spatial input | Selected epoch | Expert selection statistic | Current decision |
|---|---|---:|---:|---|
| **B18** | full field of view | 2 | `0.6654496134` | retained |
| **B19** | 90% crop + cosine vignette | 3 | `0.6581308356` | **rejected spatial formulation** |
| **B20** | 90% crop only | 2 | `0.6671593555` | retained |

The 58 expert-labelled studies are a repeatedly reused **development/checkpoint-selection surface, not independent validation**. The numerical B20-B18 difference (`+0.0017097421`) is therefore not evidence that B20 is more accurate than B18.

A same-source Grad-CAM audit showed that B19's cosine vignette created an artificial border shortcut. B20 removes that synthetic boundary, but on the audited effusion case B18 remained more focal. The current defensible conclusion is therefore:

```text
B19: rejected
B18 vs B20: unresolved
```

Independent competition evaluation is still required to decide predictive superiority.

## Canonical records

- [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) — complete experiment ledger and governance.
- [`docs/B18_FISHER_SELECTION.md`](docs/B18_FISHER_SELECTION.md) — completed B18 protocol/result.
- [`docs/B19_JOINT_FOCUS.md`](docs/B19_JOINT_FOCUS.md) — completed B19 result and vignette-shortcut finding.
- [`docs/B20_CROP_ONLY_FOCUS.md`](docs/B20_CROP_ONLY_FOCUS.md) — completed B20 result and corrected deterministic Grad-CAM comparison.
- [`docs/VISUALIZATION_GUIDE.md`](docs/VISUALIZATION_GUIDE.md) — preview/Grad-CAM/comparison command guide.
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — validation governance.
- [`docs/FINAL_ALL_DATA.md`](docs/FINAL_ALL_DATA.md) — final all-data protocol, implemented but deferred.

## Setup

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
git pull --ff-only origin main
python -m pip install -e .

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
```

## Preview versus visualization

These commands intentionally do different things.

### Preprocessing preview

`preview` shows only the image transform. It does **not** run the classifier and does not create a Grad-CAM lesion/evidence mask.

B19 crop + cosine preview:

```bash
rsna-knee-b19-preview \
  --config configs/b19_joint_focus.yaml \
  --data-root "$DATA_ROOT" \
  --out runs/b19_joint_focus/joint_focus_preview.png
```

B20 crop-only preview:

```bash
rsna-knee-b20-preview \
  --config configs/b20_crop_focus.yaml \
  --data-root "$DATA_ROOT" \
  --out runs/b20_crop_focus/crop_focus_preview.png
```

Both previews use a lightweight shared reader that decodes only one representative DICOM from at most one sagittal, coronal and axial series. B19 preview is restricted to an expert-labelled UID because that was its frozen historical preview contract. B20 preview defaults to an expert-labelled study but an explicit `--uid` may identify any training study.

### Grad-CAM visualization

`visualize` runs the selected model and produces a model-evidence Grad-CAM mask. These commands currently operate on the **58-study expert-labelled surface** so that a requested case can be verified as an expert-positive true positive. The mask is model localization, not a radiologist-drawn lesion segmentation.

B19:

```bash
rsna-knee-b19-visualize \
  --config configs/b19_joint_focus.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b19_joint_focus/b19_model.pt \
  --target effusion \
  --cam-layer 28x28 \
  --cam-threshold 0.65
```

B20:

```bash
rsna-knee-b20-visualize \
  --config configs/b20_crop_focus.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b20_crop_focus/b20_model.pt \
  --target effusion \
  --cam-layer 28x28 \
  --cam-threshold 0.65
```

A report-only/non-gold Grad-CAM mode is **not currently part of the frozen CLI**. Do not describe these gold visualizations as independent validation.

### Same-source B18/B19/B20 comparison

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

The comparison fixes one TTA view, MRI series and sampled slice across all included models. Visualization probabilities are evaluated in deterministic `eval()` mode and the comparison checks direct-view versus Grad-CAM-forward probability consistency.

## Selected-checkpoint inference smoke tests

B19:

```bash
rsna-knee-b19-submit \
  --config configs/b19_joint_focus.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b19_joint_focus/b19_model.pt \
  --out runs/b19_joint_focus/submission_smoke.csv
```

B20:

```bash
rsna-knee-b20-submit \
  --config configs/b20_crop_focus.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b20_crop_focus/b20_model.pt \
  --out runs/b20_crop_focus/submission_smoke.csv
```

Both local smoke tests passed on the three-row test surface with 15 total MRI series, matching sample-submission columns and UID order. They are schema/inference checks only, not hidden-test performance evidence.

## Frozen supervision and encoder

B18--B20 use the same downstream weak-supervision surface:

```text
B6-active studies       3120
usable B6 cells        14123
positive cells          6871
negative cells          7252
eligible MRI series    17475
```

The B16 report-aligned encoder is frozen during B18--B20 training. Its SHA-256 remained:

```text
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

## Governance

```text
B18: completed; selected epoch 2
B19: completed; rejected spatial formulation because of artificial vignette shortcut
B20: completed; selected epoch 2
B18 vs B20: unresolved
expert labels: checkpoint-selection/development surface only; never used in B18--B20 gradients
no target-specific epoch selection
no post-hoc claim that selected expert statistics are independent validation
FINAL all-data fit: deferred pending independent evaluation decision
```
