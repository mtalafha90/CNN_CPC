# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-10:** **B7.1 full-corpus weak supervision remains the retained standalone champion**, macro AUC `0.5644802945`. B8 spatial-anatomy learning (`0.5300962807`) and B9 strict semantic routing (`0.5334962669`) were both rejected. **B10 physical-scale normalization is now implemented and predeclared; its label-free geometry audit must be frozen before training.**

Canonical measured status: [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).  
B10 protocol: [`docs/B10_PHYSICAL_SCALE.md`](docs/B10_PHYSICAL_SCALE.md).

## Current software state

```text
package version       0.15.0
current leader        B7.1 full-corpus weak supervision
leader macro AUC      0.5644802945
active experiment     B10 physical-scale normalization
external pretraining  disabled
final inference       MRI-only
```

## Experiment ladder

| ID | Method | Macro AUC | Decision |
|---|---|---:|---|
| B0 | random initialization | `0.4762536432` | baseline |
| report teacher | fold-safe rules + TF-IDF | `0.49245` | rejected |
| B1 | strong competition-only MRI SSL | `0.5030284974` | retained reference |
| B2 | 0.1x encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware MIL | `0.4944652486` | rejected |
| B4 | frozen SSL + target-wise PCA/LR | `0.5137567459` | retained ablation |
| B5 | image-report SSL + unchanged B4 probe | `0.5243650851` | representation baseline |
| B6 | structured multilingual report labels | n/a | frozen weak-label source |
| B7-v1 | B5-init pathology-query model + B6 labels | `0.5397724412` | coverage ablation |
| **B7.1** | **B7 with full 3,120-study epoch coverage** | **`0.5644802945`** | **retained champion** |
| B5+B7.1 rank | fixed 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | 2x2 spatial-token anatomy model | `0.5300962807` | rejected |
| B9 | strict semantic routing | `0.5334962669` | rejected |
| **B10** | **B7.1 + physical-scale normalization** | pending | **implemented / predeclared** |

## Why B10

The existing preprocessing resizes every selected slice to `224 x 224`, which normalizes matrix size but not physical anatomy scale. Different scanners/protocols can have substantially different DICOM `PixelSpacing` and physical field of view.

B10 keeps historical B7.1 routing and inserts only:

```text
native DICOM pixels
  -> plane-specific canonical PixelSpacing
  -> canonical physical FOV by center crop/pad
  -> unchanged 224 x 224 resize
  -> unchanged B7.1 model/training/evaluation
```

Canonical geometry is derived without gold labels from the exact 3,120 active B6 weak-training studies. For sagittal, coronal and axial planes separately:

```text
target PixelSpacing = median valid training PixelSpacing
target physical FOV = median valid training physical FOV
minimum valid geometry coverage = 95%
missing PixelSpacing -> legacy resize, do not discard study
```

The audit also records scanner/model/field strength, slice thickness/spacing, and a SHA-256 signature of the exact selected-series mapping. Training refuses a policy generated from a different routing state.

## Install / update

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e .
python -c "import rsna_knee; print(rsna_knee.__version__)"
```

Expected:

```text
0.15.0
```

## Tests

```bash
python -m compileall -q src tests
pytest -q \
  tests/test_b7_weak_supervision.py \
  tests/test_b9_strict_routing.py \
  tests/test_b10_physical_scale.py
```

## B10 step 1 — label-free geometry audit

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b10-audit \
  --config configs/b10_physical_scale.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b10_physical_scale/audit
```

Inspect:

```bash
cat runs/b10_physical_scale/audit/spacing_audit.json
cat runs/b10_physical_scale/audit/physical_scale_policy.json
```

Do **not** start B10 training until the audit is inspected. The geometry coverage must be at least `0.95`, the study count should be `3120`, and `uses_gold_labels` must be `false`.

## B10 step 2 — training

```bash
rsna-knee-b10 \
  --config configs/b10_physical_scale.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --physical-policy runs/b10_physical_scale/audit/physical_scale_policy.json \
  --out-root runs/b10_physical_scale
```

Expected per complete epoch:

```text
batches                       1560
study_draws                   3120
active_supervision_cells_seen 14123
positive_cells_seen            6871
negative_cells_seen            7252
budget_limited                 false
```

## B10 step 3 — frozen gold evaluation

After four complete epochs and artifact inspection:

```bash
rsna-knee-b10-eval \
  --config configs/b10_physical_scale.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b10_physical_scale/b10_model.pt \
  --out-root runs/b10_physical_scale/gold_eval
```

Primary benchmark:

```text
B7.1 macro AUC = 0.5644802945
```

The primary comparison is an aligned B7.1 -> B10 study-level paired bootstrap with 5,000 replicates.

## Validation caution

The same 58 gold studies have informed repeated method decisions. Current scores are **development/model-selection estimates**, not pristine hidden-test estimates. Do not tune target-specific routing, physical spacing/FOV, weak-label weights, per-target model winners, or ensemble weights from these 58 labels and then present the result as independent validation.

`docs/competition.md` remains a preserved competition-summary document and is intentionally not rewritten by experiment updates.
