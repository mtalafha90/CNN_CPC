# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-11:** **B7.1 full-corpus weak supervision remains the retained standalone champion**, macro AUC `0.5644802945`. B8 spatial-anatomy (`0.5300962807`), B9 strict semantic routing (`0.5334962669`), B10 physical-scale normalization (`0.5523982721`) and B11.1 calibration-aware teacher tails (`0.5506902702`) were rejected as global replacements. **B12 variable-number-of-series modeling is now implemented and predeclared; its label-free series audit is the active next step.**

Canonical status: [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).  
B12 protocol: [`docs/B12_VARIABLE_SERIES.md`](docs/B12_VARIABLE_SERIES.md).

## Current software state

```text
package version       0.19.0
current leader        B7.1 full-corpus weak supervision
leader macro AUC      0.5644802945
active experiment     B12 variable-number-of-series model
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
| B10 | B7.1 + physical-scale normalization | `0.5523982721` | rejected globally |
| B11-v1 | absolute-threshold B7.1 teacher completion | n/a | stopped at pseudo viability gate |
| B11.1 | per-target quantile teacher tails | `0.5506902702` | rejected globally |
| **B12** | **variable number of real MRI series** | pending | **implemented / label-free audit pending** |

## Retained B7.1 benchmark

B7.1 combines B5 competition-only image-report initialization, six historical dual MRI streams, 16 distributed 2.5D positions per stream, a ConvNeXtTiny slice encoder, cross-sequence Transformer memory, 12 pathology queries and frozen B6 v1.2.1 weak labels.

```text
training studies per epoch  3120
B6 supervised cells        14123
positive cells              6871
negative cells              7252
epochs                         4
macro AUC             0.5644802945
```

The same 58 gold studies have been reused for sequential method development, so all reported gold scores are **development/model-selection estimates**, not pristine independent validation.

## B11.1 result — close the teacher-pseudo branch

B11.1 solved the calibration problem seen in B11-v1 by selecting target-wise 5/95% teacher tails. Its pseudo audit passed and training completed four full epochs, but generalization did not improve:

```text
B11.1 macro AUC          0.5506902702
95% CI                  [0.4917424630, 0.6086153876]
B7.1 macro AUC           0.5644802945
median(B11.1-B7.1)      -0.0126224565
95% paired CI           [-0.0487500119, +0.0195120537]
P(B11.1 > B7.1)          0.2184
```

Decision: reject B11.1 globally and do not construct target-wise B7.1/B11.1 winners from the reused 58-study development surface.

## Why B12

B7.1 compresses each knee into six semantic slots:

```text
sagittal_fluid
sagittal_structural
coronal_fluid
coronal_structural
axial_fluid
axial_structural
```

A study can contain repeated or additional MRI acquisitions beyond those six winners. B12 tests whether keeping those acquisitions as separate real series improves pathology discrimination.

### Single scientific change

B12 returns to the **exact original B7.1 B6 supervision surface**—no B11/B11.1 pseudo-labels.

For each B6-active study, B12:

- retains every repaired series with Sagittal, Coronal or Axial plane;
- does not choose one fluid and one structural winner per plane;
- keeps repeated acquisitions as separate series;
- adds coarse plane/fluid/fat metadata embeddings;
- uses no learned series-rank/position embedding;
- dynamically pads only to the largest series count in the current mini-batch;
- has no architecture-level maximum series count.

B5 initialization, legacy resize, B6 supervision, target balancing, optimizer, augmentation, four-epoch schedule and frozen TTA remain B7.1-equivalent.

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
0.19.0
```

## Tests

```bash
python -m compileall -q src tests
pytest -q \
  tests/test_b12_variable_series.py \
  tests/test_b7_weak_supervision.py
```

## Active next step — B12 label-free series audit

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b12-audit \
  --config configs/b12_variable_series.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b12_variable_series/audit
```

Inspect:

```bash
cat runs/b12_variable_series/audit/series_audit.json
cat runs/b12_variable_series/audit/series_policy.json
```

The predeclared audit must report `viability_passed: true` before training. It also records the exact variable-series mapping SHA-256, number of extra series retained over historical dual routing, fraction of studies gaining extra acquisitions, and the full series-count distribution.

## B12 training — only after audit pass

```bash
rsna-knee-b12 \
  --config configs/b12_variable_series.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out-root runs/b12_variable_series
```

Every full epoch must preserve:

```text
batches                        1560
study_draws                    3120
active_supervision_cells_seen 14123
positive_cells_seen            6871
negative_cells_seen            7252
full_coverage                  true
full_series_coverage           true
budget_limited                 false
```

Do not run gold evaluation unless all four epochs meet both coverage contracts.

## Frozen B12 gold evaluation

```bash
rsna-knee-b12-eval \
  --config configs/b12_variable_series.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b12_variable_series/b12_model.pt \
  --out-root runs/b12_variable_series/gold_eval
```

Primary benchmark remains:

```text
B7.1 macro AUC = 0.5644802945
```

Use the same aligned 5,000-replicate paired bootstrap. Do not tune target-wise winners, routing variants, series caps, or ensemble weights on the reused 58-study gold set.

Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.
