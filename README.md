# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-11:** **B7.1 full-corpus weak supervision remains the retained standalone champion**, macro AUC `0.5644802945`. B8 spatial-anatomy (`0.5300962807`), B9 strict semantic routing (`0.5334962669`), B10 physical-scale normalization (`0.5523982721`) and B11.1 calibration-aware teacher tails (`0.5506902702`) were rejected as global replacements. **B12 variable-number-of-series modeling has now passed its frozen label-free series audit and is ready for training.**

Canonical status: [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).  
B12 protocol: [`docs/B12_VARIABLE_SERIES.md`](docs/B12_VARIABLE_SERIES.md).

## Current software state

```text
package version       0.19.0
current leader        B7.1 full-corpus weak supervision
leader macro AUC      0.5644802945
active experiment     B12 variable-number-of-series training
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
| **B12** | **variable number of real MRI series** | pending | **audit passed / training ready** |

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

B7.1 compresses each knee into six semantic slots. B12 instead keeps **every repaired Sagittal/Coronal/Axial MRI acquisition** as a separate real series and uses plane/fluid/fat categorical embeddings with dynamic mini-batch padding. It returns to the exact original B7.1 B6 supervision surface—no B11/B11.1 pseudo-labels.

### Frozen B12 label-free audit

```text
studies                                 3120
eligible recognized-plane series      17475
historical dual unique series          15468
extra series retained                   2007
extra fraction                        12.9752%
studies with extra series               1099
fraction studies with extras          35.2244%
zero-series studies                         0
historical selected series missing         0
series/study median                         5
series/study q90                            8
series/study q95                            9
series/study q99                           10
series/study max                           14
viability_passed                         true
```

Frozen mapping SHA-256:

```text
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

The audit passes well above the predeclared minimums of `5%` extra series and `10%` of studies gaining extra acquisitions.

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

## Active next step — train B12

Use the already frozen successful series policy:

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

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
series_instances_seen         17475
expected_series_instances     17475
full_coverage                  true
full_series_coverage           true
budget_limited                 false
```

A complete epoch should also encounter the audited maximum of `14` series in at least one mini-batch. If `series_instances_seen < 17475`, do not proceed to gold evaluation; investigate unreadable/missing DICOM series.

## Frozen B12 gold evaluation

Only after four complete epochs:

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