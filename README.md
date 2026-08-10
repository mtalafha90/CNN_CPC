# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-10:** **B7.1 full-corpus weak supervision remains the retained standalone champion**, macro AUC `0.5644802945`. B8 spatial-anatomy learning (`0.5300962807`), B9 strict semantic routing (`0.5334962669`) and B10 physical-scale normalization (`0.5523982721`) were rejected as global replacements. B11-v1 was stopped before training after its pseudo-label viability gate failed. **B11.1 calibration-aware quantile teacher tails passed its label-free pseudo audit and the frozen B11.1 student is now ready for training.**

Canonical status: [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).  
B11.1 protocol: [`docs/B11_1_QUANTILE_TEACHER.md`](docs/B11_1_QUANTILE_TEACHER.md).

## Current software state

```text
package version       0.18.0
current leader        B7.1 full-corpus weak supervision
leader macro AUC      0.5644802945
active experiment     B11.1 quantile-teacher student training
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
| **B11.1** | **per-target quantile teacher tails** | pending | **pseudo audit passed; training ready** |

## Retained B7.1 benchmark

B7.1 combines B5 competition-only image-report initialization, six historical dual MRI streams, 16 distributed 2.5D positions per stream, a ConvNeXtTiny slice encoder, cross-sequence Transformer memory, 12 pathology queries and frozen B6 v1.2.1 weak labels.

```text
training studies per epoch  3120
B6 supervised cells        14123
epochs                         4
macro AUC             0.5644802945
```

The same 58 gold studies have been reused for sequential method development, so all reported gold scores are **development/model-selection estimates**, not pristine independent validation.

## B10 result — physical-scale normalization

B10 standardized in-plane MRI geometry to approximately `0.3125 mm/pixel` and `160 mm` FOV before the unchanged `224 x 224` model resize. Geometry coverage was `1.0` across all 15,468 selected weak-training series.

```text
B10 macro AUC        0.5523982721
95% CI              [0.4935605888, 0.6091548645]
median(B10-B7.1)    -0.0121030792
95% paired CI       [-0.0507382525, +0.0250750953]
P(B10 > B7.1)        0.2706
```

Decision: reject B10-v1 as a global replacement.

## B11-v1 — stopped by the pre-training gate

The first teacher-student policy used one global rule:

```text
teacher mean >= 0.90 OR <= 0.10
TTA probability range <= 0.05
```

It found 4,794 pseudo-cells and activated 880 additional studies, but only 23 pseudo-cells were positive. Medial Meniscus and Synovitis had zero accepted cells and Lateral OA had only 21, so `viability_passed = false`. **B11-v1 student training must not be run.**

## B11.1 — calibration-aware quantile teacher tails

The B11-v1 diagnostic showed that TTA predictions were generally stable but absolute teacher probability ranges differ strongly across pathologies. B11.1 therefore selects **relative target-wise tails** using no gold labels.

For each target independently, among B6-unsupervised cells:

```text
bottom 5% teacher probabilities + TTA range <= 0.05 -> target 0.10
top    5% teacher probabilities + TTA range <= 0.05 -> target 0.90
base pseudo weight                                  -> 0.10
per-target pseudo mass cap                          -> 15% of B6 mass
```

B6 cells are never overwritten.

### Frozen B11.1 pseudo-audit result

```text
B6 cells                  14123
pseudo cells                3656
combined cells              17779
B6 active studies            3120
combined active studies      3454
newly activated studies       334
pseudo low-tail cells        1864
pseudo high-tail cells       1792
viability_passed             true
```

Every target exceeded 100 pseudo-cells and had more than 50 cells in both tails. Synovitis alone hit the 15% pseudo-mass cap; all other target pseudo weights remained at `0.10`.

Frozen pseudo CSV SHA-256:

```text
94f914f3548fab17f67ae0bf1906424bac850268c09ce5febede72b2ed7246b6
```

## B11.1 student contract

The student starts from the **same B5 checkpoint as B7.1**, not from the B7.1 teacher. Historical routing, legacy resizing, architecture, optimizer, augmentation, B6 labels, B6-derived target balancing and the four-epoch schedule are retained. The only scientific change is the frozen B11.1 pseudo supervision.

Each complete epoch must contain:

```text
studies                     3454
batches                     1727
B6 cells                   14123
pseudo cells                3656
combined cells             17779
pseudo low cells            1864
pseudo high cells           1792
full_coverage               true
budget_limited              false
```

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
0.18.0
```

## Tests

```bash
python -m compileall -q src tests
pytest -q \
  tests/test_b7_weak_supervision.py \
  tests/test_b11_1_quantile_pseudo.py \
  tests/test_b11_1_student.py
```

## Active next step — train B11.1

Use the already frozen successful pseudo artifacts:

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b11-1 \
  --config configs/b11_1_quantile_teacher.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --pseudo-root runs/b11_1_quantile_teacher/pseudo \
  --out-root runs/b11_1_quantile_teacher
```

Do not run the gold evaluator unless all four epochs report `full_coverage: true` and `budget_limited: false`.

## Frozen B11.1 gold evaluation

After four complete epochs:

```bash
rsna-knee-b11-1-eval \
  --config configs/b11_1_quantile_teacher.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b11_1_quantile_teacher/b11_1_model.pt \
  --out-root runs/b11_1_quantile_teacher/gold_eval
```

Primary benchmark:

```text
B7.1 macro AUC = 0.5644802945
```

The final development comparison is the same aligned 5,000-replicate paired bootstrap. Do not tune target-specific winners, thresholds, tail fractions, pseudo weights or ensembles from the reused 58-study gold set.

Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.
