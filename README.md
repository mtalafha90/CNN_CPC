# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-10:** **B7.1 full-corpus weak supervision remains the retained standalone champion**, macro AUC `0.5644802945`. B8 spatial-anatomy learning (`0.5300962807`), B9 strict semantic routing (`0.5334962669`) and B10 physical-scale normalization (`0.5523982721`) were rejected as global replacements. B11-v1 teacher pseudo-labelling was stopped before training because its frozen viability gate failed. **B11.1 calibration-aware quantile teacher tails is now the active predeclared label-free pseudo-label audit.**

Canonical measured status: [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).  
B11 protocol: [`docs/B11_TEACHER_STUDENT.md`](docs/B11_TEACHER_STUDENT.md).  
B11.1 protocol: [`docs/B11_1_QUANTILE_TEACHER.md`](docs/B11_1_QUANTILE_TEACHER.md).

## Current software state

```text
package version       0.17.0
current leader        B7.1 full-corpus weak supervision
leader macro AUC      0.5644802945
active experiment     B11.1 calibration-aware quantile teacher tails
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
| **B11.1** | **per-target quantile teacher tails** | pending | **active / predeclared pseudo audit** |

## Retained B7.1 benchmark

B7.1 uses:

- B5 competition-only image-report encoder initialization;
- six historical dual MRI streams;
- 16 distributed 2.5D positions per stream;
- ConvNeXtTiny slice encoder;
- cross-sequence Transformer memory;
- 12 learned pathology queries;
- frozen B6 v1.2.1 weak labels;
- target-balanced weak BCE;
- 3,120 active weak-training studies per epoch;
- 4 complete epochs;
- TTA center offsets `[-1, 0, 1]`.

Measured development result:

```text
macro AUC = 0.5644802945
n         = 58 gold studies
bootstrap = 5000 study-level replicates
```

The 58-study surface has been reused for sequential development, so this is a **development/model-selection estimate**, not pristine independent validation.

## B10 result — physical-scale normalization

B10 tested the label-free geometry normalization:

```text
native DICOM pixels
  -> 0.3125 mm/pixel canonical in-plane spacing
  -> approximately 160 mm physical FOV
  -> unchanged 224 x 224 resize
  -> unchanged B7.1 model/training recipe
```

The geometry audit covered all `15,468` selected weak-training series (`geometry_coverage = 1.0`) without gold-label use.

B10 result:

```text
B10 macro AUC        0.5523982721
95% CI              [0.4935605888, 0.6091548645]
B7.1 macro AUC       0.5644802945
median(B10-B7.1)    -0.0121030792
95% paired CI       [-0.0507382525, +0.0250750953]
P(B10 > B7.1)        0.2706
```

Decision: reject B10-v1 as a global replacement. Do not construct target-specific B7.1/B10 winners from the same 58 studies.

## B11-v1 result — pseudo-label viability gate failed

B11-v1 asked whether the completed B7.1 model could add low-weight pseudo supervision to B6-unsupervised cells using one global absolute probability rule:

```text
teacher mean >= 0.90 OR teacher mean <= 0.10
TTA probability range <= 0.05
```

The label-free audit found:

```text
B6 cells                14123
pseudo cells              4794
combined cells            18917
B6 active studies          3120
combined active studies    4000
newly activated studies     880
```

However, pseudo labels were extremely asymmetric:

```text
pseudo positive cells       23
pseudo negative cells     4771
```

Additional failures:

```text
Medial Meniscus pseudo cells   0
Synovitis pseudo cells          0
Lateral OA pseudo cells        21
required minimum per target    25
```

Therefore `viability_passed = false` and **B11-v1 student training must not be run**.

## Why B11.1 exists

The follow-up label-free diagnostic showed that B11-v1 failed primarily because **teacher probability calibration differs strongly by target**, not because TTA predictions are unstable.

Examples from B6-unsupervised cells:

```text
ACL teacher range              ~0.05 to 0.51
MCL teacher range              ~0.03 to 0.40
Medial Meniscus                ~0.18 to 0.89
Synovitis                      ~0.72 to 0.89
```

Synovitis predictions are especially stable but never cross `0.90`; Medial Meniscus is also stable but never reaches the global positive threshold. A single `0.10/0.90` rule is therefore inappropriate across all 12 pathologies.

## B11.1 frozen policy

B11.1 replaces absolute probability cutoffs with **per-target relative teacher tails**, using only B6-unsupervised non-gold cells.

For each target independently:

1. compute the teacher probability 5th and 95th percentiles;
2. require TTA probability range `<= 0.05`;
3. stable bottom 5% tail -> pseudo target `0.10`;
4. stable top 5% tail -> pseudo target `0.90`;
5. pseudo base weight `0.10`;
6. cap pseudo weight mass at `15%` of the original B6 base-weight mass for that target;
7. never overwrite a B6-supervised cell.

Quantile thresholds are derived without gold labels.

### B11.1 viability gate

Before any student training, the pseudo audit must satisfy:

```text
>= 2500 pseudo cells overall
>= 100 pseudo cells per target
>= 50 stable low-tail cells per target
>= 50 stable high-tail cells per target
```

If the audit fails, do not tune the quantile fractions or TTA threshold using gold results.

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
0.17.0
```

## Tests

```bash
python -m compileall -q src tests

pytest -q \
  tests/test_b7_weak_supervision.py \
  tests/test_b9_strict_routing.py \
  tests/test_b10_physical_scale.py \
  tests/test_b11_teacher_student.py \
  tests/test_b11_1_quantile_pseudo.py
```

## Active next step — B11.1 pseudo audit

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b11-1-pseudo \
  --config configs/b11_1_quantile_teacher.yaml \
  --data-root "$DATA_ROOT" \
  --teacher-checkpoint runs/b7_1_full_coverage/b7_model.pt \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b11_1_quantile_teacher/pseudo
```

Inspect before any B11.1 student training:

```bash
cat runs/b11_1_quantile_teacher/pseudo/pseudo_summary.json
cat runs/b11_1_quantile_teacher/pseudo/pseudo_policy.json
```

Important quantities:

```text
viability_passed
pseudo_cells
combined_active_studies
newly_activated_studies
per-target low-tail counts
per-target high-tail counts
per-target quantile thresholds
per-target applied pseudo mass
```

**Do not train from the old B11-v1 pseudo artifacts.** B11.1 student training should only be implemented/run after the B11.1 label-free pseudo audit passes and its artifacts are frozen.

## Validation caution

The same 58 gold studies have informed repeated method decisions. Current scores are **development/model-selection estimates**, not pristine hidden-test estimates. Do not tune target-specific routing, physical spacing/FOV, weak-label weights, pseudo-label thresholds, per-target model winners, or ensemble weights from these 58 labels and then present the result as independent validation.

Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.

`docs/competition.md` remains a preserved competition-summary document and is intentionally not rewritten by experiment updates.
