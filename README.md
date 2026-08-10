# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a production-oriented PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training regime contains 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current experiment snapshot — 2026-08-10:** **B7.1 full-corpus weak supervision is the current best standalone development model**, with macro AUC `0.5644802945` and 95% bootstrap CI `[0.5052432984, 0.6229422178]`. The paired bootstrap favors B7.1 over B7-v1 with `P=0.8694` and over B5 with `P=0.8716`, but both 95% paired intervals still cross zero on only 58 studies. The predeclared fixed B5+B7.1 50:50 rank ensemble scored `0.5540141184` and was rejected. **B8 pathology-aware spatial anatomy learning is implemented and currently training; no B8 gold score is recorded yet.**

The canonical measured-results table is [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md). `docs/competition.md` is a preserved competition-summary document and is intentionally not rewritten by experiment updates.

## Current software state

```text
package version     0.13.0
current leader      B7.1 full-corpus weak supervision
leader macro AUC    0.5644802945
current experiment  B8 spatial anatomy learning — training in progress
external pretrained weights  disabled
final inference     MRI-only
```

## Verified data status

| Check | Verified result |
|---|---:|
| Training studies | 4,407 |
| Fully gold-labelled studies | 58 |
| Report-only studies | 4,349 |
| Training series rows | 24,371 |
| Selected training series audited | 21,886 / 21,886 decoded |
| Candidate DICOM files audited | 732,554 / 732,556 decoded |
| Selected series lost to corruption | 0 |
| Local test preflight | 3 studies, 14 / 14 selected streams decoded |
| External pretrained weights | disabled |
| Final inference | MRI-only |

Two selected series each contain one unreadable DICOM instance; both remain usable under the configured partial-corruption gate.

## Twelve targets

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture

## Six MRI streams

```text
sagittal_fluid       sagittal_structural
coronal_fluid        coronal_structural
axial_fluid          axial_structural
```

Missing streams are explicitly masked and never fabricated.

## Methodology evolution

```text
competition MRI
-> DICOM decoding / metadata repair
-> six semantic streams
-> distributed 2.5D triplets
-> ConvNeXt-Tiny encoder

B0-B3
-> neural supervised / weakly supervised baselines

B4
-> frozen strong-SSL encoder
-> mean/std/max stream features
-> target-wise PCA + balanced logistic regression

B5
-> competition-report TF-IDF/SVD alignment
-> report-aligned MRI representation
-> unchanged B4 frozen probe

B6
-> multilingual structured report states
   positive / negated / uncertain / unmentioned
-> frozen weak-label source

B7
-> B5-initialized six-stream Transformer
-> 12 pathology queries
-> direct B6 weak supervision

B7.1
-> same B7 recipe
-> full 3,120-study coverage every epoch
-> current best standalone development model

B8 — current training experiment
-> initialize from completed B7.1
-> preserve 2x2 ConvNeXt spatial grid per sampled slice
-> 384 MRI memory tokens/study instead of 96
-> fixed gentle pathology stream/slice attention priors
-> same frozen B6 supervision and full-corpus coverage
```

Reports are training supervision only. Hidden/test inference remains MRI-only.

## Completed experiment ladder

| ID | Method | Macro AUC | Decision |
|---|---|---:|---|
| B0 | random initialization | `0.4762536432` | baseline |
| report teacher | fold-safe rules + TF-IDF | `0.49245` | rejected as general teacher |
| B1 | strong competition-only SSL | `0.5030284974` | retained reference |
| B2 | 0.1x encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware MIL | `0.4944652486` | rejected globally |
| B1+B3 rank | fixed 50:50 rank average | `0.5048038179` | neutral |
| B4 | frozen SSL + target-wise PCA/LR | `0.5137567459` | retained image-only ablation |
| B4.1 | one shared policy | `0.4847792672` | rejected |
| B4.2 | four pathology-group policies | `0.4901328905` | rejected |
| B4.3 | two-way-CV target selector | `0.4966083942` | rejected |
| B1+B4 raw | fixed 50:50 probability average | `0.5050` | rejected |
| B1+B4 rank | fixed 50:50 rank average | `0.5167` | historical fixed ensemble |
| B5 | image-report SSL + unchanged B4 probe | `0.5243650851` | retained representation baseline |
| B6 | structured multilingual report labels | n/a | completed / frozen weak-label source |
| B7-v1 | B5-init pathology-query model + B6 weak labels | `0.5397724412` | retained coverage ablation |
| **B7.1** | **B7 with full 3,120-study epoch coverage** | **`0.5644802945`** | **current best standalone model** |
| B5+B7.1 rank | fixed global 50:50 rank ensemble | `0.5540141184` | rejected vs B7.1 |
| B8 | B7.1-init 2x2 spatial anatomy/pathology attention | pending | **training in progress** |

## Key current comparisons

B7-v1 -> B7.1:

```text
point delta               +0.0247078534
paired median difference  +0.0241102714
95% paired CI             [-0.0140197876, +0.0660558004]
P(B7.1 > B7-v1)            0.8694
```

B5 -> B7.1:

```text
point delta               +0.0401152095
paired median difference  +0.0399233552
95% paired CI             [-0.0301354430, +0.1092349994]
P(B7.1 > B5)               0.8716
```

B7.1 -> fixed B5+B7.1 rank ensemble:

```text
B7.1 AUC                   0.5644802945
ensemble AUC               0.5540141184
median(ensemble-B7.1)     -0.0105429030
95% paired CI             [-0.0523218181, +0.0333886570]
P(ensemble > B7.1)         0.3054
```

The fixed ensemble question is closed: no 60:40, 70:30, raw-probability, target-specific, or calibrated blend search is allowed on the same 58 development labels.

## B6/B7 supervision contract

Frozen B6 v1.2.1 training export:

```text
report-only studies             4349
active weakly labelled studies  3120
inactive zero-usable studies    1229
usable target cells            14123
positive cells                  6871
negative cells                  7252
```

B7/B7.1/B8 keep the same global asymmetric weak-label rule:

| B6 state | Soft target | Base weight |
|---|---:|---:|
| positive | `0.85` | `0.50` |
| negated | `0.05` | `1.00` |
| uncertain | ignored | `0.00` |
| unmentioned | ignored | `0.00` |

Gold labels do not enter B7/B7.1/B8 gradients or early stopping. The B6 gold audit informed the global policy, so these 58-study results are explicitly development/model-selection estimates rather than pristine independent validation.

## B8 current experiment

B8 is documented in [`docs/B8_SPATIAL_ANATOMY.md`](docs/B8_SPATIAL_ANATOMY.md).

The frozen experiment changes the MRI memory representation while retaining the successful B7.1 initialization and supervision contract:

```text
B7.1: 6 streams x 16 slices x 1 token  = 96 MRI tokens
B8:   6 streams x 16 slices x 4 regions = 384 MRI tokens
```

B8 uses a 2x2 spatial grid from the final ConvNeXt feature map and fixed gentle pathology-specific stream/slice attention priors. The in-plane fixed region prior is uniform because preprocessing does not certify a canonical left/right or anterior/posterior pixel orientation.

Current training command:

```bash
rsna-knee-b8 \
  --config configs/b8_spatial_anatomy.yaml \
  --data-root "$DATA_ROOT" \
  --b71-checkpoint runs/b7_1_full_coverage/b7_model.pt \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b8_spatial_anatomy
```

Do not record a B8 AUC until the frozen training run completes and the first gold development evaluation is performed exactly once.

## Installation

```bash
conda create -n rsna-knee python=3.12 -y
conda activate rsna-knee
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install pytest pillow
pytest -q
```

## Useful current commands

```bash
# Inspect data
python -m rsna_knee.cli inspect --data-root "$DATA_ROOT"

# B7.1 checkpoint audit
python - <<'PY'
import torch
p=torch.load('runs/b7_1_full_coverage/b7_model.pt', map_location='cpu', weights_only=False)
print(p['variant'])
print(p['completed_epochs'])
print(p['config'].get('b7_experiment_name'))
print(p['supervision']['training_studies'])
print(p['supervision']['training_usable_cells'])
PY

# B8 training
rsna-knee-b8 \
  --config configs/b8_spatial_anatomy.yaml \
  --data-root "$DATA_ROOT" \
  --b71-checkpoint runs/b7_1_full_coverage/b7_model.pt \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b8_spatial_anatomy
```

After B8 training completes, inspect `history.json` and `supervision_plan.json` before running `rsna-knee-b8-eval`.

## Documentation map

- [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) — canonical current results/status
- [`docs/strategy.md`](docs/strategy.md) — modeling strategy and current next step
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — validation protocol and campaign caveats
- [`docs/LOCAL_REAL_DATA_TRAINING.md`](docs/LOCAL_REAL_DATA_TRAINING.md) — current workstation runbook
- [`docs/TRAINING_FROM_ZERO.md`](docs/TRAINING_FROM_ZERO.md) — fresh-machine/current experiment ladder
- [`docs/B5_IMAGE_REPORT_SSL.md`](docs/B5_IMAGE_REPORT_SSL.md) — B5 representation baseline
- [`docs/B6_STRUCTURED_REPORT_LABELS.md`](docs/B6_STRUCTURED_REPORT_LABELS.md) — frozen B6 weak-label source
- [`docs/B7_WEAK_SUPERVISION.md`](docs/B7_WEAK_SUPERVISION.md) — B7-v1 experiment
- [`docs/B7_1_FULL_COVERAGE.md`](docs/B7_1_FULL_COVERAGE.md) — current leader
- [`docs/B5_B71_FIXED_RANK_ENSEMBLE.md`](docs/B5_B71_FIXED_RANK_ENSEMBLE.md) — rejected fixed ensemble
- [`docs/B8_SPATIAL_ANATOMY.md`](docs/B8_SPATIAL_ANATOMY.md) — current training experiment
- [`README_KAGGLE_METHODS.md`](README_KAGGLE_METHODS.md) — public methodology review/context
- [`docs/competition_policy.md`](docs/competition_policy.md) — conservative execution policy
- [`docs/data.md`](docs/data.md) — verified data/DICOM contract
- [`docs/references.md`](docs/references.md) — references and reviewed public work
- [`docs/competition.md`](docs/competition.md) — preserved competition summary

## Validation caution

The same 58 gold studies have now informed repeated method decisions. The campaign is **model-selection cross-validation**, not a pristine independent estimate of hidden-test performance.

Do not:

- optimize ensemble weights on the 58 gold labels;
- select target-specific post-hoc model winners;
- retune B6 parser rules from the gold audit;
- tune target-specific B7/B8 weak-label weights from observed gold AUCs;
- search B8 spatial grids, prior strengths, epochs, or target-specific priors after reading the first B8 score and still call it B8-v1;
- claim leaderboard superiority without an actual competition submission result.
