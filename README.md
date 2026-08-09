# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a production-oriented PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The project is built around the released supervision regime: 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and macro ROC AUC across 12 pathologies.

> **Current experiment snapshot — 2026-08-09:** B4 frozen strong-SSL features + target-wise PCA/logistic regression is the best clean standalone point estimate (`0.5137567459`). A fixed equal-weight B1+B4 rank ensemble is numerically highest (`0.5167`) but is statistically tied with B4 (`P=0.5544`). **B5 image-report representation learning is currently running; no B5 AUC is available yet.**

The canonical measured-results table is [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).

`docs/competition.md` is a preserved competition-summary document and is intentionally not changed by implementation or experiment updates.

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

Observed coverage:

| Stream | Selected | Missing |
|---|---:|---:|
| sagittal_fluid | 4,401 | 6 |
| sagittal_structural | 4,294 | 113 |
| coronal_fluid | 4,250 | 157 |
| coronal_structural | 3,440 | 967 |
| axial_fluid | 4,407 | 0 |
| axial_structural | 1,094 | 3,313 |

Missing streams are expected and explicitly masked; they are never fabricated.

## Current methodology

```text
COMPETITION MRI
DICOM -> metadata repair -> six semantic streams
      -> distributed 2.5D triplets
      -> ConvNeXt-Tiny encoder

B0/B1/B2
encoder + Transformer/pathology heads -> 12 logits

B3
encoder + pathology-specific low-capacity MIL

B4
strong SSL encoder frozen
-> mean/std/max stream features
-> target-specific PCA + logistic regression

B5 (running)
strong SSL encoder
+ competition reports represented by TF-IDF -> TruncatedSVD
+ image-image SSL
+ acquisition metadata loss
+ image-report alignment
-> MRI encoder only at inference
```

Reports are training supervision only. The hidden/test inference path remains MRI-only.

## Completed controlled experiments

| ID | Method | Macro AUC | Decision |
|---|---|---:|---|
| B0 | random initialization | `0.4762536432` | baseline |
| report teacher | fold-safe rules + TF-IDF | `0.49245` | rejected as general teacher |
| B1 | strong competition-only SSL | `0.5030284974` | retained reference |
| B2 | 0.1x encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware MIL | `0.4944652486` | rejected globally |
| B1+B3 rank | fixed 50:50 rank average | `0.5048038179` | neutral |
| **B4** | frozen SSL + target-wise PCA/LR | **`0.5137567459`** | **best standalone point estimate** |
| B4.1 | one shared policy | `0.4847792672` | rejected |
| B4.2 | four pathology-group policies | `0.4901328905` | rejected |
| B4.3 | two-way-CV target selector | `0.4966083942` | rejected |
| B1+B4 raw | fixed 50:50 probability average | `0.5050` | rejected |
| B1+B4 rank | fixed 50:50 rank average | `0.5167` | numerically best; tied with B4 |
| **B5** | image-report representation learning | pending | **running** |

### Current statistical interpretation

B4 versus B1:

```text
median B4-B1 difference = +0.01021
95% CI                  = [-0.05143, +0.07094]
P(B4 > B1)              = 0.6378
```

Fixed B1+B4 rank ensemble versus B4:

```text
median difference       = +0.00276
95% CI                  = [-0.03513, +0.04174]
P(ensemble > B4)        = 0.5544
```

Therefore the ensemble is not claimed as an improvement despite its higher point estimate.

## Why B4 selector tuning is closed

B4's target-wise inner selections are unstable because each inner fold contains only about 18–20 studies. Three follow-ups tested shared, grouped, and two-way-CV policy selection. All three reduced pooled OOF performance. Further selector/grid variants based on the same 58 outer labels would increasingly meta-fit the validation campaign.

The next scientific question is therefore representation quality, not another downstream selector.

## B5 — current running experiment

B5 uses only the 4,349 report-only competition studies for representation training. The 58 gold studies are excluded completely.

Text branch:

```text
competition reports
-> word TF-IDF (1-2 grams)
-> TruncatedSVD (<=256 dimensions)
-> normalized report embedding
```

MRI branch:

```text
strong competition-only SSL ConvNeXt
-> image-image SSL objective
-> plane/sequence metadata objectives
-> image-report alignment objective
```

No external language model and no external pretrained image weights are used. The report branch is discarded after training; the saved downstream artifact is an MRI encoder.

Run:

```bash
rsna-knee-b5 \
  --config configs/train_local_ssl_strong.yaml \
  --checkpoint runs/ssl_strong/ssl_encoder.pt \
  --out-root runs/b5_report_ssl
```

After B5 finishes, the first evaluation deliberately reuses the **unchanged original B4 probe**. See [`docs/B5_IMAGE_REPORT_SSL.md`](docs/B5_IMAGE_REPORT_SSL.md).

## Installation

```bash
conda create -n rsna-knee python=3.12 -y
conda activate rsna-knee
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install pytest pillow
pytest -q
```

## Useful commands

```bash
# Inspect data
python -m rsna_knee.cli inspect --data-root "$DATA_ROOT"

# Preflight
python -m rsna_knee.cli preflight \
  --data-root "$DATA_ROOT" \
  --split train \
  --sample-size 24

# Strong competition-only SSL
python -m rsna_knee.cli pretrain \
  --config configs/train_local_ssl_pretrain.yaml

# B1 Stage-1 folds
python -m rsna_knee.cli train --config configs/train_local_ssl_strong.yaml --fold 0
python -m rsna_knee.cli train --config configs/train_local_ssl_strong.yaml --fold 1
python -m rsna_knee.cli train --config configs/train_local_ssl_strong.yaml --fold 2

# B4 frozen probe
rsna-knee-b4 extract \
  --config configs/train_local_ssl_strong.yaml \
  --split train --scope gold \
  --out runs/b4_frozen_ssl/gold_features.npz

rsna-knee-b4 nested \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b4_frozen_ssl/gold_features.npz \
  --out-root runs/b4_frozen_ssl \
  --n-bootstrap 5000
```

## Documentation map

- [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) — canonical current results/status
- [`docs/data.md`](docs/data.md) — verified data/DICOM contract
- [`docs/strategy.md`](docs/strategy.md) — modeling strategy and decisions
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — validation protocol and caveats
- [`docs/competition_policy.md`](docs/competition_policy.md) — conservative execution policy
- [`docs/LOCAL_REAL_DATA_TRAINING.md`](docs/LOCAL_REAL_DATA_TRAINING.md) — current workstation runbook
- [`docs/TRAINING_FROM_ZERO.md`](docs/TRAINING_FROM_ZERO.md) — fresh-machine runbook
- [`docs/REPORT_TEACHER.md`](docs/REPORT_TEACHER.md) — report-teacher benchmark
- [`docs/SSL_STRONG.md`](docs/SSL_STRONG.md) — strong SSL experiment
- [`docs/B2_DISCRIMINATIVE_FINETUNE.md`](docs/B2_DISCRIMINATIVE_FINETUNE.md)
- [`docs/B3_PATHOLOGY_AWARE_MIL.md`](docs/B3_PATHOLOGY_AWARE_MIL.md)
- [`docs/B4_FROZEN_SSL_CLASSICAL.md`](docs/B4_FROZEN_SSL_CLASSICAL.md)
- [`docs/B4_1_SHARED_POLICY.md`](docs/B4_1_SHARED_POLICY.md)
- [`docs/B4_2_GROUPED_FROZEN_SSL.md`](docs/B4_2_GROUPED_FROZEN_SSL.md)
- [`docs/B4_3_TWO_WAY_CV_FROZEN_SSL.md`](docs/B4_3_TWO_WAY_CV_FROZEN_SSL.md)
- [`docs/B5_IMAGE_REPORT_SSL.md`](docs/B5_IMAGE_REPORT_SSL.md)
- [`README_KAGGLE_METHODS.md`](README_KAGGLE_METHODS.md) — public methodology review/context
- [`docs/references.md`](docs/references.md) — references and reviewed public work
- [`docs/competition.md`](docs/competition.md) — preserved competition summary

## Validation caution

Each individual candidate uses leakage-aware fold logic, but the same 58 gold studies have now supported multiple method decisions. The campaign as a whole is increasingly **model-selection cross-validation**, not a pristine independent estimate of hidden-test performance.

Do not:

- optimize ensemble weights on the 58 gold labels;
- select target-specific post-hoc model winners from outer OOF;
- create further B4 selector variants from observed outer results;
- report a B5 score before its fixed frozen probe completes;
- claim leaderboard superiority without an actual competition submission result.

## Competition execution policy

The conservative defaults remain:

- one GPU;
- CPU multiprocessing for DICOM/data work;
- `runtime_budget_hours: 8.5`;
- external pretrained weights disabled;
- competition-data checkpoint provenance checked;
- validation/submission contracts recorded in checkpoints;
- final inference MRI-only;
- final output exactly `submission.csv`.
