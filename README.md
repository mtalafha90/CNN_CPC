# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a production-oriented PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The project is built around the released supervision regime: 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and macro ROC AUC across 12 pathologies.

> **Current experiment snapshot — 2026-08-10:** **B5 image-report representation learning is the main standalone baseline and the highest current standalone point estimate**, with macro AUC `0.5243650851` and 95% bootstrap CI `[0.4728108406, 0.5761619105]`. Under the unchanged B4 downstream probe, the paired B5-vs-B4 bootstrap gives median difference `+0.0105821232`, 95% CI `[-0.0408197338, +0.0622131599]`, and `P(B5 > B4)=0.656`: positive but statistically inconclusive evidence of improvement. B4 (`0.5137567459`) is retained as the image-only ablation. A previously fixed B1+B4 rank ensemble scored `0.5167`; no new ensemble-weight tuning is performed on the same 58 gold studies.

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

B4 — image-only ablation
strong SSL encoder frozen
-> mean/std/max stream features
-> target-specific PCA + logistic regression

B5 — main standalone baseline
strong SSL encoder
+ competition reports represented by TF-IDF -> TruncatedSVD
+ image-image SSL
+ acquisition metadata loss
+ image-report alignment
-> B5 MRI encoder frozen
-> same mean/std/max stream features
-> unchanged B4 target-specific PCA + logistic regression
-> MRI-only at inference
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
| B4 | frozen SSL + target-wise PCA/LR | `0.5137567459` | retained image-only ablation |
| B4.1 | one shared policy | `0.4847792672` | rejected |
| B4.2 | four pathology-group policies | `0.4901328905` | rejected |
| B4.3 | two-way-CV target selector | `0.4966083942` | rejected |
| B1+B4 raw | fixed 50:50 probability average | `0.5050` | rejected |
| B1+B4 rank | fixed 50:50 rank average | `0.5167` | retained fixed ensemble; no tuning |
| **B5** | **image-report SSL + unchanged B4 probe** | **`0.5243650851`** | **main standalone baseline; best point estimate** |

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

B5 versus B4, using the same downstream probe:

```text
B4 macro AUC            = 0.5137567459
B5 macro AUC            = 0.5243650851
B5 95% CI               = [0.4728108406, 0.5761619105]
median B5-B4 difference = +0.0105821232
95% paired CI           = [-0.0408197338, +0.0622131599]
P(B5 > B4)              = 0.656
```

B5 has the highest standalone point estimate, but its superiority over B4 is not claimed as statistically established because the paired interval crosses zero.

## Why B4 selector tuning is closed

B4's target-wise inner selections are unstable because each inner fold contains only about 18–20 studies. Three follow-ups tested shared, grouped, and two-way-CV policy selection. All three reduced pooled OOF performance. Further selector/grid variants based on the same 58 outer labels would increasingly meta-fit the validation campaign.

B4 is retained as the image-only ablation. The completed B5 result shifts the primary representation baseline to report-aligned B5 without reopening the downstream selector.

## B5 — completed main representation baseline

B5 used only the 4,349 report-only competition studies for representation training. The 58 gold studies were excluded completely.

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

No external language model and no external pretrained image weights were used. The report branch is discarded after training; the saved downstream artifact is an MRI encoder.

Completed checkpoint:

```text
runs/b5_report_ssl/b5_encoder.pt
```

Training summary:

```text
epochs                  4
batches               4000
study draws          16000
active 2.5D examples 158886
loss          5.5204 -> 4.7049
report NCE    4.6031 -> 3.2901
report cosine 0.8015 -> 0.5924
budget limited          false
```

Frozen gold-feature audit:

```text
checkpoint             runs/b5_report_ssl/b5_encoder.pt
studies                58
feature shape          [58, 6, 2304]
pooling                mean + std + max
encoder frozen         true
completed epochs       4
external pretrained    false
```

Pooled B5 OOF result:

```text
macro AUC = 0.5243650851
95% CI   = [0.4728108406, 0.5761619105]
```

Per-target AUC:

| Target | B5 AUC |
|---|---:|
| ACL | `0.6678921569` |
| MCL | `0.4058956916` |
| Medial Meniscus | `0.6658653846` |
| Lateral Meniscus | `0.6173913043` |
| Medial OA | `0.6589147287` |
| Lateral OA | `0.4042553191` |
| PF OA | `0.6061776062` |
| Effusion | `0.5167701863` |
| Synovitis | `0.5555555556` |
| Baker's | `0.3858695652` |
| Contusion | `0.3994601889` |
| Fracture | `0.4083333333` |

B5 improves 8 of 12 target point estimates versus B4. The largest descriptive gains are Medial Meniscus, Synovitis, Medial OA, ACL and Effusion; the largest losses are Contusion and Fracture. These target-level differences are not used for post-hoc model selection.

See [`docs/B5_IMAGE_REPORT_SSL.md`](docs/B5_IMAGE_REPORT_SSL.md) for the full controlled experiment and interpretation.

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

# B4 image-only frozen probe
rsna-knee-b4 extract \
  --config configs/train_local_ssl_strong.yaml \
  --split train --scope gold \
  --out runs/b4_frozen_ssl/gold_features.npz

rsna-knee-b4 nested \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b4_frozen_ssl/gold_features.npz \
  --out-root runs/b4_frozen_ssl \
  --n-bootstrap 5000

# B5 frozen probe
mkdir -p runs/b5_frozen_probe
rsna-knee-b4 extract \
  --config configs/train_local_ssl_strong.yaml \
  --checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --split train --scope gold \
  --out runs/b5_frozen_probe/gold_features.npz

rsna-knee-b4 nested \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b5_frozen_probe/gold_features.npz \
  --out-root runs/b5_frozen_probe \
  --n-bootstrap 5000

python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b5_frozen_probe/oof.csv \
  --n-bootstrap 5000 \
  --out runs/b5_frozen_probe/eval.json

python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b4_frozen_ssl/oof.csv \
  --compare-oof runs/b5_frozen_probe/oof.csv \
  --n-bootstrap 5000 \
  --out runs/b4_vs_b5.json
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
- tune B5 report-loss weights, temperatures, or extra epochs from the completed outer B5 OOF result;
- use the target-level B4/B5 differences to construct a post-hoc mixed predictor;
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
