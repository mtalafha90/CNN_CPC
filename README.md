# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a production-oriented PyTorch pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The design targets the unusual supervision regime of this dataset: a very small trusted gold set, thousands of report-only studies, multiple MRI series per knee, and a macro-AUC objective across 12 pathologies.

The current implementation combines leakage-aware report supervision, six-stream MRI routing, distributed 2.5D sampling, ConvNeXt-Tiny encoding, cross-sequence Transformer fusion, pathology-query attention, nested gold validation, fold-safe image/report co-training, optional competition-data SSL, and strict one-GPU runtime control.

> `docs/competition.md` is a preserved competition-summary document and is intentionally not changed by implementation or workflow updates.

## Verified real-data status — 2026-08-08

The downloaded competition data and current pipeline have now been exercised on real files.

| Check | Verified result |
|---|---|
| Training studies | 4,407 |
| Fully gold-labeled studies | 58 |
| Report-only studies | 4,349 |
| Training series rows | 24,371 |
| Train preflight | 24 studies, 121/121 selected streams decoded, 4,045/4,045 files decoded |
| Local test preflight | all 3 studies, 14/14 selected streams decoded, 533/533 files decoded |
| Full selected-series audit | 21,886/21,886 series decoded |
| Full DICOM audit | 732,554/732,556 files decoded successfully |
| Global file decode failure rate | 2.73e-6 |
| Series lost to corruption | 0 |
| OA weak supervision | fixed and verified on the real reports |
| Smoke GPU | NVIDIA RTX A4500 Laptop GPU, BF16 |
| Ranking auxiliary | active after pair-friendly trusted sampling |

Two selected series each contained one unreadable DICOM file. Both series remained usable with 35/36 and 37/38 decoded frames respectively, well below the configured 20% per-series failure gate.

The current paired-sampler fold-0 smoke run completed end-to-end and produced every required Stage-1 artifact. Its best inner macro-AUC was `0.55135`; the outer TTA smoke score was `0.51396`. These values are **smoke diagnostics only**, not production performance claims.

Production Stage-1 results must come from the non-smoke three-fold runs and remain pending until those jobs finish.

## Core production contract

```text
SUPERVISION
report -> positive / negated / uncertain / unmentioned
       -> compartment-aware OA parsing
       -> fold-safe calibration
       -> confidence = evidence × information beyond prevalence
       -> unmentioned = zero direct weight by default
       -> finite official cells override weak supervision

MRI
DICOM -> metadata repair -> six semantic streams
      -> distributed 2.5D triplets
      -> ConvNeXt-Tiny encoder
      -> cross-sequence Transformer
      -> 12 interacting pathology queries
      -> 12 logits

LOSS
planned-epoch macro-balanced weighted BCE
+ confidence-gated pairwise ranking

SAMPLING
trusted = gold or unusually high-confidence pseudo-label
pair-friendly trusted batches for even batch sizes
preserves requested trusted-row fraction
keeps weak labels below the trusted/ranking gate unless justified

VALIDATION
outer gold fold = final OOF only
inner gold fold = epoch-count selection
Phase A discarded
fresh Phase B retrain
validation TTA == submission TTA
study bootstrap macro-AUC

STAGE 1
random initialization
or competition-data SSL initialization
candidate choice per outer fold uses INNER AUC only

STAGE 2
Phase A report-only
Phase B fresh report + fold-local cross-fitted image teacher
wrong-fold or incomplete teachers rejected
Stage 2 does not emit another weak_oof.csv

RUNTIME
one GPU
CPU multiprocessing for DICOM/data work
8.5 h software budget
10 min reserve
batch-level runtime guards
finish-time reserve includes OOF, weak OOF, bootstrap and serialization

INFERENCE
exact fold set
single checkpoint stage
identical model/stream contract
checkpoint validation TTA must match requested submission TTA
exact submission.csv schema
```

## Twelve targets

- ACL
- MCL
- Medial Meniscus
- Lateral Meniscus
- Medial OA
- Lateral OA
- PF OA
- Effusion
- Synovitis
- Baker's
- Contusion
- Fracture

## Six MRI streams

```text
sagittal_fluid       sagittal_structural
coronal_fluid        coronal_structural
axial_fluid          axial_structural
```

Observed coverage on the 4,407 training studies:

| Stream | Selected | Missing |
|---|---:|---:|
| sagittal_fluid | 4,401 | 6 |
| sagittal_structural | 4,294 | 113 |
| coronal_fluid | 4,250 | 157 |
| coronal_structural | 3,440 | 967 |
| axial_fluid | 4,407 | 0 |
| axial_structural | 1,094 | 3,313 |

Missing streams are expected and are masked by the model; they are not fabricated.

## Report teacher and OA update

The initial report lexicon was too narrow for osteoarthritis and produced zero weak-label weight for all three OA targets. The parser is now compartment-aware and recognizes explicit OA/arthrosis terminology plus cartilage loss, chondrosis/chondromalacia, osteophytes and related compartment-specific degenerative wording while avoiding generic meniscal degeneration.

Observed real-report states after the fix:

| Target | Positive | Negated | Unmentioned |
|---|---:|---:|---:|
| Medial OA | 492 | 339 | 3,576 |
| Lateral OA | 409 | 387 | 3,611 |
| PF OA | 695 | 379 | 3,333 |

The confidence remains deliberately conservative: these report-derived OA cells contribute to weighted BCE but do not automatically become gold-equivalent trusted examples.

## Pair-friendly ranking sampler

With production `batch_size: 2`, the earlier trusted/general sampler usually placed at most one trusted study in a minibatch. That made the pairwise ranking objective inactive because a trusted positive and trusted negative could not coexist in the same batch.

The current sampler groups trusted rows in pairs for even batch sizes while preserving the requested trusted-row fraction. The verified fold-0 smoke run produced:

```text
selection ranking pairs: 63
retrain ranking pairs:   61
```

All 12 targets contributed nonzero ranking pairs.

## Installation

Recommended Conda setup:

```bash
conda create -n rsna-knee python=3.12 -y
conda activate rsna-knee
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install pytest pillow
```

Check:

```bash
python -m rsna_knee.cli --help
pytest -q
```

## Local real-data workflow

For a complete fresh-machine guide, use:

- `docs/TRAINING_FROM_ZERO.md`

For the concise current-machine production sequence, use:

- `docs/LOCAL_REAL_DATA_TRAINING.md`

The high-level order is:

```text
inspect
-> validation manifests
-> train/test DICOM preflight
-> full audit
-> fold-0 smoke
-> Stage-1 random folds 0/1/2
-> optional competition-data SSL
-> Stage-1 inner-AUC candidate selection
-> Stage-2 folds 0/1/2
-> paired OOF evaluation
-> freeze final stage
-> three-fold inference
```

## Current Stage-1 random command

```bash
python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 0
```

Then folds 1 and 2 are run unchanged after the first production fold has been checked for computational correctness.

## Important artifacts per Stage-1 fold

```text
best.pt
bootstrap.json
calibration.json
calibration_selection.json
config.json
fold_assignments.csv
history.csv
metadata_repair.json
oof.csv
oof_center.csv
preflight.json
runtime.json
sampling.json
selection.json
supervision_plan.json
training_diagnostics.json
weak_oof.csv
```

`oof.csv` is the primary submission-policy TTA OOF file. `oof_center.csv` is diagnostic only. `weak_oof.csv` is the leakage-safe Stage-1 image teacher used downstream by Stage 2.

## Competition execution policy

The repository uses conservative defaults:

- one GPU only;
- no DDP or `torchrun`;
- CPU multiprocessing for data work;
- `runtime_budget_hours: 8.5`;
- ten-minute reserve;
- external pretrained weights off by default;
- competition-data SSL provenance checked;
- Internet-independent final inference;
- output file exactly `submission.csv`.

See `docs/competition_policy.md`.

## Repository map

```text
configs/train.yaml
src/rsna_knee/
  audit.py
  budget.py
  calibration.py
  cli.py
  constants.py
  cotrain.py
  data.py
  dataset.py
  dicom.py
  dicom_meta.py
  evaluation.py
  inference.py
  model.py
  policy.py
  preflight.py
  report_labels.py
  runtime.py
  sampling.py
  ssl.py
  training.py

docs/
  TRAINING_FROM_ZERO.md
  LOCAL_REAL_DATA_TRAINING.md
  VALIDATION.md
  data.md
  strategy.md
  competition_policy.md
  references.md
  competition.md              # preserved

fixtures/external_validation/
README_KAGGLE_METHODS.md
main.tex
```

## Methodological guarantees

- report silence is not converted into a negative;
- official finite labels override weak labels cell-by-cell;
- OA report rules are compartment-aware;
- outer gold never selects its own epoch count;
- Phase B starts from a fresh model;
- random-vs-SSL Stage-1 selection uses inner AUC only;
- Stage-2 Phase A does not use the image teacher;
- Stage-2 fold `k` can consume only safe fold-`k` weak OOF predictions;
- Stage 2 never re-exports its teacher rows as fresh OOF;
- validation TTA is predeclared and matches submission TTA;
- weighted BCE is macro-balanced over planned epoch supervision;
- ranking pairs and effective supervision are recorded per pathology;
- trusted sampling is pair-friendly without lowering confidence gates;
- duplicate normalized reports remain grouped;
- DICOM decode quality is audited before long GPU work;
- checkpoints self-describe fold, stage, architecture, stream order and TTA contract;
- final inference is MRI-only.

This repository does **not** claim leaderboard superiority. Production AUC, runtime and leaderboard results should be reported only after the corresponding real runs have completed.