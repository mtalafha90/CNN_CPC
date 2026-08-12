# B6/B15 reused-gold diagnostic package

> **Status — 2026-08-12:** IMPLEMENTED / READY TO RUN. Package `0.24.2`. This is a no-GPU, no-training diagnostic on the already-reused 58-study gold development surface.

## Purpose

B15 produced a large improvement on the frozen weak-v2 surface (`0.5652498118 -> 0.7319060415`) but did not improve the reused-gold development score (`B13=0.6293565948`, `B15=0.6209002783`). The weak-v2 surface therefore remains useful as a measure of B6-teacher agreement, but it must not be treated as a surrogate selector for expert-label improvement.

This package asks a narrower question:

```text
Did B15 move toward the frozen B6 report process specifically on cells where B6 disagrees with expert gold?
```

It performs no optimization and must not be used to tune B15 or construct target-specific model mixtures.

## Four diagnostics

### 1. Coverage-conditioned B6 teacher AUC

Only cells satisfying the actual downstream B6 supervision eligibility are used:

```text
state in {positive, negated}
confidence >= 0.75
positive score = 0.85
negated score  = 0.05
```

The result is reported with coverage and both strict-all-12-target and relaxed bootstrap diagnostics.

**Important:** this AUC is conditional on B6 coverage. It is not directly comparable as a ceiling to B13/B15, which are evaluated on all `58 x 12 = 696` expert-labelled cells.

### 2. Full-surface B6 state-only ranking baseline

All 696 gold cells receive a frozen diagnostic score:

```text
positive      0.85
negated       0.05
uncertain     0.50
unmentioned   0.50
```

This produces a full 12-target macro AUC on exactly the same gold surface as the models. It is a descriptive state-only ranking baseline, not a theoretical teacher ceiling.

### 3. B6 state -> expert truth audit

For every target and every parser state the package exports:

```text
n
gold positive count
gold negative count
P(gold=1 | state)
mean / median confidence
count with confidence >= 0.75
```

A pooled state summary is also included. This directly quantifies whether `positive`, `negated`, `uncertain`, and `unmentioned` have different expert-truth meaning by pathology.

### 4. B13 -> B15 noise-alignment audit

On every high-confidence B6 positive/negated gold cell, the package records:

```text
expert truth
B6 state / binary class / soft score
B6 correct vs wrong
B13 prediction
B15 prediction
B15-B13 probability movement
movement toward B6
movement toward expert truth
change in absolute distance to B6 soft target
change in absolute distance to expert truth
```

The primary diagnostic subset is **B6-wrong cells**.

Strong evidence of B6-error imitation would require the B6-wrong group to show both:

```text
B13 -> B15 mean movement toward B6 > 0
B13 -> B15 mean distance from expert truth increases > 0
```

The package adds 5,000-replicate **study-cluster bootstrap** intervals for these movement statistics so multiple target cells from the same knee remain clustered.

This remains a reused-gold descriptive diagnostic, not independent validation.

## Inputs

Existing artifacts only:

```text
competition train.csv
runs/b6_report_labels_v121/structured_labels.csv
runs/b13_imagenet/gold_eval/gold_predictions.csv
runs/b15_mri_ssl/gold_confirmation/gold_predictions.csv
```

No checkpoint loading and no GPU are required.

## Run

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b6-b15-diagnostic \
  --data-root "$DATA_ROOT" \
  --structured runs/b6_report_labels_v121/structured_labels.csv \
  --b13-predictions runs/b13_imagenet/gold_eval/gold_predictions.csv \
  --b15-predictions runs/b15_mri_ssl/gold_confirmation/gold_predictions.csv \
  --out-root runs/b6_b15_gold_diagnostic \
  --min-confidence 0.75 \
  --n-bootstrap 5000
```

## Outputs

```text
runs/b6_b15_gold_diagnostic/
├── diagnostic.json
├── state_truth_audit.csv
├── high_confidence_alignment_cells.csv
└── alignment_by_target.csv
```

### `diagnostic.json`

Contains:

- coverage-conditioned teacher AUC and coverage;
- full-surface state-only baseline AUC;
- reproduced B13/B15 full-gold scores and paired comparison;
- alignment summaries for all high-confidence cells, B6-correct cells and B6-wrong cells;
- study-cluster bootstrap intervals;
- explicit evidence flags.

### `state_truth_audit.csv`

Per-target and pooled truth distribution for all four B6 states.

### `high_confidence_alignment_cells.csv`

Cell-level audit trail for every B6-supervised gold cell.

### `alignment_by_target.csv`

Descriptive per-target movement summary. **Do not use it to build target-specific B13/B15 winners.**

## Interpretation discipline

The diagnostic can support statements such as:

> B15 moved more strongly toward the B6 report process on cells where the frozen B6 teacher disagreed with expert gold.

It cannot by itself prove:

```text
all B6 noise is class-conditional or instance-dependent
B13/B15 have reached a theoretical teacher ceiling
no future architecture or representation can improve expert AUC
weak-v2 is an expert-validation surface
```

B13 and B15 were also trained on different B6 study surfaces (`3120` versus `2497` downstream studies), so B13->B15 movement is descriptive rather than a pure encoder-only causal contrast.

The next training experiment should be chosen only after this diagnostic is inspected globally, without target-wise model mixing or B15 retuning from the reused gold set.
