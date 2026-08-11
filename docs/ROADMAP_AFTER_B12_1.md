# Roadmap after B12.1

> **Status — 2026-08-11:** B14 is completed and rejected globally. B13 remains the retained development champion. Before B15 training, package `0.22.1` adds a corrected exact B13 slice-exposure audit and a frozen report-group weak holdout. The reserved next representation hypothesis remains **B15: ImageNet -> competition knee-MRI self-supervised adaptation -> B13 hierarchical aggregation**.

## Current reference state

```text
B7.1 macro AUC        0.5644802945
B12 macro AUC         0.5660915179
B13 macro AUC         0.6293565948   retained development champion
B14 macro AUC         0.6197914249   completed / rejected globally

B13 vs B12
median difference      +0.0638674720
95% paired CI          [+0.0127183837,+0.1144643292]
P(B13 > B12)            0.9920

B13 vs B7.1
median difference      +0.0652260946
95% paired CI          [+0.0039768779,+0.1266069220]
P(B13 > B7.1)           0.9808

B14 vs B13
raw macro difference    -0.0095651699
median difference       -0.0093726931
95% paired CI           [-0.0469823411,+0.0250137870]
P(B14 > B13)             0.2924
```

## Governing rules

1. The 58 fully labelled studies are a repeatedly reused development/model-selection surface, not independent validation.
2. Primary model selection remains global macro ROC AUC across 12 targets.
3. Paired aligned bootstrap remains required for controlled local comparisons.
4. Do not construct target-specific winners from per-target AUCs.
5. Do not tune slice counts, series caps, thresholds, pooling heads, ImageNet variants, normalization, LR, epoch count or ensemble weights from the reused gold surface.
6. No gold labels enter gradients, early stopping or checkpoint selection.
7. Any B15 SSL stage must exclude all 58 gold studies from SSL optimization.
8. Any model scored on the new weak holdout must have been trained with those holdout UIDs excluded.
9. The independent Kaggle hidden-test/leaderboard remains more valuable than repeated local tuning.

## Completed B13

B13 combines:

```text
torchvision ConvNeXt-Tiny IMAGENET1K_V1
standard ImageNet mean/std normalization
16 slice tokens / series -> one learned series token
K series tokens -> study Transformer -> pathology queries
```

Frozen gold macro AUC: `0.6293565948`, 95% CI `[0.5789896351,0.6775867717]`.

B13 remains the retained global model.

## Completed B14 — full slice-token memory

B14 tested whether B13's one-token-per-series compression discarded useful slice-level information by retaining all `K x 16` slice tokens.

```text
epoch 1 loss  0.7346330162
epoch 2 loss  0.6606430862
epoch 3 loss  0.6074723502
epoch 4 loss  0.5822778610

B14 macro AUC      0.6197914249
95% CI            [0.5706800512,0.6693542716]
raw B14-B13        -0.0095651699
median difference  -0.0093726931
95% paired CI      [-0.0469823411,+0.0250137870]
P(B14 > B13)        0.2924
```

B14 fit B6 more strongly than B13 but did not improve macro AUC. The decision is:

```text
B14 -> REJECT GLOBALLY
B13 -> RETAIN
```

Do not run B14 epoch 5 and do not construct target-wise B13/B14 mixtures.

## Pre-B15 diagnostic gate

### A. Exact B13 slice-exposure audit

The original `16 / number_of_slices` diagnostic was invalid because B13 uses 16 **2.5D triplets**, training gap choices `[1,2]`, center jitter `+/-2`, and evaluation TTA offsets `[-1,0,1]`.

The corrected audit must reconstruct and verify the exact non-gold B13 surface:

```text
3120 active B6 studies
17475 eligible real MRI series
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

It computes actual unique frame exposure and orientation-correct through-plane spacing.

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-slice-audit \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out runs/slice_audit_b13
```

Use the full 17,475-series result for decisions. A `--limit` run is smoke-test only.

Decision interpretation:

```text
near-complete eval exposure / no multi-slice gaps
    -> slice count is not supported as the primary bottleneck

material multi-slice gaps across many series
    -> slice exposure remains a plausible later global experiment
```

No target-wise slice policies are allowed.

### B. Freeze the weak B6 holdout

The earlier estimate that a 20% holdout would have ~3,120 validation studies was wrong. It has roughly 624 studies, and B6 contains only 14,123 usable cells across all 12 targets. Therefore uncertainty must be measured empirically on the actual sparse holdout.

Freeze the report-group-safe split **before any new control/candidate training**:

```bash
rsna-knee-weak-holdout \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --holdout-fraction 0.20 \
  --seed 2026 \
  --out-root runs/weak_holdout_v1
```

The manifest is hashed and records actual study/cell/per-target class counts. Report groups are mandatory and cannot straddle train/holdout.

Critical rule:

> Existing B13/B14 checkpoints are **not** valid weak-holdout validation models because they were trained on the full 3,120-study B6 surface.

For future weak-surface ranking, retrain a matched control and candidate on the same weak-train partition and compare them with aligned bootstrap on the frozen holdout.

```text
weak holdout   -> biased teacher-agreement ranking
58 gold        -> one development confirmation only
Kaggle hidden  -> independent signal
```

## Next major hypothesis — B15

### Intended structure

```text
ImageNet ConvNeXt-Tiny
        |
        v
competition knee-MRI self-supervised adaptation
        |
        v
B13 one-token-per-series hierarchical architecture
        |
        v
frozen B6 weak-supervision downstream recipe
```

### Clean B15 comparison

To use the weak holdout correctly, train both models from scratch on the same frozen weak-train partition:

```text
control:   ImageNet -> B13 hierarchy
candidate: ImageNet -> MRI SSL -> B13 hierarchy
```

Then perform the paired comparison on `weak_holdout_v1`. Only one predeclared winner should proceed to the reused 58-study gold surface.

### Required safeguards

Before B15 implementation/training, freeze the SSL objective and data policy:

```text
58 gold studies excluded from SSL optimization
no gold labels in SSL
no B6 labels in SSL
no report labels in SSL unless explicitly declared as a different experiment
no gold-based SSL checkpoint selection
no gold-based SSL hyperparameter sweep
same B13 downstream hierarchy unless separately predeclared
same frozen weak-train partition for B13-control and B15
same downstream B6 policy
same all-series policy
same downstream epoch/TTA policy unless a new experiment explicitly changes it
```

Candidate competition-only MRI SSL families may be considered **before** B15 is frozen:

```text
same-study / cross-sequence contrastive learning
masked slice/token reconstruction
cross-plane consistency
teacher-student self-distillation
```

Only one B15 protocol should be selected and frozen before evaluation.

## Supervision-quality hypothesis

B6 audit values (`specificity 0.606`, `coverage 0.361`, balanced accuracy `0.790`) establish noisy/incomplete supervision, but they do **not** establish a numerical downstream AUC ceiling such as `0.75-0.80`.

If B15 stalls, improved report parsing should become a separate new frozen B6 version with its own audit before any downstream run. Do not alter historical B6 v1.2.1 in place.

## Later candidates

- A slice/resolution experiment only if the corrected exposure audit shows a material global gap.
- A larger foundation-encoder change can be reserved for **B16** if B15 is unsuccessful and development budget still justifies it.
- Scanner/protocol robustness becomes **B17** and remains optional/diagnostic.
- Multi-seed/global ensembling comes only after structure is settled and must not use gold-selected target weights.

## Independent competition signal

```text
B13 retained
   |
   +--> Kaggle test inference / submission
   |
   +--> corrected diagnostics
            |
            v
          B15 controlled representation experiment
```

A leaderboard result is the next genuinely independent signal and should not be turned into a high-frequency tuning loop.

## Explicitly not allowed

```text
B14 epoch extension
target-wise B13/B14 mixture
gold-selected slice count
gold-selected thresholds
gold-selected ensemble weights
retrospective weak validation of checkpoints trained on the holdout
calling the weak surface expert truth
calling the reused 58 studies independent validation
claiming a 0.75-0.80 supervision ceiling from B6 balanced accuracy
```

The goal remains a higher global macro AUC through interpretable, reproducible representation or supervision improvements rather than increasingly fine tuning to 58 repeatedly reused cases.
