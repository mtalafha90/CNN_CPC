# Roadmap after B12.1

> **Status — 2026-08-11:** B13 remains the retained development champion. B14 is completed/rejected. The full B13 slice audit is complete and rejects slice-count undersampling as a primary bottleneck. Weak holdout v1 is superseded before model training because its Synovitis holdout had `70 positive / 1 negative`. Package `0.23.0` introduces stratified weak holdout v2 plus strict all-12-target bootstrap. The reserved next representation hypothesis remains **B15: ImageNet -> competition knee-MRI self-supervised adaptation -> B13 hierarchy**.

## Current reference state

```text
B7.1 macro AUC        0.5644802945
B12 macro AUC         0.5660915179
B13 macro AUC         0.6293565948   retained development champion
B14 macro AUC         0.6197914249   completed / rejected globally

B14 vs B13
raw macro difference    -0.0095651699
median difference       -0.0093726931
95% paired CI           [-0.0469823411,+0.0250137870]
P(B14 > B13)             0.2924
```

## Governing rules

1. The 58 fully labelled studies are a repeatedly reused development/model-selection surface, not independent validation.
2. Primary selection remains global macro ROC AUC across 12 targets.
3. Do not construct target-specific winners from per-target AUCs.
4. Do not tune slice counts, thresholds, normalization, LR, epoch count or ensemble weights from the reused gold surface.
5. No gold labels enter gradients, early stopping or checkpoint selection.
6. Any B15 SSL stage must exclude all 58 gold studies from SSL optimization.
7. Any model scored on weak holdout v2 must be trained with every v2 holdout UID excluded.
8. Weak-surface bootstrap is strict: accepted replicates must define all 12 target AUCs.
9. The independent Kaggle hidden evaluation is more valuable than repeated local tuning.

## Completed B13 / B14

```text
B13
ImageNet ConvNeXt-Tiny
one learned token per series
macro AUC 0.6293565948
-> RETAIN

B14
same ImageNet protocol
full K x 16 slice-token memory
macro AUC 0.6197914249
-> REJECT GLOBALLY
```

B14 reached lower B6 training loss (`0.5822778610`) than B13 (`0.6132239342`) without improving macro AUC. Do not extend B14 or create B13/B14 target-wise mixtures.

## Completed diagnostic — exact B13 slice exposure

The corrected audit reproduced the actual B13 2.5D sampler on all 17,475 eligible non-gold series:

```text
series audited/readable  17475 / 17475
slices/series median     30 (p95 50, max 320)

eval unique fraction     median 100.0% (p25 100.0%)
eval max skipped run     median 0.0 slices (p95 0.0)
training expected/view   median 87.0%
complete eval exposure   95.9%
eval run >=2 slices      3.9%
eval run >=3 slices      3.8%

Axial      n=4455   eval=100.0% max-run=0.0 train/view=85.2%
Coronal    n=5815   eval=100.0% max-run=0.0 train/view=87.0%
Sagittal   n=7205   eval=100.0% max-run=0.0 train/view=87.0%
```

Decision:

```text
slice-count undersampling as primary B13 bottleneck -> REJECT
```

No global slice-count sweep is justified from the current evidence. In-plane resolution remains a separate possible future question.

See `docs/B13_SLICE_EXPOSURE_AUDIT.md`.

## Weak holdout v1 — superseded

The first report-group-safe 20% split produced:

```text
train studies              2496
holdout studies             624
report-group overlap          0
holdout usable cells       2697
holdout positive cells     1257
holdout negative cells     1440
Synovitis                  70 positive / 1 negative
manifest SHA
fdbc02f88e5a4eff31783b4242890e943609d5c783bd54aca38af8a89e7e0968
```

No B15 or matched B13-control training used v1. It is superseded before model fitting because the single Synovitis negative makes a stable fixed 12-target macro bootstrap impossible without a high undefined-replicate rate.

## Weak holdout v2 — immediate gate

Package `0.23.0` freezes a better split using only frozen B6 labels and normalized report groups.

```text
surface                 weak_b6_holdout_v2
holdout fraction        0.20
seed                    2026
report grouping         mandatory
minimum class count     4 per side where globally feasible
candidate splits        4096
uses gold labels        false
uses model predictions  false
```

The split objective balances holdout size and all 24 target/class weak-label counts. For global Synovitis negatives (`17`), at least four must be in holdout and at least four must remain in weak training.

Freeze v2 **before any new control/candidate training**:

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-weak-holdout \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --holdout-fraction 0.20 \
  --seed 2026 \
  --min-class-count 4 \
  --search-candidates 4096 \
  --out-root runs/weak_holdout_v2
```

Once the manifest is successfully frozen, do not regenerate it based on model performance.

See `docs/WEAK_HOLDOUT_V2.md`.

## Strict weak-surface comparison

For every model evaluated on v2:

```text
study bootstrap
-> all 12 target AUCs computed
-> replicate rejected if any target is undefined
-> macro = mean of exactly 12 AUCs
```

A clean B15 comparison therefore requires two newly trained models on the same v2 weak-train partition:

```text
control:   ImageNet -> B13 hierarchy
candidate: ImageNet -> MRI SSL -> B13 hierarchy
```

Existing B13/B14 checkpoints are invalid for this weak-holdout comparison because they trained on all 3,120 active B6 studies.

## Next major hypothesis — B15

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
frozen B6 downstream recipe on v2 weak-train studies
```

Before B15 implementation/training, freeze the SSL objective and data policy:

```text
58 gold studies excluded from SSL optimization
no gold labels in SSL
no B6 labels in SSL
no report labels in SSL unless explicitly declared as another experiment
no gold-based SSL checkpoint selection
no gold-based SSL hyperparameter sweep
same B13 downstream hierarchy unless separately predeclared
same v2 weak-train partition for B13-control and B15 downstream training
same downstream B6 policy
same all-series policy
```

Candidate SSL families can be considered **before** freezing B15:

```text
same-study / cross-sequence contrastive learning
masked slice/token reconstruction
cross-plane consistency
teacher-student self-distillation
```

Only one B15 protocol should be selected and frozen before evaluation.

## Decision chain

```text
freeze weak holdout v2
       |
       +--> matched B13-control
       `--> B15 candidate
                 |
                 v
      paired strict 12-target weak bootstrap
                 |
                 v
      one predeclared winner to reused gold
                 |
                 v
       development confirmation only
                 |
                 v
          Kaggle hidden signal
```

## Later hypotheses

- Supervision-quality improvement must use a separately versioned/frozen B6 successor; do not alter B6 v1.2.1 in place.
- In-plane resolution can be a later global experiment, but slice-count undersampling itself is closed.
- Larger foundation encoders remain B16 territory if B15 fails and budget justifies it.
- Scanner/protocol robustness remains B17/diagnostic.
- Multi-seed/global ensembling comes only after structure is settled and cannot use gold-selected target weights.

## Explicitly not allowed

```text
B14 epoch extension
target-wise B13/B14 mixture
gold-selected slice count
gold-selected thresholds
gold-selected ensemble weights
retrospective weak validation of checkpoints trained on holdout studies
regenerating v2 based on model results
calling weak teacher agreement expert truth
calling the reused 58 studies independent validation
claiming a 0.75-0.80 supervision ceiling from B6 balanced accuracy
```

The goal remains a higher global macro AUC through controlled, reproducible representation or supervision improvements rather than increasingly fine tuning to 58 repeatedly reused cases.
