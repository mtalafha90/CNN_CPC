# Raising macro AUC beyond B13

> **Status — 2026-08-11.** B13 remains the development champion at macro AUC `0.6293565948`. B14 completed at `0.6197914249` and was rejected globally. The full corrected B13 slice audit is complete and rejects slice-count undersampling as a primary bottleneck. Package `0.23.0` supersedes weak holdout v1 with a stratified v2 plus strict all-12-target bootstrap.

## Current evidence

```text
B13 macro AUC        0.6293565948
B14 macro AUC        0.6197914249

paired B14-B13
median difference    -0.0093726931
95% paired CI        [-0.0469823411,+0.0250137870]
P(B14 > B13)          0.2924
```

B14 fit the B6 weak labels more strongly than B13 (`0.5822778610` versus `0.6132239342` final training loss) but did not improve global macro AUC. This argues against simply increasing downstream token capacity or fitting B6 harder.

## What the B6 audit implies

```text
sensitivity          0.975
specificity          0.606
positive precision   0.690
balanced accuracy    0.790
coverage             0.361
```

These values establish noisy and incomplete weak supervision. They do **not** establish a numerical downstream macro-AUC ceiling. Supervision quality remains a plausible limiting factor, but historical B6 v1.2.1 must stay frozen for controlled B7-B15 comparisons.

## Diagnostic 1 complete — exact B13 slice exposure

The retired `16 / n_slices` proxy was wrong because B13 uses 16 2.5D triplets, training gaps `[1,2]` with center jitter `+/-2`, and evaluation TTA offsets `[-1,0,1]`.

The full corrected audit on the exact 17,475-series non-gold B13 surface found:

```text
series audited/readable  17475 / 17475
slices/series median     30 (p95 50, max 320)

eval unique fraction     median 100.0% (p25 100.0%)
eval max skipped run     median 0.0 slices (p95 0.0)
training expected/view   median 87.0%
complete eval exposure   95.9%
eval run >=2 slices      3.9%
eval run >=3 slices      3.8%
skipped-run length       median 0.0 mm (p95 0.0 mm)

Axial      n=4455   eval=100.0% max-run=0.0 train/view=85.2%
Coronal    n=5815   eval=100.0% max-run=0.0 train/view=87.0%
Sagittal   n=7205   eval=100.0% max-run=0.0 train/view=87.0%
```

Decision:

```text
slice-count undersampling as primary B13 bottleneck -> REJECT
```

Do not launch a 24/32/48-slice sweep from the reused gold surface. This result does not rule out in-plane resolution loss at `224x224`.

Canonical record: `docs/B13_SLICE_EXPOSURE_AUDIT.md`.

## Diagnostic 2 — weak holdout v1 superseded

The first report-group-safe 20% split had the correct global size and zero leakage:

```text
holdout studies          624
holdout usable cells    2697
report-group overlap       0
gold studies               0
manifest SHA
fdbc02f88e5a4eff31783b4242890e943609d5c783bd54aca38af8a89e7e0968
```

However, Synovitis had:

```text
70 positive / 1 negative
```

With only one negative, a substantial fraction of ordinary study-bootstrap replicates omit that class. Allowing undefined targets to drop out would change the macro estimand across replicates.

No B15 candidate or matched B13 control was trained on v1. It is therefore superseded **before model fitting**, without using gold performance or model predictions.

## Weak holdout v2

Package `0.23.0` makes `rsna-knee-weak-holdout` freeze `weak_b6_holdout_v2` by default.

The split policy uses only frozen B6 labels and normalized report groups:

```text
holdout fraction        0.20
seed                    2026
report groups           mandatory
candidate splits        4096
minimum class count     4 per side where globally feasible
split objective         match holdout size + all 24 target/class counts
uses gold labels        false
uses model predictions  false
```

For the 17 global Synovitis negatives, v2 requires at least 4 in holdout and at least 4 in weak training.

Freeze v2 before any B15/control training:

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

Once v2 is frozen successfully, its manifest SHA is part of the experiment contract and must not be regenerated based on model results.

## Strict weak-surface bootstrap

Weak evaluation now keeps the estimand fixed:

```text
bootstrap studies with replacement
-> compute all 12 target AUCs
-> discard replicate if any target AUC is undefined
-> accepted macro = mean of exactly 12 target AUCs
```

Always report both `n_valid_replicates` and `valid_replicate_fraction`. The weak surface still measures teacher agreement, not expert truth.

## B15 remains the next representation hypothesis

```text
ImageNet ConvNeXt-Tiny
        |
        v
competition knee-MRI self-supervised adaptation
        |
        v
B13 one-token-per-series hierarchy
        |
        v
frozen downstream B6 recipe
```

For a valid weak-surface comparison, train two new models on the same v2 weak-train partition:

```text
control:   ImageNet -> B13 hierarchy
candidate: ImageNet -> MRI SSL -> B13 hierarchy
```

Existing B13/B14 checkpoints were trained on all 3,120 B6-active studies and cannot be retrospectively scored on v2 as validation.

The intended decision chain is:

```text
freeze v2
   -> matched B13-control + B15 candidate
   -> paired strict 12-target weak bootstrap
   -> one predeclared winner to reused 58-study gold surface
   -> gold used as development confirmation only
   -> Kaggle hidden evaluation as independent signal
```

## Priority order

1. **Freeze weak holdout v2.** This is now the immediate gate.
2. **Implement B15 plus a matched B13-control trainer that respects the v2 manifest.**
3. **Compare on v2 with paired strict all-12-target bootstrap.**
4. **Take only the predeclared winner to one reused-gold development confirmation.**
5. **Investigate supervision quality** under a separately versioned/frozen label experiment if B15 stalls.
6. **Investigate in-plane resolution** only as a separately predeclared global experiment; slice-count undersampling itself is closed.
7. **Multi-seed/global ensembling** only after structure is settled and without gold-selected target weights.

## Still prohibited

```text
target-wise B13/B14 winners
gold-selected slice counts
gold-selected thresholds
gold-selected ensemble weights
retrospective weak-holdout evaluation of checkpoints trained on holdout studies
regenerating v2 based on model performance
claiming the weak holdout is expert truth
claiming the reused 58 studies are independent validation
claiming a 0.75-0.80 supervision ceiling from B6 balanced accuracy
```

The objective remains higher **global macro ROC AUC** through controlled, reproducible representation or supervision improvements rather than increasingly fine tuning to the repeatedly reused 58 labelled studies.
