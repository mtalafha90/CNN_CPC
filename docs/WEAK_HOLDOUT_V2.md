# Weak B6 validation holdout v2

> **Status — 2026-08-11:** v1 is superseded before model training. Package `0.23.0` freezes a stratified report-group-safe v2 and uses strict all-12-target bootstrap.

## Why v1 was superseded

The first report-group-safe 20% split was frozen before any B15/control training and produced:

```text
surface                    weak_b6_holdout_v1
active studies             3120
train studies              2496
holdout studies             624
train report groups        2430
holdout report groups       609
report-group overlap          0
holdout usable cells       2697
holdout positive cells     1257
holdout negative cells     1440
manifest SHA-256
fdbc02f88e5a4eff31783b4242890e943609d5c783bd54aca38af8a89e7e0968
```

Most targets had usable class counts, but Synovitis did not:

| Target | Positive | Negative |
|---|---:|---:|
| ACL | 93 | 228 |
| MCL | 42 | 208 |
| Medial Meniscus | 217 | 114 |
| Lateral Meniscus | 89 | 235 |
| Medial OA | 86 | 60 |
| Lateral OA | 69 | 67 |
| PF OA | 131 | 71 |
| Effusion | 259 | 163 |
| **Synovitis** | **70** | **1** |
| Baker's | 106 | 98 |
| Contusion | 60 | 97 |
| Fracture | 35 | 98 |

The global frozen B6 corpus contains only 17 explicit Synovitis negatives. A holdout with one negative gives an unstable AUC: ordinary study bootstrap frequently omits that single negative, making Synovitis undefined. If undefined targets are silently omitted from the macro, different bootstrap replicates estimate different quantities.

No B15 candidate or matched B13 control was trained on v1. Therefore v1 can be safely superseded before model fitting without using gold performance or model predictions.

## v2 split policy

`weak_b6_holdout_v2` preserves the same scientific purpose but improves the split design using only frozen B6 labels and normalized report groups.

The v2 algorithm:

1. starts from the exact 3,120 active non-gold B6 studies and 14,123 usable cells;
2. groups duplicate normalized reports so no report group can straddle train/holdout;
3. represents each group by 24 weak-label counts: positive and negative counts for each of 12 targets;
4. generates a frozen deterministic set of candidate group-safe 20% splits using seed `2026`;
5. rejects candidates that do not satisfy the rare-class floor in both train and holdout when globally feasible;
6. among feasible candidates, chooses the split that best matches the requested holdout size and approximately 20% of every target/class count;
7. uses no gold labels, no MRI predictions, and no model performance during split selection.

Frozen defaults:

```text
surface                 weak_b6_holdout_v2
holdout fraction        0.20
seed                    2026
minimum class count     4 per side where globally feasible
candidate splits        4096
report-group overlap    0 required
uses gold labels        false
uses model predictions  false
```

For Synovitis negatives (`17` globally), the v2 hard floor requires at least four negatives in the holdout and at least four in weak training. The desired 20% count is approximately 3.4, so the floor deliberately raises the holdout target to four.

## Strict macro-AUC bootstrap

Weak-surface scoring in `0.23.0` uses a strict study bootstrap:

```text
one bootstrap replicate
    -> sample holdout studies with replacement
    -> compute all 12 target AUCs
    -> accept replicate only if all 12 are defined
    -> macro = mean of exactly those 12 AUCs
```

A replicate with an undefined target is discarded. The estimand therefore remains the same 12-target macro AUC in every usable replicate.

Reported diagnostics include:

```text
n_bootstrap
n_valid_replicates
valid_replicate_fraction
strict_all_12_targets = true
```

The weak surface still measures **agreement with the B6 report teacher, not expert truth**. Absolute weak AUC is not a gold or leaderboard estimate.

## Freeze v2 before training

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

Outputs:

```text
runs/weak_holdout_v2/weak_holdout_manifest.csv
runs/weak_holdout_v2/weak_holdout.json
```

Once v2 is successfully frozen, **do not regenerate it based on model performance**. Its manifest SHA becomes part of the B15/control experiment contract.

## Training contract after v2 is frozen

A valid B15 comparison requires two newly trained models on the same weak-train partition:

```text
B13-control
ImageNet -> B13 hierarchy
trained only on v2 weak-train studies

B15-candidate
ImageNet -> knee-MRI SSL -> B13 hierarchy
trained downstream only on the same v2 weak-train studies
```

Every v2 holdout StudyInstanceUID must be absent from downstream weak-supervision training for both models.

For the B15 SSL stage, all 58 gold studies remain excluded from SSL optimization. The exact SSL data policy must be frozen separately before training.

Existing B13/B14 checkpoints were trained on all 3,120 active B6 studies and cannot be retrospectively evaluated on v2 and called validation.

## Decision flow

```text
v2 weak holdout
    -> paired strict 12-target bootstrap
    -> rank B13-control vs B15 candidate by teacher agreement
    -> take only the predeclared winner to the reused 58-study gold surface
    -> treat gold as development confirmation only
    -> use Kaggle hidden evaluation as the independent signal
```
