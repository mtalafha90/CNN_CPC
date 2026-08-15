# Competition summary

Competition: **RSNA Knee Abnormality Detection** (Kaggle, 2026).

> **Repository experiment snapshot — 2026-08-12:** B13 is the reused-gold development champion at macro AUC `0.6293565948`. B15 passed the frozen weak-v2 teacher-agreement gate but reached `0.6209002783` on its one-look reused-gold confirmation and did not replace B13. These are local development results, not leaderboard scores.

## Objective

Predict 12 abnormalities from a complete knee MRI study:

```text
ACL
MCL
Medial Meniscus
Lateral Meniscus
Medial OA
Lateral OA
PF OA
Effusion
Synovitis
Baker's
Contusion
Fracture
```

## Metric

The primary metric is **macro-averaged ROC AUC across all 12 targets**. Each target therefore has equal weight regardless of prevalence.

## Released training-data structure used by this repository

```text
training studies       4407
fully gold-labelled      58
report-only studies    4349
training series rows  24371
reports present        4407
```

`train_series.csv` supplies series-level metadata including anatomical plane and fluid-sensitive / fat-suppression flags. Final inference is MRI-only; reports are training supervision.

## Validation structure in this repository

Three evidence sources must be distinguished:

1. **58-study expert-gold surface** — repeatedly reused development/model-selection data, not independent validation.
2. **623-study frozen weak-v2 holdout** — B6 report-teacher agreement only, not expert truth.
3. **Kaggle hidden evaluation** — the next genuinely independent model-performance signal.

Frozen weak-v2:

```text
weak-train studies      2497
holdout studies          623
holdout usable cells    2875
report-group overlap       0
manifest SHA
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

## Current model evidence

```text
B13 gold             0.6293565948  retained champion
B14 gold             0.6197914249  rejected globally
B15 weak-v2          0.7319060415  gate passed
B15 reused gold      0.6209002783  no global improvement
```

B15's matched weak-v2 control was `0.5652498118`; paired B15-control median difference was `+0.1675245839`, 95% CI `[+0.1124433208,+0.2165156305]`, `P=1.0`.

The large teacher-agreement gain did not transfer to a higher expert-gold global macro AUC.

## Practical implication

The problem is closer to weakly/semi-supervised multi-series medical imaging than conventional fully supervised classification. A useful pipeline should exploit report/MRI structure while preserving strict leakage and validation boundaries.

The current next diagnostic is a B6 report-state audit (`positive`, `negated`, `uncertain`, `unmentioned`) against expert truth. Do not blindly treat report silence as negative.

Because the competition is active, this repository does not call any local method a winning solution and does not present unverified leaderboard claims as established performance.

See [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) and [`VALIDATION.md`](VALIDATION.md).