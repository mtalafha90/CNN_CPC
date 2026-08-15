# B6 — structured multilingual report labels

> **Status — 2026-08-12:** **COMPLETE / FROZEN at v1.2.1.** B6 remains the historical weak-label source through B15. The completed B15 experiment makes a direct audit of B6's four report states the next evidence-driven step; B6 itself must not be edited in place from downstream outcomes.

## Goal

B6 converts each competition training report into auditable target-level weak states for all 12 abnormalities:

```text
positive
negated
uncertain
unmentioned
```

Each cell stores a fixed soft probability, confidence, reason, and evidence snippet. The 58 gold studies are audit-only and are excluded from `training_targets.csv`.

## Frozen leakage contract

B6 v1.2.1:

- uses only competition `train.csv` reports;
- uses no external model or external language resource;
- does not fit calibration on gold;
- does not convert report silence to a negative;
- excludes every gold row from the weak-training export;
- uses the audited confidence threshold `0.75`.

## Frozen report-only corpus

At confidence `>=0.75`, B6 produced:

```text
report-only studies   4349
active studies        3120
inactive studies      1229
usable cells         14123
positive cells        6871
negative cells        7252
```

Per target:

| Target | Positive | Negative | Usable |
|---|---:|---:|---:|
| ACL | 572 | 1,089 | 1,661 |
| MCL | 271 | 1,089 | 1,360 |
| Medial Meniscus | 1,126 | 536 | 1,662 |
| Lateral Meniscus | 448 | 1,182 | 1,630 |
| Medial OA | 484 | 334 | 818 |
| Lateral OA | 402 | 382 | 784 |
| PF OA | 682 | 372 | 1,054 |
| Effusion | 1,338 | 757 | 2,095 |
| Synovitis | 399 | 17 | 416 |
| Baker's | 557 | 476 | 1,033 |
| Contusion | 389 | 466 | 855 |
| Fracture | 203 | 552 | 755 |

## Frozen gold audit

Across 251 high-confidence usable gold cells:

```text
TP = 116
TN = 80
FP = 52
FN = 3

positive precision = 0.690476
sensitivity        = 0.974790
specificity        = 0.606061
NPV                = 0.963855
accuracy           = 0.780876
balanced accuracy  = 0.790425
coverage           = 0.360632
```

The audit motivated the fixed global downstream policy:

```text
B6 positive -> target 0.85, weight 0.50
B6 negated  -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
```

Because this global policy was informed by the gold audit, later gold scores are development/model-selection estimates even though gold labels did not enter model gradients or early stopping.

These audit metrics demonstrate noisy, sparse and asymmetric supervision. They do **not** establish a numerical downstream AUC ceiling.

## Downstream evidence through B15

Representative reused-gold results:

```text
B7-v1   0.5397724412
B7.1    0.5644802945
B11.1   0.5506902702
B12     0.5660915179
B13     0.6293565948  retained champion
B14     0.6197914249
B15     0.6209002783
```

B15 introduced a frozen weak-v2 teacher-agreement gate. On that surface:

```text
B13-v2 control  0.5652498118
B15            0.7319060415
raw delta      +0.1666562297
paired median  +0.1675245839
95% CI         [+0.1124433208,+0.2165156305]
P(B15>control)  1.0000
```

The gate passed decisively. Yet B15's one-look expert-gold macro AUC was `0.6209002783`, slightly below B13 `0.6293565948`.

This is the most important new B6-era lesson: **a large improvement in reproducing the frozen B6 target ordering did not translate to a global expert-gold improvement.**

## Current interpretation

The next question is not whether B6 should be tuned blindly. It is whether its four states contain different amounts of expert-label information by target.

The next diagnostic should therefore quantify, for each target/state:

```text
count
expert-positive fraction
expert-negative fraction
coverage
precision / NPV where meaningful
```

with particular attention to:

```text
P(expert positive | positive)
P(expert positive | negated)
P(expert positive | uncertain)
P(expert positive | unmentioned)
```

Do **not** assume `unmentioned = negative`. Report silence is distinct from explicit negation.

## Frozen artifacts

```text
runs/b6_report_labels_v121/
├── structured_labels.csv
├── training_targets.csv
├── review_queue.csv
├── audit.json
├── policy.json
└── gold_audit/
```

## Decision

**B6 v1.2.1 remains frozen.** Do not create another parser revision, confidence threshold, or state weighting based directly on the existing downstream gold results and still call it B6 v1.2.1.

If the state audit supports a new policy, define a separately versioned/frozen supervision successor before training.

Current campaign status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). Post-B15 roadmap: [`RAISING_AUC.md`](RAISING_AUC.md).