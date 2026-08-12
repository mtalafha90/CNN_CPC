# B11 — conservative B7.1 teacher–student completion

> **Status — 2026-08-12:** **B11-v1 STOPPED AT ITS PREDECLARED VIABILITY GATE.** The absolute-confidence pseudo-label policy was not trained as a student model. B11.1 later tested quantile tails and was rejected globally. B13 is the current reused-gold champion; B15 has since shifted the next diagnostic toward direct report-state quality.

## Motivation

The frozen B6 export contains 4,349 report-only studies but only 3,120 have at least one usable report-derived target cell. Across all report-only studies there are `4,349 x 12 = 52,188` possible target cells, while B6 directly supervises only 14,123. B11 asked whether a completed B7.1 MRI model could conservatively add information to a subset of remaining B6-unsupervised cells.

## Frozen B11-v1 policy

For each B6-unsupervised target cell, the B7.1 teacher was evaluated with TTA `[-1,0,1]`.

Acceptance required:

```text
teacher mean >= 0.90 OR teacher mean <= 0.10
AND
max(view probability) - min(view probability) <= 0.05
```

The pseudo-target was the teacher mean. Base pseudo weight was `0.20`, with per-target total pseudo mass capped at `25%` of the original B6 base-weight mass. B6-supervised cells were never overwritten.

Predeclared viability gates:

```text
>= 500 accepted pseudo cells overall
>= 25 accepted pseudo cells for every target
```

## Actual viability outcome

B11-v1 generated 4,794 pseudo-cells overall but failed the target-wise gate. Only 23 accepted cells were positive, and Medial Meniscus and Synovitis had zero accepted cells under the absolute threshold policy.

Decision:

```text
B11-v1 -> STOP before student training
```

The thresholds were not loosened in place after the failure.

## Successor B11.1

B11.1 used target-wise probability quantile tails rather than absolute `0.10/0.90` cutoffs. It passed its pseudo-label viability checks and trained, but the reused-gold result was:

```text
B11.1 macro AUC        0.5506902702
B7.1 macro AUC         0.5644802945
median(B11.1-B7.1)    -0.0126224565
95% paired CI         [-0.0487500119,+0.0195120537]
P(B11.1 > B7.1)        0.2184
```

B11.1 was rejected globally.

## Successor context through B15

```text
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249  rejected
B15 gold  0.6209002783  no global improvement
```

B15 is especially informative: it improved frozen weak-v2 B6-teacher agreement from matched-control `0.5652498118` to `0.7319060415` with paired median `+0.1675245839`, yet did not improve global expert-gold AUC.

That result makes another teacher-derived completion rule lower priority than directly auditing the report states that generate the weak supervision.

## Current decision

B11-v1 remains a **failed viability experiment**, not a trained model result. Do not relax its thresholds retrospectively and call the result B11-v1.

The immediate next evidence step is a B6 report-state audit (`positive`, `negated`, `uncertain`, `unmentioned`) against expert truth. Only if the audit supports a new use of ignored states should a separately versioned/frozen supervision experiment be defined.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). B11.1 record: [`B11_1_QUANTILE_TEACHER.md`](B11_1_QUANTILE_TEACHER.md). Post-B15 roadmap: [`RAISING_AUC.md`](RAISING_AUC.md).