# B11.1 — calibration-aware quantile teacher tails

> **Status — 2026-08-12:** **COMPLETED / REJECTED GLOBALLY.** B11.1 remains a historical pseudo-label experiment. B13 is now the reused-gold development champion, and completed B15 points the next diagnostic toward report-state quality rather than another teacher-tail sweep.

## Motivation

B11-v1 found 4,794 pseudo-cells but failed its predeclared viability gate because one absolute `0.10/0.90` teacher-confidence rule was badly mismatched to target-specific probability calibration. Only 23 accepted cells were positive; Medial Meniscus and Synovitis had zero accepted cells.

B11.1 therefore used **relative per-target teacher tails** rather than one global probability cutoff.

## Frozen pseudo policy

For each target separately, among cells with B6 weight exactly zero:

1. derive the teacher-probability 5th and 95th percentiles from all 4,349 non-gold studies;
2. require TTA probability range `<=0.05`;
3. stable bottom 5% tail -> pseudo target `0.10`;
4. stable top 5% tail -> pseudo target `0.90`;
5. base pseudo weight `0.10`;
6. cap total pseudo weight mass per target at `15%` of original B6 base-weight mass;
7. never overwrite B6 supervision.

## Frozen pseudo audit

```text
B6 cells                 14123
pseudo cells               3656
combined cells             17779
B6 active studies           3120
combined active studies     3454
newly activated studies      334
pseudo low cells            1864
pseudo high cells           1792
viability_passed             true
```

Frozen pseudo CSV SHA-256:

```text
94f914f3548fab17f67ae0bf1906424bac850268c09ce5febede72b2ed7246b6
```

## Completed training

```text
epoch 1 loss 0.7522255329
epoch 2 loss 0.6619129086
epoch 3 loss 0.6250558991
epoch 4 loss 0.5935560781
```

Each epoch covered exactly 3,454 studies, 14,123 original B6 cells and 3,656 pseudo cells.

## Reused-gold result

```text
B11.1 macro AUC        0.5506902702
95% CI                [0.4917424630,0.6086153876]
B7.1 macro AUC         0.5644802945
median(B11.1-B7.1)    -0.0126224565
95% paired CI         [-0.0487500119,+0.0195120537]
P(B11.1 > B7.1)        0.2184
```

The paired interval crosses zero, so a statistically decisive degradation is not claimed. Operationally, B11.1 supplied no evidence to replace B7.1 and the teacher-derived pseudo-label completion branch was closed.

## Successor context through B15

```text
B12 gold  0.5660915179
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249  rejected
B15 gold  0.6209002783  no global improvement
```

B15 is especially relevant to the interpretation of B11.1. B15 raised frozen weak-v2 B6-teacher agreement from `0.5652498118` to `0.7319060415`, with paired median `+0.1675245839`, yet did not improve global expert-gold AUC. This strengthens the case for auditing the information content of the report states themselves before another pseudo-label completion strategy.

## Current decision

**B11.1 remains rejected globally.** Do not revive it by changing quantiles, pseudo weights, target-specific thresholds or mixing B11.1 with B13/B15 based on reused gold.

The next evidence-driven step is a B6 state audit (`positive`, `negated`, `uncertain`, `unmentioned`) against expert truth. Only if that audit supports additional information should a separately versioned supervision successor be defined.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). Post-B15 roadmap: [`RAISING_AUC.md`](RAISING_AUC.md).