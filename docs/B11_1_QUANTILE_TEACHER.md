# B11.1 — calibration-aware quantile teacher tails

> **Status — 2026-08-11:** **COMPLETED / REJECTED GLOBALLY.** B7.1 remains the retained development champion at macro AUC `0.5644802945`.

## Motivation

B11-v1 found 4,794 pseudo-cells but failed its predeclared viability gate because a single absolute `0.10/0.90` teacher-confidence rule was badly mismatched to target-specific probability calibration. Only 23 accepted cells were positive; Medial Meniscus and Synovitis had zero accepted cells.

A label-free diagnostic showed that TTA predictions were generally stable while absolute probability ranges varied strongly by pathology. B11.1 therefore used **relative per-target teacher tails** rather than one global probability cutoff.

## Frozen pseudo policy

For each target separately, among cells with B6 weight exactly zero:

1. derive the teacher-probability 5th and 95th percentiles from all 4,349 non-gold studies;
2. require TTA probability range `<= 0.05`;
3. stable bottom 5% tail -> pseudo target `0.10`;
4. stable top 5% tail -> pseudo target `0.90`;
5. base pseudo weight `0.10`;
6. cap total pseudo weight mass per target at `15%` of original B6 base-weight mass;
7. never overwrite B6 supervision.

The B7.1 teacher and all thresholds were label-free with respect to the 58 gold studies.

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

Every target passed the predefined gates. Synovitis alone reached the 15% pseudo-mass cap, scaling its per-cell pseudo weight to approximately `0.08242`; all other targets retained weight `0.10`.

Frozen pseudo CSV SHA-256:

```text
94f914f3548fab17f67ae0bf1906424bac850268c09ce5febede72b2ed7246b6
```

## Completed training

The student started from the same B5 encoder initialization as B7.1, not from the B7.1 teacher. Four complete full-coverage epochs finished with no budget limitation:

```text
epoch 1  loss 0.7522255329
epoch 2  loss 0.6619129086
epoch 3  loss 0.6250558991
epoch 4  loss 0.5935560781
```

Each epoch covered exactly:

```text
studies        3454
batches        1727
B6 cells      14123
pseudo cells   3656
combined      17779
pseudo low     1864
pseudo high    1792
```

## Frozen gold result

```text
B11.1 macro AUC        0.5506902702
95% CI                [0.4917424630, 0.6086153876]
B7.1 macro AUC         0.5644802945
median(B11.1-B7.1)    -0.0126224565
95% paired CI         [-0.0487500119, +0.0195120537]
P(B11.1 > B7.1)        0.2184
```

The paired interval crosses zero, so the reused 58-study development set does not establish a statistically decisive degradation. Operationally, however, B11.1 provides no evidence to replace B7.1 and its point estimate is lower.

## Decision

**Reject B11.1 globally and close the teacher-derived pseudo-label completion branch for now.** Do not construct target-wise B7.1/B11.1 winners from the repeatedly reused 58-study development set.

The next experiment is **B12 variable-number-of-series modeling**, documented in [`B12_VARIABLE_SERIES.md`](B12_VARIABLE_SERIES.md).
