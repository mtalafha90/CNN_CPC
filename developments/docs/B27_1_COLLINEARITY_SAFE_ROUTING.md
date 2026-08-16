# B27.1 — collinearity-safe pathology routing

> **Status — 2026-08-16:** FROZEN / READY FOR FIXED-E2 TRAINING. **B20 remains the active working model.** No B27 or B27.1 expert/gold outcome has been inspected.

## Why B27.1 exists

B27 completed its fixed-E2 training surface correctly, but a pre-outcome inspection of its learned routing tables showed that the fluid-sensitivity and fat-suppression routing tables were numerically identical for every target. The training metadata audit then confirmed the cause on the complete 17,475-series B20 gradient surface:

```text
series                         17475
fluid_id == fat_id             17475
fluid_id != fat_id                 0
fraction identical               1.0

pair (1,1)                     7459
pair (2,2)                    10016
```

Thus `Fluid_Sensitive` and `Fat_Suppression` carry only one empirical degree of freedom on this training surface. B27 represented that same degree of freedom twice and added both biases to the same attention logit.

This was discovered from training metadata and trained routing parameters only, before any B27 reused-expert evaluation. B27.1 is therefore a pre-outcome structural correction rather than an outcome-driven retune.

## Single correction

B27 used:

```text
route = plane + fluid + fat
12 x (3 + 2 + 2) = 84 parameters
```

B27.1 uses:

```text
route = plane + paired_sequence
12 x (3 + 2) = 60 parameters
```

The paired sequence categories are exactly those observed on the frozen training surface:

```text
1 = structural + not-fat-suppressed
2 = fluid-sensitive + fat-suppressed
```

If a future/test series has unknown or discordant fluid/fat metadata, the B27.1 paired-sequence routing contribution is fixed at zero. This is a conservative inference policy; it does not invent a mapping for a metadata combination that was absent from training.

## Frozen controls

Everything else remains B20/B27:

```text
training studies            3120
usable B6 cells            14123
positive / negative        6871 / 7252
eligible MRI series        17475
series signature           5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
B16 encoder                 frozen
encoder SHA                 b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
B20 crop                    90% post-resize crop only
slice sampling              unchanged
optimizer / LR              unchanged
augmentation                unchanged
loader seed                 unchanged
scheduler horizon           5 epochs
training endpoint           fixed E2
expert labels in gradients  0
expert checkpoint selection none
```

All 60 routing parameters are zero-initialised. With the same shared state and zero routing, B27.1 is functionally identical to B20.

## Runtime

B27.1 retains the conservative execution guard:

```text
hard budget       <= 8.25 h
internal reserve   >= 30 min
competition limit      9 h
```

B27 itself completed E2 in about 58.9 minutes on the RTX A4500 Laptop GPU, so the 60-parameter B27.1 routing layer should remain far below the platform limit.

## Required training-time metadata assertion

B27.1 refuses to train unless the frozen training surface still reproduces:

```text
(1,1)  7459
(2,2) 10016
other      0
```

This prevents the corrected experiment from silently changing identity.

## Governance

Do not inspect B27 reused-expert performance as a way to choose between B27 and B27.1. B27.1 was defined before that outcome and is the scientifically cleaner representation of the observed metadata geometry.

After B27.1 training:

```text
1. verify exact E1/E2 coverage and frozen encoder SHA
2. inspect learned 60-parameter routing table
3. optional one-call Ollama plausibility review only
4. then perform the paired reused-expert B20-vs-B27.1 diagnostic
```

The reused 58-study expert surface remains development/post-hoc evidence, not independent validation. Hidden competition evaluation remains the independent performance signal.
