# B3 — pathology-aware stream MIL

> **Status — 2026-08-12:** **COMPLETED / REJECTED.** B3 remains a historical architecture ablation. B13 is now the reused-gold development champion; B15 completed without replacing it.

B3 tested whether a lower-capacity pathology-aware MIL head could use the strong competition-only SSL encoder more effectively than the global Transformer/pathology-query architecture.

## Architecture

B3 removed the global MRI Transformer/pathology-to-pathology Transformer and used target-specific attention first over sampled 2.5D positions within each stream, then over the six MRI streams. Soft anatomical/sequence priors were fixed before outer evaluation and were not hard masks.

B3 returned to the B1 supervised optimizer rather than inheriting the B2 lower-encoder-LR change.

## Result

```text
B3 macro AUC       0.4944652486
95% CI            [0.4314514263,0.5578825232]
median(B3-B1)     about -0.00806
95% paired CI     [-0.06105,+0.04045]
P(B3 > B1)         0.3808
```

B3 improved some target point estimates but degraded others. Target-specific B1/B3 winner selection was not permitted.

A fixed 50:50 B1+B3 rank ensemble reached `0.5048038179`, effectively neutral versus B1.

Decision: **rejected as a global replacement for B1**.

## Current successor context

The campaign subsequently progressed through direct weak supervision, all-series modeling and ImageNet initialization:

```text
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249
B15 gold  0.6209002783
```

B15 passed frozen weak-v2 teacher agreement decisively but did not improve expert-gold macro AUC. Current development prioritizes B6 report-state diagnosis rather than returning to target-specific architecture choices derived from reused gold.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).