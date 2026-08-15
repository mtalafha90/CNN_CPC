# B4 — frozen SSL + classical pathology classifiers

> **Status — 2026-08-12:** **COMPLETED / RETAINED HISTORICAL ABLATION.** B4 gold macro AUC was `0.5137567459`. B13 is now the reused-gold development champion at `0.6293565948`; B15 completed afterward without replacing B13.

## Scientific question

B4 froze the strong competition-only SSL ConvNeXt encoder and used the 58 gold labels only in low-capacity target-specific PCA + logistic-regression classifiers. It tested whether useful pathology signal was already present in the representation but obscured by high-variance end-to-end fine-tuning.

Verified gold feature cache:

```text
studies  58
features [58, 6, 2304]
streams  6
pooling  mean + std + max
encoder frozen true
```

## Nested probe

For each outer fold/target, B4 selected among a fixed grid on non-outer gold only:

```text
feature mode   all / fixed prior subset
PCA components 4 / 8 / 12 / 16
logistic C     0.1 / 1.0
```

Outer labels were not used for policy selection.

## Completed result

```text
B4 macro AUC       0.5137567459
95% CI            [0.4619827141,0.5642366629]
B1 macro AUC       0.5030284974
paired median      +0.0102107449
95% paired CI      [-0.0514266147,+0.0709432872]
P(B4 > B1)          0.6378
```

The paired interval was wide. B4 was retained as a useful frozen-representation ablation rather than a proven improvement.

B4.1-B4.3 tested shared/grouped/two-way policy selection and all scored lower, so the selector branch was frozen.

## Successor context

B5 changed the representation while reusing this probe and reached `0.5243650851`. Later direct weak-supervision/all-series/ImageNet experiments ultimately produced:

```text
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249
B15 gold  0.6209002783
```

B15 also raised frozen weak-v2 teacher agreement to `0.7319060415` but did not improve expert-gold macro AUC. This reinforces the decision not to return to increasingly fine target-wise classifier selection on the same 58 labels.

## Decision

B4 remains a historical representation-separability diagnostic. Do not reopen target-specific PCA/C/grid design or use B4/B13/B15 target winners from reused gold.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).