# B46 frozen gold-fold manifest record

**Status:** manifest created after the B46 code/unit-test gate and before any B46 training or OOF prediction.

The B46 unit tests passed before manifest construction:

```text
pytest -q developments/tests/test_b46_gold_crossfit.py
3 passed in 4.10s
```

The one generated manifest is:

```text
runs/079_Experiment_B46_gold_anchored_crossfit/
└── b46_gold_anchored_crossfit/
    └── gold_folds.json
```

Frozen SHA-256:

```text
054c4ce9ab808af714cd4b86f159ef02a2b7e67de0c80e5c930d29fa5fb22e03
```

This fingerprint is now the B46 fold identity. Do not recreate or replace the manifest after this record.

## Completed use and outcome

All five fixed-E2 fold checkpoints were trained against this exact manifest and
passed the pooled 58-study OOF leakage audit. The completed B46 result was
`0.678174` macro AUC versus the B42 parent at `0.683120`
(`B46 − B42 = −0.004946`; paired 95% CI `[−0.014664, +0.003402]`,
`P(B46 > B42) = 0.1296`). It is therefore **no support for gold anchoring at
the frozen 4.0 cell weight**. See
[`B46_GOLD_ANCHORED_CROSSFIT.md`](B46_GOLD_ANCHORED_CROSSFIT.md) for the full
predeclared decision and governance.

## Fold sizes

```text
fold 0  12 studies
fold 1  12 studies
fold 2  12 studies
fold 3  11 studies
fold 4  11 studies
```

## Global expert-label prevalence

| Target | Positive | Negative |
|---|---:|---:|
| ACL | 24 | 34 |
| MCL | 9 | 49 |
| Medial Meniscus | 26 | 32 |
| Lateral Meniscus | 23 | 35 |
| Medial OA | 15 | 43 |
| Lateral OA | 11 | 47 |
| PF OA | 21 | 37 |
| Effusion | 35 | 23 |
| Synovitis | 27 | 31 |
| Baker's | 12 | 46 |
| Contusion | 19 | 39 |
| Fracture | 18 | 40 |

## Held-out positive counts by fold

| Target | F0 | F1 | F2 | F3 | F4 |
|---|---:|---:|---:|---:|---:|
| ACL | 4 | 4 | 7 | 5 | 4 |
| MCL | 0 | 0 | 1 | 4 | 4 |
| Medial Meniscus | 3 | 5 | 6 | 6 | 6 |
| Lateral Meniscus | 3 | 1 | 4 | 7 | 8 |
| Medial OA | 2 | 0 | 3 | 4 | 6 |
| Lateral OA | 0 | 0 | 2 | 1 | 8 |
| PF OA | 3 | 4 | 3 | 4 | 7 |
| Effusion | 5 | 8 | 7 | 7 | 8 |
| Synovitis | 6 | 3 | 5 | 5 | 8 |
| Baker's | 0 | 0 | 1 | 5 | 6 |
| Contusion | 4 | 2 | 6 | 5 | 2 |
| Fracture | 6 | 0 | 5 | 3 | 4 |

## Interpretation of the realized split

The manifest satisfies the frozen structural contract: exactly 58 unique studies, fixed capacities `12/12/12/11/11`, deterministic assignment, complete 12-target labels and one held-out fold per study.

The realized multilabel balance is imperfect for several low-prevalence targets. In particular, folds 0 and 1 contain no positive MCL, Lateral OA or Baker's cases, and fold 1 contains no positive Fracture cases. Fold 4 contains 8 of the 11 Lateral OA positives. This is recorded before training and must not be hidden or retroactively repaired.

This does not make the primary 58-study pooled OOF macro AUC undefined: every target has both classes on the complete concatenated OOF surface. However, it increases fold-to-fold composition heterogeneity and therefore the variance of a five-model cross-fit comparison. Per-fold target AUC is not a valid required metric when a held-out fold contains one class only.

The protocol remains unchanged: no fold regeneration, target-specific reassignment or gold-weight change is allowed. Every fold checkpoint must record the manifest SHA above, and the final evaluator must verify that each OOF study was excluded from the gold gradients of the model that produced its prediction.
