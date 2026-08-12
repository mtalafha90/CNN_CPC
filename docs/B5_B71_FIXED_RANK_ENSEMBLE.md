# Fixed B5 + B7.1 rank ensemble

> **Status — 2026-08-12:** **COMPLETED / REJECTED.** This fixed global ensemble remains closed. B13 is now the reused-gold development champion, and no target-specific B13/B14/B15 mixture is permitted from reused gold.

## Purpose

B5 and B7.1 are materially different models: B5 is a report-aligned frozen representation with classical target probes; B7.1 is an end-to-end pathology-query MRI model trained from B6 weak labels.

A fixed rank-space ensemble avoided probability-calibration and blend-weight tuning on the 58-study development set.

## Frozen rule

For each target independently:

```text
percentile-rank B5 predictions
percentile-rank B7.1 predictions
output = 0.5 * B5 rank + 0.5 * B7.1 rank
```

No target-specific model selection, raw-vs-rank search, blend-weight search or calibration fit was allowed.

## Completed result

```text
ensemble macro AUC      0.5540141184
95% CI                 [0.4959089903,0.6097647325]
B7.1 macro AUC          0.5644802945
point difference       -0.0104661761
paired median          -0.0105429030
95% paired CI          [-0.0523218181,+0.0333886570]
P(ensemble > B7.1)      0.3054
```

Decision: reject the fixed ensemble as campaign leader and do not search alternative weights on the same gold surface.

## Successor context through B15

```text
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249
B15 gold  0.6209002783
```

B15 passed frozen weak-v2 teacher agreement strongly but did not improve expert-gold macro AUC. This does not reopen ensemble-weight tuning. Target-wise B13/B14/B15 mixtures would be especially vulnerable to reused-gold overfitting.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). Validation rules: [`VALIDATION.md`](VALIDATION.md).