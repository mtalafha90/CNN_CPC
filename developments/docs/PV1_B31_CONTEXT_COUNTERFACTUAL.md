# PV1 B31 local-context inference counterfactual

> **Frozen after the original PV1 result and after the B29 mechanistic addendum, before running this counterfactual.** This is a post-result global mechanism audit. It is not a new prospective model-selection experiment, not independent clinical validation, and not a basis for target-wise tuning.

## Motivation

The completed PV1 architecture-selection result remains:

```text
B31  macro weighted soft BCE  0.5743066
B33  macro weighted soft BCE  0.5849691
B20  macro weighted soft BCE  0.6155808
```

The later frozen-B29 addendum produced:

```text
B29  macro weighted soft BCE  0.5959239
```

with the global mechanistic ordering:

```text
B31 > B33 > B29 > B20
```

B29 was clearly better than B20, but B31 was also clearly better than B29. That establishes an association between the B31 local-context pathway and the stronger final result. It does not yet show whether the trained local-context operation itself is required at inference, because the branch could also alter the training/optimization trajectory while ending with only a very small direct perturbation of the attention distribution.

This audit therefore asks one narrower question:

> **If the already-trained B31 checkpoint is left completely unchanged except that `local_context.weight` is set to exact zero at inference, does its PV1 performance materially change?**

## Exact intervention

Normal B31 computes

```text
H = X + DWConv1d_k3(LN0(X))
w = softmax(q^T H / sqrt(D))
C = LN0(sum_i w_i X_i)
T = A + tanh(g) * (C - A)
```

where the values remain the original B20 slice tokens `X`; local context changes only the scores.

The counterfactual uses the same trained B31 checkpoint and performs exactly one in-memory intervention after loading:

```text
model.local_context.weight[:] = 0
```

No checkpoint is retrained or saved. The following remain identical to the normal PV1 B31 run:

```text
encoder weights
B31 complementary query
B31 complementary gate
study Transformer
pathology heads
metadata embeddings
B20 post-resize 90% crop
624 validation StudyInstanceUIDs and order
3544 validation MRI series
TTA [-1,0,1]
primary metric
secondary metric
```

## Governance

```text
original PV1 result already observed        yes
B29 addendum result already observed        yes
new training                                none
expert labels read                          no
independent clinical validation             no
prospective model selection                 no
primary purpose                              global mechanism audit
```

The original PV1 selection remains `B31 > B33 > B20`. This audit cannot demote or promote the active model by itself.

No target-wise context masking, pathology-specific context switches, target-specific blends, kernel-size retuning, B29.1, or B31.1 is allowed from the outcome.

## Frozen comparisons

The primary counterfactual comparison is:

```text
B31-context-zero - B31-normal
```

Two additional global comparisons are frozen for mechanism placement:

```text
B31-context-zero - B29
B31-context-zero - B33
```

All differences use:

```text
candidate macro weighted soft BCE - reference macro weighted soft BCE
```

so negative favors the candidate.

The paired uncertainty calculation uses 5,000 study-level bootstrap replicates, the same metric implementation used by PV1, and the same frozen 624-study weak-label validation surface.

## Predeclared interpretation of the primary comparison

Let the primary paired interval describe `B31-context-zero - B31-normal`.

```text
95% interval entirely above zero:
    the trained local-context operation directly improves the final B31 inference function.

95% interval includes zero:
    a direct inference contribution is unresolved; an optimization/training-path effect remains plausible.

95% interval entirely below zero:
    the trained local-context operation is harmful at inference even though the B31 training path produced the strongest PV1 model.
```

The B29 and B33 comparisons are secondary mechanism-placement diagnostics only. They must not be used to design target-wise variants.

## Memory policy

Only the trained B31 checkpoint is loaded. Normal B31, B29, and B33 reference predictions are read from already completed and fingerprint-validated PV1 artifacts. Evaluation retains the low-memory policy:

```text
batch_size                 1
num_workers                1
prefetch_factor            1
persistent_workers         false
series_cache_mb_per_worker 0
```

## Command

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.prospective_weak_v1_b31_context_counterfactual \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --split-manifest runs/prospective_weak_v1/split_manifest.json \
  --b6-root "$B6_ROOT" \
  --b31-checkpoint runs/prospective_weak_v1/b31/model.pt \
  --reference-eval-root runs/prospective_weak_v1/eval \
  --b29-addendum-eval-root runs/prospective_weak_v1/b29_addendum/eval \
  --out-root runs/prospective_weak_v1/b31_context_counterfactual \
  --n-bootstrap 5000
```

Expected artifacts:

```text
runs/prospective_weak_v1/b31_context_counterfactual/
├── b31_context_zero_predictions.csv
├── b31_context_zero_prediction_meta.json
├── paired_predictions.csv
└── comparison.json
```

## Decision after completion

This audit is the final planned mechanism check before defining B34. After the result is recorded, B34 should be formulated from the global mechanism conclusion rather than from per-target PV1 outcomes. Independent hidden competition or external expert validation remains necessary before replacing B20 as the active predictive model.
