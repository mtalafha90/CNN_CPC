# PV1 B29 mechanistic addendum

> **Frozen after the original B20/B31/B33 PV1 result and before running B29 on PV1.** This is a post-result global mechanism-decomposition addendum. It is not a fourth original prospective PV1 control, not independent clinical validation, and not a basis for target-wise tuning.

## Why this addendum exists

The completed original PV1 comparison ranked the three predeclared controls by the frozen primary metric:

```text
B31  macro weighted soft BCE  0.5743066
B33  macro weighted soft BCE  0.5849691
B20  macro weighted soft BCE  0.6155808
```

with lower better. The paired intervals excluded zero for B31-vs-B20, B33-vs-B20, and B33-vs-B31. B31 is therefore the PV1-selected downstream development architecture.

However, B31 and B33 have almost identical secondary weak-state macro AUC (`0.7567309` versus `0.7565223`) even though B31 is clearly better on the predeclared soft-label loss. The already-frozen B29 architecture provides the cleanest missing mechanism comparator because it predates PV1 and lies structurally between B33 and B31:

```text
B33 = exact uniform complementary mean, no learned query, no local context
B29 = learned complementary query, no local context
B31 = learned complementary query + local-context score perturbation
```

The addendum therefore asks one global question: which part of the B31 pathway explains the PV1 loss advantage relative to the simple B33 mean?

## Governance boundary

This protocol is intentionally separate from `prospective_weak_v1_eval.py`.

The original B20/B31/B33 PV1 result was already observed before this B29 run was requested. Therefore:

```text
original PV1 prospective selection result    remains B31 > B33 > B20
B29 architecture frozen before PV1           yes
B29 PV1 matched retraining prospective?       no; defined after original result
B29 addendum role                             global mechanism decomposition only
expert labels read                            no
independent clinical validation               no
target-wise adaptation allowed                no
B29.1/B31.1 from target outcomes              no
blend/model switching from target outcomes    no
```

B20 remains the active historical model until independent hidden or external expert evidence supports replacement.

## Frozen B29 training contract

B29 is retrained on exactly the same PV1 training partition used by B20/B31/B33:

```text
training studies                 2496
validation studies                624  never loaded during training
training series                 13931
training weak cells             11303
positive / negated              5559 / 5744
encoder                          exact frozen B16 encoder
encoder SHA256                   b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
crop                             B20 post-resize 90% center crop
series policy                    frozen B12/B13 all-series policy
fixed endpoint                   E2
scheduler horizon                5
construction seed                19002026
loader seed                      19102026
post-construction training seed  19202026
validation checkpoint selection  none
expert labels                    none
```

The post-construction RNG reset is required so B29's extra query construction does not shift the subsequent dropout/random-training path relative to the original PV1 controls.

## Frozen B29 architecture

For the sixteen B20 slice tokens `X_i`, B29 retains the historical B20 attention-pooled token `A` and forms

```text
w_i = softmax(q^T X_i / sqrt(768))
C   = LN0(sum_i w_i X_i)
T   = A + tanh(g) * (C - A)
```

where `q` is a learned 768-vector and `g` is a zero-initialized 768-vector feature-wise gate. The addendum uses the exact existing B29 implementation with 1,536 new parameters. No architecture edits are permitted before the addendum result.

## Frozen evaluation contract

Only B29 is newly inferred. The evaluator reuses the persisted original PV1 B20/B31/B33 prediction CSV files after verifying:

```text
same split SHA256
same 624 StudyInstanceUIDs and order
same encoder SHA256
same PV1 evaluation version 1.1.0
same original primary ranking B31 > B33 > B20
recomputed reference metrics match original comparison.json
```

B29 is evaluated with the same:

```text
624 studies
3544 validation series
TTA [-1,0,1]
batch size 1
one worker
prefetch factor 1
no persistent worker
zero raw-series worker cache
primary metric = macro per-target B6-weighted soft-label BCE
secondary metric = macro weak-state ROC AUC
5000 study-level bootstrap replicates
```

## Predeclared global comparisons

Exactly three paired primary-loss comparisons are allowed:

```text
1. B29 - B20
   tests the frozen learned complementary-summary effect relative to B20.

2. B29 - B33
   isolates learned complementary query versus exact uniform mean,
   with neither model using local-context scoring.

3. B31 - B29
   isolates the incremental B31 local-context score perturbation
   conditional on the learned complementary query architecture.
```

Difference definition is always:

```text
candidate macro weighted soft BCE - reference macro weighted soft BCE
```

so negative is better for the candidate.

No new hypothesis may be selected from per-target B29 outcomes. The global result may motivate B34 only after this addendum is complete, and B34 must be defined without returning to target-wise PV1 winners.

## Commands

Training:

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.prospective_weak_v1_b29_training \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --split-manifest runs/prospective_weak_v1/split_manifest.json \
  --b6-root "$B6_ROOT" \
  --series-policy "$SERIES_POLICY" \
  --report-ssl-checkpoint "$B16_ENCODER" \
  --out-root runs/prospective_weak_v1/b29_addendum/train
```

Evaluation:

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.prospective_weak_v1_b29_eval \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --split-manifest runs/prospective_weak_v1/split_manifest.json \
  --b6-root "$B6_ROOT" \
  --b29-checkpoint runs/prospective_weak_v1/b29_addendum/train/model.pt \
  --reference-eval-root runs/prospective_weak_v1/eval \
  --out-root runs/prospective_weak_v1/b29_addendum/eval \
  --n-bootstrap 5000
```

Expected artifacts:

```text
runs/prospective_weak_v1/b29_addendum/
├── train/
│   ├── model.pt
│   ├── training_audit.json
│   └── history.json
└── eval/
    ├── b29_predictions.csv
    ├── b29_prediction_meta.json
    ├── paired_predictions.csv
    └── comparison.json
```

## Decision rule after completion

The addendum does not change the original fact that B31 won the predeclared PV1 selection. Its purpose is interpretation:

```text
If B29 approximately matches B31 and beats B33:
    learned complementary query explains most of the B31-vs-B33 advantage;
    local context adds little.

If B29 approximately matches B33 and B31 beats B29:
    B31 local-context scoring explains most of the B31-vs-B33 advantage.

If B29 lies clearly between B33 and B31:
    both the learned query and local context contribute.

If B29 behaves unexpectedly outside this ordering:
    report the result as-is and do not tune a B29.1/B31.1 from the same PV1 surface.
```

Regardless of the outcome, independent hidden competition or new external expert validation remains necessary before replacing B20 as the active predictive model.
