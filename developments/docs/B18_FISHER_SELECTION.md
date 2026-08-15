# B18 — Fisher-style expert-guided epoch selection

> **Status — 2026-08-13:** COMPLETED. Package `0.28.0`. Epoch 2 selected by the predeclared global expert-selection rule. The selected expert score is a checkpoint-selection statistic only and is **not** independent validation evidence.

B18 is a checkpoint-selection experiment built directly on B17. It does **not**
change the MRI encoder, downstream architecture, B6 supervision, augmentation,
resolution, slice count, optimizer, or five-epoch training trajectory. Its only
intervention is to use the repeatedly reused 58-study expert set to select one
global epoch among five frozen-encoder B6-only training passes.

## Frozen B18 question

```text
Among five otherwise identical B17 training epochs, does selecting one epoch
using the global 12-target macro AUC on the repeatedly reused 58-study expert
set provide a better checkpoint for independent test transfer than always
keeping epoch 5?
```

## Completed result

The run completed all five predeclared training/selection cycles with exact
coverage and an unchanged frozen encoder.

```text
epoch    B6 loss       global expert-selection macro AUC
1        0.7371836930  0.6187157061
2        0.6336947483  0.6654496134   <- selected
3        0.6087776578  0.6511148368
4        0.5862506992  0.6394162186
5        0.5667051629  0.6425890153
```

Frozen selection rule:

```text
selected epoch                    2
selected selection statistic      0.6654496134
fixed epoch-5 endpoint            0.6425890153
selection-statistic difference    +0.0228605982
selection metric                  global 12-target macro ROC AUC
tie break                         earliest epoch
```

The B6 optimization loss decreased monotonically through epoch 5, while the
expert-selection statistic peaked at epoch 2. This is the behavior B18 was
specifically designed to test: the best checkpoint for the noisy/report-derived
B6 objective need not be the best checkpoint on the expert-selection surface.

Importantly, epoch 5 reproduces the B17 reused-gold point estimate
(`0.6425890153`) to numerical precision. This is consistent with B18 following
the intended B17 trajectory and changing only the checkpoint-selection rule.

## Integrity audit

Every epoch completed exactly:

```text
batches                       1560 / 1560
studies                       3120 / 3120
active B6 cells              14123 / 14123
positive cells                6871 / 6871
negative cells                7252 / 7252
series                       17475 / 17475
encoder LR                    0
encoder frozen                true
encoder training mode         false
encoder gradients             false
full coverage                 true
full series coverage          true
budget limited                false
```

The encoder SHA-256 remained unchanged throughout:

```text
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

## Unchanged B17 gradient surface

```text
training studies                    3120
usable B6 cells                    14123
positive cells                      6871
negative cells                      7252
eligible real MRI series           17475
batches / epoch                     1560
batch size                             2
```

B6 targets remain:

```text
positive -> target 0.85, weight 0.50
negated  -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
minimum confidence -> 0.75
```

No additional generic label smoothing is applied because B6 positive/negative
targets are already soft (`0.85/0.05`).

## Unchanged model

```text
initializer                 completed B16 report-aligned encoder
encoder                     fully frozen
encoder LR                  0
encoder optimizer member    false
encoder training mode       eval
resolution                  224 x 224
positions / series          16
series aggregation          learned one-token-per-series
series-pool heads           8
study Transformer           2 layers / 8 heads
pathology-query layers      1
dropout                     0.25
head LR                     1e-4
minimum LR                  1e-6
weight decay                1e-4
grad clip                   1.0
candidate epochs            5
```

## Expert selection surface

```text
expert studies               58
expert target cells          696
expert MRI series            336
TTA                           [-1,0,1]
selection metric              global 12-target macro ROC AUC
candidate epochs              1,2,3,4,5
tie break                     earliest epoch
expert labels in gradients    no
```

Only the single global macro AUC was used for selection. Per-target epoch
selection, target-specific model mixing, bootstrap-based selection, and
post-result tuning were forbidden by the predeclared protocol.

## Critical interpretation

Because the 58 studies selected the checkpoint:

```text
DO NOT report 0.6654496134 as independent validation performance.
DO NOT claim B18 improved B17 by +0.0228605982 on validation.
DO NOT select a different epoch per target.
DO NOT tune smoothing, LR, architecture, resolution or TTA from this curve.
DO NOT bootstrap the five candidate scores to manufacture significance.
```

The correct statement is:

> Expert-guided checkpoint selection chose epoch 2 (selection macro AUC
> `0.6654496134`) instead of the fixed epoch-5 endpoint. Whether that selected
> checkpoint improves generalization must be established on a genuinely
> independent competition evaluation surface.

## Outputs

```text
runs/b18_fisher_selection/
  policy.json
  supervision_plan.json
  history.json
  selection_history.json
  selection.json
  candidates/
    epoch_1.pt
    epoch_2.pt
    epoch_3.pt
    epoch_4.pt
    epoch_5.pt
  b18_model.pt                  # selected epoch-2 checkpoint
```

Selected checkpoint:

```text
runs/b18_fisher_selection/b18_model.pt
```

## Local inference smoke test

The selected epoch-2 checkpoint passed the local three-study inference/schema
smoke test on 2026-08-13:

```text
test rows                       3
test series                    15
series / study                  5 / 5 / 5
TTA                             [-1,0,1]
sample columns match            true
sample UID order match          true
metadata repairs required       0
```

Command:

```bash
rsna-knee-b18-submit \
  --config configs/b18_fisher_selection.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b18_fisher_selection/b18_model.pt \
  --out runs/b18_fisher_selection/submission_smoke.csv
```

The local three-study output is a **schema/inference smoke test only**. The
submission manifest now uses the neutral experiment label
`B18_submission_inference` so merely running local inference cannot be mistaken
for hidden-test evaluation.

## Governance after B18

```text
B16/B17: closed to post-gold retuning
B18: completed; epoch 2 frozen as selected checkpoint
B18: expert labels never entered gradients
B18: selected expert score is not validation evidence
B18: no target-specific epoch choice or target mixing
B18: no smoothing/robust-loss/LR/architecture/resolution/TTA retuning from selection curve
next performance signal: genuinely independent competition evaluation
```
