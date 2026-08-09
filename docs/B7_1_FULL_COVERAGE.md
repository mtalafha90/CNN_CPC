# B7.1 — full-corpus weak-supervision coverage

> **Status — 2026-08-10:** **PREDECLARED / TRAINING PENDING.**

## Motivation

B7-v1 produced the highest standalone development point estimate so far:

```text
B7-v1 macro AUC = 0.5397724412
95% CI           = [0.4733481702, 0.6035621405]
```

versus the previous B5 baseline:

```text
B5 macro AUC     = 0.5243650851
exact point delta = +0.0154073561
```

The paired 5,000-replicate study bootstrap comparing B5 (A) with B7-v1 (B) gave:

```text
median(B7 - B5) = +0.0155102430
95% paired CI   = [-0.0607472600, +0.0889531461]
P(B7 > B5)      = 0.6678
valid replicates = 5000/5000
```

Interpretation: B7-v1 has the best standalone point estimate, but the paired evidence is still statistically inconclusive on the 58-study development set.

Before the B7-v1 gold result was inspected, its supervision audit had already exposed a coverage limitation: B7-v1 trained on 3,120 active weakly labelled studies but capped each epoch at 500 batches with batch size 2. That is only 1,000 study draws per epoch and 4,000 draws over four epochs, or about 1.28 nominal corpus passes.

B7.1 tests that pre-identified limitation directly.

## Single scientific change

B7.1 changes only:

```text
b7_max_batches_per_epoch: 500 -> 1560
```

With 3,120 active studies and batch size 2, 1,560 batches correspond to one full shuffled pass through the active weak-training pool per epoch. Four epochs therefore provide four nominal full corpus passes.

Everything else remains fixed from B7-v1:

- B5 encoder initialization;
- frozen B6 v1.2.1 labels;
- no gold labels in gradient or early stopping;
- positive soft target `0.85`, weight `0.50`;
- negative soft target `0.05`, weight `1.00`;
- uncertain and unmentioned cells ignored;
- same target-balance multipliers computed from the B6 training pool;
- same six-stream 2.5D ConvNeXt + cross-sequence Transformer + pathology-query architecture;
- four epochs;
- encoder LR `1e-5`;
- head LR `1e-4`;
- cosine LR schedule;
- same MRI augmentations;
- same three-view gold evaluation `[-1,0,1]`;
- 5,000 bootstrap replicates.

This is intentionally a coverage experiment, not a hyperparameter search.

## Configuration

```text
configs/b7_1_full_coverage.yaml
```

The config records:

```text
b7_experiment_name: B7.1_full_coverage
```

The internal implementation variant remains `b7_b5_init_b6_asymmetric_weak_v1` because the model and weak-supervision code are unchanged; the experiment identity is carried by the config and output directory.

## Train

```bash
rsna-knee-b7 \
  --config configs/b7_1_full_coverage.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b7_1_full_coverage
```

Expected training pool from B7-v1 audit:

```text
active studies = 3120
usable cells   = 14123
batch size     = 2
batches/epoch  = 1560
study draws/epoch = 3120
nominal corpus passes over 4 epochs = 4.0
```

Do not alter epochs, learning rates, weak-label weights, target multipliers, architecture, or augmentation after the B7-v1 result and still call the run B7.1.

## Evaluation

Use a runtime-only workers=0 evaluation config if desired to avoid DataLoader teardown noise; this does not alter the scientific experiment.

```bash
rsna-knee-b7-eval \
  --config /tmp/b7_1_eval.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b7_1_full_coverage/b7_model.pt \
  --out-root runs/b7_1_full_coverage/gold_eval
```

Then compare B7-v1 and B7.1 with the same paired bootstrap machinery. Because the same 58 gold studies have already informed B6 and model-development decisions, B7.1 results are development estimates rather than independent validation.
