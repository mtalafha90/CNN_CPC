# B26.2 fixed-E2 training result

> **Status — 2026-08-16:** TRAINING COMPLETE / EXACT SURFACE VERIFIED / PERFORMANCE NOT YET EVALUATED. B20 remains the active working model pending development evaluation.

## Training command/result

B26.2 was trained on the exact historical B20 gradient surface with only the 171 quality-approved Synovitis fill cells added.

```text
device                      NVIDIA RTX A4500 Laptop GPU
precision                   bf16
workers                     6
training studies            3120
series instances            17475
fixed endpoint              E2
runtime                     58m08.697s
```

Loss trajectory:

```text
E1 loss   0.7543412205   28.2 min
E2 loss   0.6484235386   29.7 min
```

Training loss is an optimization diagnostic only and is not interpreted as predictive performance.

## Supervision contract

```text
B6 usable cells             14123
B26.2 accepted additions      171
  positive                     76
  negated                      95
final usable cells           14294
final positive cells          6947
final negative cells          7347

Synovitis final
  positive                    475
  negative                    112

B6 cells dropped                0
B6 cells overridden             0
```

## Full-coverage verification

Both epochs completed the identical intended surface:

```text
batches per epoch            1560
studies per epoch            3120
series per epoch            17475
supervision cells           14294
positive cells               6947
negative cells               7347
full_coverage                true
```

The frozen encoder fingerprint after training is:

```text
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

This matches the historical B16 report-aligned encoder used by B20.

Canonical local checkpoint:

```text
runs/b26_2_training/b26_2_model.pt
```

## Evaluation consequence

The completed B26.2 model used all 3,120 historical B20-gradient studies. Therefore the 623-study weak-v2 set used by the earlier B25X protocol is **not a holdout for this checkpoint**. B25X achieved leakage safety by training on the complementary 2,497 studies and keeping those 623 studies out of gradients.

Accordingly, B26.2 must **not** be scored on weak-v2 as though it were an independent or leakage-safe holdout.

The next available labelled diagnostic is the reused 58-study expert surface. This is explicitly post-hoc development evidence, not independent validation, and the historical B20 control was itself selected on that same surface. A paired evaluator is provided in:

```text
developments/src/rsna_knee/b26_2_gold_eval.py
```

No automatic promotion is allowed from the reused-gold result alone. Hidden competition evaluation remains the independent predictive-performance signal.
