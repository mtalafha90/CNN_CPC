# B7 — B5-initialized MRI model trained on frozen B6 weak labels

> **Status — 2026-08-12:** **COMPLETED / RETAINED COVERAGE ABLATION.** B7-v1 gold macro AUC was `0.5397724412`; B7.1 improved it to `0.5644802945` by full-corpus coverage. B13 is now the reused-gold champion, and B15 has clarified that better B6-teacher agreement alone is not enough for expert-gold improvement.

## B7-v1 result

Training completed four predefined epochs with 500 batches/epoch and no runtime-budget limiting:

```text
epoch 1 loss 0.8393994069
epoch 2 loss 0.7084098365
epoch 3 loss 0.6809013321
epoch 4 loss 0.6433874173
```

The run made 4,000 study draws total, about `1.28` nominal passes over the 3,120 active weakly labelled studies.

Frozen supervision scope:

```text
report-only studies             4349
active weakly labelled studies  3120
inactive zero-usable studies    1229
usable target cells            14123
positive cells                  6871
negative cells                  7252
```

Reused-gold evaluation:

```text
macro AUC = 0.5397724412
95% CI   = [0.4733481702,0.6035621405]
```

Paired B5 -> B7-v1:

```text
median difference = +0.0155102430
95% paired CI     = [-0.0607472600,+0.0889531461]
P(B7-v1 > B5)     = 0.6678
```

## Architecture

```text
6 MRI streams
-> 2.5D ConvNeXt slice encoder initialized from B5
-> slice-position + stream embeddings
-> cross-sequence Transformer
-> 12 pathology queries
-> cross-attention to MRI memory
-> 12 logits
```

Gold studies were excluded from the B7 training loss and were not used for early stopping.

## Frozen asymmetric B6 supervision

The B6 audit showed high sensitivity/NPV but lower positive precision. B7 therefore used one global policy across all targets:

| B6 state | Soft target | Base weight |
|---|---:|---:|
| positive | `0.85` | `0.50` |
| negated | `0.05` | `1.00` |
| uncertain | ignored | `0.00` |
| unmentioned | ignored | `0.00` |

Only B6 cells with confidence `>=0.75` were used. Target balancing normalized total base supervision mass across targets.

## B7.1 full-coverage successor

B7.1 changed only:

```text
500 -> 1560 batches/epoch
```

and reached:

```text
macro AUC = 0.5644802945
```

Paired B7-v1 -> B7.1:

```text
median difference = +0.0241102714
95% paired CI     = [-0.0140197876,+0.0660558004]
P(B7.1 > B7-v1)   = 0.8694
```

Thus B7-v1 remains the direct weak-supervision/coverage ablation.

## Successor context through B15

Later experiments eventually produced:

```text
B12 gold  0.5660915179
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249
B15 gold  0.6209002783
```

The most important new supervision result is B15's frozen weak-v2 comparison:

```text
B13-v2 control  0.5652498118
B15            0.7319060415
paired median  +0.1675245839
95% CI         [+0.1124433208,+0.2165156305]
```

Despite that very large increase in B6-teacher agreement, B15 did not improve global expert-gold AUC over B13. This indicates that the current weak-label surface and expert target surface are not interchangeable.

## Current interpretation

B7 established that direct report-derived target supervision is useful, and B7.1 established that complete weak-corpus coverage matters. B15 now shows the limit of interpreting weak-target fit as a proxy for expert-label performance.

The immediate next step is therefore a B6 state audit on the already-reused gold set:

```text
positive
negated
uncertain
unmentioned
```

Do not retune B7/B6 target weights, convert unmentioned states to negative by assumption, or select target-wise B7/B13/B15 winners from reused gold.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). Validation governance: [`VALIDATION.md`](VALIDATION.md).