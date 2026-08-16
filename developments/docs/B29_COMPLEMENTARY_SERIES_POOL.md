# B29 — zero-gated complementary learned series summary

> **Status — 2026-08-16:** SAFETY TESTS PASSED / FULL FIXED-E2 TRAINING COMPLETED / REUSED-EXPERT DIAGNOSTIC POSITIVE / **FROZEN PROMISING CANDIDATE, NOT PROMOTED**. **B20 remains the active working model.** See `B29_REUSED_GOLD_RESULT.md` for the frozen result and hidden-evaluation protocol.

## Why B29 exists

B20 compresses each real MRI series as:

```text
16 sampled 2.5D slice tokens
  -> frozen ConvNeXt encoder
  -> one learned attention-pooled series token
  -> study Transformer
  -> pathology queries
```

B28 tested whether an element-wise max summary could recover sparse evidence that the single learned series token might miss. The B28 training run was valid, but on the reused 58-study expert development surface its macro AUC was 0.638346 versus B20 0.667407 (delta -0.029061; paired P[B28>B20]=0.0586). B28 is therefore not promoted and its max-evidence residual formulation is closed.

B29 asks a different representation question:

> Can one learned slice summary per series be an information bottleneck even when an aggressive element-wise maximum is not a good universal companion?

B29 is defined after the B28 result, so it is not claimed to be independent of the experiment sequence. However, it was frozen before any B29 performance outcome and does not use B28 target-wise gains/losses to choose target-specific behavior.

## Single architectural intervention

For each real series, define:

```text
A = historical B20 learned attention-pooled token
C = second learned softmax-weighted summary of the same B20 slice tokens
```

The B29 token is

```text
series_token = A + tanh(g) * (C - A)
```

where `g` is a feature-wise gate.

The complementary summary `C` is deliberately small:

```text
score_s = <x_s, q> / sqrt(D)
weight_s = softmax(score_s)
C = parameter_free_LayerNorm(sum_s weight_s * x_s)
```

with one new learned query vector `q`.

### New parameters

The frozen ConvNeXt-Tiny representation dimension is 768:

```text
complementary query q   768
feature-wise gate g     768
                       ----
new parameters         1536
```

No second MHA block, projection matrix, target-specific route, or image-encoder pass is added.

## Zero-gate safety and RNG preservation

The gate is initialised to exactly zero. Therefore at initialization:

```text
tanh(g) = 0
series_token = A
```

and B29 is functionally identical to B20.

The complementary branch contains **no dropout or random operation**. The historical B20 attention pool is executed first and unchanged. Consequently, when the gate is zero, merely computing the complementary branch does not consume RNG and does not shift downstream B20 dropout masks. Safety tests passed for both evaluation-mode and training-mode B20 equivalence with an identical RNG reset.

The new query is constructed after every historical B20 parameter so the historical construction seed preserves all shared random initialization.

## Staged learning behavior

At exact zero gate, the first backward pass behaves as intended:

```text
gate gradient   nonzero
query gradient  zero
```

After the gate moves away from zero, the complementary query becomes coupled to the loss and receives a finite nonzero gradient. The completed fixed-E2 run recorded both gate and query coupling in each epoch.

## Frozen controls

```text
training studies            3120
usable B6 cells            14123
positive / negative        6871 / 7252
eligible MRI series        17475
B16 report-aligned encoder  frozen
encoder SHA                 b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
B20 crop                    90% post-resize crop only
slice sampling              16 positions / series
B20 learned series pool     unchanged
study Transformer           unchanged
pathology-query heads       unchanged
optimizer / LR              unchanged
augmentation                unchanged
loader seed                 unchanged
scheduler horizon           5 epochs
training endpoint           fixed E2
expert labels in gradients  0
expert checkpoint selection none
```

## Completed runtime

```text
hard budget          <= 8.25 h
internal reserve      >= 30 min
observed training     ~61.6 min
competition limit          9 h
```

## Training outputs

```text
runs/b29_complementary_series_pool/
├── b29_model.pt
├── training_audit.json
├── history.json
└── complementary_pool_state.json
```

The final effective feature-wise gate remained small and unsaturated (max absolute value ~0.0225, mean absolute value ~0.00516).

## Reused expert result

On the 58-study reused expert development surface:

```text
B20 macro AUC              0.6674066371
B29 macro AUC              0.6768879224
raw delta                 +0.0094812853
paired median delta       +0.0094213679
paired 95% CI             [-0.0037494185, +0.0241875594]
P(B29 > B20)               0.9188
```

This is encouraging but **not independent validation**. B20 itself was historically selected using these expert studies. B29 was fixed before its outcome was inspected, but this surface remains reused development evidence.

See:

```text
developments/docs/B29_REUSED_GOLD_RESULT.md
```

for the full target-wise result, freeze decision, and hidden competition protocol.

## Evaluation governance

B29 trains on all 3,120 historical B20 weak-supervision studies, so the historical 623-study weak-v2 partition is not a holdout.

No target-specific gate, target-specific query, endpoint change, selective ensemble, probability blend, or architecture retuning may be derived from the reused expert result.

The next independent performance signal is the hidden competition comparison of:

```text
Submission A = canonical B20
Submission B = exact frozen B29 fixed-E2 checkpoint
```

No blending is used in this first comparison. B20 remains the active model until that independent signal justifies promotion.
