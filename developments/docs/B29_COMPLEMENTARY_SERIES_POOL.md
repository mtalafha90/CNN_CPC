# B29 — zero-gated complementary learned series summary

> **Status — 2026-08-16:** IMPLEMENTED / FROZEN BEFORE B29 OUTCOME / READY FOR SAFETY TESTS. **B20 remains the active working model.** B27/B27.1 and B28 are closed and not promoted.

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

B29 is defined after the B28 result, so it is not claimed to be independent of the experiment sequence. However, it is frozen before any B29 performance outcome and does not use B28 target-wise gains/losses to choose target-specific behavior.

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

The complementary branch contains **no dropout or random operation**. The historical B20 attention pool is executed first and unchanged. Consequently, when the gate is zero, merely computing the complementary branch does not consume RNG and does not shift downstream B20 dropout masks. The safety tests check equivalence in both evaluation mode and training mode with dropout enabled and an identical RNG reset.

The new query is constructed after every historical B20 parameter so the historical construction seed preserves all shared random initialization.

## Staged learning behavior

At exact zero gate, the first backward pass behaves as intended:

```text
gate gradient   nonzero
query gradient  zero
```

After the gate moves away from zero, the complementary query becomes coupled to the loss and must receive a finite nonzero gradient. The trainer refuses the run if that coupling never occurs during an epoch.

This staged activation prevents a randomly initialized second summary from perturbing B20 before the model has learned that any complementary contribution is useful.

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

## Runtime

```text
hard budget       <= 8.25 h
internal reserve   >= 30 min
competition limit      9 h
```

B29 adds only one 768-dimensional scoring query and one 768-dimensional gate over already-computed slice embeddings. The encoder workload and number of study tokens remain unchanged, so B29 should stay in the same roughly one-hour runtime class as recent B20-family fixed-E2 runs on the RTX A4500 Laptop GPU. Actual runtime must still be measured.

## Safety tests before full training

The B29 tests pin:

- exactly 1,536 new parameters;
- exactly zero gate initialization;
- B20 functional equivalence at zero gate;
- B20 **training-mode RNG-path equivalence** at zero gate with dropout enabled;
- finite normalized complementary softmax weights;
- bounded tanh gate;
- expected staged gradient behavior: gate first, then query;
- bf16 finite behavior;
- empty-study finite behavior.

## Training outputs

```text
runs/b29_complementary_series_pool/
├── b29_model.pt
├── training_audit.json
├── history.json
└── complementary_pool_state.json
```

The state audit records query norm, query cosine similarity to the historical primary pooling query, gate magnitude, exact surface coverage, frozen encoder SHA, and runtime.

## Evaluation governance

B29 trains on all 3,120 historical B20 weak-supervision studies, so the historical 623-study weak-v2 partition is not a holdout.

The 58 expert studies are heavily reused development data and were historically used to select B20 checkpoints. A B20-vs-B29 paired comparison on those studies is therefore descriptive/post-hoc only. It cannot independently promote B29.

No target-specific gate, target-specific query, endpoint change, selective ensemble, or architecture retuning may be derived from that reused expert result. If B29 shows a convincing broad improvement there, the next independent performance signal should come from hidden competition evaluation rather than another target-wise retune on the 58 studies.
