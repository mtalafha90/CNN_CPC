# B30 — projected complementary attention

> **Status — 2026-08-16:** IMPLEMENTED / FROZEN BEFORE B30 OUTCOME / READY FOR SAFETY TESTS. **B20 remains the active working model. B29 remains a frozen promising candidate and is not modified.**

## Why B30 exists

B29 was the first post-B20 candidate to show a broad positive direction on the reused 58-study expert development surface:

```text
B20 macro AUC                 0.6674066371
B29 macro AUC                 0.6768879224
raw B29 - B20                +0.0094812853
paired 95% CI                [-0.0037494185, +0.0241875594]
P(B29 > B20)                  0.9188
```

That result is not independent validation and is not used to tune any target-specific behavior. B29 is frozen exactly.

B29's complementary summary, however, is structurally simpler than B20's historical series pool. B20 uses an 8-head `MultiheadAttention` block with learned Q/K/V and output projections, while B29 scores the same slice tokens with a direct single-vector dot product.

B30 asks one prospectively frozen question:

> Does a second learned series query become more useful when it operates through the same learned B20 attention coordinate system instead of a raw dot-product scorer?

## Single architectural intervention

For each real MRI series:

```text
A  = historical B20 learned attention-pooled token
C2 = complementary projected attention token
T  = A + tanh(g) * (C2 - A)
```

The complementary token uses one new learned 768-D query `q2`, but reuses the **current B20 series-pool Q/K/V projections, output projection and LayerNorm affine parameters as detached operators**.

```text
q2 ---------------------> detached B20 Q projection ----\
slice tokens -----------> detached B20 K projection ----- attention -> detached B20 V/out/LN -> C2
```

The reused B20 projection and norm parameters are detached **only in the complementary branch**. The historical B20 `A` branch is unchanged and continues to train exactly as in the frozen B20 recipe.

The complementary branch has no dropout.

### New trainable parameters

```text
complementary query q2    768
feature-wise gate g       768
                         ----
new parameters           1536
```

No second encoder, no target-specific routing, no new projection matrices, and no new normalization parameters are added.

## Zero-gate and RNG safety

The gate is initialized to exactly zero:

```text
tanh(g) = 0
T = A
```

Therefore B30 starts as the exact B20 function.

The complete historical B20 model is constructed before `q2` and `g`, preserving shared parameter initialization under the historical construction seed. The historical B20 series pool runs first. The complementary branch contains no dropout or random operation, so merely computing it at zero gate does not shift the downstream B20 RNG path.

Safety tests require B20/B30 equivalence in both evaluation mode and training mode with dropout enabled and an identical RNG reset.

## Detached shared-projection rule

B30 deliberately prevents the complementary path from directly changing B20's Q/K/V, output-projection or series-pool LayerNorm parameters.

Gradients through `C2` may flow into:

```text
q2
slice token inputs
upstream trainable non-encoder token/metadata components
```

They may not flow through the complementary branch into:

```text
B20 series-pool Q/K/V projection weights
B20 series-pool Q/K/V projection biases
B20 series-pool output projection
B20 series-pool LayerNorm affine parameters
```

Those historical parameters still receive their ordinary gradients through the unchanged `A` branch.

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
historical B20 series pool  unchanged
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

B30 is trained from the same historical B20/B16 initialization recipe. It is **not** a fine-tune of the B29 checkpoint. This keeps the comparison mechanistically interpretable.

## Prospective mechanism audit

Before any B30 expert outcome is inspected, the trainer records attention-complementarity diagnostics on the first **64 deterministic frozen-loader batches of each epoch**. The subset size and rule are fixed here before B30 performance inspection.

For the primary and complementary no-dropout attention distributions, B30 records:

```text
normalized primary attention entropy
normalized complementary attention entropy
normalized Jensen-Shannon divergence
top-1 slice agreement
top-3 slice overlap fraction
mean effective residual norm ||g(C2-A)|| / ||A||
maximum effective residual norm ratio
```

These are mechanism diagnostics only. No threshold is used to tune B30. They answer whether the second query is actually complementary or merely duplicating the historical pool.

The diagnostic primary attention is the deterministic no-dropout attention distribution implied by the current B20 query and current B20 projections. It is not claimed to be the exact stochastic dropout realization used by the training forward pass.

## Runtime

```text
hard budget       <= 8.25 h
internal reserve   >= 30 min
competition limit      9 h
```

B30 adds a second projected attention calculation over already-computed slice embeddings but no encoder pass. Runtime must be measured rather than assumed.

## Safety tests before full training

The B30 tests pin:

- exactly 1,536 new trainable parameters;
- exactly zero gate initialization;
- B20 functional equivalence at zero gate;
- B20 training-mode RNG-path equivalence at zero gate;
- complementary shared-projection detachment;
- finite normalized per-head attention weights;
- bounded tanh gate;
- staged gate-then-query gradient coupling;
- finite/bounded mechanism-audit metrics;
- bf16 finite behavior;
- empty-study finite behavior.

## Training outputs

```text
runs/b30_projected_complementary_series_pool/
├── b30_model.pt
├── training_audit.json
├── complementary_pool_state.json
├── attention_audit.json
└── history.json
```

## Evaluation governance

B30 trains on all 3,120 historical B20 weak-supervision studies, so the historical 623-study weak-v2 partition is not a holdout.

The reused 58-study expert surface remains development-only. The B30 architecture, endpoint, audit rule, gate, query, detached-projection policy and all controls are frozen before looking at B30 expert performance.

After a valid training/mechanism audit, the reused expert comparison may be run descriptively. It cannot independently promote B30. Do not derive B30.1, target-specific gates, target-specific queries, selective B29/B30 mixing, endpoint changes, or blend weights from that reused result.

Independent promotion still requires hidden competition evidence.
