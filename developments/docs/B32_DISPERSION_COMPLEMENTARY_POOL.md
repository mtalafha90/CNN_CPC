# B32 — complementary dispersion summary

> **Status — 2026-08-16:** IMPLEMENTED / FROZEN BEFORE B32 OUTCOME / READY FOR SAFETY TESTS. **B20 remains the active reference. B29 and B31 remain frozen development candidates. B30 is closed.**

## Motivation

B29 added a simple learned complementary softmax summary and improved the reused 58-study development macro from 0.667407 (B20) to 0.676888. B31 added a local-context scoring perturbation and reached 0.682280 on the same reused surface, but its prospective mechanism audit showed that the context-aware attention remained almost identical to raw B29 attention (E2 normalized JS divergence 3.56e-9; top-1 agreement 98.9%; top-3 overlap 99.63%).

This suggests that the useful B29/B31 complementary representation may behave more like a broad global series statistic than a sharply selective diagnostic-slice pool.

B32 therefore asks a new, orthogonal question:

> Does the through-series **dispersion** of B20 slice-token features provide complementary pathology information beyond B29's mean-like summary?

## Governance: B32 branches from B29, not B31

B31 was already inspected on the reused expert surface. To avoid carrying forward a local-context mechanism after seeing that outcome, B32 is a separate prospective branch from the frozen B29 architecture.

B32 does **not** include B31's depthwise Conv1d local-context module. It uses exactly B29's simple query weights for both first- and second-order statistics.

## Frozen architecture

For the 16 original B20 slice tokens `X_i`, B29 produces deterministic softmax weights:

```text
w_i = softmax(<X_i, q> / sqrt(768))
```

B32 keeps the B29 weighted mean:

```text
mu_raw = sum_i w_i X_i
C_mu   = LN0(mu_raw)
```

and adds a weighted feature standard deviation:

```text
sigma_raw = sqrt(sum_i w_i (X_i - mu_raw)^2 + 1e-6)
C_sigma   = LN0(sigma_raw)
```

The final series token is:

```text
T = A
  + tanh(g_mu)    * (C_mu - A)
  + tanh(g_sigma) * C_sigma
```

where:

- `A` is the unchanged historical B20 learned-attention series token;
- `q` is B29's learned 768-D query;
- `g_mu` is B29's existing zero-init 768-D feature gate;
- `g_sigma` is a new zero-init 768-D feature gate;
- `LN0` is parameter-free LayerNorm.

The same B29 weights are used for both `mu_raw` and `sigma_raw`.

## Parameter contract

```text
B29 query q                  768
B29 mean gate g_mu           768
B32 dispersion gate g_sigma  768
                              ---
total new parameters        2304
```

No projection matrix, local-context convolution, target-specific route, additional encoder pass, or trainable normalization is added.

## Zero-init safety

The complete B29 model is constructed first. The dispersion gate is then added and initialized to exact zero without consuming a random draw.

At initialization:

```text
g_mu    = 0
g_sigma = 0
```

therefore:

```text
T = A
```

and B32 is exactly the historical B20 function.

The extra B32 branch is deterministic and contains no dropout, so it must preserve the historical B20 training RNG path at zero gates.

If `g_sigma=0`, B32 reduces exactly to B29 for any fixed B29 state.

## Staged gradient behavior

At the first exact-zero-gate backward pass:

```text
mean-gate gradient        nonzero
dispersion-gate gradient  nonzero
query gradient            zero
```

After either gate moves, the shared B29 query can receive gradient through the weighted mean and weighted dispersion statistics. The trainer requires finite nonzero mean-gate, dispersion-gate and query gradients during every epoch.

## Frozen controls

```text
training studies            3120
eligible MRI series        17475
usable B6 cells            14123
positive / negative        6871 / 7252
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
variance epsilon            1e-6
expert labels in gradients  0
expert checkpoint selection none
```

B32 is trained from the same historical B20/B16 initialization recipe. It is not a fine-tune of B29 or B31 checkpoints.

## Prospective mechanism audit

Before any B32 expert outcome is inspected, the trainer records diagnostics on the first 64 frozen-loader batches of each epoch.

The audit records:

```text
B29 attention entropy
||weighted mean - uniform mean|| / ||uniform mean||
||weighted std - uniform std|| / ||uniform std||
||raw weighted std|| / ||raw weighted mean||
mean-residual norm / ||A||
dispersion-residual norm / ||A||
combined residual norm / ||A||
maximum combined residual norm / ||A||
cosine(mean residual, dispersion residual)
```

These are mechanism diagnostics only. No threshold selects a checkpoint or changes training.

The weighted-vs-uniform statistics are included because B29/B31 attention was observed to be nearly uniform. They quantify whether the learned query meaningfully changes the first- and second-order statistics relative to simple uniform aggregation.

## Runtime

```text
hard budget       <= 8.25 h
internal reserve   >= 30 min
competition limit      9 h
```

B32 adds only weighted moment arithmetic over already computed 16x768 slice-token sequences. No additional image-encoder pass is introduced. Runtime must still be measured.

## Outputs

```text
runs/b32_dispersion_complementary_pool/
├── b32_model.pt
├── training_audit.json
├── mean_complementary_pool_state.json
├── dispersion_gate_state.json
├── dispersion_audit.json
└── history.json
```

## Evaluation governance

The historical 623-study weak-v2 partition is not a holdout because B32 trains on all 3,120 historical B20 weak-supervision studies.

After a valid training/mechanism audit, the reused 58-study expert surface may be used only descriptively. The frozen evaluator compares:

```text
B20
B29
B31
B32
```

and reports paired B32-vs-B20, B32-vs-B29 and B32-vs-B31 bootstrap differences.

This reused surface cannot independently promote B32. Do not derive a B32.1 dispersion statistic, variance epsilon, gate formulation, endpoint, target-specific route, or blend from that outcome.
