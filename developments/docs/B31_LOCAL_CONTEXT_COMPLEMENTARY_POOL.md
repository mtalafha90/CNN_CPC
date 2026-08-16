# B31 — local-context complementary pooling

> **Status — 2026-08-16:** IMPLEMENTED / FROZEN BEFORE B31 OUTCOME / READY FOR SAFETY TESTS. **B20 remains the active reference. B29 remains the frozen promising candidate. B30 is closed and not promoted.**

## Motivation

B29's complementary branch uses a simple learned dot-product query over each of the 16 B20 slice tokens independently, then forms a weighted sum of the original slice tokens. B29 showed an encouraging reused-expert direction but is frozen and must not be altered.

B30 tested a more sophisticated projected complementary attention mechanism. B30 completed a valid fixed-E2 run but declined on the reused 58-study expert development surface (macro 0.654703 vs B20 0.667407; delta -0.012703; P[B30>B20]=0.1422). Its projected-attention formulation is closed.

B31 asks an orthogonal question:

> Can B29's simple complementary scorer improve if each candidate slice is scored with immediate through-plane neighbour context, while the value representation remains the original B20 slice token?

B31 does **not** use B30 target-wise outcomes, B29 target-wise gains, or expert labels to choose a pathology-specific mechanism.

## Frozen architecture

Let `X` be the 16 original B20 slice tokens for one real MRI series. Define parameter-free per-slice normalization and a depthwise local context residual:

```text
H = X + DWConv1d_k3(LN0(X))
```

where the Conv1d:

```text
in_channels   768
out_channels  768
kernel_size   3
padding       1
groups        768   # depthwise
bias          false
initialization exactly zero
```

The B29 query then scores the contextualized tokens:

```text
score_i = <H_i, q> / sqrt(768)
w       = softmax(score)
```

but the value sum deliberately remains the **original B20 tokens**:

```text
C = LN0(sum_i w_i X_i)
```

The final series token remains:

```text
T = A + tanh(g) * (C - A)
```

where `A` is the unchanged historical B20 learned-attention series token.

Thus local context affects **attention scores only**, not the values being pooled.

## Parameter contract

B31 constructs the complete B29 model first, then adds the depthwise convolution. Therefore all B20 parameters and B29's query/gate preserve the historical construction random draws.

```text
B29 complementary query q       768
B29 feature-wise gate g         768
B31 depthwise Conv1d           2304
                               ----
total new parameters vs B20    3840
```

The depthwise Conv1d is initialized to exact zeros after construction.

At initialization:

```text
DWConv = 0  ->  H = X
```

so the complementary branch is exactly B29. The outer B29 gate is also exactly zero:

```text
g = 0  ->  T = A
```

so the complete B31 network starts as the exact B20 function.

## Staged gradient behavior

At the first exact-zero-gate backward pass:

```text
gate gradient       nonzero
query gradient      zero
context gradient    zero
```

After the gate moves, both the B29 query and local-context Conv1d must become coupled to the loss. The trainer rejects an epoch if it never observes nonzero finite gradients for any of these three components.

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
expert labels in gradients  0
expert checkpoint selection none
```

B31 is trained from the historical B20/B16 initialization recipe. It is **not** a fine-tune of the B29 checkpoint.

## Prospective mechanism audit

Before any B31 expert outcome is inspected, the trainer records diagnostics on the first 64 frozen-loader batches of each epoch. No audit threshold changes training or selects a checkpoint.

The audit compares the raw B29 attention weights against the B31 context-aware weights using the same learned query:

```text
raw B29 attention entropy
context attention entropy
normalized Jensen-Shannon divergence
raw-vs-context top-1 slice agreement
raw-vs-context top-3 overlap
raw adjacent attention absolute-difference mean
context adjacent attention absolute-difference mean
context feature delta norm ||H-X|| / ||X||
effective final residual norm ||g(C-A)|| / ||A||
```

The local-context state also records Conv1d max absolute weight, mean absolute weight and L2 norm.

These are mechanism diagnostics only. B31 is not trained to achieve a target divergence, smoothness, overlap, or context norm.

## Safety tests

Before full training, B31 tests pin:

- exact 3,840 new-parameter contract;
- depthwise Conv1d geometry `k=3`, groups=768, no bias;
- exact zero initialization of context and outer gate;
- exact B20 functional equivalence at zero gate;
- B20 training-mode RNG-path equivalence at zero gate;
- exact B29 complementary scoring/summary when context weights are zero;
- context affects scores while values remain original B20 slice tokens;
- staged gate -> query/context gradient coupling;
- finite and bounded mechanism-audit metrics;
- bf16 finite behavior;
- empty-study finite behavior.

## Runtime

```text
hard budget       <= 8.25 h
internal reserve   >= 30 min
competition limit      9 h
```

The new operation is a depthwise Conv1d over already computed 16x768 slice-token sequences. No additional image-encoder pass is introduced. Runtime must still be measured.

## Outputs

```text
runs/b31_local_context_complementary_pool/
├── b31_model.pt
├── training_audit.json
├── complementary_pool_state.json
├── local_context_state.json
├── attention_audit.json
└── history.json
```

## Evaluation governance

The historical 623-study weak-v2 surface is not a holdout because B31 trains on all 3,120 historical B20 weak-supervision studies.

After a valid training/mechanism audit, B31 may be evaluated descriptively on the reused 58-study expert surface. The evaluator performs a single three-way paired comparison:

```text
B20 vs B31
B29 vs B31
```

This reused surface cannot independently promote B31. Do not derive a B31.1 kernel size, context strength, endpoint, target-specific route, B29/B31 target selection, or blend weight from that outcome.
