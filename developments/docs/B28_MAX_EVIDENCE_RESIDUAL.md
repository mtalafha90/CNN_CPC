# B28 — zero-gated max-evidence series residual

> **Status — 2026-08-16:** IMPLEMENTED / PRE-OUTCOME / READY FOR SAFETY TESTS. **B20 remains the active working model.** B27/B27.1 are closed and not promoted.

## Motivation

B20/B12.1 represents each real MRI acquisition hierarchically:

```text
16 sampled slice triplets
  -> frozen ConvNeXt encoder
  -> learned single-query attention pooling
  -> one series token
  -> study Transformer
  -> 12 pathology queries
```

The next experiment asks a new imaging-side question that is independent of the reused 58-study B27.1 outcome:

> Does compressing all 16 slice embeddings to only the learned attention summary discard sparse/extreme within-series evidence that would be useful to the downstream study model?

This is motivated by two established aggregation ideas rather than by target-level tuning:

1. **MRNet** (Bien et al., PLOS Medicine, 2018) classified knee MRI by extracting per-slice CNN features and applying element-wise max pooling across slices before classification. That architecture demonstrates that max-across-slice evidence is a clinically relevant baseline for knee MRI.
2. **Attention-based MIL** (Ilse et al., ICML 2018) provides a learnable permutation-invariant bag aggregation mechanism. B20 already uses this family of idea for within-series pooling.
3. **Set Transformer** (Lee et al., ICML 2019) formalizes attention-based pooling of set-structured inputs. B20's series compression is consistent with this general set-pooling view.

B28 does **not** replace B20 attention pooling. It preserves it exactly and adds a residual max-evidence path so that the experiment isolates whether complementary extreme evidence is useful.

## Single architectural intervention

For each real series, let

```text
A = historical B20 learned attention-pooled series token
M = element-wise max over encoder image-content slice embeddings
```

B28 uses

```text
series_token = A + tanh(g) ⊙ LayerNorm(M)
```

where `g` is a feature-wise vector with one value per encoder feature dimension.

Important constraints:

- `g` is initialized to **exactly zero**;
- `tanh(g)` bounds every residual coefficient to `[-1, +1]`;
- the max branch uses **encoder image-content features only**;
- B20 slice-position and plane/fluid/fat metadata additions are subtracted before the max operation;
- the max branch contains no dropout and no additional attention module;
- therefore B28 at zero gate is functionally identical to B20 without changing B20's stochastic forward path.

The frozen ConvNeXt-Tiny representation dimension is 768, so B28 adds exactly:

```text
768 trainable gate parameters
```

No target-specific routing is introduced.

## Why this is cleaner than adding another attention network

A two-query or multi-token series pool would change study-token count and introduce another randomly initialized attention path. B28 instead keeps the study memory shape unchanged and begins exactly at the B20 function.

The residual is also computationally negligible because the expensive slice encoder is unchanged. `amax` and feature-wise gating operate only on already-computed slice embeddings.

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

## Runtime guard

```text
hard budget       <= 8.25 h
internal reserve   >= 30 min
competition limit      9 h
```

Recent B20-family fixed-E2 runs take about one hour on the RTX A4500 Laptop GPU. B28 adds no encoder pass and should remain in the same runtime class.

## Safety checks before full training

The B28 tests pin:

- exact 768-parameter zero initialization;
- B20 functional equivalence at zero gate;
- correct removal of slice-position and metadata terms from the max branch;
- bounded tanh gate;
- nonzero finite gradient reaching the gate from zero initialization;
- bf16 finite behavior;
- empty-study finite behavior.

## Training outputs

```text
runs/b28_max_evidence_residual/
├── b28_model.pt
├── training_audit.json
├── history.json
└── max_residual_gate.json
```

The audit records exact coverage, gate growth, frozen encoder SHA and runtime budget.

## Evaluation governance

B28 trains on all 3,120 historical B20 weak-supervision studies, so the 623-study weak-v2 partition is not a holdout.

The 58 expert studies are reused development data and historically selected B20. B28 therefore uses a fixed E2 endpoint and may only be compared post hoc on that surface after the training/gate audit is complete.

No automatic promotion is allowed from that result alone, and no post-hoc target-specific gating or selective ensembling may be derived from it.

## References

1. Bien N, Rajpurkar P, Ball RL, et al. **Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet.** *PLOS Medicine*. 2018;15(11):e1002699. doi:10.1371/journal.pmed.1002699.
2. Ilse M, Tomczak JM, Welling M. **Attention-based Deep Multiple Instance Learning.** *Proceedings of the 35th International Conference on Machine Learning*. PMLR 80:2127–2136, 2018.
3. Lee J, Lee Y, Kim J, Kosiorek A, Choi S, Teh YW. **Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.** *Proceedings of the 36th International Conference on Machine Learning*. PMLR 97:3744–3753, 2019.
