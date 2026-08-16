# B27 — pathology-specific acquisition routing

> **Status — 2026-08-16:** IMPLEMENTED / READY FOR FIXED-E2 TRAINING. **B20 remains the active working model.**

## Motivation

B26 closed the supervision-repair branch. The next intervention returns to the imaging side while preserving the frozen B20 weak-label semantics.

A naive proposal to "add pathology-specific series attention" would duplicate functionality already present in B20/B12.1: B20 has 12 learned pathology tokens that cross-attend to the contextualised real-series memory.

B27 therefore makes a much narrower change. It exposes the already-available acquisition metadata directly to that final pathology-to-series attention as a learned additive logit bias.

```text
B20
series image/content + metadata embeddings
 -> shared study Transformer
 -> pathology-query cross attention

B27
same path
 + 12 pathology-specific additive attention biases from
   plane / fluid sensitivity / fat suppression
```

## Single architectural change

For target `t` and real series `k`, B27 adds

```text
routing_bias(t,k)
  = plane_bias(t, plane_k)
  + fluid_bias(t, fluid_k)
  + fat_bias(t, fat_k)
```

to the existing B20 pathology-query attention logits before softmax.

No anatomical preference is hard-coded. All new values start at exactly zero.

Known categories:

```text
plane:   Sagittal / Coronal / Axial
fluid:   structural / fluid-sensitive
fat:     not-fat-suppressed / fat-suppressed
```

Unknown metadata and padding have a permanently fixed routing bias of exactly zero, preventing B27 from learning a direct missing-metadata/site shortcut through these new parameters.

Total new trainable parameters:

```text
12 x (3 + 2 + 2) = 84
```

This is deliberately tiny relative to the ConvNeXt/Transformer model and adds negligible compute compared with MRI encoding.

## Zero-initialisation contract

When B27 shares the same B20 state and all routing tables are zero, B27 is functionally equivalent to B20. The unit tests explicitly check this property.

That matters because B27 starts from the existing imaging decision function rather than introducing a second randomly initialised routing network.

## Frozen elements

```text
B6 v1.2.1 supervision       unchanged
training studies            3120
usable cells                14123
positive / negative         6871 / 7252
eligible MRI series         17475
B16 report-aligned encoder  unchanged and frozen
encoder SHA                 b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
B20 crop                    90% post-resize crop only
slice sampling              16 positions / series
optimizer / LR              unchanged
augmentation                unchanged
loader seed                 unchanged
scheduler horizon           5 epochs
training endpoint           fixed E2
expert labels in gradients  0
expert checkpoint selection none
```

The historical B20 active checkpoint remains untouched.

## Runtime policy

The user-supplied competition ceiling is 9 hours. B27 uses a stricter experiment guard:

```text
hard budget          <= 8.25 h
internal reserve      >= 30 min
```

The recent B20-family fixed-E2 run on the RTX A4500 Laptop GPU completed in about 58 minutes, so B27 has a very large safety margin. The 84 routing-bias parameters do not add another image encoder pass.

The existing repository-wide `RuntimeBudget` also refuses any budget `>= 9 h`.

## Ollama use

The local `qwen3:14b` Ollama model is **not** placed in B27 training or competition inference.

Instead, after training B27 exports `routing_biases.json`. A one-call audit utility:

```text
developments/scripts/review_b27_routes_with_ollama.py
```

asks the pinned local Ollama model to review whether the *learned* routing preferences are clinically interpretable. The review:

- is descriptive only;
- is never fed into the MRI model;
- does not change routing values, labels, thresholds or epochs;
- is not part of the competition runtime path;
- records the Ollama model digest/provenance.

This gives us useful LLM-assisted interpretation without paying LLM latency inside the 9-hour submission window or injecting an LLM-derived anatomical prior into B27.

## Training outputs

```text
runs/b27_pathology_routing/
├── b27_model.pt
├── training_audit.json
├── history.json
└── routing_biases.json
```

The training audit records exact study/series/cell coverage, frozen encoder SHA, routing norm growth and the conservative runtime budget.

## Evaluation governance

Because B27 trains on all 3,120 B20 studies, the historical 623-study weak-v2 partition is not a holdout for B27.

The existing 58 expert studies are also heavily reused development data and selected the historical B20 epoch. The provided B27 paired evaluation is therefore post-hoc development evidence only:

```text
python -m rsna_knee.b27_gold_eval ...
```

No automatic promotion is allowed from that result. Hidden competition evaluation remains the independent performance signal.

## Run order

```text
1. unit tests
2. B27 fixed-E2 training
3. inspect training_audit.json and routing_biases.json
4. optional one-call Ollama route plausibility audit
5. only then run the reused-expert paired diagnostic
```

Do not tune B27 routing tables or metadata categories from the reused 58-study outcome.
