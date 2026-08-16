# B29 reused-expert result and pre-competition freeze

> **Status — 2026-08-16:** B29 is a **PROMISING FROZEN CANDIDATE**, but it is **NOT PROMOTED**. B20 remains the active working model until an independent hidden competition signal supports replacement.

## Frozen candidate

B29 is the fixed-E2 zero-gated complementary learned series-summary model:

```text
A = historical B20 learned attention-pooled series token
C = second learned softmax summary of the same 16 B20 slice tokens
series_token = A + tanh(g) * (C - A)
```

The exact local candidate checkpoint is:

```text
runs/b29_complementary_series_pool/b29_model.pt
```

Frozen training contract:

```text
training studies            3120
eligible MRI series        17475
usable B6 cells            14123
positive / negative        6871 / 7252
new parameters              1536
complementary query          768
feature-wise gate            768
training endpoint        fixed E2
expert labels in gradient      0
expert checkpoint selection    no
encoder SHA256
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

The final effective gate remained small and unsaturated:

```text
max |tanh(g)|   0.0225038
mean |tanh(g)|  0.00516436
L2              0.181919
```

## Reused 58-study expert diagnostic

This surface is heavily reused development data. It is **not independent validation**. B20 itself was historically selected using these expert studies. B29 was fixed before this B29 outcome was inspected, but the comparison remains post-hoc development evidence.

```text
B20 macro AUC                 0.6674066371
B29 macro AUC                 0.6768879224
raw B29 - B20                +0.0094812853
paired median difference     +0.0094213679
paired 95% CI                [-0.0037494185, +0.0241875594]
P(B29 > B20)                  0.9188
bootstrap replicates          5000
```

Per-target result:

| Target | B20 AUC | B29 AUC | B29 - B20 |
|---|---:|---:|---:|
| ACL | 0.526961 | 0.520833 | -0.006127 |
| MCL | 0.462585 | 0.469388 | +0.006803 |
| Medial Meniscus | 0.677885 | 0.730769 | +0.052885 |
| Lateral Meniscus | 0.744099 | 0.740373 | -0.003727 |
| Medial OA | 0.694574 | 0.745736 | +0.051163 |
| Lateral OA | 0.671180 | 0.673114 | +0.001934 |
| PF OA | 0.674389 | 0.688546 | +0.014157 |
| Effusion | 0.864596 | 0.869565 | +0.004969 |
| Synovitis | 0.837515 | 0.816010 | -0.021505 |
| Baker's | 0.711957 | 0.762681 | +0.050725 |
| Contusion | 0.520918 | 0.520918 | 0.000000 |
| Fracture | 0.622222 | 0.584722 | -0.037500 |

B29 improved seven targets, tied one, and declined on four. The broad positive direction is more encouraging than the preceding B27.1 and B28 experiments, but the confidence interval still crosses zero and the reused expert surface cannot authorize promotion.

## Decision frozen before hidden evaluation

```text
B20   ACTIVE REFERENCE MODEL
B29   PROMISING FROZEN CANDIDATE
B29.1 DO NOT CREATE FROM THIS RESULT
```

Do not change any of the following after seeing the reused expert result:

- B29 query formulation;
- gate formulation or initialization;
- fixed-E2 endpoint;
- target-specific gating;
- target-specific routing;
- selective B20/B29 target substitution;
- probability blending weights;
- supervision;
- crop geometry;
- encoder.

The historical 623-study weak-v2 partition is **not** a holdout because B29 trained on all 3,120 historical B20 weak-supervision studies.

## Independent next signal: hidden competition evaluation

The next comparison is strictly:

```text
Submission A = canonical B20 checkpoint
Submission B = exact frozen B29 fixed-E2 checkpoint
```

No blending is used for the first comparison.

Before the first hidden submission, record locally:

```text
SHA256(B20 checkpoint)
SHA256(B29 checkpoint)
SHA256(B20 submission CSV)
SHA256(B29 submission CSV)
```

The B29 competition inference module writes the checkpoint and submission hashes into its manifest. This creates a byte-level freeze of the candidate used for hidden evaluation.

A hidden competition result may be used as the next independent performance signal. It should be interpreted alongside runtime validity and submission integrity. Only after that result should promotion of B29 over B20 be considered.
