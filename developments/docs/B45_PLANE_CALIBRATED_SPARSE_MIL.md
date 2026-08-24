# B45 — Plane-Calibrated Target-Conditioned Sparse MIL

## Status

**COMPLETED / EXPERT-58 DIAGNOSTIC NEGATIVE-NEUTRAL / NO KAGGLE SUBMISSION.**

B45 completed its prospectively frozen fixed-E2 training endpoint and the one allowed post-training Expert-58 descriptive diagnostic. The user explicitly decided not to submit B45 to Kaggle. B45 is closed as a mechanistic experiment and must not be retuned from the reused Expert-58 result.

Final checkpoint SHA-256:

```text
bd7fbc94b49d45b2cf7fe97a1a7ab371a175dc63b9ee6551a56e251e13e6bc61
```

## Mechanistic basis

B43 decomposed the frozen B42 sparse-MIL evidence by target, series and anatomical plane on the reused 58-study expert surface. The diagnostic showed strong systematic axial selection enrichment despite axial series representing only a minority of the available acquisitions. For ACL in particular, sagittal best-series evidence was substantially more discriminative than axial evidence, while the unrestricted B42 global top-k overwhelmingly selected axial tokens. B44 then doubled deterministic center coverage from 32 to a nested 64 while preserving the original first 32 exactly. ACL, MCL and Contusion combined AUCs were unchanged and Fracture moved only minimally, so center-count coverage was not supported as the primary weak-target mechanism.

These diagnostics were mechanistic reuse of Expert-58, not independent test evidence. They motivated one architectural change but were not used to tune B45.

## Frozen hypothesis

B42 adds plane metadata directly to every local token before pathology-specific evidence scoring and performs one top-k selection across all series, planes, slices and regions. A learned plane-dependent score offset can therefore dominate cross-plane ranking even when another plane carries more discriminative pathology information.

B45 factors anatomical plane identity out of the token evidence score. The local feature, normalized slice position, region embedding, fluid-sensitivity embedding and fat-suppression embedding remain available to the evidence classifier. Evidence is pooled separately within sagittal, coronal and axial tokens using the unchanged top-k=8 log-mean-exp operator. A learned target-specific three-plane router then fuses the available plane logits.

For target `t` and plane `p`,

```text
L[t,p] = LME(TopK_8(E[t,n] for tokens n in plane p))
```

and

```text
alpha[t,p] = softmax(q[t,p]) over planes present in the study
L[t]       = sum_p alpha[t,p] * L[t,p]
```

The router logits `q[t,p]` initialize to zero, so every available plane receives equal initial weight. No target-specific clinical plane preference is hard coded.

## Frozen unchanged B42/B37 contract

- 4,349 report-only training studies.
- 24,035 eligible recognized-plane MRI series.
- 34,010 supervision cells.
- Zero expert/gold gradients and zero gold labels.
- Full-native percentile normalization.
- Fixed 90% native center crop.
- Constant-area native-aspect resize with reference area `448^2`.
- Reflection padding only to stride 32.
- Ragged per-series encoding.
- 32 deterministic 2.5D centers, gap 1.
- ConvNeXt local grid 6x6.
- Sparse top-k=8 and temperature 1.0.
- Local auxiliary loss weight 1.0.
- Zero-start target-wise residual gate.
- Final ConvNeXt stage/output norm trainable.
- Head learning rate `1e-4`.
- Encoder-tail learning rate `5e-6`.
- Weight decay `1e-4`.
- Gradient clipping 1.0.
- Effective batch size 2.
- Exactly two epochs, no checkpoint selection.
- Evaluation TTA offsets `[-1,0,+1]`.

## B45-only frozen choices

- Three explicit anatomical plane pools: sagittal, coronal and axial.
- Top-k=8 is applied independently inside every available plane.
- Plane router is target-specific but study-independent apart from the available-plane mask.
- Plane router logits initialize at zero.
- Router temperature is 1.0.
- Plane embedding is excluded from token evidence scoring and frozen.
- Fluid-sensitivity and fat-suppression metadata remain in token evidence scoring.
- No hard-coded target-plane priors.

## Completed training audit

B45 completed exactly two epochs and passed the final training audit. The endpoint used all 4,349 report-only studies, 24,035 eligible series and 34,010 supervision cells, with zero expert/gold gradients and no checkpoint selection. The encoder fingerprint moved, confirming that the declared final ConvNeXt stage was updated. The learned router remained close to uniform rather than collapsing into hard plane priors.

### Learned all-plane router weights

| Target | Sagittal | Coronal | Axial |
|---|---:|---:|---:|
| ACL | 0.329068 | 0.344869 | 0.326063 |
| MCL | 0.324091 | 0.342470 | 0.333439 |
| Medial Meniscus | 0.334139 | 0.334904 | 0.330957 |
| Lateral Meniscus | 0.329352 | 0.340822 | 0.329826 |
| Medial OA | 0.320470 | 0.329022 | 0.350508 |
| Lateral OA | 0.328181 | 0.330055 | 0.341764 |
| PF OA | 0.322485 | 0.337275 | 0.340240 |
| Effusion | 0.327627 | 0.325356 | 0.347017 |
| Synovitis | 0.326206 | 0.337483 | 0.336310 |
| Baker's | 0.328262 | 0.335894 | 0.335845 |
| Contusion | 0.327461 | 0.340807 | 0.331732 |
| Fracture | 0.333707 | 0.331402 | 0.334892 |

`plane_embedding_used_in_token_score=false` and `hard_coded_target_plane_priors=false` were verified by the final audit.

## Expert-58 descriptive diagnostic

Evaluation role: reused post-training Expert-58 descriptive diagnostic; not independent test evidence and not a B45 tuning or checkpoint-selection criterion.

Surface:

```text
studies       58
series        336
TTA offsets   [-1, 0, +1]
```

All three anatomical planes were available for every Expert-58 TTA view:

```text
available_plane_view_counts = [174, 174, 174]
```

Consequently, the mean effective router weights on Expert-58 were exactly the learned all-plane weights above; missing-plane masking did not influence this comparison.

### Macro AUC

| Endpoint | Macro AUC |
|---|---:|
| base 224 | 0.66875427195 |
| B37 combined | **0.68581779163** |
| B42 combined | 0.68312037480 |
| B45 global | 0.67615048763 |
| B45 combined | 0.67917640580 |
| B45 - B42 | **-0.00394396901** |
| B45 - B37 | **-0.00664138583** |

### Focal-six AUC

Focal six: ACL, MCL, Medial Meniscus, Lateral Meniscus, Contusion and Fracture.

| Endpoint | Focal-six AUC |
|---|---:|
| base 224 | 0.57162759875 |
| B37 combined | **0.58416487723** |
| B42 combined | 0.58009784257 |
| B45 global | 0.57689557635 |
| B45 combined | 0.57933407005 |

### Paired bootstrap

B45 combined minus B42 combined, 5,000 valid study-level replicates:

```text
median difference      -0.00346437957
95% CI                  [-0.01461292950, +0.00354830031]
P(B45 > B42)            0.1886
```

B45 combined minus B37 combined:

```text
median difference      -0.00620912158
95% CI                  [-0.01593680503, +0.00039182423]
P(B45 > B37)            0.0346
```

The intervals cross zero, so the 58-study surface does not establish a statistically definitive loss, but it provides no positive evidence for B45 and the paired mass is strongly tilted away from an improvement over B37.

### Per-target AUC versus B42

| Target | B42 | B45 | B45 - B42 |
|---|---:|---:|---:|
| ACL | 0.475490 | 0.462010 | **-0.013480** |
| MCL | 0.412698 | 0.414966 | +0.002268 |
| Medial Meniscus | 0.790865 | 0.781250 | -0.009615 |
| Lateral Meniscus | 0.667081 | 0.680745 | **+0.013665** |
| Medial OA | 0.813953 | 0.809302 | -0.004651 |
| Lateral OA | 0.692456 | 0.673114 | **-0.019342** |
| PF OA | 0.709138 | 0.697555 | -0.011583 |
| Effusion | 0.844720 | 0.844720 | 0.000000 |
| Synovitis | 0.816010 | 0.808841 | -0.007168 |
| Baker's | 0.840580 | 0.840580 | 0.000000 |
| Contusion | 0.533063 | 0.539811 | +0.006748 |
| Fracture | 0.601389 | 0.597222 | -0.004167 |

The two clearest positive target movements were Lateral Meniscus and Contusion. ACL, which motivated the plane-routing hypothesis, moved in the wrong direction.

## Interpretation

B45 does not support the simple hypothesis that B42's weak ACL performance was primarily caused by useful sagittal evidence being crowded out by axial tokens in one global top-k pool. The router itself stayed very close to uniform, so B45 mostly tested the effect of giving each plane an independent top-k evidence quota before near-uniform fusion. That intervention did not improve the reused overall or focal-six endpoints and worsened ACL.

This does **not** prove that plane or sequence information is irrelevant. It shows that static study-independent target-wise plane weights applied after independently pooled plane logits are too weak a mechanism for the observed failure. A useful next model should learn interactions among sequences/planes/features rather than merely allocate each plane a fixed scalar weight.

## Decision

```text
B45 status                 completed_not_promoted
Kaggle submission          intentionally not run
B45 vs B42 macro delta     -0.0039439690
B45 vs B37 macro delta     -0.0066413858
P(B45 > B42)               0.1886
P(B45 > B37)               0.0346
```

Close the B45 formulation. Do not tune router temperature, target-specific plane weights, target subsets, pooling, geometry, learning rates or epoch count from this Expert-58 result. Preserve the checkpoint and diagnostic as a negative mechanistic experiment.

## Governance

The B43 and B44 Expert-58 diagnostics are mechanistic evidence only. They are not a B45 tuning surface. The completed B45 Expert-58 evaluation remains descriptive reuse only. No later target-wise switching, router-prior injection, thresholding, blending, or checkpoint selection is authorized from these 58 studies.
