# B45 — Plane-Calibrated Target-Conditioned Sparse MIL

## Status

Prospective fixed endpoint.  B45 has not been evaluated on Expert-58 or the Kaggle hidden test at the time this protocol is frozen.

## Mechanistic basis

B43 decomposed the frozen B42 sparse-MIL evidence by target, series and anatomical plane on the reused 58-study expert surface.  The diagnostic showed strong systematic axial selection enrichment despite axial series representing only a minority of the available acquisitions.  For ACL in particular, sagittal best-series evidence was substantially more discriminative than axial evidence, while the unrestricted B42 global top-k overwhelmingly selected axial tokens.  B44 then doubled deterministic center coverage from 32 to a nested 64 while preserving the original first 32 exactly.  ACL, MCL and Contusion combined AUCs were unchanged and Fracture moved only minimally, so center-count coverage is not supported as the primary weak-target mechanism.

These diagnostics are mechanistic reuse of Expert-58, not independent test evidence.  They motivate one architectural change but cannot be used to tune B45 or promote it.

## Frozen hypothesis

B42 adds plane metadata directly to every local token before pathology-specific evidence scoring and performs one top-k selection across all series, planes, slices and regions.  A learned plane-dependent score offset can therefore dominate cross-plane ranking even when another plane carries more discriminative pathology information.

B45 factors anatomical plane identity out of the token evidence score.  The local feature, normalized slice position, region embedding, fluid-sensitivity embedding and fat-suppression embedding remain available to the evidence classifier.  Evidence is pooled separately within sagittal, coronal and axial tokens using the unchanged top-k=8 log-mean-exp operator.  A learned target-specific three-plane router then fuses the available plane logits.

For target `t` and plane `p`,

```text
L[t,p] = LME(TopK_8(E[t,n] for tokens n in plane p))
```

and

```text
alpha[t,p] = softmax(q[t,p]) over planes present in the study
L[t]       = sum_p alpha[t,p] * L[t,p]
```

The router logits `q[t,p]` are initialized to zero, so every available plane receives equal initial weight.  No target-specific clinical plane preference is hard coded.

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

## Governance

The B43 and B44 Expert-58 diagnostics are recorded mechanistic evidence only.  They are not a B45 tuning surface.  After this protocol is frozen, do not change plane pooling, router temperature, target-plane weights, token metadata, top-k, grid size, crop, geometry, learning rates, target subset or epoch count based on Expert-58.  B45 must first pass code/unit/preflight checks, then train for exactly two epochs.  Any later Expert-58 evaluation is descriptive/reused diagnostic evidence and cannot by itself promote B45 over the hidden champion.
