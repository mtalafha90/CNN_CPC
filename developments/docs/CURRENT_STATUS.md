# Current project status

**Snapshot:** 2026-08-24
**Primary metric:** macro ROC AUC across 12 targets
**Best independent displayed Kaggle score:** **0.714**

This file is the living project-status record. Earlier snapshots remain available in Git history and in the frozen experiment documents.

## Independent hidden evidence

The current successful hidden submissions are:

| Endpoint | Main distinction | Kaggle displayed macro AUC |
|---|---|---:|
| B37 | direct-square 448 sparse MIL | **0.714** |
| B41 | aspect-preserving 448 square-pad sparse MIL | **0.714** |
| B42 | constant-area native-aspect ragged sparse MIL | **0.714** |

These are displayed ties. Kaggle rounds the leaderboard value, so identical displayed scores do not establish identical unrounded AUC.

B41 originally failed during the hidden notebook rerun. The same frozen scientific endpoint later completed after inference was changed to hidden-safe streaming: one TTA study view at a time, native normalization once per series, host trimming after each study, and runtime prediction converted from a possible exception into telemetry. The resulting hidden score was `0.714`, demonstrating that the original B41 failure was operational rather than a model result.

B42 completed normally and also scored `0.714`.

B39 remains an inference-only five-offset B37 refinement whose earlier hidden notebook failed operationally; a hidden-safe streaming implementation exists, but no corrected B39 score is recorded here yet.

## B45 closed

B45 tested plane-calibrated target-conditioned sparse MIL on top of B42. It was frozen before training, completed exactly two epochs, used zero expert/gold gradients, and passed its final training audit.

Final checkpoint SHA-256:

```text
bd7fbc94b49d45b2cf7fe97a1a7ab371a175dc63b9ee6551a56e251e13e6bc61
```

The learned target-plane router stayed close to uniform. The fixed post-training Expert-58 diagnostic returned:

```text
                         macro AUC      focal-six
B37 combined             0.685818       0.584165
B42 combined             0.683120       0.580098
B45 combined             0.679176       0.579334

B45 - B42 macro         -0.003944
B45 - B37 macro         -0.006641
```

Paired bootstrap:

```text
B45 - B42
median                  -0.003464
95% CI                  [-0.014613, +0.003548]
P(B45 > B42)             0.1886

B45 - B37
median                  -0.006209
95% CI                  [-0.015937, +0.000392]
P(B45 > B37)             0.0346
```

ACL, which motivated the plane-routing hypothesis, changed from `0.475490` for B42 to `0.462010` for B45. Lateral Meniscus and Contusion improved, but the aggregate and focal-six endpoints did not.

**Decision:** B45 is `completed_not_promoted`. No Kaggle submission will be made for B45 by explicit project decision. Do not tune router temperature, plane weights, target subsets, top-k, geometry, learning rates or epoch count from the reused Expert-58 result.

Full B45 record: [`B45_PLANE_CALIBRATED_SPARSE_MIL.md`](B45_PLANE_CALIBRATED_SPARSE_MIL.md).

## What the late experiments now say

The B37-B45 line has reached a plateau rather than a missing-small-hyperparameter problem.

- B38 showed that higher-resolution global-only encoder-tail training did not improve the reused Expert-58 endpoint.
- B40 showed that an additional training epoch reduced the weak-supervision objective while failing to improve expert AUC.
- B41/B42 showed that correcting in-plane geometry did not create visible hidden separation from B37.
- B44 showed that doubling deterministic center coverage from 32 to 64 did not rescue the weak targets.
- B45 showed that independent per-plane top-k pooling plus static target-plane fusion did not repair ACL and did not improve the aggregate endpoint.

The project should therefore stop spending experiments on nearby resolution, crop, center-count, top-k, static plane-router or duration variants.

## Main diagnosis

The strongest remaining bottlenecks are now judged to be:

1. **report-to-expert target mismatch** — later models keep optimizing report-derived labels even when lower training loss no longer tracks expert AUC;
2. **adaptive use of the 58 expert studies** — they remain out of gradients but have been inspected too many times to act as a clean architecture-design surface;
3. **unused clean supervision** — all `58 x 12 = 696` official gold cells have been withheld from training despite being the only direct competition-target labels;
4. **insufficient ordered volumetric reasoning** — 2.5D/top-k models sample depth but do not explicitly model long-range slice continuity;
5. **late/static plane handling** — B45 fuses per-plane logits with study-independent target scalars rather than learning feature-level cross-sequence interactions;
6. **multicenter/domain shift** — existing UID-hash weak splits are not explicit site/scanner holdouts;
7. **underpowered report semantics** — B16's full-report teacher is TF-IDF + SVD rather than a modern multilingual semantic representation.

The detailed evidence and corrective plan are frozen in [`POST_B45_PLATEAU_RETROSPECTIVE.md`](POST_B45_PLATEAU_RETROSPECTIVE.md).

## Next experiment

### B46 — gold-anchored cross-fitted supervision

B46 should test the largest unresolved assumption **without changing the image architecture first**:

> Does clean official target supervision improve the frozen parent when used prospectively and cross-fitted, or is the ceiling primarily representational?

The proposed protocol is five-fold OOF use of the 58 gold studies. For each fold, the model trains on all report-only weak studies plus the other gold folds and predicts only gold studies excluded from that fold's gradients. The gold-source weight, architecture, epoch count and fold assignment must be frozen before the OOF result is inspected.

If B46 gives a coherent OOF gain that is not driven by one target, train one final all-gold-anchored model. If B46 is negative/small, move to B47 rather than tuning the weighting.

### B47 — explicit within-series slice sequence modeling

B47 should test ordered through-plane representation while keeping center density fixed. A lightweight slice Transformer/temporal block should operate on ordered per-slice features before study-level series aggregation. This tests depth relationships rather than more slice samples.

### B48 — dynamic cross-series/cross-sequence attention

Only after the preceding question is resolved should plane/sequence modeling be revisited. The next plane mechanism must be study-dependent feature interaction, not a static target-plane scalar router.

## Current direction in one line

```text
stop small sparse-MIL/geometry tweaks
-> test clean-label anchoring
-> then test explicit volumetric sequence reasoning
-> then dynamic cross-sequence interaction
```
