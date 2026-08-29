# Current project status

**Snapshot:** 2026-08-29
**Primary metric:** macro ROC AUC across 12 targets
**Best independent displayed Kaggle score:** **0.714**

This file is the living project-status record. Earlier snapshots remain available in Git history and in the frozen experiment documents.

## Independent hidden evidence

The recorded completed hidden submissions are:

| Endpoint | Main distinction | Kaggle displayed macro AUC |
|---|---|---:|
| B37 | direct-square 448 sparse MIL | **0.714** |
| B41 | aspect-preserving 448 square-pad sparse MIL | **0.714** |
| B42 | constant-area native-aspect ragged sparse MIL | **0.714** |
| B49 candidate | full-FOV native tiled local branch, dual-T4 hidden-safe inference | 0.707 |

The three `0.714` entries are displayed ties. Kaggle rounds leaderboard values,
so identical displayed scores do not establish identical unrounded AUC.

B41 originally failed during the hidden notebook rerun. The same frozen scientific endpoint later completed after inference was changed to hidden-safe streaming: one TTA study view at a time, native normalization once per series, host trimming after each study, and runtime prediction converted from a possible exception into telemetry. The resulting hidden score was `0.714`, demonstrating that the original B41 failure was operational rather than a model result.

B42 completed normally and also scored `0.714`.

The one permitted exploratory B49 candidate-only hidden submission completed on
the dual-T4 hidden-safe path and scored `0.707`. This is `−0.007` versus B42's
displayed `0.714`; it does not promote B49 or authorise B49 tuning, blending,
calibration, or another hidden submission.

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

## Completed B46–B49 evidence

| Experiment | Locked primary comparison | Completed result | Decision |
|---|---|---|---|
| B46 | B46 cross-fitted official-gold OOF − B42 on 58 studies | `−0.004946`; 95% CI `[−0.014664, +0.003402]`; `P=0.1296`; 5/12 targets improved | no support for gold anchoring at weight 4.0 |
| B48 | candidate − control on unseen scanners | `+0.0000749`; 95% CI `[−0.0000972, +0.0002786]`; `P=0.8010`; 7/12 targets improved | no support for global conditioning |
| B49 | candidate − control on unseen scanners | `+0.0005468`; 95% CI `[+0.0003146, +0.0008120]`; `P=1.0`; 10/12 targets improved | no support: effect below predeclared `+0.010` threshold |

B49's native full-FOV tile representation deliberately removed the B42 local
crop/resize loss, but its matched result remains a non-promotion result. The
candidate-only `0.707` Kaggle score is independent hidden evidence for that one
frozen endpoint; it is not a selector for another B49 variant.

Full records: [`B46_GOLD_ANCHORED_CROSSFIT.md`](B46_GOLD_ANCHORED_CROSSFIT.md),
[`B48_GLOBAL_CONDITIONED_SPARSE_MIL.md`](B48_GLOBAL_CONDITIONED_SPARSE_MIL.md),
and [`B49_NATIVE_TILED_MULTISCALE_MIL.md`](B49_NATIVE_TILED_MULTISCALE_MIL.md).

## What the B37–B49 sequence now says

The B37-B45 line has reached a plateau rather than a missing-small-hyperparameter problem.

- B38 showed that higher-resolution global-only encoder-tail training did not improve the reused Expert-58 endpoint.
- B40 showed that an additional training epoch reduced the weak-supervision objective while failing to improve expert AUC.
- B41/B42 showed that correcting in-plane geometry did not create visible hidden separation from B37.
- B44 showed that doubling deterministic center coverage from 32 to 64 did not rescue the weak targets.
- B45 showed that independent per-plane top-k pooling plus static target-plane fusion did not repair ACL and did not improve the aggregate endpoint.
- B46 found no support for the fixed clean-gold anchor, despite leakage-free cross-fitted OOF predictions.
- B48 found no meaningful support for post-cross-attention global-to-local conditioning on the frozen scanner surface.
- B49 preserved the full native local FOV in tiles, yet its matched effect was far below the predeclared practical threshold and its exploratory hidden score was lower than B42.

The project should therefore stop spending experiments on nearby resolution, crop, center-count, top-k, static plane-router or duration variants.

## Main diagnosis

The strongest remaining bottlenecks are now judged to be:

1. **report-to-expert target mismatch** — later models keep optimizing report-derived labels even when lower training loss no longer tracks expert AUC;
2. **adaptive use of the 58 expert studies** — they remain out of gradients but have been inspected too many times to act as a clean architecture-design surface;
3. **limited clean-supervision evidence** — B46's fixed 4.0 gold-cell anchor did not improve leakage-free OOF ranking; a different use of clean labels would require a new predeclared mechanism rather than a weight sweep;
4. **insufficient ordered volumetric reasoning** — 2.5D/top-k models sample depth but do not explicitly model long-range slice continuity;
5. **late/static plane handling** — B45 fuses per-plane logits with study-independent target scalars rather than learning feature-level cross-sequence interactions;
6. **multicenter/domain shift** — existing UID-hash weak splits are not explicit site/scanner holdouts;
7. **underpowered report semantics** — B16's full-report teacher is TF-IDF + SVD rather than a modern multilingual semantic representation.

The detailed evidence and corrective plan are frozen in [`POST_B45_PLATEAU_RETROSPECTIVE.md`](POST_B45_PLATEAU_RETROSPECTIVE.md).

## Current governance

- B46, B48, and B49 are completed frozen experiments. Do not retune their
  gold weight, folds, spatial preprocessing, tile geometry, query source, rank,
  loss, seed, scanner split, target subset, calibration, or blend from these
  results.
- B49's one candidate-only Kaggle endpoint is complete. The `0.707` hidden
  score is a recorded exploratory result, not a reason to submit the control
  arm or create another B49 endpoint.
- The repository still contains an unrun B47 native-grid implementation. It is
  not an automatically approved successor; any future training must begin with
  a separate prospective question, locked endpoint, and resource preflight.

## Current direction in one line

```text
preserve the completed B46/B48/B49 records
-> do not tune closed mechanisms from their results
-> define any future hypothesis and endpoint prospectively
```
