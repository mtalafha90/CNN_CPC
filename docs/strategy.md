# Modeling strategy

> **Snapshot: 2026-08-10.** Canonical measured results are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). **B7.1 full-corpus weak supervision is the current best standalone development model at macro AUC `0.5644802945`. B8 spatial-anatomy learning is implemented and currently training; no B8 gold score is recorded yet.**

## Core principle

`CNN_CPC` treats the challenge as a **weakly supervised multi-sequence MRI problem with a tiny trusted development set**. The strategy prioritizes supervision quality, leakage control, metric alignment, representation quality, pathology-specific evidence extraction and runtime discipline before increasing model scale.

Verified release:

```text
training studies       4407
gold studies             58
report-only studies     4349
targets                   12
```

## What the experiment ladder now shows

| Candidate | Macro AUC | Interpretation |
|---|---:|---|
| B0 random | `0.4763` | weak baseline |
| B1 strong MRI SSL | `0.5030` | useful in-domain representation signal |
| B2 lower encoder LR | `0.4993` | optimizer tweak not the main bottleneck |
| B3 pathology MIL | `0.4945` | head simplification insufficient |
| B4 frozen SSL + classical | `0.5138` | image-only representation ablation |
| B5 image-report SSL | `0.524365` | report-aligned representation helped modestly |
| B7-v1 direct B6 supervision | `0.539772` | direct weak supervision helped |
| **B7.1 full coverage** | **`0.564480`** | **current best standalone development model** |
| B5+B7.1 fixed rank | `0.554014` | rejected versus B7.1 |
| B8 spatial anatomy | pending | current training experiment |

The strongest strategic evidence is now:

1. **representation quality matters**;
2. **direct pathology-specific weak supervision from reports is more useful than generic report alignment alone**;
3. **covering the entire weakly labelled corpus each epoch materially improved the point estimate**;
4. the next substantive lever is **retaining/localizing spatial evidence before pathology-query attention**, not more blend tuning.

## 1. Reports are training supervision, not inference inputs

Final inference is MRI-only. Reports are used only during training.

B5 uses report semantics for representation alignment. B6 converts reports to structured pathology states for B7/B7.1/B8:

```text
positive
negated
uncertain
unmentioned
```

Report silence is not negative.

## 2. B6 weak-label contract is frozen

Frozen B6 v1.2.1 training export:

```text
report-only rows                  4349
active weakly labelled studies    3120
inactive zero-usable studies      1229
usable cells                     14123
positive cells                    6871
negative cells                    7252
```

The completed gold audit showed high sensitivity/NPV but lower positive precision. One global asymmetric policy was therefore frozen:

| state | soft target | base weight |
|---|---:|---:|
| positive | 0.85 | 0.50 |
| negated | 0.05 | 1.00 |
| uncertain | ignored | 0.00 |
| unmentioned | ignored | 0.00 |

Target balancing equalizes total expected supervision mass across the 12 pathologies.

Do not tune parser rules or target-specific weak-label weights from subsequent gold results.

## 3. Six-stream MRI contract

```text
sagittal_fluid       sagittal_structural
coronal_fluid        coronal_structural
axial_fluid          axial_structural
```

Missing streams are explicit and masked. Each active series is represented with distributed 2.5D triplets and encoded with ConvNeXt-Tiny.

## 4. Strong competition-only MRI SSL

The strong SSL run used only the 4,349 non-gold MRI studies:

```text
8 epochs
8,000 batches
24,000 study draws
~5.52 corpus passes
238,274 active 2.5D examples
```

This improved the Stage-1 point estimate from B0 `0.4763` to B1 `0.5030` and supplied the representation foundation for B4/B5.

## 5. B4 branch: useful representation probe, closed selector search

B4 froze the strong SSL encoder and used deterministic mean/std/max stream features plus target-wise PCA/logistic regression:

```text
macro AUC = 0.5137567459
```

B4.1/B4.2/B4.3 selector-stabilization attempts all reduced pooled OOF. The B4 selector branch is closed; do not create further policy/grid variants from the same 58 labels.

## 6. B5: report-aligned representation

B5 excluded all 58 gold studies from representation training and aligned competition MRI with TF-IDF/SVD report semantics.

Result under the unchanged B4 probe:

```text
macro AUC = 0.5243650851
95% CI   = [0.4728108406, 0.5761619105]
```

Paired B4 -> B5 favored B5 but remained inconclusive:

```text
median difference = +0.0105821232
95% paired CI     = [-0.0408197338, +0.0622131599]
P(B5 > B4)        = 0.656
```

B5 remains the representation baseline and the initialization source for B7.

## 7. B7: direct pathology-specific weak supervision

B7 is the first experiment to train the MRI classifier directly on frozen B6 target cells.

Architecture:

```text
6 MRI streams
-> 16 sampled 2.5D slices/stream
-> ConvNeXt slice features
-> slice-position + stream embeddings
-> cross-sequence Transformer
-> 12 interacting pathology queries
-> cross-attention to MRI memory
-> 12 logits
```

B7-v1 reached:

```text
macro AUC = 0.5397724412
```

but its `500`-batch epoch cap yielded only 1,000 study draws/epoch, about 1.28 nominal corpus passes over four epochs.

## 8. B7.1: full-corpus coverage

The coverage limitation was identified before the B7-v1 gold result was used to design the follow-up. B7.1 changed only:

```text
batches/epoch 500 -> 1560
```

With batch size 2, every epoch covered all 3,120 active studies and all 14,123 usable cells.

Training loss:

```text
0.752419 -> 0.665171 -> 0.639117 -> 0.612758
```

Gold development result:

```text
macro AUC = 0.5644802945
95% CI   = [0.5052432984, 0.6229422178]
```

Paired B7-v1 -> B7.1:

```text
median difference = +0.0241102714
95% paired CI     = [-0.0140197876, +0.0660558004]
P(B7.1 > B7-v1)   = 0.8694
```

Paired B5 -> B7.1:

```text
median difference = +0.0399233552
95% paired CI     = [-0.0301354430, +0.1092349994]
P(B7.1 > B5)      = 0.8716
```

B7.1 is the current leader, but its superiority is not called statistically established because both paired intervals cross zero.

## 9. Ensemble question is closed

A single predeclared global B5+B7.1 50:50 rank ensemble scored:

```text
0.5540141184
```

below B7.1 `0.5644802945`.

Paired comparison:

```text
median(ensemble-B7.1) = -0.0105429030
95% paired CI         = [-0.0523218181, +0.0333886570]
P(ensemble > B7.1)     = 0.3054
```

Do not search alternative blend weights, raw averages, calibration transforms or target-specific mixtures on these 58 labels.

## 10. B8 strategy: preserve spatial evidence

B7.1 globally pools each sampled ConvNeXt feature map to one vector before pathology attention. B8 tests whether this discards useful localization information.

B8 changes MRI memory from:

```text
B7.1: 6 x 16 x 1   = 96 tokens
B8:   6 x 16 x 2x2 = 384 tokens
```

B8 keeps all compatible B7.1 weights and adds:

- 2x2 adaptive spatial pooling from the final ConvNeXt feature map;
- learned region-position embeddings;
- fixed soft pathology-specific stream preferences;
- broad center-slice preference only for selected focal structures;
- no hard stream/slice masking;
- no fixed medial/lateral/anterior/posterior quadrant assumption.

The in-plane fixed region prior is uniform because preprocessing does not certify canonical in-plane orientation across all series.

B8 retains the frozen B6 supervision policy, target balancing, 3,120-study full coverage, four epochs, learning rates and MRI-only inference contract.

**Current status: B8 training is in progress. Its first gold development evaluation must remain frozen and one-shot.**

## 11. Validation campaign caveat

The same 58 gold studies have now supported repeated method decisions. Each individual procedure may be leakage-aware internally, but the campaign as a whole is **model-selection cross-validation**.

Do not:

- select per-target winners after viewing gold predictions;
- optimize ensemble weights;
- retune B6 parser rules or global weak-label policy;
- derive target-specific B7/B8 weights from observed gold AUCs;
- search B8 grid sizes, anatomy-prior strengths, epochs or target-specific priors after the first B8 score and still treat that score as untouched;
- describe the best development AUC as a hidden-test or leaderboard guarantee.

## 12. Runtime and competition policy

Long GPU runs use one GPU, CPU multiprocessing for DICOM/data work, a bounded software runtime budget, no external pretrained image weights in the conservative path, and explicit checkpoint provenance.

See [`competition_policy.md`](competition_policy.md), [`B8_SPATIAL_ANATOMY.md`](B8_SPATIAL_ANATOMY.md) and [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).
