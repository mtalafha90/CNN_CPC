# Modeling strategy

## Core principle

`CNN_CPC` treats the challenge as a **weakly supervised multi-sequence MRI problem with a very small trusted gold set**. The production strategy therefore prioritizes supervision quality, leakage control, metric alignment and runtime discipline before increasing model scale.

The current baseline has now passed real-data inspection, real DICOM preflight, a complete selected-series audit, OA weak-label verification and an end-to-end GPU smoke run.

## 1. Report supervision is a teacher, not an inference input

Radiology reports are available during training and are converted into per-target states:

```text
positive
negated
uncertain
unmentioned
```

The report teacher is deterministic and multilingual. Report silence is not treated as a negative. By default:

```text
unmentioned -> zero direct weight
```

Finite official labels override the report teacher cell-by-cell.

Final inference is MRI-only.

## 2. Fold-safe calibration

For each training phase, calibration uses only gold studies allowed in that phase. For target `c` and report state `s`, a smoothed state probability is estimated from the permitted gold subset. Confidence combines:

- evidence: how much gold support the state has;
- informativeness: how far the state-conditioned probability moves beyond target prevalence.

A common or noisy state can therefore remain low-weight even when it is frequently observed.

The calibration path is intentionally conservative. The production trusted threshold is not lowered merely to increase pseudo-label count.

## 3. Compartment-aware OA supervision

The first real-data audit exposed a concrete problem: all three OA targets had zero report-teacher weight because the original lexicon recognized only narrow explicit OA phrases.

The parser was expanded in a controlled, compartment-aware way. It now recognizes evidence such as:

- explicit osteoarthritis / arthrosis / gonarthrosis;
- cartilage loss;
- chondrosis / chondromalacia;
- osteophytes;
- compartment-specific degenerative cartilage wording;
- patellar/trochlear cartilage disease for PF OA.

The parser does **not** turn a generic meniscal degeneration statement or a bare compartment mention into OA.

Verified state counts on the 4,407 real reports are:

| Target | Positive | Negated | Unmentioned |
|---|---:|---:|---:|
| Medial OA | 492 | 339 | 3,576 |
| Lateral OA | 409 | 387 | 3,611 |
| PF OA | 695 | 379 | 3,333 |

The resulting OA cells contribute weak weighted BCE supervision but remain below the trusted/gold confidence regime.

## 4. Nested outer/inner validation

The 58 official gold studies are deterministically balanced across three folds.

For outer fold `k`:

```text
outer gold fold       -> final OOF evaluation only
inner gold fold       -> Phase-A epoch-count selection
remaining gold fold   -> Phase-A trusted training
```

Observed role sizes:

| Outer fold | Gold train | Inner selection | Outer validation |
|---|---:|---:|---:|
| 0 | 20 | 20 | 18 |
| 1 | 18 | 20 | 20 |
| 2 | 20 | 18 | 20 |

Every target has at least one positive and one negative in every outer fold, so per-target AUC is defined in all three folds.

Phase A is discarded. Phase B starts from a **freshly initialized model** and trains for exactly the selected epoch count using all non-outer gold studies.

## 5. Macro-metric-aligned BCE

The competition metric is macro ROC AUC across 12 targets. Weak-label coverage differs strongly by pathology, so ordinary cell averaging would allow heavily mentioned targets to dominate optimization.

For each planned epoch, the deterministic sampler is expanded first and the total target-specific supervision mass is computed. Batch BCE contributions are divided by that target's planned epoch denominator before the 12 target objectives are macro-averaged.

This keeps optimizer influence aligned with the macro target structure rather than raw mention frequency.

## 6. Confidence-gated ranking objective

A pairwise ranking auxiliary is added to weighted BCE. A cell must satisfy the configured confidence gate and positive/negative target threshold before it may participate.

Defaults:

```yaml
rank_loss_weight: 0.10
rank_pairs_per_target: 32
rank_min_confidence: 0.35
rank_positive_threshold: 0.75
rank_negative_threshold: 0.25
```

The first smoke run revealed zero ranking pairs for every target. This was not a ranking-loss bug; it was a minibatch composition problem.

## 7. Pair-friendly trusted sampling

With `batch_size: 2` and `trusted_fraction: 0.30`, the original sampler spread trusted rows so evenly that batches usually contained one trusted study and one general study. Most weak cells are below the ranking confidence gate, so a trusted positive and trusted negative rarely coexisted in a minibatch.

The current sampler preserves the requested trusted-row fraction but, for even batch sizes, emits trusted rows in pairs. For production batch size 2, batches are therefore typically either:

```text
[trusted, trusted]
```

or

```text
[general, general]
```

according to the deterministic cumulative quota.

This does **not** lower `trusted_pseudo_threshold` or `rank_min_confidence`.

Verified paired-sampler fold-0 smoke diagnostics:

```text
selection ranking pairs = 63
retrain ranking pairs   = 61
```

All 12 targets contributed at least one ranking pair.

## 8. Trusted/general pools

A study belongs to the trusted pool when it is:

- an official gold study; or
- a study with at least one pseudo-label weight above the configured trusted threshold.

Current default:

```yaml
trusted_fraction: 0.30
trusted_pseudo_threshold: 0.60
```

The real report audit showed that ordinary report pseudo-labels remain below 0.60, so the trusted pool is currently anchored primarily by official gold examples. This is intentional rather than a reason to weaken the threshold.

## 9. Six-stream MRI representation

The model consumes up to six semantic streams:

- sagittal fluid-sensitive;
- sagittal structural;
- coronal fluid-sensitive;
- coronal structural;
- axial fluid-sensitive;
- axial structural.

Missing streams are masked. Real-data coverage confirms this is essential because `axial_structural` is present in only about one quarter of training studies.

Each active series contributes distributed 2.5D triplets. The production tensor contract is conceptually:

```text
[B, K, S, 3, H, W]
```

with `K=6` semantic streams and `S=16` positions per stream by default.

## 10. ConvNeXt + cross-sequence Transformer + pathology queries

Active 2.5D triplets are encoded by ConvNeXt-Tiny. Learned position and stream embeddings are added before a cross-sequence Transformer.

Twelve learnable pathology tokens then:

1. interact with one another through a pathology-context Transformer;
2. cross-attend to MRI memory;
3. produce one target-specific logit each.

This allows ACL, menisci, OA, effusion and other targets to attend to different sequence/slice evidence while still sharing the study representation.

## 11. MRI augmentation and reproducibility

Training uses mild acquisition-compatible perturbations:

- center jitter;
- triplet gap 1 or 2;
- small affine changes;
- gamma variation;
- low-frequency bias field;
- Gaussian noise;
- slice dropout.

Worker randomness is seeded from PyTorch worker seeds. The pipeline aims for reproducible sampling for a fixed environment/settings but does not claim bitwise-deterministic GPU kernels.

## 12. DICOM quality gate

Long GPU training is blocked until preflight and audit succeed.

The verified real-data audit found:

```text
21,886 / 21,886 selected series decoded
732,554 / 732,556 DICOM files decoded
2 selected series with one failed file each
0 selected series failed
```

The global failure rate is far below the configured threshold. Both partially affected series remain usable.

## 13. Optional competition-data SSL

An optional self-supervised Stage-1 candidate uses only non-gold competition MRI by default. Same-study sequences provide anatomy-related positive pairs, with auxiliary plane and sequence-type heads.

External pretrained weights remain off in the conservative production config unless the exact current competition rules are explicitly verified to permit them.

## 14. Leakage-safe Stage-1 candidate selection

Random initialization and competition-data SSL are candidate Stage-1 methods.

For outer fold `k`, downstream candidate selection uses only that candidate's `inner_macro_auc` for fold `k`. `outer_macro_auc` is deliberately ignored.

This prevents the outer fold from choosing which teacher is subsequently used in its own Stage-2 experiment.

## 15. Leakage-safe Stage 2

Each non-gold report group receives a deterministic cross-fit fold.

Stage-1 fold `k` excludes:

- outer-gold fold `k`;
- non-gold `crossfit_fold=k` rows.

After Phase B, it predicts the excluded weak subset to `fold{k}/weak_oof.csv`.

Stage-2 fold `k` may use only that safe fold-local image teacher. Wrong-fold, incomplete, wrong-stage or validation-contract-incompatible teachers are rejected.

Stage-2 Phase A remains report-only. Phase B starts fresh and combines report and fold-local image evidence. Strong agreement is emphasized; conflict is downweighted; very confident image predictions can modestly supervise report-silent cells.

Stage 2 intentionally does not write another `weak_oof.csv`.

## 16. TTA policy

Primary validation and final submission use the same predeclared center offsets:

```yaml
validation_tta_offsets: [-1, 0, 1]
tta_center_offsets: [-1, 0, 1]
```

The paired-sampler smoke run happened to show center-only outer AUC slightly above TTA outer AUC. This **does not justify changing TTA**, because the result came from a tiny smoke fold. `oof_center.csv` remains diagnostic only.

## 17. Runtime policy

Every long GPU run uses:

```yaml
runtime_budget_hours: 8.5
runtime_reserve_minutes: 10
```

The finish estimator reserves time for:

```text
remaining retraining
+ outer OOF TTA inference
+ Stage-1 weak OOF inference
+ bootstrap
+ loader startup
+ serialization
```

Prediction also checks the deadline batch-by-batch.

The verified smoke runtime resolved to an NVIDIA RTX A4500 Laptop GPU using BF16, with modest peak allocated GPU memory. Production runtime results are reported only from non-smoke runs.

## 18. Statistical reporting

Final OOF evaluation reports:

- per-target AUC;
- macro AUC;
- study bootstrap confidence intervals;
- paired bootstrap differences for controlled comparisons.

Once outer OOF is used to choose a final method, it must be described as **model-selection cross-validation**, not an untouched independent estimate.

## Recommended execution order

```text
real-data inspect
-> validation manifests
-> train/test preflight
-> full selected-series audit
-> fold-0 Stage-1 random smoke
-> Stage-1 random production folds 0/1/2
-> optional competition-data SSL
-> Stage-1 SSL folds 0/1/2
-> per-fold Stage-1 candidate selection using inner AUC only
-> Stage-2 folds 0/1/2
-> controlled OOF comparisons
-> freeze final stage
-> stage/fold/TTA-validated inference
```

Do not add model scale or weaken supervision thresholds before the controlled baseline experiments establish what is actually limiting performance.