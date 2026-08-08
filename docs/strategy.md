# Modeling strategy

## Core principle

This is a weakly supervised multi-sequence MRI problem with a very small trusted gold set. Production therefore prioritizes **supervision quality, leakage control, and runtime discipline before model scale**.

## 1. PU-aware weak supervision

Reports are a training teacher, never a required inference modality. Each target/report pair is classified as positive, negated, uncertain, or unmentioned. Fold-safe calibration estimates `P(y=1 | state)` only from gold studies allowed in the current training phase.

Confidence combines calibration evidence with information beyond target prevalence. A frequent but uninformative report state remains low-weight. `unmentioned` receives zero direct BCE weight by default; report silence is unlabeled, not negative. Official target cells override the teacher cell-by-cell.

## 2. Nested outer/inner validation

For outer fold `k`:

```text
outer gold fold       -> untouched final OOF evaluation
inner gold fold       -> epoch-count selection only
remaining gold fold   -> Phase-A trusted training
```

After epoch selection, Phase A is discarded. A fresh model is initialized and trained for exactly the selected duration using all non-outer gold studies. Only then is the outer fold evaluated.

The wall-clock guard may stop Phase-A exploration early if another selection epoch would leave insufficient estimated time for Phase B and evaluation.

## 3. Macro-metric-aligned loss

The competition metric is macro ROC-AUC, so BCE is normalized separately within each target and then averaged across the 12 targets. A pathology with more pseudo-label mass cannot dominate simply because it has more supervised cells.

A confidence-gated pairwise ranking term is retained as an AUC-oriented auxiliary objective. Because production uses one GPU with a small study batch, its effective use is **measured, not assumed**: every fold writes per-pathology pair counts to `ranking_pairs.json`. If rare targets receive essentially no ranking pairs, the term should be disabled or redesigned rather than credited with performance.

## 4. One-GPU trusted/general sampling

Training uses two pools:

- trusted: official gold plus unusually reliable pseudo-labeled studies;
- general: remaining weakly supervised studies.

A deterministic single-GPU batch sampler enforces the requested trusted fraction in expectation. CPU worker processes handle DICOM/data work; the model itself runs in one CUDA process.

## 5. In-domain MRI self-supervision

Optional SSL uses only non-gold competition MRI studies by default. Different sequences from the same knee form anatomy-related positive pairs; auxiliary heads predict plane and fluid/structural sequence type.

The conservative competition config does **not** use external pretrained weights unless the exact competition-specific rules are independently verified to permit them. Therefore the meaningful production ablation is:

```text
random ConvNeXt initialization
vs
competition-data SSL initialization
```

on identical Stage-1 outer OOF folds.

## 6. MRI representation and fusion

The model consumes up to six semantic streams:

- sagittal fluid-sensitive;
- sagittal structural;
- coronal fluid-sensitive;
- coronal structural;
- axial fluid-sensitive;
- axial structural.

Each selected series is represented by distributed 2.5D triplets. Training varies triplet gap and center position mildly. ConvNeXt encodes active triplets only; missing streams are masked before the backbone.

All MRI slice/sequence tokens then interact through a Transformer before classification. Twelve pathology query tokens self-attend and cross-attend to the MRI memory.

## 7. Deterministic stochastic training

Augmentation uses acquisition-compatible perturbations: center jitter, gap 1-2, small affine transforms, gamma variation, low-frequency bias field, noise, and slice dropout.

Worker randomness is derived from the seeded PyTorch worker generator. NumPy slice-jitter generators are seeded from that worker-local torch RNG, making controlled experiments reproducible for fixed settings and worker count.

## 8. DICOM caching and one-pass TTA

Persistent DataLoader workers hold a bounded LRU of recently decoded DICOM volumes. This reduces repeat decoding without requiring a huge disk cache.

At inference, a selected series is decoded once. All requested slice-center TTA views are generated from that decoded volume, and all fold models use the same batch before it is released. TTA therefore multiplies GPU forward passes but not DICOM reads.

If projected multi-view inference approaches the runtime budget, the pipeline automatically falls back to the central view. It fails early if even the reduced path cannot safely finish.

## 9. Leakage-safe Stage-2 co-training

Every non-gold report group receives a deterministic `crossfit_fold`.

Stage-1 outer fold `k` excludes:

- outer-gold fold `k`; and
- non-gold `crossfit_fold=k` studies.

It writes predictions for those weak studies to:

```text
stage1/fold{k}/weak_oof.csv
```

For Stage-2 outer fold `k`, **that is the only permitted image teacher**. Predictions from Stage-1 folds `j != k` are rejected because those models may have trained on gold fold `k`, creating indirect validation leakage.

Thus Stage-2 uses image/report consensus only for the fold-local weak subset; other weak cells remain report-supervised. This sacrifices some image-teacher coverage in exchange for an unbiased outer OOF estimate.

## 10. Full pre-run audit

Before long GPU runs, `rsna-knee audit` measures:

- report-state counts per pathology;
- calibrated confidence distributions;
- gold-fold positive/negative counts;
- six-stream selection/missing rates;
- full selected-series DICOM decode status;
- partial file-decode failures.

The full pixel audit runs in CPU processes and must complete within its configured budget or it is marked incomplete and fails.

## 11. Runtime policy

Every Kaggle execution is its own bounded job:

```text
full audit
optional SSL
Stage-1 fold 0
Stage-1 fold 1
Stage-1 fold 2
Stage-2 fold 0
Stage-2 fold 1
Stage-2 fold 2
final inference
```

Production uses `runtime_budget_hours: 8.5`, giving margin below a 9-hour GPU notebook ceiling. Do not combine all folds into one notebook.

## 12. Statistical reporting

Outer OOF uses raw official cells with NaNs preserved. Report:

- per-target AUC;
- macro AUC;
- study-level bootstrap interval;
- paired bootstrap delta between controlled runs.

No SSL, Stage-2, augmentation, or architecture change should be called an improvement until it is measured on the same outer OOF studies.

## Recommended execution order

```text
full audit
  -> 3-fold Stage-1 smoke
  -> Stage-1 random-init production
  -> optional SSL run
  -> Stage-1 SSL production
  -> paired OOF comparison
  -> freeze best Stage-1 initialization
  -> Stage-2 fold-local co-training
  -> paired Stage2 vs Stage1 OOF
  -> one-pass MRI-only submission
```

Avoid report-silence negatives, outer-fold early stopping, wrong-fold image teachers, multi-GPU launchers, network dependencies, and any notebook plan that consumes the full 9-hour ceiling without safety margin.
