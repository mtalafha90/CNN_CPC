# Modeling strategy

## Core principle

This is a weakly supervised multi-sequence MRI problem with a very small trusted gold set. Production therefore prioritizes **supervision quality, leakage control, metric alignment, and runtime discipline before model scale**.

## 1. PU-aware report supervision

Reports are a training teacher, never an inference requirement. Each target/report pair is classified as positive, negated, uncertain, or unmentioned. Fold-safe calibration estimates `P(y=1 | state)` only from gold studies permitted in the current training phase.

Confidence combines calibration evidence with information beyond target prevalence. A frequent but uninformative state remains low-weight. `unmentioned` receives zero direct report weight by default; report silence is unlabeled, not negative. Finite official target cells override the teacher cell-by-cell.

## 2. Nested outer/inner validation

For outer fold `k`:

```text
outer gold fold       -> final OOF evaluation only
inner gold fold       -> Phase-A epoch-count selection
remaining gold fold   -> Phase-A trusted training
```

Phase A is discarded. A fresh Phase-B model is initialized and trained for exactly the selected duration using all non-outer gold studies. Only then is the outer fold evaluated.

The primary inner and outer validation predictions use the same TTA offsets as the planned submission. `oof.csv` is therefore the production-policy OOF prediction; `oof_center.csv` is a diagnostic only.

## 3. Nested Stage-1 method selection

Random ConvNeXt initialization and competition-data SSL are treated as **candidate Stage-1 methods**, not as a global choice made from outer OOF.

For outer fold `k`, candidate selection reads only fold `k`'s `inner_macro_auc`. It deliberately ignores `outer_macro_auc`. Candidates must share the same inner fold and validation-TTA contract.

This prevents the current outer fold from deciding which Stage-1 teacher is used inside its downstream Stage-2 experiment.

## 4. Macro-metric-aligned loss

The competition metric is macro ROC-AUC. Production training first builds the deterministic batch plan for the epoch and computes each pathology's total planned supervision weight. Each batch then contributes its weighted BCE against that **epoch-level target denominator**.

This prevents a pathology from receiving systematically more optimizer influence merely because it appears with non-zero weak-label weight in more batches. The 12 valid pathology objectives are macro-averaged.

A confidence-gated pairwise ranking term remains an auxiliary AUC-oriented objective. Because the study batch is small, its contribution is measured rather than assumed. `training_diagnostics.json` records per-target ranking pairs, weight mass, non-zero supervised cells, and participating batches.

## 5. One-GPU trusted/general sampling

Training uses two study pools:

- trusted: official gold plus unusually reliable pseudo-labeled studies;
- general: remaining weakly supervised studies.

A deterministic one-GPU batch sampler maintains the requested trusted fraction. CPU worker processes handle DICOM/data work; the neural network runs in one CUDA process.

The sampler receives an explicit absolute work deadline, so direct `train_fold()` calls have the same stop behavior as CLI launches.

## 6. In-domain MRI self-supervision

Optional SSL uses only non-gold competition MRI studies by default. Different sequences from the same knee form anatomy-related positive pairs; auxiliary heads predict plane and fluid/structural sequence type.

External pretrained weights remain disabled in the conservative production configuration unless competition-specific permission is explicitly verified. SSL checkpoints must carry competition-data provenance.

## 7. MRI representation and fusion

The model consumes up to six semantic streams:

- sagittal fluid-sensitive;
- sagittal structural;
- coronal fluid-sensitive;
- coronal structural;
- axial fluid-sensitive;
- axial structural.

Each selected series is represented by distributed 2.5D triplets. Training varies triplet gap and center position mildly. ConvNeXt encodes active triplets only; missing streams are masked before the backbone.

All MRI slice/sequence tokens interact through a Transformer. Twelve pathology query tokens then self-attend and cross-attend to MRI memory before target-specific readout.

## 8. Reproducible stochastic data sampling

Augmentation uses small acquisition-compatible perturbations: center jitter, gap 1-2, affine changes, gamma variation, low-frequency bias field, Gaussian noise, and slice dropout.

Worker randomness is derived from seeded PyTorch workers, and NumPy center jitter is seeded from the worker-local torch RNG. Sampling is reproducible for fixed seeds/settings/worker count, although accelerated GPU kernels are not claimed to be bitwise deterministic.

## 9. DICOM caching and one-decode TTA

Persistent workers own a bounded LRU of decoded volumes. Validation/inference construct all requested center-offset views immediately after one DICOM decode.

Thus TTA multiplies model forwards, not DICOM reads.

## 10. Leakage-safe Stage-2 co-training

Every non-gold report group receives a deterministic `crossfit_fold`.

Stage-1 outer fold `k` excludes outer-gold fold `k` and non-gold `crossfit_fold=k` studies. After Phase B it predicts those weak studies into `fold{k}/weak_oof.csv`.

For Stage-2 outer fold `k`, only the safe fold-`k` image teacher may be used. Wrong-fold, incomplete, non-Stage-1, or validation-contract-incompatible candidates are rejected.

Stage-2 phases remain distinct:

1. **Phase A — report-only epoch selection.** Image predictions are disabled so the Stage-1 model cannot indirectly influence inner epoch selection.
2. **Phase B — fresh co-training.** All non-outer gold are permitted and the fold-local weak subset receives independent image/report supervision.

Strong report/image agreement receives high weight. Direct conflicts are strongly downweighted. A very confident cross-fitted image teacher (`>=0.95` or `<=0.05` by default) may add a modest BCE weight (`0.20`) when report confidence is near zero. This lets Stage 2 learn from some report omissions without promoting those cells into the trusted sampler or ranking loss.

`stage2_supervision.json` records, per pathology, how many cells move from zero to non-zero weight and how much probabilities change. Stage 2 never writes another `weak_oof.csv` because those weak rows are now in-sample.

## 11. Full pre-run data gate

Before long GPU runs, `rsna-knee audit` measures:

- report-state counts per pathology;
- calibrated confidence distributions;
- gold-fold positive/negative counts;
- six-stream selection/missing rates;
- every selected training series' decode status;
- per-series and global partial file-decode failure rates.

The full pixel audit uses CPU processes and is a hard gate. Incomplete audits, undecodable selected series, or corruption rates above thresholds fail.

## 12. Sub-nine-hour runtime policy

Every long operation is an independent bounded job. Production uses `runtime_budget_hours: 8.5` with an additional final reserve.

Training does not estimate only the next epoch. The remaining-work model reserves time for:

```text
remaining Phase-B epochs
+ outer OOF TTA inference
+ Stage-1 weak OOF generation
+ bootstrap
+ loader startup / serialization
```

Inner validation supplies measured seconds-per-study, which updates the finish estimate. A conservative safety factor is applied. Prediction itself also checks the budget before each batch.

This closes the previous failure mode in which training could finish safely but the much larger Stage-1 weak-OOF inference could push the total run past the competition ceiling.

## 13. Checkpoint and submission identity

Production checkpoints include:

- outer fold;
- stage (`stage1` or `stage2`);
- model specification;
- stream order;
- saved training config;
- validation TTA offsets.

Final ensemble inference requires exactly the configured fold set (normally `{0,1,2}`), a single checkpoint stage, identical model/stream contracts, and checkpoint validation TTA equal to requested submission TTA.

## 14. Statistical reporting

Outer OOF keeps raw official NaNs and reports per-target AUC, macro AUC, study bootstrap intervals, and paired bootstrap differences.

`oof_center.csv` exists only to quantify the predeclared TTA policy. It must not be used to change submission TTA after seeing outer labels.

Once an outer OOF result is used to choose among final competition methods, call that result **model-selection cross-validation**, not an untouched independent estimate.

## Recommended execution order

```text
full data audit
  -> Stage-1 random fold-0 smoke
  -> remaining random smoke folds
  -> Stage-1 random production folds
  -> competition-data SSL
  -> Stage-1 SSL production folds
  -> per-outer-fold Stage-1 selection using inner AUC only
  -> Stage-2 fold-local co-training
  -> diagnostic/competition OOF comparisons
  -> freeze final stage
  -> stage/fold/TTA-validated one-pass submission
```

Do not add model scale until this baseline has real audit, smoke, OOF, runtime, and submission evidence.
