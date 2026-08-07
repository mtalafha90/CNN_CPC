# Modeling strategy and experiment roadmap

## Baseline
The implemented baseline is a multi-series MRNet-style model: shared ResNet18 slice encoder, attention over slices, learned stream embeddings, attention over MRI streams, and a 12-logit multi-label head. Training uses confidence-weighted report pseudo-labels plus higher weight for the 58 gold cases; model selection is gold-only macro AUC.

## Measuring a change honestly

Two pieces of machinery exist so that experiment comparisons mean something.

**Bootstrap intervals.** `rsna-knee evaluate --train-csv ... --oof runs/*/oof.csv` resamples the gold studies and reports a 95% interval alongside the macro AUC. At 58 studies that interval spans roughly 0.08, so a change of 0.01 is not a result. Add `--compare-oof` to compare two runs on the *same* resampled studies: pairing cancels the shared study-selection noise, so it detects real differences that two overlapping individual intervals would hide.

**Fold-safe teacher calibration.** With `calibrate_teacher: true` the fixed rule probabilities are replaced by `P(y = 1 | rule state)` learned from the gold studies outside the validation fold, smoothed towards each target's prevalence. Calibrating on all 58 and then validating on a subset of the same 58 makes validation optimistic; `calibration_split_mask` enforces the split. Each fold writes a `calibration.json` so the learned mapping can be inspected — it is also the quickest way to audit the teacher, since it shows directly what a "positive mention" is worth per target.

## Running locally on a GPU

`rsna-knee runtime` prints the resolved device, precision and worker count before you commit to a long run. Precision defaults to bf16 on Ampere or newer and fp16 on older cards; `num_workers` defaults to CPU cores minus one. DICOM decoding, not the GPU, is the bottleneck, so the worker count matters more than the backbone. With several GPUs, `data_parallel: true` splits each batch across them, and `batch_size` must be at least the number of cards. See `configs/local_gpu.yaml`.

## Recommended experiment order
1. **Audit the text teacher.** Measure per-class AUC and inspect errors against the 58 gold studies before trusting pseudo-labels.
2. **Establish an honest image baseline.** Train three gold-validation folds and save OOF predictions.
3. **Improve pseudo-labels.** If current competition rules permit, evaluate a stronger multilingual clinical-text teacher. Do not send restricted competition data to external APIs unless explicitly allowed.
4. **Improve MRI representations.** Compare random initialization with permitted pretrained or self-supervised MRI encoders.
5. **Compare sequence routing.** Test 3-stream best-series vs 6-stream fluid+structural routing and eventually target-specific routing.
6. **Backbone diversity.** Compare ConvNeXt, EfficientNetV2, Swin, and 2.5D/3D networks after the pipeline is verified.
7. **3D context.** Evaluate adjacent-slice 2.5D input before moving to more expensive full 3D networks.
8. **Ensembling.** Mean/rank-average diverse folds and backbones. Because the metric is AUC, ordering matters more than perfect calibration.

## Avoid
- treating unlabeled target cells as negatives;
- putting duplicate reports in both train and validation;
- tuning dozens of hyperparameters on only 58 gold cases;
- claiming unmeasured CV or leaderboard scores;
- committing competition DICOMs, reports, Kaggle credentials, or API keys.
