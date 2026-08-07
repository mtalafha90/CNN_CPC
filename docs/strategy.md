# Modeling strategy and experiment roadmap

## Baseline
The implemented baseline is a multi-series MRNet-style model: shared ResNet18 slice encoder, attention over slices, learned stream embeddings, attention over MRI streams, and a 12-logit multi-label head. Training uses confidence-weighted report pseudo-labels plus higher weight for the 58 gold cases; model selection is gold-only macro AUC.

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
