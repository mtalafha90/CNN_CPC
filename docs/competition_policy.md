# Competition execution policy

`docs/competition.md` is the preserved competition-description document. This file describes the **conservative execution policy enforced by the current code**.

## Authority and conservative defaults

The active Kaggle competition rules are the authority. The repository deliberately defaults to a stricter configuration whenever an allowance has not been independently verified.

Current production defaults:

- one GPU only;
- no DDP and no `torchrun`;
- CPU multiprocessing only for DICOM/data work;
- `runtime_budget_hours: 8.5`;
- `runtime_reserve_minutes: 10`;
- Internet-independent training/inference where required by the execution environment;
- final output exactly `submission.csv`;
- external pretrained weights disabled by default;
- optional SSL checkpoints must have permitted provenance;
- validation TTA must match requested submission TTA.

These are checked by `rsna_knee.policy`, `rsna_knee.budget`, training-time checkpoint contracts, and inference validation.

## Verified local execution environment

The paired-sampler smoke run on 2026-08-08 resolved to:

```text
device    NVIDIA RTX A4500 Laptop GPU
precision bf16
GPU count 1
```

The smoke completed the full Stage-1 path: preflight, nested selection, fresh retraining, outer OOF, weak OOF, bootstrap, serialization and checkpoint writing.

This is an engineering verification only. Production runtime and AUC are reported from non-smoke folds after they finish.

## Runtime decomposition

Do not run the entire research workflow as one long notebook/job. Treat major stages as independent bounded runs:

1. full data audit;
2. optional competition-data SSL;
3. Stage-1 fold 0;
4. Stage-1 fold 1;
5. Stage-1 fold 2;
6. Stage-2 fold 0;
7. Stage-2 fold 1;
8. Stage-2 fold 2;
9. final inference.

Each GPU training/inference run is independently protected by the configured sub-nine-hour software budget.

## Finish-time protection

Training does not reserve only enough time for the next epoch. It estimates the remaining complete workflow, including:

```text
remaining Phase-B retraining
+ outer OOF inference
+ Stage-1 weak OOF inference
+ bootstrap
+ DataLoader startup
+ checkpoint/JSON/CSV serialization
```

The estimator is updated from measured prediction speed. Prediction itself checks the runtime guard before each batch.

The point of this policy is to prevent a model from finishing training but exceeding the competition time ceiling while generating OOF or submission predictions.

## One-GPU rule

In competition mode:

```yaml
requested_gpus: 1
```

The code rejects incompatible multi-GPU assumptions. CPU workers may still decode and preprocess DICOM data in parallel.

## External pretrained weights

Conservative defaults:

```yaml
pretrained: false
allow_external_pretrained: false
```

Do not enable a public external checkpoint merely because it is technically downloadable. First verify that the exact current competition rules permit that source and use, then document the source and provenance.

Competition-data self-supervision is handled separately through the repository's SSL path.

## Report supervision policy

Reports are training-only supervision. They are not required for final inference.

Important safeguards:

- report silence receives zero direct weight by default;
- official finite target cells override weak labels;
- report-state calibration is fold-safe;
- OA parsing is compartment-aware;
- ordinary weak labels do not become trusted merely to increase batch counts;
- `trusted_pseudo_threshold` and ranking confidence gates remain explicit configuration values.

## Trusted-pair sampling policy

The production batch size is currently 2. To allow the confidence-gated ranking auxiliary to operate, the sampler groups trusted rows in pairs for even batch sizes while preserving the requested trusted-row fraction.

This changes minibatch composition, not the trust definition. It does **not** lower:

```yaml
trusted_pseudo_threshold: 0.60
rank_min_confidence: 0.35
```

The corrected fold-0 smoke produced nonzero ranking pairs for all 12 targets.

## Stage-1 leakage rule

For outer fold `k`:

- outer gold fold `k` is never used for epoch selection;
- inner gold selects the training duration;
- Stage-1 non-gold `crossfit_fold=k` rows are excluded and later predicted as weak OOF;
- Phase B starts from a fresh model.

## Random-versus-SSL candidate rule

If both Stage-1 random and Stage-1 SSL candidates exist, the candidate for outer fold `k` must be chosen from **fold-`k` inner AUC only**.

Outer AUC is not an allowed candidate-selection signal for that fold.

## Stage-2 leakage rule

For Stage-2 outer fold `k`, the only permitted image teacher is a safe Stage-1 fold-`k` weak OOF file:

```text
<stage1_root>/fold{k}/weak_oof.csv
```

The loader rejects:

- wrong-fold predictions;
- missing expected weak rows;
- extra unsafe rows;
- non-Stage-1 sources;
- incompatible validation-TTA contracts.

Stage-2 Phase A remains report-only. The image teacher is enabled only in the fresh Phase-B co-training model.

Stage 2 does not export another `weak_oof.csv` because those teacher rows have become in-sample training rows.

## Validation/submission TTA contract

Default:

```yaml
tta_center_offsets: [-1, 0, 1]
validation_tta_offsets: [-1, 0, 1]
```

A checkpoint stores its validation offsets. Final inference rejects checkpoints whose validation TTA differs from the requested submission policy unless an explicitly supported runtime fallback is invoked.

`oof_center.csv` is a diagnostic file and is not an authorization to retune TTA from outer labels.

## Checkpoint identity

Production checkpoints include enough metadata to reconstruct and validate the ensemble contract:

- model state;
- model specification;
- stream order;
- training config;
- outer fold;
- inner fold;
- stage;
- selected epoch;
- validation TTA offsets;
- Stage-1 teacher source metadata when relevant.

Final inference requires the expected fold set and one checkpoint stage.

## DICOM quality policy

Real-data audit results have verified that the current data are well within configured decode limits:

```text
21,886 / 21,886 selected series decoded
2 failed files out of 732,556 candidate files
0 selected series lost
```

Small partial corruption is permitted only below the configured per-series and global gates. A fully undecodable selected series remains a hard failure.

## Submission path

The final Kaggle template writes:

```text
/kaggle/working/submission.csv
```

The file must contain exactly:

```text
StudyInstanceUID + 12 target columns
```

with finite probabilities in `[0,1]` and the expected test study set/order.

## Reporting policy

Do not convert engineering smoke results into scientific or leaderboard claims.

Use the following labels accurately:

- **preflight result** — DICOM/data-path gate;
- **audit result** — full data-quality/supervision inventory;
- **smoke result** — short end-to-end software/GPU test;
- **OOF result** — completed validation prediction;
- **model-selection CV** — OOF after it has been used to choose the final method;
- **leaderboard result** — actual Kaggle submission score.

Production results remain pending until the corresponding non-smoke runs complete.