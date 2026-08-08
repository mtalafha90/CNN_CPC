# Competition execution policy

`docs/competition.md` is the preserved competition-description document. This file describes the **conservative execution policy implemented by the code**.

## Why conservative defaults

The active Kaggle competition page is the authority. The repository therefore defaults to settings that remain safe even when a competition-specific allowance has not been independently verified from the rule text.

Production defaults are:

- one GPU only;
- no `torchrun`, DDP, or multi-GPU execution;
- CPU multiprocessing only for DICOM/data preparation;
- wall-clock budget `8.5 h`, strictly below a `9 h` GPU notebook ceiling;
- ten-minute runtime reserve;
- Internet-independent runtime;
- output file name exactly `submission.csv`;
- external pretrained weights disabled by default;
- self-produced SSL/checkpoints may use only supplied competition training data unless the current rules explicitly allow more.

The policy is enforced by `rsna_knee.policy` and `rsna_knee.budget`; it is not merely documentation.

## Runtime decomposition

Do **not** run the full research workflow in one Kaggle notebook. Use separate committed runs:

1. full data audit;
2. optional competition-data SSL;
3. Stage-1 fold 0;
4. Stage-1 fold 1;
5. Stage-1 fold 2;
6. Stage-2 fold 0;
7. Stage-2 fold 1;
8. Stage-2 fold 2;
9. final submission inference.

Each run has its own `<9 h` wall-clock guard.

## Stage-2 leakage rule

For Stage-2 outer fold `k`, the only allowed image teacher is:

```text
stage1_root/fold{k}/weak_oof.csv
```

The corresponding Stage-1 model excluded both:

- outer-gold fold `k`; and
- non-gold `crossfit_fold=k` studies.

Predictions from Stage-1 folds `j != k` are rejected for Stage-2 fold `k` because those models may have trained on outer-gold fold `k`.

## Pretrained weights

`pretrained: false` and `allow_external_pretrained: false` are the production defaults. If the exact current competition-specific rules are later verified to permit a particular public pretrained model, that allowance should be documented with its source before changing these flags.

## Submission path

The final template writes:

```text
/kaggle/working/submission.csv
```

Inference reconstructs the architecture from checkpoint metadata and requires no report text or Internet connection.
