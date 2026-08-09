# Competition execution policy

`docs/competition.md` is the preserved competition-description document. This file describes the **conservative execution policy enforced by the current code and experiment workflow**.

> **Snapshot: 2026-08-09.** B0-B4.3 and fixed B1/B4 ensembles have completed; B5 representation training is running. Current scores are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Conservative defaults

The active competition rules remain the authority. The repository defaults to the stricter option whenever a permission has not been independently verified.

```yaml
requested_gpus: 1
runtime_budget_hours: 8.5
runtime_reserve_minutes: 10
pretrained: false
allow_external_pretrained: false
```

Additional defaults:

- no DDP / no `torchrun` in the production competition path;
- CPU multiprocessing only for DICOM/data work;
- Internet-independent final inference;
- output exactly `submission.csv`;
- competition-data SSL checkpoint provenance checked;
- final inference MRI-only.

## Verified local environment

```text
device    NVIDIA RTX A4500 Laptop GPU
precision bf16
GPU count 1
```

The real-data pipeline has completed DICOM preflight/audit and multiple non-smoke OOF experiments.

## Runtime protection

Long jobs are independent bounded stages. Runtime estimation reserves time not only for the next training step but also for required finishing work such as prediction, bootstrap and serialization.

Prediction checks the deadline batch by batch. A run that cannot safely complete the next unit of work stops cleanly rather than intentionally exceeding the configured budget.

## External-pretraining policy

Do not enable a public image checkpoint or external language model merely because it is available. Verify the exact competition allowance first and record provenance.

Current strong SSL and B5 use only competition training data:

- strong SSL: competition MRI only;
- B5 MRI branch: competition MRI only;
- B5 text branch: competition reports only, TF-IDF -> TruncatedSVD;
- no ImageNet weights;
- no external clinical language model.

## Report supervision policy

Reports are training-only information.

Safeguards:

- report silence is not a negative;
- finite official labels override weak labels;
- calibration is fold-safe where gold labels are used;
- OA parsing is compartment-aware;
- ordinary weak labels are not promoted to trusted merely to increase counts;
- B5 excludes all gold studies from report/MRI representation training.

## Gold-validation policy

The 58 official gold studies are the scientific validation resource. For individual candidates, outer folds are protected according to the experiment's predefined protocol.

However, many sequential method decisions have now been made from the same 58-study OOF set. Therefore the campaign-level result must be described as **model-selection cross-validation**, not a pristine independent hidden-test estimate.

Do not:

- tune ensemble weights on the 58 gold labels;
- select target-specific post-hoc model winners;
- create more B4 selector variants from observed outer performance;
- tune B5 after reading its OOF without declaring a new experiment;
- call an OOF result a leaderboard result.

## B4/B5 controlled-comparison policy

B4 is currently the best clean standalone point estimate at `0.5137567459` macro AUC.

B5 changes the representation only. Its first evaluation must reuse the original B4 frozen-feature/classifier probe unchanged. This avoids confounding representation improvement with another downstream hyperparameter search.

B5 currently has **no performance result** because training/probing has not completed.

## TTA policy

For neural checkpoint inference, validation and submission TTA contracts are stored/checked explicitly. Diagnostic center-only OOF is not permission to retune TTA after reading outer labels.

Frozen B4/B5 representation probes use their own deterministic extraction contract and must be compared like-for-like.

## DICOM quality policy

Verified audit:

```text
21,886 / 21,886 selected series decoded
732,554 / 732,556 candidate files decoded
2 partial one-file failures
0 selected series lost
```

Partial corruption is permitted only below configured per-series/global thresholds. A fully undecodable required selected series remains a hard failure.

## Submission schema

The final competition file must contain exactly:

```text
StudyInstanceUID + 12 target columns
```

with the expected study set/order and finite probabilities in `[0,1]`.

Default output:

```text
/kaggle/working/submission.csv
```

## Reporting vocabulary

Use these labels accurately:

- **preflight** — data-path/DICOM technical gate;
- **audit** — full data-quality inventory;
- **smoke** — short engineering test;
- **OOF result** — completed cross-validation predictions;
- **model-selection CV** — OOF after it has informed method choice;
- **leaderboard result** — actual Kaggle submission score;
- **running/pending** — implemented experiment without completed evaluation, currently B5.

See [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) for the current measured table.
