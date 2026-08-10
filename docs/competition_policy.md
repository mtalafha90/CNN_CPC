# Competition execution policy

`docs/competition.md` is the preserved competition-description document. This file describes the **conservative execution policy enforced by the current code and experiment workflow**.

> **Snapshot: 2026-08-10.** Package `0.13.0`. B7.1 full-corpus weak supervision is the current best standalone development model at macro AUC `0.5644802945`. The fixed B5+B7.1 rank ensemble is rejected. **B8 spatial-anatomy learning is currently training; no B8 gold score has been recorded yet.** Current scores are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

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
- competition-data checkpoint provenance checked;
- final inference MRI-only.

## Verified local environment

```text
device    NVIDIA RTX A4500 Laptop GPU
precision bf16
GPU count 1
```

The real-data pipeline has completed DICOM preflight/audit and multiple full development experiments.

## Runtime protection

Long jobs are independent bounded stages. Runtime estimation reserves time not only for the next training step but also for required finishing work such as prediction, bootstrap and serialization.

Training checkpoints are written after completed epochs where the experiment implementation supports it. A budget-limited run must be reported as such rather than silently treated as the predefined full recipe.

## External-pretraining policy

Do not enable a public image checkpoint or external language model merely because it is available. Verify the exact competition allowance first and record provenance.

Current representation/weak-supervision path uses only competition training data:

- strong SSL: competition MRI only;
- B5 MRI branch: competition MRI only;
- B5 text branch: competition reports only, TF-IDF -> TruncatedSVD;
- B6: competition reports only, no external language model/resource;
- B7/B7.1/B8: competition MRI + frozen B6 competition-report supervision;
- no ImageNet weights in the conservative path;
- no external clinical language model.

## Report supervision policy

Reports are training-only information.

Safeguards:

- report silence is not a negative;
- B6 states are positive / negated / uncertain / unmentioned;
- B6 v1.2.1 is frozen after its gold audit;
- B7/B7.1/B8 use only confidence `>=0.75` positive/negated cells;
- global positive soft target/weight = `0.85 / 0.50`;
- global negated soft target/weight = `0.05 / 1.00`;
- uncertain/unmentioned cells are ignored;
- gold studies are excluded from B6 `training_targets.csv`;
- B7/B7.1/B8 do not use gold labels in gradient or early stopping.

The B6 gold audit informed the global B7 policy, so later B7/B7.1/B8 scores on the same 58 studies are explicitly development/model-selection estimates.

## Gold-validation policy

The 58 official gold studies are the scientific development resource. For individual candidates, leakage protections follow the experiment's predefined protocol.

However, many sequential method decisions have now been made from the same 58-study set. Therefore campaign-level results must be described as **model-selection cross-validation**, not pristine independent hidden-test estimates.

Do not:

- tune ensemble weights on the 58 gold labels;
- select target-specific post-hoc model winners;
- create more B4 selector variants from observed outer performance;
- retune B5 report-alignment hyperparameters from its completed gold probe;
- retune B6 parser rules from its gold audit or later B7/B8 target results;
- tune target-specific B7/B8 weak-label weights from observed gold AUCs;
- search B8 spatial grid size, anatomy-prior strength, target-specific priors or extra epochs after the first B8 gold score and still call it B8-v1;
- call a development OOF/gold result a leaderboard result.

## Current model-selection decisions

Retained roles:

```text
B4    image-only frozen representation ablation
B5    report-aligned representation baseline / B7 encoder source
B6    frozen structured weak-label source
B7-v1 direct weak-supervision coverage ablation
B7.1 current best standalone development model
B8    active predeclared spatial-anatomy experiment
```

Closed branches:

```text
B4 selector redesign
B5/B7.1 ensemble-weight search
raw-vs-rank blend search
target-specific model mixtures
post-audit B6 parser tuning from gold
```

## B8 execution policy

B8 must initialize from the completed named B7.1 full-coverage checkpoint and preserve the frozen weak-supervision/training contract.

Frozen B8-v1 architecture direction:

```text
B7.1 MRI memory = 6 x 16 x 1    = 96 tokens/study
B8 MRI memory   = 6 x 16 x 2x2  = 384 tokens/study
```

B8 adds coarse spatial tokens and gentle fixed pathology stream/slice attention priors. It does not hard-code medial/lateral/anterior/posterior in-plane quadrants because canonical pixel orientation is not guaranteed by the current preprocessing contract.

Current B8 status:

```text
implementation complete
real-data training in progress
gold evaluation not yet run
benchmark B7.1 = 0.5644802945
```

When training completes, inspect `history.json` and `supervision_plan.json` before gold evaluation.

## TTA policy

For neural checkpoint inference, validation and submission TTA contracts are stored/checked explicitly. Diagnostic center-only evaluation is not permission to retune TTA after reading gold labels.

B7.1/B8 use the predeclared three-view center offsets `[-1,0,1]` for gold development evaluation.

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
- **audit** — full data-quality or weak-label inventory;
- **training run** — optimization on the predefined training data;
- **gold development result** — result on the repeatedly used 58-study set;
- **paired comparison** — study-level bootstrap comparison of aligned predictions;
- **model-selection CV** — campaign-level use of the 58 studies after they inform method choice;
- **leaderboard result** — actual Kaggle submission score;
- **training in progress** — current B8 state; no final B8 score exists yet.

See [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) for the current measured table.
