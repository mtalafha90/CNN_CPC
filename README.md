# CNN_CPC — RSNA Knee Abnormality Detection

Production PyTorch pipeline for the **2026 RSNA Knee Abnormality Detection** challenge.

The repository exposes one conservative production methodology: PU-aware report supervision, optional competition-data MRI SSL, 2.5D multi-sequence Transformer fusion, nested validation, fold-local image/report co-training, one-GPU execution, and explicit sub-nine-hour runtime protection.

## Production contract

```text
SUPERVISION
report -> positive / negated / uncertain / unmentioned
       -> fold-safe calibration
       -> confidence = evidence × information beyond prevalence
       -> unmentioned = unlabeled by default
       -> official finite gold cells override weak supervision

MRI
DICOM -> metadata repair -> six semantic streams
      -> stochastic 2.5D triplets
      -> ConvNeXt-Tiny encoder
      -> cross-sequence Transformer
      -> 12 interacting pathology queries
      -> 12 logits

LOSS
planned-epoch per-target weighted BCE
+ confidence-gated ranking auxiliary
+ effective supervision/ranking diagnostics

VALIDATION
outer gold fold = untouched OOF evaluation
inner gold fold = epoch-count selection
Phase A discarded
fresh Phase B retrain for selected epoch count
validation TTA == submission TTA
OOF bootstrap macro-AUC

STAGE-1 CANDIDATES
random vs competition-data SSL
outer fold k candidate chosen by fold-k INNER AUC only
outer OOF is never used to choose the Stage-1 teacher for that fold

STAGE 2
Phase A = report-only epoch selection
Phase B = fresh report + fold-local cross-fitted image teacher
very confident image teacher may modestly supervise report-silent cells
wrong-fold/incomplete/legacy-incompatible teachers are rejected
Stage 2 never exports another weak_oof.csv

RUNTIME
one GPU
CPU multiprocessing for DICOM/data work
8.5 h software budget
training and prediction batch deadlines
measured finish-time reserve includes outer OOF + weak OOF + bootstrap + serialization

INFERENCE
exactly folds {0,1,2}
one checkpoint stage only
checkpoint validation TTA must equal requested submission TTA
one DICOM decode supplies all requested TTA views
exact submission.csv
```

## Competition execution policy

The production configuration is intentionally stricter than unverified allowances:

- **one GPU only**;
- no DDP / no `torchrun`;
- CPU multiprocessing for data work;
- `runtime_budget_hours: 8.5`, below the 9 h ceiling supplied for this workflow;
- Internet-independent submission runtime;
- external pretrained weights **off by default**;
- competition-data SSL provenance is checked;
- final output is exactly `submission.csv`.

See `docs/competition_policy.md`. `docs/competition.md` is preserved separately and is not modified by production-policy changes.

## Repository layout

```text
configs/train.yaml
src/rsna_knee/
  audit.py          full teacher/fold/stream/DICOM audit
  budget.py         absolute work/hard deadline and remaining-work guard
  calibration.py    fold-safe report calibration
  cli.py            production commands + nested Stage-1 selector
  constants.py      target and stream contracts
  cotrain.py        fold-safe candidate selection and image/report fusion
  data.py           CSV validation, folds, metadata repair, routing
  dataset.py        worker RNG, DICOM LRU, augmentation/TTA
  dicom.py          decoding, slice ordering, 2.5D preprocessing
  dicom_meta.py     plane/contrast metadata recovery
  evaluation.py     NaN-safe AUC/bootstrap/paired comparison
  inference.py      fold/stage/TTA-validated ensemble inference
  model.py          ConvNeXt + MRI Transformer + pathology queries
  policy.py         competition execution safeguards
  preflight.py      sampled real-pixel DICOM gate
  report_labels.py  deterministic multilingual report teacher
  runtime.py        one GPU + CPU multiprocessing runtime
  sampling.py       trusted/general deterministic batch sampler
  ssl.py            competition-data MRI SSL
  training.py       nested Stage-1/Stage-2 production training
```

## Install and test

```bash
git clone https://github.com/mtalafha90/CNN_CPC.git
cd CNN_CPC
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
pytest -q
```

The test suite includes unit contracts, a two-process DataLoader test, checkpoint round-trip validation, fold/stage/TTA inference-contract tests, and a synthetic DICOM -> nested train-fold -> OOF/artifact integration test.

For real data, follow **`docs/LOCAL_REAL_DATA_TRAINING.md`** rather than improvising commands.

## 1. Full data audit

```bash
rsna-knee audit --config configs/train.yaml --out-dir runs/audit
```

The audit reports report-state/confidence coverage, gold fold balance, six-stream availability, every selected series' decode status, and per-series/global partial DICOM corruption. Incomplete audits or failures beyond configured gates raise errors.

## 2. Stage-1 random baseline

Run one fold per job. Start with fold-0 smoke:

```bash
rsna-knee train --config configs/train.yaml --fold 0 --smoke
```

Then run production folds sequentially:

```bash
rsna-knee train --config configs/train.yaml --fold 0
rsna-knee train --config configs/train.yaml --fold 1
rsna-knee train --config configs/train.yaml --fold 2
```

Each Stage-1 fold writes at least:

- `best.pt`;
- `oof.csv` — primary TTA OOF, using the same offsets as submission;
- `oof_center.csv` — center-only diagnostic;
- `weak_oof.csv` — cross-fitted non-gold image teacher;
- `selection.json`;
- `history.csv`;
- `training_diagnostics.json`;
- `supervision_plan.json`;
- `bootstrap.json`;
- `runtime.json` and data/calibration metadata.

`training_diagnostics.json` contains per-pathology ranking-pair counts and effective supervision participation/weight mass.

## 3. Optional competition-data SSL Stage-1 candidate

```bash
rsna-knee pretrain --config configs/train.yaml
```

Attach the resulting encoder in a second Stage-1 config and train all three folds using the **same fold definitions and validation TTA policy** as random initialization.

Do not choose random vs SSL globally from the three outer OOF folds and then reuse that decision inside the same OOF estimate.

## 4. Leakage-safe Stage-1 candidate selection

After random and SSL Stage-1 folds exist:

```bash
rsna-knee select-stage1 \
  --candidate-root /path/to/runs/stage1_random \
  --candidate-root /path/to/runs/stage1_ssl \
  --n-folds 3 \
  --out runs/stage1_selection.json
```

For outer fold `k`, this command uses only that candidate fold's `inner_macro_auc`. `outer_macro_auc` is deliberately ignored. Candidates must have the same inner-fold and validation-TTA contracts.

## 5. Stage-2 leakage-safe co-training

Recommended candidate configuration:

```yaml
output_dir: runs/stage2
cotrain_stage1_root: null
cotrain_stage1_candidates:
  - /path/to/runs/stage1_random
  - /path/to/runs/stage1_ssl
```

For each outer fold, the appropriate Stage-1 candidate is selected from inner AUC only. Only its fold-local `fold{k}/weak_oof.csv` may enter Stage-2 fold `k`.

Stage-2 Phase A remains report-only. Fresh Phase B uses all non-outer gold plus the independent fold-local image teacher. Strong image/report agreement gets high weight; conflict is suppressed; a very confident cross-fitted image prediction may receive a modest BCE weight when report confidence is near zero.

Every Stage-2 fold writes `stage2_supervision.json`, including `zero_to_nonzero_weight` counts per pathology, so additional image-only supervision is measured rather than assumed.

Stage 2 intentionally does **not** write another `weak_oof.csv`.

## 6. OOF evaluation

Primary `oof.csv` already uses submission-matched TTA. `oof_center.csv` is only a diagnostic comparison and must not be used to retroactively change the algorithm after looking at outer labels.

```bash
rsna-knee evaluate \
  --train-csv /path/to/train.csv \
  --oof runs/stage2/fold0/oof.csv \
        runs/stage2/fold1/oof.csv \
        runs/stage2/fold2/oof.csv \
  --out runs/stage2/evaluation.json
```

Use paired bootstrap for controlled comparisons. Once an outer OOF comparison is used to choose the final competition method, treat the resulting score as **model-selection CV**, not a pristine independent generalization estimate.

## 7. Final inference

A final inference config should declare the expected stage, for example:

```yaml
expected_checkpoint_stage: stage2
```

Then:

```bash
rsna-knee infer \
  --config configs/final_infer.yaml \
  --checkpoints runs/stage2/fold0/best.pt \
                runs/stage2/fold1/best.pt \
                runs/stage2/fold2/best.pt \
  --out submission.csv
```

Inference rejects duplicate/missing folds, mixed checkpoint stages, mismatched architecture/stream order, and any checkpoint whose validation TTA differs from requested submission TTA. Runtime prediction is guarded batch-by-batch; if configured to allow it, a time-critical submission may fall back to the center view to guarantee completion.

## Kaggle templates

```text
kaggle/audit_template.py      separate CPU audit run
kaggle/pretrain_template.py   separate SSL run
kaggle/train_template.py      exactly one Stage-1/Stage-2 fold per run
kaggle/submit_template.py     final stage-validated submission run
```

## Methodological guarantees

- report silence is not converted into a negative;
- finite official labels override weak labels cell-by-cell;
- outer gold never selects its own epoch;
- Phase B reinitializes before outer evaluation;
- random-vs-SSL fold selection uses inner AUC only;
- Stage-2 Phase A never uses the image teacher;
- Stage-2 outer fold `k` can only consume safe fold-`k` weak OOF predictions;
- Stage-2 image-teacher rows are not re-exported as OOF;
- validation TTA is predeclared and matches submission TTA;
- planned-epoch BCE balances pathology supervision mass;
- ranking/effective-supervision utilization is recorded per pathology;
- normalized duplicate reports stay in the same held-out report group;
- training and prediction have explicit wall-clock guards;
- Stage-1 finish budgeting includes weak OOF generation;
- DICOM decode failures are audited before long GPU work;
- checkpoints self-describe stage/fold/architecture/TTA contract;
- submission inference is image-only and Internet-independent.

The repository does **not** claim leaderboard superiority or a 9.5/10 empirical model score until real-data audit, smoke runs, OOF experiments, runtime measurements, and the actual competition submission have been executed.
