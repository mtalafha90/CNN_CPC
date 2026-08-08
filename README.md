# CNN_CPC — RSNA Knee Abnormality Detection

Production PyTorch pipeline for the **2026 RSNA Knee Abnormality Detection** challenge.

The repository exposes one production methodology: PU-aware report supervision, optional competition-data MRI SSL, 2.5D multi-sequence Transformer fusion, leakage-safe nested validation, and fold-local image/report co-training.

## Production contract

```text
SUPERVISION
report -> positive / negated / uncertain / unmentioned
       -> fold-safe calibration
       -> reliability = evidence × information beyond prevalence
       -> unmentioned = unlabeled (zero direct BCE by default)
       -> official gold cells override teacher cells

MRI
DICOM -> metadata repair -> six semantic streams
      -> stochastic 2.5D triplets
      -> ConvNeXt-Tiny slice encoder
      -> cross-sequence Transformer
      -> 12 interacting pathology queries
      -> 12 logits

LOSS
per-target confidence-weighted BCE
+ confidence-gated ranking loss
+ per-pathology ranking-pair diagnostics

VALIDATION
outer gold fold untouched
inner gold fold selects epoch count
fresh model retrains for fixed selected duration
outer gold fold evaluated once
study bootstrap macro-AUC

STAGE 2
outer fold k uses ONLY stage1/fold{k}/weak_oof.csv
wrong-fold weak teachers are rejected

INFERENCE
one GPU
CPU multiprocessing for DICOM/data work
all TTA views generated from one DICOM decode
all fold models consume the same decoded batch
exact submission.csv
```

## Competition execution policy

The production config is deliberately conservative:

- **one GPU only**;
- no DDP / no `torchrun`;
- CPU multiprocessing for DICOM/data preparation;
- `runtime_budget_hours: 8.5`, strictly below a 9 h notebook ceiling;
- Internet-independent runtime;
- external pretrained weights **off by default**;
- final file exactly `submission.csv`.

See `docs/competition_policy.md`. `docs/competition.md` remains preserved and is not modified by the execution-policy work.

## Repository layout

```text
configs/train.yaml
src/rsna_knee/
  audit.py          full teacher/fold/stream/DICOM audit
  budget.py         <9 h wall-clock guard
  calibration.py    fold-safe reliability-aware report calibration
  cli.py            production commands
  constants.py      targets and six-stream contract
  cotrain.py        fold-local leakage-safe image/report teacher
  data.py           CSV validation, folds, metadata repair, routing
  dataset.py        deterministic worker RNG, DICOM LRU, augmentation/TTA
  dicom.py          decoding, physical ordering, 2.5D sampling
  dicom_meta.py     DICOM-derived plane/contrast metadata
  evaluation.py     NaN-safe AUC/bootstrap/paired comparisons
  inference.py      one-pass fold ensemble/TTA inference
  model.py          ConvNeXt + MRI Transformer + pathology queries
  policy.py         competition execution safeguards
  preflight.py      sampled real-pixel safety audit
  report_labels.py  multilingual deterministic report teacher
  runtime.py        one GPU + CPU multiprocessing runtime
  sampling.py       trusted/general single-GPU batch sampler
  ssl.py            non-gold competition-data MRI SSL
  training.py       nested Stage-1/Stage-2 training
```

## Install

```bash
git clone https://github.com/mtalafha90/CNN_CPC.git
cd CNN_CPC
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
pytest -q
```

Edit machine-specific paths in `configs/train.yaml`.

## 1. Full data audit

Run this before expensive experiments:

```bash
rsna-knee audit \
  --config configs/train.yaml \
  --out-dir runs/audit
```

It reports:

- report-state counts for all 12 targets;
- calibrated confidence histograms;
- gold fold class counts;
- selected/missing six-stream counts;
- every selected series' DICOM decode status;
- candidate-file failures and partial-series corruption.

The audit uses CPU processes and must complete within the configured budget or it fails rather than presenting a partial audit as complete.

## 2. Optional competition-data SSL

Run SSL separately:

```bash
rsna-knee pretrain --config configs/train.yaml
```

Then attach the generated model artifact and set:

```yaml
ssl_encoder_checkpoint: /path/to/ssl_encoder.pt
```

The conservative production config starts ConvNeXt without external pretrained weights. Compare random initialization versus competition-data SSL on identical Stage-1 outer OOF before deciding whether SSL is beneficial.

## 3. Stage-1 smoke test

Run each fold separately:

```bash
rsna-knee train --config configs/train.yaml --fold 0 --smoke
rsna-knee train --config configs/train.yaml --fold 1 --smoke
rsna-knee train --config configs/train.yaml --fold 2 --smoke
```

Smoke mode caps selection to two epochs, shortens bootstrap, and uses a shorter runtime budget. It verifies the complete Phase-A -> Phase-B -> OOF -> weak-OOF path.

## 4. Stage-1 production training

Use **one fold per notebook/job**:

```bash
rsna-knee train --config configs/train.yaml --fold 0
```

Repeat in separate runs for folds 1 and 2. Do not put all folds into one Kaggle GPU notebook.

Each fold writes:

- `best.pt`;
- untouched outer-gold `oof.csv`;
- fold-local non-gold `weak_oof.csv`;
- `selection.json`;
- `history.csv`;
- `ranking_pairs.json`;
- calibration/sampling/preflight/runtime metadata;
- bootstrap uncertainty.

## 5. Stage-2 leakage-safe co-training

After all Stage-1 folds are available, use a new output directory and set only the Stage-1 root:

```yaml
output_dir: runs/cotrain
cotrain_stage1_root: /path/to/stage1/runs/model
```

For outer fold `k`, training automatically reads only:

```text
cotrain_stage1_root/fold{k}/weak_oof.csv
```

This is intentional. Using all three weak-OOF files for every outer fold would leak outer-gold information indirectly and is therefore rejected by the production API.

Run Stage-2 folds separately, again one run per fold.

## 6. Evaluate and compare

```bash
rsna-knee evaluate \
  --train-csv /path/to/train.csv \
  --oof runs/cotrain/fold0/oof.csv \
        runs/cotrain/fold1/oof.csv \
        runs/cotrain/fold2/oof.csv \
  --compare-oof runs/model/fold0/oof.csv \
                runs/model/fold1/oof.csv \
                runs/model/fold2/oof.csv \
  --out runs/cotrain/evaluation.json
```

Use paired bootstrap differences; do not infer an improvement from a tiny raw AUC change on 58 gold studies.

## 7. Submission inference

```bash
rsna-knee infer \
  --config configs/train.yaml \
  --checkpoints runs/cotrain/fold0/best.pt \
                runs/cotrain/fold1/best.pt \
                runs/cotrain/fold2/best.pt \
  --out submission.csv
```

Inference decodes each MRI series once per study, constructs all requested TTA views from that decode, and evaluates every fold model before releasing the batch. If projected multi-view runtime threatens the budget, it automatically falls back to the central slice view. It fails if even that projected path cannot finish safely.

## Kaggle templates

```text
kaggle/audit_template.py      separate audit run
kaggle/pretrain_template.py   separate SSL run
kaggle/train_template.py      exactly one fold per run
kaggle/submit_template.py     final one-pass submission run
```

## Methodological guarantees

- report silence is not a negative label;
- teacher confidence measures informativeness as well as calibration sample size;
- missing official target cells remain unknown;
- outer gold folds never choose their own epoch/model;
- Phase B reinitializes before outer evaluation;
- Stage-2 outer fold `k` can only use the Stage-1 fold-`k` weak teacher;
- duplicate normalized reports stay inside held-out report-group boundaries;
- BCE is normalized per pathology before macro averaging;
- ranking-pair utilization is recorded per pathology;
- missing sequence metadata stays unknown until DICOM backfill;
- worker RNG is deterministic for fixed seeds/settings;
- DICOM decoding has a bounded per-worker LRU cache;
- TTA does not multiply DICOM reads;
- model checkpoints self-describe the Transformer architecture;
- submission inference is image-only and Internet-independent.

The repository does not claim leaderboard superiority until the real audit, smoke runs, Stage-1 ablations, and official OOF comparisons have been executed.
