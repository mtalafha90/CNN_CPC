# CNN_CPC — RSNA Knee Abnormality Detection

Production PyTorch pipeline for the **2026 RSNA Knee Abnormality Detection** challenge.

The repository exposes one methodology: uncertainty-aware report supervision + in-domain MRI representation learning + multi-sequence 2.5D Transformer fusion + leakage-safe nested validation + cross-fitted co-training.

## Production methodology

```text
TRAINING SUPERVISION
radiology report
  -> multilingual states: positive / negated / uncertain / unmentioned
  -> fold-safe empirical-Bayes calibration
  -> confidence = evidence certainty × information beyond prevalence
  -> unmentioned = PU/unlabeled (zero direct BCE by default)
  -> official gold cells override every teacher target cell-by-cell

MRI REPRESENTATION
non-gold MRI studies
  -> optional in-domain SSL
  -> same-knee cross-sequence contrastive anatomy objective
  -> plane + fluid/structural metadata objectives
  -> SSL ConvNeXt encoder initialization

SUPERVISED MRI MODEL
study
  -> independent DICOM metadata repair
  -> physical slice ordering
  -> sagittal/coronal/axial × fluid/structural routing
  -> stochastic 2.5D triplets during training
  -> shared ConvNeXt-Tiny slice encoder
  -> slice + sequence embeddings
  -> Transformer across all active MRI slice tokens
  -> 12 interacting pathology query tokens
  -> pathology-to-MRI cross-attention
  -> 12 logits

OBJECTIVE
  -> confidence-weighted BCE normalized independently per pathology
  -> equal mean across the 12 targets (macro-metric aligned)
  -> confidence-gated pairwise AUC surrogate
  -> DDP global batch for meaningful rare-target ranking pairs

VALIDATION
  -> outer gold fold is never used for model/epoch selection
  -> inner gold fold chooses training duration
  -> model is reinitialized
  -> retrain for that fixed duration using all non-outer gold data
  -> evaluate outer fold once
  -> bootstrap macro-AUC uncertainty

CO-TRAINING
stage-1 fold model
  -> holds out ~1/3 of non-gold report groups
  -> writes weak_oof.csv
three weak_oof files
  + fold-calibrated report teacher
  -> agreement strengthens pseudo-labels
  -> disagreement is downweighted
  -> stage-2 image student

INFERENCE
  -> MRI only
  -> three-fold ensemble
  -> deterministic slice-center TTA
```

## Repository layout

```text
configs/train.yaml
src/rsna_knee/
  calibration.py   fold-safe reliability-aware report calibration
  cli.py           commands
  constants.py     targets and canonical six-stream contract
  cotrain.py       cross-fitted image/report consensus teacher
  data.py          CSV validation, folds, metadata repair, routing
  dataset.py       stochastic MRI study dataset and augmentation
  dicom.py         decoding, physical ordering, 2.5D sampling
  dicom_meta.py    DICOM-derived plane/contrast metadata
  evaluation.py    NaN-safe AUC/bootstrap/paired comparisons
  inference.py     versioned checkpoint reconstruction + TTA
  model.py         ConvNeXt + MRI Transformer + pathology queries
  preflight.py     real-pixel safety audit
  report_labels.py multilingual deterministic report teacher
  runtime.py       AMP + NCCL DistributedDataParallel runtime
  sampling.py      globally sharded trusted/general batch sampler
  ssl.py           non-gold in-domain MRI self-supervision
  training.py      nested CV, DDP, co-training workflow
```

`docs/competition.md` is intentionally preserved as the competition-description document. `README_KAGGLE_METHODS.md` remains the public-method review/background.

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

Edit machine paths in `configs/train.yaml`.

## 1. Inspect and preflight

```bash
rsna-knee inspect --data-root /path/to/rsna-knee-abnormality-detection
rsna-knee preflight --data-root /path/to/rsna-knee-abnormality-detection --split train --sample-size 24
```

Preflight performs actual DICOM pixel decoding and the real production 2.5D transform. Missing protocol streams are reported separately from selected-stream decode/path failures.

## 2. Optional in-domain SSL

SSL uses **non-gold studies only** by default.

Single GPU:

```bash
rsna-knee pretrain --config configs/train.yaml
```

Multiple GPUs:

```bash
torchrun --standalone --nproc_per_node=4 -m rsna_knee.cli pretrain --config configs/train.yaml
```

Then set:

```yaml
ssl_encoder_checkpoint: runs/ssl/ssl_encoder.pt
```

## 3. Stage-1 nested/cross-fit training

Single GPU:

```bash
for fold in 0 1 2; do
  rsna-knee train --config configs/train.yaml --fold "$fold"
done
```

Multi-GPU example:

```bash
for fold in 0 1 2; do
  torchrun --standalone --nproc_per_node=4 \
    -m rsna_knee.cli train --config configs/train.yaml --fold "$fold"
done
```

Each fold writes `best.pt`, outer-gold `oof.csv`, non-gold `weak_oof.csv`, `selection.json`, `history.csv`, calibration/sampling/preflight/runtime metadata, and bootstrap uncertainty.

## 4. Stage-2 cross-fitted co-training

After all stage-1 folds finish, change to a new output directory and supply all three independent weak-study prediction files:

```yaml
output_dir: runs/cotrain
cotrain_image_oof:
  - runs/model/fold0/weak_oof.csv
  - runs/model/fold1/weak_oof.csv
  - runs/model/fold2/weak_oof.csv
```

Then rerun all three folds. The image teacher is cross-fitted: every non-gold prediction came from a model that excluded that study's normalized report group.

## 5. Evaluate untouched outer OOF

```bash
rsna-knee evaluate \
  --train-csv /path/to/train.csv \
  --oof runs/cotrain/fold0/oof.csv \
        runs/cotrain/fold1/oof.csv \
        runs/cotrain/fold2/oof.csv \
  --out runs/cotrain/evaluation.json
```

Use `--compare-oof` for paired bootstrap comparison against the stage-1 model.

## 6. MRI-only submission inference

```bash
rsna-knee infer \
  --config configs/train.yaml \
  --checkpoints runs/cotrain/fold0/best.pt \
                runs/cotrain/fold1/best.pt \
                runs/cotrain/fold2/best.pt \
  --out submission.csv
```

The default TTA averages slice-center offsets `[-1, 0, +1]`. Reports are never required at inference.

## Methodological guarantees

- Report silence is not treated as a negative label.
- Teacher confidence measures informativeness as well as calibration sample size.
- Missing official target cells remain unknown.
- Gold labels override pseudo-labels target-by-target.
- Outer gold folds never select their own epoch or model.
- Final outer-fold models retrain on all non-outer gold cases after inner selection.
- Duplicate normalized reports cannot cross held-out report-group boundaries.
- Weak-study image predictions used for co-training are cross-fitted.
- BCE gives equal aggregate weight to each pathology.
- DDP ranking operates on the differentiably gathered global batch.
- DDP trusted/general sampling is generated globally and sharded without independent-rank duplication.
- Missing fluid/fat metadata remains unknown until DICOM backfill.
- Submission checkpoints explicitly record the Transformer/pathology architecture.
- Submission inference is image-only.

## Competition-rule caution

Verify current competition rules before using pretrained/external weights. The repository does not claim leaderboard performance until the real data runs and OOF evaluation are completed.
