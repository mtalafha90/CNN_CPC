# CNN_CPC — RSNA Knee Abnormality Detection

Clean PyTorch pipeline for the **2026 RSNA Knee Abnormality Detection** challenge.

The repository now exposes one production path only. Historical model branches and runnable experiment configs were removed so training, validation and submission use the same data/model contract.

## Production design

```text
training report
  -> multilingual rule states
  -> fold-safe empirical calibration
  -> soft target + confidence

MRI study
  -> metadata repair
  -> physical DICOM ordering
  -> sagittal/coronal/axial × fluid/structural routing (up to six streams)
  -> uniformly sampled 2.5D [z-1,z,z+1] triplets
  -> shared pretrained ConvNeXt-Tiny
  -> target-specific attention over slices
  -> target-specific attention over MRI streams
  -> 12 study-level logits

objective
  -> confidence-weighted BCE
  -> small confidence-gated pairwise ranking term

validation
  -> official gold cells only
  -> NaNs preserved for unannotated cells
  -> macro ROC AUC + per-target AUC
  -> study bootstrap confidence interval
```

Radiology reports are used **only to create training supervision**. Submission inference is MRI-only.

This architecture is the repository's best-current engineering/methodology choice; it is not claimed to be leaderboard-optimal until real OOF experiments establish that.

## Repository layout

```text
configs/train.yaml                 single supported configuration
src/rsna_knee/
  calibration.py                   fold-safe report-teacher calibration
  cli.py                           command line interface
  constants.py                     targets/submission schema
  data.py                          CSV validation, folds, series routing
  dataset.py                       study-level DICOM dataset
  dicom.py                         DICOM decoding + 2.5D preprocessing
  dicom_meta.py                    metadata fallback inference
  evaluation.py                    NaN-safe AUC/bootstrap comparisons
  inference.py                     checkpoint-driven MRI-only inference
  model.py                         hierarchical ConvNeXt-Tiny MIL model
  preflight.py                     real-pixel DICOM audit
  report_labels.py                 multilingual rule teacher
  runtime.py                       single-device AMP/DataLoader runtime
  training.py                      leakage-safe fold training
kaggle/
  train_template.py
  submit_template.py
tests/
docs/
```

`docs/competition.md` is the preserved competition-description document. The public-code research review remains in `README_KAGGLE_METHODS.md` as background methodology, not as runnable architecture choices.

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

## Configure

Edit only the machine-specific paths in:

```text
configs/train.yaml
```

The default model settings are deliberately explicit and checkpointed. Inference reconstructs preprocessing/model settings from each checkpoint, preventing a stale YAML from changing the trained input contract.

## Inspect and preflight

```bash
rsna-knee inspect --data-root /path/to/rsna-knee-abnormality-detection

rsna-knee preflight \
  --data-root /path/to/rsna-knee-abnormality-detection \
  --split train \
  --sample-size 24
```

Preflight performs actual pixel decoding. It reports legitimate missing MRI streams separately from selected-stream path/decode failures; only the latter trip the strict failure threshold.

## Train all folds

```bash
for fold in 0 1 2; do
  rsna-knee train --config configs/train.yaml --fold "$fold"
done
```

Each fold writes:

- `best.pt` — self-describing model checkpoint;
- `oof.csv` — gold-fold predictions;
- `history.csv`;
- `config.json`;
- `fold_assignments.csv`;
- `calibration.json` when calibration is active;
- `metadata_repair.json`;
- `preflight.json`;
- `bootstrap.json`;
- `runtime.json`.

## Evaluate OOF

```bash
rsna-knee evaluate \
  --train-csv /path/to/train.csv \
  --oof runs/model/fold0/oof.csv \
        runs/model/fold1/oof.csv \
        runs/model/fold2/oof.csv \
  --out runs/model/evaluation.json
```

For a future controlled alternative, pass its OOF files with `--compare-oof` to obtain a paired bootstrap difference.

## Inference

```bash
rsna-knee infer \
  --config configs/train.yaml \
  --checkpoints runs/model/fold0/best.pt \
                runs/model/fold1/best.pt \
                runs/model/fold2/best.pt \
  --out submission.csv
```

Inference fails loudly for missing/incompatible checkpoints, mismatched stream ordering, duplicate IDs, unreadable required DICOM series, non-finite predictions or an invalid submission schema.

## Runtime and memory

```bash
rsna-knee runtime --config configs/train.yaml
```

The current runtime is intentionally **single-device**. It uses BF16 where natively supported, otherwise FP16 on CUDA, optimized DICOM DataLoader workers, TF32, bounded ConvNeXt encoder micro-batches, and gradient checkpointing to control memory.

`nn.DataParallel` was deliberately removed. **The next implementation step is proper multi-GPU DistributedDataParallel (DDP)** so data sharding, synchronization and validation aggregation are explicit and scalable.

## Correctness guarantees enforced in code

- Unlabeled target cells are never converted to negatives.
- Official gold cells override pseudo-labels target-by-target.
- Report-teacher calibration never uses the validation fold.
- Duplicate normalized reports cannot cross the fold's train/validation boundary.
- Validation uses raw official target cells with NaNs preserved.
- Test-time report fusion does not exist in the production path.
- Dual routing avoids duplicate fluid/structural series when alternatives exist.
- DICOM preflight distinguishes absent sequences from decode failures.
- Pretrained input normalization is preserved at checkpoint inference without re-downloading weights.

## Competition-rule caution

Before competition submission, verify the current Kaggle rules for pretrained/external weights and package availability. Do not upload competition DICOMs, reports, credentials or private data to this repository.

See `docs/data.md`, `docs/strategy.md`, `docs/references.md`, and the preserved `docs/competition.md` for supporting documentation.
