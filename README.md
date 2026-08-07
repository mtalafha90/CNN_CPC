# CNN_CPC — RSNA Knee Abnormality Detection

Reproducible PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** Kaggle competition.

> **Competition status (snapshot: 2026-08-07):** the competition is still open. This repository therefore avoids winner claims and records only measured OOF/leaderboard results.

## Documentation

- [Public competition methodology review](README_KAGGLE_METHODS.md)
- [Current repository technical review](docs/REPO_REVIEW_2026-08-07.md)
- [Competition notes](docs/competition.md)
- [Dataset handling](docs/data.md)
- [Experiment strategy](docs/strategy.md)
- [References](docs/references.md)

## Core problem

Each MRI study contains multiple DICOM series and must receive 12 probabilities:

`ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`.

The metric is **macro ROC-AUC across the 12 targets**.

The important supervision structure is unusual:

- 4,407 training studies;
- only 58 studies with explicit gold labels;
- 4,349 report-supervised studies;
- multi-plane/multi-sequence MRI metadata in `train_series.csv`.

The pipeline therefore treats radiology reports as a **training teacher**, while validation is strictly gold-only and final inference is image-only by default.

## Current architecture and capabilities

The baseline remains intentionally simple, but the same pipeline now supports controlled upgrades:

```text
Radiology report (training only)
  -> multilingual rule states
  -> fold-safe empirical calibration
  -> soft targets + per-target confidence

MRI study
  -> DICOM metadata repair
  -> physical slice ordering
  -> 3-stream or 6-stream sequence routing
  -> 2D slices OR 2.5D [z-gap,z,z+gap] triplets
  -> shared ResNet18 / ConvNeXt-Tiny / timm backbone
  -> mean / max / attention / Top-K slice pooling
  -> shared stream attention OR 12 target-specific queries
  -> 12 logits

Loss
  -> confidence-weighted BCE
  -> optional pairwise ranking loss

Validation
  -> gold-only OOF
  -> per-target AUC
  -> macro AUC
  -> bootstrap interval
  -> paired bootstrap run comparison
```

## DICOM safety

Long training runs should begin with a real pixel-decode audit:

```bash
rsna-knee preflight \
  --data-root /path/to/rsna-knee-abnormality-detection \
  --split train \
  --sample-size 24
```

The preflight checks selected streams, path resolution, DICOM decoding, frame counts, preprocessing, and metadata repair. Training runs this automatically by default and refuses to start if the sampled stream failure rate exceeds the configured threshold.

DICOM handling includes:

- physical ordering from `ImageOrientationPatient` + `ImagePositionPatient`;
- `InstanceNumber` fallback;
- suffix-less/`.ima`/`.dicom` instances;
- enhanced multi-frame DICOM;
- rescale slope/intercept;
- `MONOCHROME1` inversion;
- mixed in-plane size normalization;
- robust percentile intensity normalization.

## Experiment configs

Use the frozen configs instead of manually changing several knobs at once:

| Experiment | Main change |
|---|---|
| `configs/e01_baseline.yaml` | 2D ResNet18, 3 streams, shared attention |
| `configs/e02_2p5d_resnet18.yaml` | E01 + true neighboring-slice 2.5D input |
| `configs/e03_target_attention.yaml` | E02 + per-target stream queries |
| `configs/e04_dual_stream.yaml` | E03 + six fluid/structural streams |
| `configs/e05_convnext_tiny.yaml` | E04 + ConvNeXt-Tiny backbone |
| `configs/e06_rank_loss.yaml` | E05 + pairwise AUC-surrogate ranking loss |

The model layer also supports timm backbones using:

```yaml
backbone: timm:<model_name>
```

This makes DINOv2-style timm encoders available when allowed weights are accessible. Verify current competition external-model rules before enabling pretrained external weights.

## Installation

```bash
git clone https://github.com/mtalafha90/CNN_CPC.git
cd CNN_CPC
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
pytest -q
```

GitHub Actions also runs the test suite on pushes to `main` and on pull requests.

## Expected data layout

```text
DATA_ROOT/
├── train.csv
├── train_series.csv
├── test.csv
├── test_series.csv
├── train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*
└── test_series/<StudyInstanceUID>/<SeriesInstanceUID>/*
```

## Inspect data

```bash
rsna-knee inspect --data-root DATA_ROOT
```

## Train E01

Change only `data_root` in the experiment config, then run:

```bash
for fold in 0 1 2; do
  rsna-knee train --config configs/e01_baseline.yaml --fold "$fold"
done
```

Each fold writes:

- `best.pt`
- `oof.csv`
- `history.csv`
- `config.json`
- `fold_assignments.csv`
- `calibration.json` when calibration is active
- `metadata_repair.json`
- `bootstrap.json`

## Evaluate the full OOF baseline

```bash
rsna-knee evaluate \
  --train-csv DATA_ROOT/train.csv \
  --oof runs/e01_baseline/fold0/oof.csv \
        runs/e01_baseline/fold1/oof.csv \
        runs/e01_baseline/fold2/oof.csv \
  --out runs/e01_baseline/evaluation.json
```

## Compare E02 against E01

```bash
rsna-knee evaluate \
  --train-csv DATA_ROOT/train.csv \
  --oof runs/e01_baseline/fold0/oof.csv \
        runs/e01_baseline/fold1/oof.csv \
        runs/e01_baseline/fold2/oof.csv \
  --compare-oof runs/e02_2p5d_resnet18/fold0/oof.csv \
                runs/e02_2p5d_resnet18/fold1/oof.csv \
                runs/e02_2p5d_resnet18/fold2/oof.csv
```

The paired bootstrap estimates the median macro-AUC difference, an interval for that difference, and how often the second run wins under study resampling.

## Inference and submission

Inference is **image-only by default**:

```bash
rsna-knee infer \
  --config configs/e01_baseline.yaml \
  --checkpoints /path/fold0.pt /path/fold1.pt /path/fold2.pt \
  --alpha 1.0 \
  --out submission.csv
```

Report fusion is intentionally rejected unless `allow_test_report_fusion: true` is explicitly set after verifying that a test-like dataset genuinely contains report text.

Submission columns are exactly:

```text
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
```

## Development policy

1. Freeze folds and preserve OOF predictions.
2. Change one major factor per experiment.
3. Validate only on official gold labels.
4. Use paired bootstrap before accepting small improvements.
5. Treat pretrained/external models as rule-dependent features, not assumptions.
6. Do not commit competition DICOMs, credentials, or large checkpoints.
7. Do not report synthetic, teacher-only, or README demonstration metrics as competition results.

## Next data-dependent step

The code-side review recommendations are implemented. The remaining work that cannot be completed without the mounted competition data/GPU is to **run E01-E06**, collect OOF predictions/runtime/memory, and use those measured results to decide which architecture to retain.
