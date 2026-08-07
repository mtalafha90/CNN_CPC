# CNN_CPC — RSNA Knee Abnormality Detection

Reproducible PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** Kaggle competition.

> **Competition status (snapshot: 2026-08-07):** the competition is still open. This repository avoids winner claims and records only measured OOF/leaderboard results.

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

The supervision structure is unusual:

- 4,407 training studies;
- only 58 studies with explicit gold labels;
- 4,349 report-supervised studies;
- multi-plane/multi-sequence MRI metadata in `train_series.csv`.

Radiology reports are therefore a **training teacher**. Validation is gold-only and final inference is image-only by default.

## Current architecture and capabilities

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
  -> ResNet18 / ConvNeXt-Tiny / timm-DINOv2-compatible / compact 3D encoder
  -> mean / max / attention / Top-K feature pooling
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

The preflight checks selected streams, path resolution, actual pixel decoding, frame counts, preprocessing, and metadata repair. Training runs this gate automatically by default and refuses to start if the sampled stream failure rate exceeds the configured threshold.

DICOM handling includes:

- physical ordering from `ImageOrientationPatient` + `ImagePositionPatient`;
- `InstanceNumber` fallback;
- suffix-less/`.ima`/`.dicom` instances;
- enhanced multi-frame DICOM;
- rescale slope/intercept;
- `MONOCHROME1` inversion;
- mixed in-plane size normalization;
- robust percentile intensity normalization;
- decode/file statistics for preflight.

## Frozen experiment configs

Use the frozen configs instead of manually changing several knobs at once:

| Experiment | Main change |
|---|---|
| `configs/e01_baseline.yaml` | 2D ResNet18, 3 streams, shared attention |
| `configs/e02_2p5d_resnet18.yaml` | E01 + true neighboring-slice 2.5D input |
| `configs/e03_target_attention.yaml` | E02 + per-target stream queries |
| `configs/e04_dual_stream.yaml` | E03 + six fluid/structural streams |
| `configs/e05_convnext_tiny.yaml` | E04 + ConvNeXt-Tiny backbone |
| `configs/e06_rank_loss.yaml` | E05 + pairwise AUC-surrogate ranking loss |
| `configs/e07_topk_mil.yaml` | E05 + Top-K feature pooling, ranking loss off |
| `configs/e08_dinov2_timm.yaml` | DINOv2 ViT-S/14 through timm |
| `configs/e09_small3d.yaml` | compact 3D complementary arm |

The model layer also supports generic timm backbones:

```yaml
backbone: timm:<model_name>
```

The DINOv2 config uses `timm:vit_small_patch14_dinov2.lvd142m`. Check that this exact name exists in the installed timm release and verify the current competition rules before enabling pretrained external weights. Kaggle submission notebooks have Internet disabled, so allowed pretrained weights must be available offline.

The 3D arm is deliberately small. Its purpose is to test **complementary inter-slice signal**, not to assume that 3D is automatically superior to 2.5D.

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

GitHub Actions runs the test suite on pushes to `main` and on pull requests.

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
- `preflight.json`
- `bootstrap.json`
- `runtime.json` with elapsed time, device, workers, and peak GPU memory

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

## Heterogeneous model ensemble

After producing aligned prediction files from different model families, rank-average them:

```bash
rsna-knee ensemble \
  --predictions submission_2p5d.csv submission_3d.csv \
  --method rank \
  --out submission_ensemble.csv
```

Rank averaging is useful for an AUC competition because it reduces probability-scale differences between heterogeneous models. Use OOF predictions first to decide whether an ensemble actually helps before submitting it.

## Inference and submission

Inference is **image-only by default**:

```bash
rsna-knee infer \
  --config configs/e01_baseline.yaml \
  --checkpoints /path/fold0.pt /path/fold1.pt /path/fold2.pt \
  --alpha 1.0 \
  --out submission.csv
```

Report fusion is rejected unless `allow_test_report_fusion: true` is explicitly set after verifying that a test-like dataset genuinely contains report text.

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
8. Keep the 3D arm only if it improves OOF itself or improves a 2.5D+3D ensemble beyond validation noise.

## What remains data-dependent

The code-side recommendations are implemented. What cannot be completed without the mounted competition data and suitable GPU is the **actual E01-E09 experiment campaign**: training the folds, collecting OOF predictions/runtime/memory, comparing runs with paired bootstrap, and retaining only the methods supported by measured results.
