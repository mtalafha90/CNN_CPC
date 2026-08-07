# CNN_CPC — RSNA Knee Abnormality Detection

Reproducible PyTorch research baseline for the **2026 RSNA Knee Abnormality Detection** Kaggle competition.

> **Competition status (snapshot: 2026-08-07):** the competition is still open, so there are no winning solutions to reproduce yet. This repository therefore documents the current challenge, reviews public early approaches and the MRNet literature, and implements an honest baseline that can be trained and submitted through Kaggle.

## Public Kaggle/code methodology review

A detailed review of the currently discoverable public competition code, methodology, and recurring techniques is now available in:

**[README_KAGGLE_METHODS.md](README_KAGGLE_METHODS.md)**

It covers weak supervision from reports, multilingual/LLM pseudo-labeling, DICOM geometry, sequence routing, 2.5D triplets, DINOv2, ConvNeXt/EfficientNet, target-specific MIL, ranking loss, leakage-safe validation, efficiency engineering, 3D complementary models, and a prioritized experiment matrix for `CNN_CPC`.

## What the challenge asks

Given a knee MRI study containing multiple DICOM series, predict 12 binary abnormalities:

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture

The score is the **macro-average ROC AUC across the 12 targets**, so every class contributes equally.

## The key data insight

The public competition files have an unusual supervision structure:

- 4,407 training studies
- only 58 studies with explicit gold 0/1 labels
- 4,349 studies without gold labels
- radiology report text is present for the training studies
- `train_series.csv` identifies each series' anatomical plane and whether it is fluid-sensitive / fat-suppressed

This makes the problem primarily **semi-supervised / weakly supervised**, not a conventional fully supervised CNN task. The baseline here therefore uses the reports to create soft pseudo-labels for the unlabeled MRI studies while keeping evaluation strictly on the gold-labeled studies.

See [docs/competition.md](docs/competition.md) and [docs/data.md](docs/data.md) for the full breakdown.

## Baseline architecture

```text
Radiology report ──> multilingual rule labels ─────────────┐
                                                          │ soft targets
Sagittal DICOM series ─> shared ResNet18 ─> slice attention│
Coronal DICOM series  ─> shared ResNet18 ─> slice attention├─> series attention ─> 12 logits
Axial DICOM series    ─> shared ResNet18 ─> slice attention│
                                                          │
Gold labels (58) ──────────────────────────────────────────┘ higher loss weight + validation only
```

The image model is deliberately MRNet-like: a 2D CNN encodes slices, attention pools slices into a series representation, and attention then fuses anatomical planes/series into one study-level prediction.

### Why this baseline

- Uses the large unlabeled portion rather than pretending 58 labeled studies are enough.
- Keeps duplicate normalized reports in the same validation group to reduce leakage.
- Validates only against explicit gold labels.
- Uses the competition-provided series metadata instead of guessing MRI planes from pixels.
- Supports 3-stream (`best`) and 6-stream (`dual`) series selection.
- Produces the exact 13-column Kaggle submission schema.
- Does **not** claim an unmeasured leaderboard score.

## Repository layout

```text
CNN_CPC/
├── configs/baseline.yaml
├── docs/
│   ├── competition.md
│   ├── data.md
│   ├── strategy.md
│   └── references.md
├── kaggle/
│   ├── train_template.py
│   └── submit_template.py
├── scripts/
│   └── report_only_submission.py
├── src/rsna_knee/
│   ├── constants.py
│   ├── data.py
│   ├── dicom.py
│   ├── dataset.py
│   ├── report_labels.py
│   ├── model.py
│   ├── metrics.py
│   ├── training.py
│   ├── fusion.py
│   ├── inference.py
│   └── cli.py
└── tests/
```

## Installation

```bash
git clone https://github.com/mtalafha90/CNN_CPC.git
cd CNN_CPC
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install -e .
```

For development/tests:

```bash
pip install pytest
PYTHONPATH=src pytest -q
```

## Expected Kaggle data layout

Set `data_root` in `configs/baseline.yaml` to the mounted competition dataset, normally:

```text
/kaggle/input/rsna-knee-abnormality-detection/
├── train.csv
├── train_series.csv
├── test.csv
├── test_series.csv
├── train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
└── test_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
```

The loader also accepts a few common alternative image directory names. See [docs/data.md](docs/data.md).

## Inspect the competition files

```bash
rsna-knee inspect --data-root /kaggle/input/rsna-knee-abnormality-detection
```

This prints study counts, gold/unlabeled counts, series count, and gold positive counts per target.

## Generate report pseudo-labels

```bash
rsna-knee pseudo-label \
  --train-csv /kaggle/input/rsna-knee-abnormality-detection/train.csv \
  --out /kaggle/working/report_pseudo_labels.csv
```

The included extractor is intentionally conservative and transparent. It understands target terms and common negation expressions across several languages represented in public descriptions of the challenge. It is a baseline, not a substitute for a validated multilingual clinical NLP model.

## Train image folds

The default is 3 folds because the gold validation set is extremely small.

```bash
for fold in 0 1 2; do
  rsna-knee train --config configs/baseline.yaml --fold "$fold"
done
```

Outputs are written under `output_dir/foldN/`:

- `best.pt`
- `oof.csv`
- `history.csv`
- `config.json`

Validation is gold-only and selected by macro AUC.

## Tune image/report fusion

```bash
rsna-knee tune-fusion \
  --train-csv /kaggle/input/rsna-knee-abnormality-detection/train.csv \
  --oof /kaggle/working/runs/fold0/oof.csv \
        /kaggle/working/runs/fold1/oof.csv \
        /kaggle/working/runs/fold2/oof.csv \
  --out /kaggle/working/fusion.json
```

Because only 58 gold cases exist, treat the tuned fusion coefficient as high variance. Do not over-optimize it against the same tiny validation surface.

## Inference and submission

```bash
rsna-knee infer \
  --config configs/baseline.yaml \
  --checkpoints /kaggle/input/YOUR-MODEL-DATASET/fold0.pt \
                /kaggle/input/YOUR-MODEL-DATASET/fold1.pt \
                /kaggle/input/YOUR-MODEL-DATASET/fold2.pt \
  --alpha 0.70 \
  --out /kaggle/working/submission.csv
```

The output columns are exactly:

```text
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
```

Templates for Kaggle are in `kaggle/train_template.py` and `kaggle/submit_template.py`.

## Report-only sanity baseline

Before spending GPU time, create a report-only submission to validate the text path and CSV schema:

```bash
python scripts/report_only_submission.py \
  --test-csv /kaggle/input/rsna-knee-abnormality-detection/test.csv \
  --out /kaggle/working/submission.csv
```

This is a plumbing/sanity baseline, not the intended final model.

## What to improve next

The highest-value experiments are documented in [docs/strategy.md](docs/strategy.md). In order:

1. audit report-derived labels against the 58 gold studies, per class and per language;
2. establish a measured gold-only CV baseline;
3. replace rules with a stronger multilingual clinical-text teacher if competition rules permit;
4. add self-supervised MRI pretraining / allowed pretrained image encoders;
5. compare `best` vs `dual` multi-sequence routing;
6. test 3D or 2.5D backbones and plane-specific encoders;
7. ensemble diverse folds/backbones and calibrate only when supported by validation.

## Important cautions

- **No winner claims:** the competition is ongoing as of this repository snapshot.
- **No fabricated score:** no leaderboard/CV number is reported until actually measured.
- **Tiny gold set:** 58 labeled studies makes per-class AUC noisy; some fold/class combinations may be poorly estimated.
- **Pseudo-label noise:** radiology reports are useful supervision but are not identical to expert binary annotations.
- **Competition rules change:** verify the current Kaggle rules before using external data, pretrained weights, APIs, or internet-dependent code.
- **Do not commit competition data or DICOMs** to this repository.

## Background

The design is inspired by MRNet, which demonstrated that knee MRI can be modeled by processing individual slices and combining information across sagittal, coronal and axial series. This repository extends that idea to the current 12-label, multilingual, weakly supervised challenge.

See [docs/references.md](docs/references.md) for sources and public early competition implementations reviewed during development.