# CNN_CPC — Current Knee MRI Model

This repository now exposes **one clean working model** at the top level and preserves the complete research history under [`developments/`](developments/README.md).

The active working model remains **B20 (`B20_crop_only_joint_focus`)**. The restructure changes repository organization only; it does **not** promote a new model or alter the canonical checkpoint.

## Current model

```text
MRI study
  -> all eligible real MRI series
  -> 16 sampled slice positions per series
  -> adjacent-slice 2.5D triplets
  -> 224 x 224 image tensors
  -> deterministic centered 90% crop + resize
  -> frozen ConvNeXt-Tiny CNN encoder
  -> learned attention pooling to one token per series
  -> study-level Transformer
  -> 12 pathology queries
  -> 12 sigmoid probabilities
```

Canonical checkpoint:

```text
runs/b20_crop_focus/b20_model.pt
```

Canonical epoch: **2**.

See [`docs/CURRENT_MODEL.md`](docs/CURRENT_MODEL.md) for the complete current-model description.

## Clean project structure

```text
CNN_CPC/
├── config/          current-model configuration
├── data/            dataset interface
├── model/           architecture and preprocessing
├── training/        model training
├── validation/      development validation
├── testing/         competition test inference
├── docs/            current-model documentation
├── developments/    complete B0--B25X research archive
├── requirements.txt
└── pyproject.toml
```

The root no longer exposes every historical experiment as if it were part of the current model. Historical modules, scripts, tests, configurations, papers, Kaggle notes and experiment records are preserved intact under `developments/`.

## Installation

```bash
conda activate rsna-knee
pip install -e .
```

The RSNA dataset and run artifacts are intentionally not stored in Git.

## Training

```bash
python -m training.train \
  --data-root /path/to/rsna-knee-abnormality-detection \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --report-ssl-checkpoint runs/b16_v2_safe_report/report_ssl/b16_v2_report_encoder.pt \
  --out-root runs/current_model
```

This reproduces the recorded B20 recipe through the preserved verified implementation.

## Validation

```bash
python -m validation.validate \
  --data-root /path/to/rsna-knee-abnormality-detection \
  --checkpoint runs/b20_crop_focus/b20_model.pt
```

**Important:** the 58 expert-labelled studies are a repeatedly reused development/checkpoint-selection surface. Validation on them is not independent test evidence.

## Testing / submission inference

```bash
python -m testing.test \
  --data-root /path/to/rsna-knee-abnormality-detection \
  --checkpoint runs/b20_crop_focus/b20_model.pt \
  --out submission.csv
```

This runs the active model on released test metadata and writes the competition submission plus the existing inference manifest.

## Model information

```bash
python -m model.architecture
```

## Development archive

All historical work is retained at:

```text
developments/
```

This includes the full B0--B25X experiment lineage, including B21/B22, B23/B24X, B25X, the supervision-balance audit, old CLI scripts, tests, documentation and manuscript material.

For exact rollback/reference, the pre-restructure repository tree is also saved in the Git branch:

```text
archive/pre-clean-structure-2026-08-15
```

No historical work was deleted.
