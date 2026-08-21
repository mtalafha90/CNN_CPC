# Native-resolution DICOM audit before B37

## Purpose

Before freezing a 288-resize or native/padding B37 input policy, audit the actual
training acquisition geometry.  A matrix size alone is not physical resolution:
`PixelSpacing` and field of view (FOV) must be considered together.

This audit is intentionally **header-only**.  It does not decompress `PixelData`,
does not alter any DICOM file, and does not train/evaluate a model.

## What is scanned

Every series row in `train_series.csv` is located through the same directory
routing used by the model.  All DICOM headers in each series are read with
`stop_before_pixels=True`.

Per-series output records:

- native `Rows x Columns`;
- within-series matrix consistency;
- `PixelSpacing` (or `ImagerPixelSpacing` fallback);
- physical row/column FOV;
- slice thickness and spacing between slices;
- manufacturer and scanner model;
- magnetic field strength;
- series/sequence description;
- bit depth and photometric interpretation;
- header-read coverage/failures;
- repaired anatomical plane and current model eligibility.

The summary also evaluates a fixed **90% native center crop** and asks what square
canvas can retain those cropped source pixels by **padding only**, with candidate
canvases 288, 320, 384, 448, 464, 512, 576 and 640.

Important: padding-only feasibility does not imply physical-scale equivalence.
Two 512x512 scans with different `PixelSpacing` represent different physical
sampling and FOV.

## Run

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
export PYTHONPATH="$PWD/developments/src:${PYTHONPATH:-}"

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

python -m rsna_knee.native_resolution_audit \
  --data-root "$DATA_ROOT" \
  --workers 4 \
  --out-root runs/native_resolution_audit
```

A small plumbing run can be done first:

```bash
python -m rsna_knee.native_resolution_audit \
  --data-root "$DATA_ROOT" \
  --workers 2 \
  --max-series 100 \
  --out-root runs/native_resolution_audit_smoke
```

## Outputs

```text
runs/native_resolution_audit/
├── series_geometry.csv
├── summary.json
└── REPORT.md
```

`series_geometry.csv` is the auditable per-series table. `summary.json` contains
machine-readable distributions and padding coverage. `REPORT.md` is the compact
human-readable decision report.

## Decision use

Do not freeze a native/padded B37 input shape until the complete audit is read.
The key quantities are:

1. matrix distribution among model-eligible series;
2. percentage of the fixed 90% native crop fitting each canvas without resizing;
3. p05/median/p95 `PixelSpacing` by model-eligible series;
4. physical FOV distribution;
5. matrix/spacing inconsistencies within a series;
6. scanner/manufacturer distribution.

If a practical canvas preserves nearly all cropped matrices but `PixelSpacing`
varies materially, native padding is still pixel-preserving but the model must
learn acquisition-dependent physical scale.  That is a separate scientific
choice from interpolation loss and must be made explicitly.
