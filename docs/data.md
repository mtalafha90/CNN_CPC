# Dataset and DICOM handling

## CSV contracts

`train.csv` must contain `StudyInstanceUID`, `Report`, and all 12 target columns. `test.csv` must contain `StudyInstanceUID`; report text is not required for inference.

`train_series.csv` / `test_series.csv` must contain:

- `StudyInstanceUID`
- `SeriesInstanceUID`
- `Fluid_Sensitive`
- `Fat_Suppression`
- `Anatomical_Plane`

UIDs are normalized to strings on load and duplicate study/series rows are rejected.

## Production series routing

The production model always builds up to six semantic MRI streams:

```text
sagittal_fluid       sagittal_structural
coronal_fluid        coronal_structural
axial_fluid          axial_structural
```

The routing table is built with a single `groupby` over the series CSV rather than repeatedly scanning the full table for every study. Within each plane, fluid/fat-suppressed metadata ranks the fluid candidate while non-fat-suppressed/non-fluid metadata ranks the structural candidate. When multiple candidates exist, the router avoids placing the same series in both semantic slots.

Missing streams remain explicitly absent and are masked by the model; they are never replaced with a fabricated MRI signal.

## Metadata repair

The series CSV is authoritative when `Anatomical_Plane` is populated. Blank plane entries are repaired from DICOM headers before routing:

- plane from `ImageOrientationPatient` geometry;
- sequence hints from TE/TR/TI when available.

Boolean metadata is parsed explicitly; string values such as `"False"` are never interpreted with Python's generic truthiness.

## DICOM discovery and ordering

`read_dicom_series` supports mixed layouts containing `.dcm`, `.dicom`, `.ima`, and suffix-less instances in the same series directory.

Slices are ordered from `ImageOrientationPatient` + `ImagePositionPatient` when possible, with `InstanceNumber` fallback. Enhanced multi-frame instances are expanded while preserving frame order.

The reader also applies:

- `RescaleSlope` / `RescaleIntercept`;
- `MONOCHROME1` inversion;
- center crop/pad normalization for mixed in-plane dimensions;
- finite-value validation.

## MRI preprocessing

Each selected series is normalized globally using its finite 1st/99th intensity percentiles, clipped to that interval, and mapped to `[0,1]`.

The production representation is 2.5D. Uniformly distributed center slices are selected over the original series and each sample becomes:

```text
[z - gap, z, z + gap]
```

with edge indices clipped to the valid range. The three neighboring slices become the three ConvNeXt input channels.

## Preflight

Run before expensive training:

```bash
rsna-knee preflight --data-root DATA_ROOT --split train --sample-size 24
```

Preflight performs real pixel decoding and reports two different quantities:

- **missing-stream rate** — semantic stream slots that legitimately have no selected series;
- **decode-failure rate** — selected series that cannot be found or decoded.

Only decode failures are used by the strict training gate. This prevents normal protocol variability from being mistaken for corrupted data.

## Leakage control

Gold folds group identical normalized reports together. For a validation fold, every training study sharing a validation report hash is removed from the fold's training set.

Report-teacher calibration uses only gold studies outside the current validation fold. Validation targets are the raw official target cells; NaNs remain NaNs and are ignored by the AUC calculation rather than converted to negatives or pseudo-labels.

## Training versus inference

Reports are used only during training to generate weak supervision. Final inference depends only on MRI data and self-describing model checkpoints.
