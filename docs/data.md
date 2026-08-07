# Dataset and DICOM handling

## CSV files
`train.csv` contains `StudyInstanceUID`, `Report`, and the 12 target columns. `train_series.csv` maps studies to series and provides `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, and `Anatomical_Plane`.

## DICOM preprocessing
The baseline reads DICOMs with pydicom, orders slices from orientation/position metadata when possible, applies rescale slope/intercept, handles MONOCHROME1, clips robustly at the 1st/99th percentiles, samples a fixed number of slices, and resizes them for the CNN.

## Series routing
`best` selects one representative series per sagittal/coronal/axial plane. `dual` keeps both a fluid-sensitive and a structural series per plane, yielding up to six streams.

## Leakage control
Gold folds group identical normalized reports together. When a gold report is in validation, every training row with the same normalized report hash is excluded from that fold's training set. Validation itself is restricted to explicit gold labels.

## Limitation
Report-derived pseudo-labels are weak supervision, not independent expert image annotations. Reports may omit incidental findings or use uncertain language. Audit the report teacher on the 58 gold studies before investing heavily in image training.
