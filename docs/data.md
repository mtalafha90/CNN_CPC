# Dataset and DICOM handling

## CSV files
`train.csv` contains `StudyInstanceUID`, `Report`, and the 12 target columns. `train_series.csv` maps studies to series and provides `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, and `Anatomical_Plane`.

## DICOM preprocessing
The baseline reads DICOMs with pydicom, orders slices from orientation/position metadata when possible, applies rescale slope/intercept, handles MONOCHROME1, clips robustly at the 1st/99th percentiles, samples a fixed number of slices, and resizes them for the CNN.

## Series routing
`best` selects one representative series per sagittal/coronal/axial plane. `dual` keeps both a fluid-sensitive and a structural series per plane, yielding up to six streams.

## Metadata fallbacks
`train_series.csv` is authoritative wherever it is populated. Two fallbacks cover the gaps, because a series with a blank `Anatomical_Plane` is invisible to `select_series`, which then hands the model a stream of zeros — a silent loss of data rather than an error.

- `load_series_csv` parses the flag columns with `coerce_bool` rather than `astype(bool)`. The latter is wrong twice over: missing values become `True`, and so does the string `"False"`, since any non-empty string is truthy. Either would mark structural series as fluid-sensitive and corrupt the routing.
- `backfill_series_metadata(series_df, data_root, split)` fills only the blank rows, reading one DICOM header per affected series. The plane comes from the cross product of the `ImageOrientationPatient` direction cosines, and the weighting from TE/TR/TI. Both are language-independent, which matters when series descriptions arrive in twelve languages. It returns a count of what it repaired, so the size of the gap is visible rather than assumed.

## Awkward DICOM layouts
`read_dicom_series` also accepts instances stored without a `.dcm` suffix, expands enhanced multi-frame instances into slices, and normalises mixed in-plane sizes (which occur when a localiser is stored beside the series) instead of raising.

## Leakage control
Gold folds group identical normalized reports together. When a gold report is in validation, every training row with the same normalized report hash is excluded from that fold's training set. Validation itself is restricted to explicit gold labels.

## Limitation
Report-derived pseudo-labels are weak supervision, not independent expert image annotations. Reports may omit incidental findings or use uncertain language. Audit the report teacher on the 58 gold studies before investing heavily in image training.
