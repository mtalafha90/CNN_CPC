# Completed dataset contract audit — Phase 3 DICOM header result

> **Descriptive data audit only.** This phase does not define B35, alter B6, select a model, or use target-wise PV1/PV2 outcomes.

## Coverage and header integrity

The uploaded Phase-3 audit inspected one representative DICOM header from every listed training MRI series using `stop_before_pixels=True`.

```text
listed training series                 24,371
representative headers read            24,371
header failures                              0
orientation available                  24,371
orientation / supplied-plane matches   24,371
orientation mismatches                      0
multi-frame representative headers          0
```

The supplied sagittal/coronal/axial labels are therefore fully consistent with the representative ImageOrientationPatient geometry in this local training release.

## Acquisition dimensionality and the long-slice tail

Representative MR Acquisition Type:

```text
2D       22,329   91.62%
3D          836    3.43%
missing    1,206    4.95%
```

The Phase-2 long slice-count tail is almost exactly a 3D-acquisition phenomenon. In fact:

```text
all 836 known-3D series have >48 DICOM slices
763 / 836 known-3D series have >78 slices      91.27%
709 / 836 known-3D series have >100 slices     84.81%
 88 / 836 known-3D series have >200 slices     10.53%

all series with >78 slices are labelled 3D
all series with >100 slices are labelled 3D
all series with >200 slices are labelled 3D
```

The 763 series above 78 slices comprise 385 sagittal, 377 axial and one coronal series. Their median SliceThickness is about 0.8 mm; where SpacingBetweenSlices is available its median is about 0.9 mm.

The 709 series above 100 slices comprise 373 axial and 336 sagittal series. Their median SliceThickness is about 0.8 mm and median available SpacingBetweenSlices is about 0.9 mm.

The 88 series above 200 slices are especially homogeneous:

```text
sagittal                     85
axial                         3
fluid-sensitive              85
non-fluid-sensitive           3
median SliceThickness      ~0.8 mm
median SpacingBetweenSlices ~0.4 mm
```

Eighty-six of those 88 series use the `Achieva dStream` model label and two use `Aera`. This establishes that the extreme 320-slice/long-volume tail is not simply ordinary 2D knee MRI with more slices; it is a thin-slice 3D acquisition family.

## Consequence for the frozen 16-position sampler

The current preprocessing uses 16 distributed 2.5D centers. A single evaluation view can touch at most 48 distinct source slices, while the frozen three-view center-offset TTA `[-1,0,+1]` can touch at most 78 distinct source slices.

For the complete 24,371-series release:

```text
series fully coverable by one evaluation view     95.93%
series fully coverable by three-view TTA           96.87%
mean per-series source coverage, one view          97.78%
mean per-series source coverage, three-view TTA    98.52%
slice-weighted source coverage, one view           89.04%
slice-weighted source coverage, three-view TTA     92.03%
```

For known 3D series only, mean source-slice coverage is much lower:

```text
one view     ~36.43%
three-view TTA ~56.88%
```

For the >78-slice 3D tail it is about 32.46% and 52.75%, respectively. The 88 series above 200 slices receive only about 15.23% mean source coverage in one view and 24.75% across the current TTA.

**Decision:** there is no data-based justification for globally increasing the 16-position count. Approximately 97% of all series are already fully coverable by the existing three-view evaluation policy. Any future sampling experiment should be acquisition-aware and predeclared specifically for long 3D series rather than changing all series.

## Spatial and scanner heterogeneity

The training release is strongly heterogeneous in geometry:

```text
PixelSpacing median                  0.3125 mm
PixelSpacing range             ~0.073–1.172 mm
row FOV median                         160 mm
row FOV range                       ~70–320 mm
column FOV median                      160 mm
column FOV range                    ~70–361 mm
Rows range                          160–1280
Columns range                       160–1444
SliceThickness median                    3 mm
SliceThickness range                 ~0.6–6 mm
```

Obliquity relative to the nearest canonical plane has median ~6.8 degrees, 95th percentile ~20.3 degrees and maximum ~41.4 degrees. By supplied plane, the 95th-percentile obliquity is approximately 10.6 degrees axial, 19.6 degrees sagittal and 30.1 degrees coronal. Thus the plane labels are geometrically correct but the data contain substantial obliquity.

Field-strength tags are also heterogeneous:

```text
1.5 T      14,108
3.0 T       8,922
1.16 T        130
1.0 T           5
missing      1,206
```

The manufacturer/model fields contain many Siemens, Philips, GE, Toshiba/Canon and Hitachi/Fujifilm variants, confirming broad scanner diversity.

## Transfer-syntax finding and deployment risk

Every representative training header reports:

```text
TransferSyntaxUID = 1.2.840.10008.1.2.1
Explicit VR Little Endian
```

Thus the local training data do **not** exercise compressed-pixel decoding, despite the competition data contract stating that hidden data may include JPEG Lossless and JPEG 2000 transfer syntaxes. This is a deployment-contract risk rather than a modelling result. The current reader relies on `pydicom.Dataset.pixel_array`, so codec availability must be tested in the final competition environment before submission.

## Gold versus report-only descriptive note

Using the already-frozen 58 official gold UIDs only for descriptive acquisition composition:

```text
gold studies with any known 3D series        4 / 58   6.90%
report-only studies with any known 3D       655 / 4349 15.06%
```

The gold surface therefore contains relatively few 3D acquisitions. This does not invalidate it, but it is another reason not to interpret the repeatedly reused 58-study surface as a representative hidden-test distribution.

## Governance

Phase 3 supports three data-level conclusions only:

1. the long slice-count tail is a thin-slice 3D acquisition family;
2. a global slice-count increase is not justified from coverage alone;
3. compressed-DICOM decode capability and the intersection between B6 supervision coverage and acquisition domain should be audited before any new architecture is defined.

No B35, adaptive sampler, new TTA offsets, target-specific switch, or B6 modification is authorized directly from this result.