# Dataset and DICOM handling

> **Snapshot: 2026-08-10.** Data/audit facts below are verified on the downloaded competition release. B7.1 is the current development leader; B8 is rejected; B9 strict semantic routing is the active predeclared experiment. Scores live in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## CSV contract

`train.csv` contains `StudyInstanceUID`, `Report`, and 12 targets. `test.csv` contains study UIDs. Series metadata contain:

- `StudyInstanceUID`;
- `SeriesInstanceUID`;
- `Fluid_Sensitive`;
- `Fat_Suppression`;
- `Anatomical_Plane`.

Duplicate study/series rows and missing UIDs are rejected.

## Verified release snapshot

```text
training studies       4,407
fully gold-labelled       58
report-only studies    4,349
reports present        4,407
training series rows  24,371
local test studies          3
local test series          15
```

All 58 gold studies have all 12 official target cells populated.

## Six-stream semantic contract

```text
sagittal_fluid       sagittal_structural
coronal_fluid        coronal_structural
axial_fluid          axial_structural
```

In the released metadata, `Fluid_Sensitive` and `Fat_Suppression` are perfectly coupled. Populated CSV metadata are authoritative; DICOM-derived metadata are fallback only.

## Historical routing coverage

The historical `mode="dual"` selector selected:

| Stream | Selected | Missing |
|---|---:|---:|
| sagittal_fluid | 4,401 | 6 |
| sagittal_structural | 4,294 | 113 |
| coronal_fluid | 4,250 | 157 |
| coronal_structural | 3,440 | 967 |
| axial_fluid | 4,407 | 0 |
| axial_structural | 1,094 | 3,313 |
| **Total** | **21,886** | — |

Those 21,886 streams were all successfully audited for DICOM decoding; two selected series each contained one unreadable instance, but no selected series was lost.

## B9 routing audit: semantic mismatch in the historical selector

The historical selector tries to populate both fluid and structural slots when a plane has multiple series. If a study has multiple acquisitions from only one contrast class, one series can be placed in the opposite semantic slot.

Full 4,407-study metadata audit:

| Stream | Historical selected | Strict selected | Historical wrong-slot assignments removed |
|---|---:|---:|---:|
| sagittal_fluid | 4,401 | 4,150 | 251 |
| sagittal_structural | 4,294 | 4,266 | 28 |
| coronal_fluid | 4,250 | 4,248 | 2 |
| coronal_structural | 3,440 | 3,406 | 34 |
| axial_fluid | 4,407 | 4,407 | 0 |
| axial_structural | 1,094 | 857 | 237 |
| **Total** | **21,886** | **21,334** | **552** |

Therefore:

```text
historical semantic mismatches  552
historical selected streams   21886
wrong-slot fraction            2.52%
strict semantic mismatches         0
strict selected streams       21334
```

Because removing a false assignment can also change which valid same-class series occupies the remaining slot, 805 stream assignments differ between the historical and strict indexes in total.

## Local test metadata routing audit

The provided three-study test surface contains one analogous wrong-slot assignment:

```text
historical selected streams 14
strict selected streams     13
historical mismatches         1
strict mismatches             0
```

One study has two sagittal structural series and no sagittal fluid series. Historical routing fabricates `sagittal_fluid`; B9 leaves it missing.

This test audit uses no labels and is not scientific validation.

## B9 strict routing rule

```text
*_fluid:
    candidates = Fluid_Sensitive == True only

*_structural:
    candidates = Fluid_Sensitive == False only

if no candidate exists:
    slot = None
    presence mask = False
```

Unknown contrast after metadata repair is not promoted into either slot.

The historical selector remains unchanged for B7.1 reproducibility; B9 uses `src/rsna_knee/strict_routing.py`.

## Metadata repair

Missing plane/sequence metadata can be backfilled from DICOM headers. The current production-selected surface required no repair. CSV values remain authoritative when present.

DICOM fallback uses:

- plane from `ImageOrientationPatient`;
- weighting from TE/TR/inversion time;
- fat suppression cues from acquisition tags.

The uploaded train/test DICOM examples confirm that scanner vendor, matrix size, pixel spacing and sequence timings vary across acquisitions. The current pipeline handles these through DICOM decoding, percentile intensity normalization and resizing; B9 does not change physical-scale preprocessing.

## DICOM decoding

The reader supports single-frame and enhanced/multi-frame arrays and performs:

1. physical slice ordering from orientation/position;
2. `InstanceNumber` fallback;
3. deterministic filename fallback;
4. rescale slope/intercept;
5. `MONOCHROME1` inversion;
6. mixed-size center crop/pad;
7. finite percentile clipping/normalization.

Verified historical audit:

```text
selected series checked  21,886
selected series decoded  21,886
candidate DICOM files   732,556
failed DICOM files            2
selected series failed        0
```

## 2.5D representation

Each active series is normalized and sampled as distributed triplets:

```text
[I_(c-gap), I_c, I_(c+gap)]
```

Production values:

```text
n_slices       16
image_size     224
triplet_gap      1
train gaps     [1,2]
center jitter    2
```

B9 keeps this representation unchanged from B7.1.

## B6 weak-supervision data contract

```text
report-only rows                  4349
active studies                    3120
inactive zero-usable studies      1229
usable cells                     14123
positive cells                    6871
negative cells                    7252
confidence threshold              0.75
gold rows in training_targets.csv    0
```

B6 is frozen. B9 changes no report parsing, labels, confidence thresholds or target weights.

## B7.1/B9 training data contract

B7.1 uses all 3,120 report-only studies with at least one usable B6 cell. Every training study has axial fluid imaging in the released metadata, so strict routing is not expected to remove all MRI streams from any active weak-training study.

B9 keeps:

```text
batch size          2
batches/epoch    1560
study draws/epoch 3120
epochs              4
```

The actual active-pool routing audit is written to:

```text
runs/b9_strict_routing/routing_audit.json
```

and must report zero strict semantic mismatches before gold evaluation.

## Validation role

The 58 gold studies do not enter B9 gradients or early stopping. However, the broader campaign has repeatedly used these studies for method decisions, so B9's eventual score is a development/model-selection estimate rather than untouched independent validation.
