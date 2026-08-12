# Dataset and DICOM handling

> **Snapshot — 2026-08-12.** Data/audit facts below are verified on the downloaded competition release. B13 is the reused-gold development champion. B15 is completed and uses the frozen weak-v2 split plus a stricter SSL image-holdout policy. Scores live in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

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
training studies       4407
fully gold-labelled      58
report-only studies    4349
reports present        4407
training series rows  24371
local test studies        3
local test series        15
```

All 58 gold studies have all 12 official target cells populated.

## Historical six-stream contract and routing audit

```text
sagittal_fluid       sagittal_structural
coronal_fluid        coronal_structural
axial_fluid          axial_structural
```

Historical selected streams: `21,886`. Strict semantic audit found `552` wrong-slot assignments (`2.52%`) and `21,334` strictly valid selected streams. B9 tested strict routing and scored `0.5334962669`; it was rejected globally, but the data-quality audit remains valid.

## DICOM decoding

The reader supports single-frame and enhanced/multi-frame arrays and performs:

1. physical slice ordering from orientation/position;
2. `InstanceNumber` fallback;
3. deterministic filename fallback;
4. rescale slope/intercept;
5. `MONOCHROME1` inversion;
6. mixed-size center crop/pad;
7. finite percentile clipping/normalization.

Historical full audit:

```text
selected series checked  21886
selected series decoded  21886
candidate DICOM files   732556
failed DICOM files           2
selected series failed       0
```

## All-real-series contract

B12 replaced the six fixed semantic slots with every repaired Sagittal/Coronal/Axial acquisition as a real series. That mapping became the basis of B13-B15 hierarchy experiments.

```text
B6-active studies        3120
eligible real series    17475
historical dual unique  15468
extra series             2007
series/study min/median/max 3 / 5 / 14
series SHA
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

B13/B14 require this frozen full mapping.

## 2.5D representation

Production B13-B15 downstream values:

```text
n_slices       16
image_size     224
triplet_gap      1
train gaps     [1,2]
center jitter    2
TTA offsets  [-1,0,1]
```

The exact B13 slice-exposure audit found median evaluation exposure `100%` and complete evaluation exposure for `95.9%` of all 17,475 series. Slice-count undersampling is therefore rejected as a primary B13 bottleneck.

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

Frozen state treatment:

```text
positive -> target 0.85, weight 0.50
negated -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
```

Do not interpret report silence as an explicit negative.

## Frozen weak-v2 split

B15-era model ranking uses an exact report-group-safe split frozen before training:

```text
active studies            3120
weak-train studies        2497
holdout studies            623
actual holdout fraction   0.1996794872
train report groups       2426
holdout report groups      613
report-group overlap         0
holdout usable cells      2875
holdout positive          1407
holdout negative          1468
manifest SHA
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

The split uses no gold labels or model predictions. It must not be regenerated based on B15 results.

## B15 SSL image pool

B15 MRI-domain SSL used a stricter data boundary than downstream training:

```text
competition studies   4407
minus gold               58
minus weak-v2 holdout   623
SSL studies            3726
eligible SSL series   20534
```

All gold images and all weak-v2 holdout images were excluded from SSL optimization.

## B15 downstream data contract

Matched B13-v2 control and B15 candidate both used exactly:

```text
weak-train studies      2497
eligible real series   13974
usable B6 cells        11248
positive cells          5464
negative cells          5784
batches/epoch           1249
epochs                     4
```

This exact match is essential to attributing the weak-v2 difference to the encoder initialization path.

## Current validation role

The 58 gold studies remain repeatedly reused development/model-selection data. Weak-v2 is B6 teacher agreement only. The hidden Kaggle test is the next independent performance signal.

B15 weak-v2 improved from control `0.5652498118` to `0.7319060415`, but B15 gold was `0.6209002783` versus B13 `0.6293565948`. This makes direct auditing of B6 report-state information the next data/supervision priority.

See [`WEAK_HOLDOUT_V2.md`](WEAK_HOLDOUT_V2.md), [`B15_MRI_SSL.md`](B15_MRI_SSL.md), and [`VALIDATION.md`](VALIDATION.md).