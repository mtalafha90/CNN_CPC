# Official dataset contract audit

## Purpose

Architecture development is paused after the successful frozen B34/PV2 mechanism test. Before defining B35 or making a hidden-test submission, the current work is a descriptive audit of the competition data contract and supervision coverage.

The audit does not promote a checkpoint, alter frozen B6 v1.2.1, or use PV1/PV2 target-wise outcomes for architecture tuning.

## Phase 1 — COMPLETE: labels, reports and supplied series metadata

Recorded in `developments/docs/DATASET_CONTRACT_AUDIT_PHASE1_RESULT.md`.

```text
training studies                         4407
fully labelled studies                     58
partially labelled studies                  0
report-only studies                      4349
B6-active report-only studies            3120
B6 usable cells                         14123
Latin-script share of report-only       87.70%
Latin-script share of B6-active         98.75%
Latin-script share of usable B6 cells   99.72%
listed MRI series                       24371
```

Greek- and Cyrillic-script reports account for about 12.3% of report-only studies but only 40 of 14,123 usable B6 cells. `Fluid_Sensitive` and `Fat_Suppression` are perfectly redundant in the supplied training metadata even though the competition contract permits discordant values elsewhere.

## Phase 2 — COMPLETE: physical DICOM slice counts

Recorded in `developments/docs/DATASET_CONTRACT_AUDIT_PHASE2_RESULT.md`.

```text
median slices/series                      30
95th percentile                           45
99th percentile                          160
maximum                                  320
series >78 slices                        763  (3.13%)
series >100 slices                       709  (2.91%)
series >200 slices                        88  (0.36%)
```

The current 16-center gap-1 2.5D policy can touch up to 48 distinct source slices in one deterministic view and up to 78 across frozen TTA `[-1,0,+1]`. About 96.87% of series are fully coverable by that three-view policy. A global increase above 16 centers is therefore NO-GO from slice counts alone.

## Phase 3 — COMPLETE: DICOM scanner/header heterogeneity

Recorded in `developments/docs/DATASET_CONTRACT_AUDIT_PHASE3_RESULT.md`.

```text
known 2D series                         22329
known 3D series                           836
acquisition type missing                 1206
all >78-slice series are 3D
all >100-slice series are 3D
all >200-slice series are 3D
```

The extreme tail is a thin-slice 3D family rather than ordinary 2D MRI. All 24,371 representative training headers use Explicit VR Little Endian, so compressed DICOM pixel decoding remains a deployment capability that must be tested separately before hidden submission.

## Phase 4 — COMPLETE: B6 supervision × acquisition-domain intersection

Recorded in `developments/docs/DATASET_CONTRACT_AUDIT_PHASE4_RESULT.md`.

| Metric | B6 active | B6 inactive |
|---|---:|---:|
| studies | 3120 | 1229 |
| studies with any known 3D series | 614 (19.68%) | 41 (3.34%) |
| studies with any >78-slice series | 546 (17.50%) | 41 (3.34%) |
| studies with any >100-slice series | 524 (16.79%) | 37 (3.01%) |
| studies with any >200-slice series | 87 (2.79%) | 0 |

The manufacturer-family mixture also shifts substantially. Script and acquisition domain are associated in this exact release, so the report-supervision gap is also an MRI acquisition-domain selection problem. This is not an institution/site inference.

Only six gold studies directly anchor the non-Latin script groups: three Greek and three Cyrillic. Script-specific accuracy claims must therefore remain cautious.

## Phase 5 — COMPLETE: actual report-supervision failure modes

Recorded in `developments/docs/DATASET_CONTRACT_AUDIT_PHASE5_RESULT.md`.

The deterministic local sample contained 79 reports, including 36 report-only studies with zero usable B6 cells: 12 Latin-script, 12 Greek-script and 12 Cyrillic-script.

Direct inspection found target-relevant diagnostic content in **all 36** inactive examples. Zero B6 cells therefore do not indicate clinically silent reports; they indicate language/terminology coverage failure.

The current B6 rule set is multilingual but largely Latin-script. Its normalizer does not transliterate Greek or Cyrillic, and its target/normality/negation lexicons contain no effective native Greek/Cyrillic coverage.

The active controls confirm the mechanism: sampled Greek B6-active reports were activated by embedded English `bone bruise`, while the single Cyrillic active control was activated by embedded English fracture wording. Latin-script failure is also substantial, including South-Slavic Latin-script, Turkish and Spanish terminology missed by the frozen rules.

Raw Phase-5 report text remains a local-only artifact and must not be committed.

## Phase 6 — COMPLETE/PASS: deterministic translation -> frozen B6

Protocol: `developments/docs/DATASET_CONTRACT_AUDIT_PHASE6_TRANSLATION_RESCUE.md`

Result: `developments/docs/DATASET_CONTRACT_AUDIT_PHASE6_RESULT.md`

```text
translation failures                    0
overall inactive rescue             31/36 = 86.11%
Latin inactive rescue               12/12 = 100%
Greek inactive rescue                7/12 = 58.33%
Cyrillic inactive rescue            12/12 = 100%
added usable cells                    112
added positive cells                   81
added negative cells                   31
active B6 controls preserved        25/25
```

Phase 6 established coverage-mechanism feasibility, not independent clinical accuracy.

## Phase 7 — COMPLETE: full 1,229-study inactive-population audit

Protocol: `developments/docs/DATASET_CONTRACT_AUDIT_PHASE7_FULL_TRANSLATION_RESCUE.md`

Result: `developments/docs/DATASET_CONTRACT_AUDIT_PHASE7_RESULT.md`

The exact Phase-6 translator was applied to all 1,229 originally zero-cell report-only studies:

```text
successful translations                1229 / 1229
translation failures                       0
rescued studies                         1053 / 1229 = 85.68%
new usable cells                        3901
new positive cells                      2719
new negative cells                      1182
candidate active report-only studies    4173 / 4349 = 95.95%
candidate usable cells                 18024
```

By script:

```text
Latin       610/733 = 83.22%
Greek       228/280 = 81.43%
Cyrillic    215/216 = 99.54%
```

The rescue also greatly reduces the acquisition-domain coverage gap. Combining original B6-active and rescued studies gives supervision for 652/655 report-only studies with known 3D series, 584/587 with >78 slices, 558/561 with >100 slices, and all 87 with >200 slices.

Target-level recovery is not balanced. Synovitis adds 35 positives and zero negatives, while OA additions are strongly positive-skewed. These observed imbalances are recorded but may not be repaired through post-hoc target filtering from Phase-7 outcomes.

## Phase 8 — COMPLETE/PASS: global merged supervision artifact

Protocol: `developments/docs/DATASET_CONTRACT_AUDIT_PHASE8_MERGED_SUPERVISION.md`

Result: `developments/docs/DATASET_CONTRACT_AUDIT_PHASE8_RESULT.md`

Implementation: `developments/src/rsna_knee/translation_rescue_supervision_merge.py`

The generated artifact passed the frozen population and fingerprint checks:

```text
training_targets.csv SHA-256
c59d78c74743112f09946fd18b64d7726947e6f75b83aabd1f585389a89d045a

report-only studies                  4349
candidate active studies             4173
candidate inactive studies            176
candidate usable cells              18024
candidate positive cells             9590
candidate negative cells             8434
gold studies in output                  0
```

The merge preserves the original global rule:

```text
3120 originally B6-active studies
    -> frozen B6 unchanged

1053 Phase-7 rescued zero-cell studies
    -> all frozen recovered cells added

176 still-unrecovered studies
    -> remain zero-weight for supervised BCE
```

No target/script-specific filtering is permitted from the Phase-8 result. Phase 8 freezes a supervision candidate; it does not establish MRI performance or authorize model promotion.

## Phase 9 — FROZEN, READY TO RUN: matched B34 supervision experiment

Protocol: `developments/docs/DATASET_CONTRACT_AUDIT_PHASE9_MATCHED_SUPERVISION.md`

Implementation:

```text
developments/src/rsna_knee/phase9_supervision.py
developments/src/rsna_knee/phase9_matched_supervision_training.py
developments/src/rsna_knee/phase9_matched_supervision_eval.py
```

The old PV2 training split cannot test Phase 8 because it contains only original B6-active studies, and Phase 8 leaves those rows unchanged. Phase 9 therefore keeps the complete report-only MRI exposure identical in both arms:

```text
studies per arm                        4349
eligible real MRI series per arm      24035
batch size                                 2
batches/full epoch                      2175
fixed endpoint                            E2
architecture                              B34
encoder                         same frozen B16
crop                                same B20 90%
```

Only supervision changes:

```text
CONTROL: original B6 v1.2.1
active studies        3120
usable cells         14123
positive cells        6871
negative cells        7252

CANDIDATE: frozen Phase-8 merge
active studies        4173
usable cells         18024
positive cells        9590
negative cells        8434
```

The 1,229 B6-inactive studies remain inside the control dataloader with zero supervised BCE weight rather than being removed. This holds MRI acquisition-domain exposure constant between arms. Target-balance multipliers are recomputed mechanically from each arm using the unchanged frozen B7 formula; no manual target-specific adjustment is allowed.

Both arms use the same B34 construction, optimizer, scheduler horizon, augmentation, seeds and fixed E2 endpoint. The 58 gold studies remain outside gradients and checkpoint selection. A paired reused-gold macro-AUC evaluation is diagnostic only; hidden competition or new external expert-labelled evidence remains required for promotion.

## Current decision boundary

```text
globally increase 16 slice positions                    NO-GO
create adaptive 3D sampler now                           NO-GO
modify frozen B6 v1.2.1                                  NO-GO
fill partially silent B6-active studies                  NO-GO
target/script-specific rescue tuning                     NO-GO
add 58 gold studies to matched training                  NO-GO
define B35                                                NO-GO
rebuild/refilter frozen Phase-8 supervision               NO-GO
run Phase-9 matched B34 control/candidate                 GO
same 4349-study / 24035-series exposure in both arms      REQUIRED
reuse PV2 active-only training split for Phase 9           NO-GO
promotion from Phase 8 or reused-gold Phase 9 alone       NO-GO
verify compressed-DICOM codec capability                  GO before hidden submission
independent hidden competition comparison                 GO after Phase-9 checkpoints are frozen
```

B6 v1.2.1, PV1, PV2 and the Phase-8 supervision artifact remain frozen historical evidence.
