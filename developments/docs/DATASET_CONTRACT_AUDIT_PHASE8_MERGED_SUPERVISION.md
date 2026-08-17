# Phase 8 — frozen global B6 + Phase-7 merged supervision artifact

## Status

**FROZEN AFTER PHASE-7 RESULTS, BEFORE MRI TRAINING.**

Phase 7 recovered 3,901 definite cells from 1,053 of the 1,229 originally zero-cell report-only studies. Phase 8 turns that completed audit into one reproducible supervision artifact over all 4,349 report-only studies.

This stage still does **not** train an MRI model.

## Global merge rule

```text
4349 report-only studies
|
+-- 3120 originally B6-active
|      -> keep frozen B6 v1.2.1 exactly
|      -> do not add translated cells
|
+-- 1229 originally B6-inactive
       |
       +-- 1053 Phase-7 rescued
       |      -> add every frozen Phase-7 definite cell
       |
       +-- 176 unrecovered
              -> remain zero-usable-cell studies
```

The first downstream supervision experiment must use this rule globally. No Phase-7 target or script may be selectively accepted or rejected after seeing the population result.

## Frozen Phase-7 input

```text
recovered_cells.csv SHA-256
ed094e5d6f77b1558fe63921f2f22b8e1006443c506f00f921d842cde72025d0

expected recovered studies      1053
expected recovered cells        3901
expected positive cells         2719
expected negative cells         1182
```

## Expected merged surface

```text
report-only studies             4349
candidate active studies        4173
candidate inactive studies       176
candidate active fraction      95.95%

original B6 usable cells       14123
Phase-7 added cells             3901
candidate usable cells         18024
```

The 58 official gold studies remain excluded from the merged training-target artifact.

## Important class-balance warning

Phase-7 recovery is not balanced target by target. In particular, Synovitis adds 35 positives and zero negatives; OA additions are also strongly positive-skewed.

This is recorded as a risk, but **must not be repaired by post-hoc target filtering** in Phase 8. Doing so would use Phase-7 outcomes to choose a target-specific supervision policy. The first MRI experiment must compare one global merged policy against the original global B6 policy.

## Implementation

```text
developments/src/rsna_knee/translation_rescue_supervision_merge.py
```

The builder verifies:

- frozen B6 v1.2.1 population counts;
- exact Phase-7 recovered-cell SHA-256;
- no duplicate study/target recovered cells;
- recovered cells belong only to originally zero-cell studies;
- every original B6-active row remains exactly unchanged;
- no original usable B6 cell is overwritten;
- no target/script-specific filtering is performed;
- no gold study enters the output.

## Run

```bash
cd /media/talafha/Disk_1/CNN_CPC_current
conda activate rsna-knee
git pull --ff-only origin main

export B6_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/b6_report_labels_v121"
export PHASE7_ROOT="runs/report_translation_rescue_full"

PYTHONPATH=developments/src \
python -m rsna_knee.translation_rescue_supervision_merge \
  --b6-root "$B6_ROOT" \
  --phase7-root "$PHASE7_ROOT" \
  --out-root runs/translation_rescue_supervision_v1
```

Outputs:

```text
runs/translation_rescue_supervision_v1/
├── training_targets.csv
├── merge_audit.json
└── policy.json
```

These contain no raw report text or translation text.

## What comes after Phase 8

After the merged artifact is generated and its hashes/counts are checked, define one matched same-architecture MRI experiment:

```text
CONTROL
same architecture + original B6 v1.2.1

CANDIDATE
same architecture + frozen Phase-8 merged supervision
```

All MRI-side settings must be identical. The 58 gold studies remain out of gradients for this matched comparison.

The evaluation design must acknowledge that PV1/PV2 weak labels are B6-derived and therefore are not independent truth for a supervision method intended to repair B6. Reused gold is diagnostic only. Hidden competition evaluation remains the strongest available independent signal after the candidate is fully frozen.

## Current decision boundary

```text
build/fingerprint Phase-8 merged supervision       GO
post-hoc target filtering                          NO-GO
post-hoc script filtering                          NO-GO
fill silent targets inside B6-active studies       NO-GO
add 58 gold studies to matched training             NO-GO
change architecture simultaneously                 NO-GO
matched original-B6 vs merged-supervision training GO after Phase-8 artifact check
promotion from Phase 8 alone                        NO-GO
```
