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

The active controls confirm the mechanism:

```text
12/12 sampled Greek B6-active reports:
    exactly one usable cell each
    Contusion positive
    driven by embedded English `bone bruise`

single Cyrillic B6-active report:
    exactly one usable cell
    Fracture positive
    driven by embedded English `subchondral insufficiency fracture`
```

Latin-script failure is also substantial. The 12 sampled inactive Latin-script reports included South-Slavic Latin-script, Turkish and Spanish reports with clear target findings that the frozen rule vocabulary did not resolve.

Consequently:

```text
modify B6 v1.2.1 in place                         NO-GO
Greek/Cyrillic-only lexicon patch                  NO-GO
revive unrestricted B23 replacement                NO-GO
define B35                                          NO-GO
constrained translation -> frozen-B6 rescue pilot   GO
```

Raw Phase-5 report text remains a local-only artifact and must not be committed.

## Phase 6 — FROZEN, READY TO RUN: translation -> frozen-B6 rescue

Protocol recorded in:

```text
developments/docs/DATASET_CONTRACT_AUDIT_PHASE6_TRANSLATION_RESCUE.md
```

Implementation:

```text
developments/src/rsna_knee/report_translation_rescue_pilot.py
```

The hypothesis is deliberately narrow:

```text
original report
  -> deterministic pinned local translation to English
  -> unchanged frozen B6 v1.2.1
  -> translated-B6 cells eligible ONLY when original study has zero usable B6 cells
```

Original B6-active studies remain unchanged by construction. The translator is not asked to classify the 12 targets.

Frozen feasibility criteria, declared before translation results are observed:

```text
translation failures                                      0
overall rescue among 36 inactive sample reports          >=75%
rescue in each Latin/Greek/Cyrillic inactive stratum     >=50%
each inactive script stratum recovers positive + negative cells
all original B6-active control cells preserved            yes
```

Gold metrics are diagnostic safety information only because the gold surface is reused and Phase 5 directly inspected the report texts.

Run:

```bash
cd /media/talafha/Disk_1/CNN_CPC_current
conda activate rsna-knee
git pull --ff-only origin main

PYTHONPATH=developments/src \
python -m rsna_knee.report_translation_rescue_pilot \
  --sample-jsonl runs/report_supervision_gap_audit/report_text_sample.jsonl \
  --out-root runs/report_translation_rescue_pilot \
  --model qwen3:14b \
  --num-ctx 8192 \
  --max-new-tokens 4096
```

The translated-text output is local-only and must not be committed.

## Current decision boundary

```text
globally increase 16 slice positions              NO-GO
create adaptive 3D sampler now                     NO-GO
modify frozen B6 v1.2.1                            NO-GO
define B35                                         NO-GO
run frozen Phase-6 translation rescue pilot        GO
full 1229-study translation rescue                 only if Phase 6 passes
MRI training using translation rescue              NOT YET AUTHORIZED
verify compressed-DICOM codec capability           GO before hidden submission
```

B6 v1.2.1, PV1 and PV2 remain frozen historical evidence.
