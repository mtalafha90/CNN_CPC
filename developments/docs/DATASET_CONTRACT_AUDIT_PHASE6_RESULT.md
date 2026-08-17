# Phase 6 result — translation to frozen-B6 rescue feasibility

## Status

**COMPLETE — PREDECLARED FEASIBILITY GATE PASSED.**

This result is a supervision-coverage mechanism result only. It does not modify B6 v1.2.1, train an MRI model, define B35, or authorize model promotion.

Input pilot:

```text
version                   translation_to_frozen_b6_rescue_pilot_v1
Phase-5 sample version    report_supervision_gap_sample_v1
sample SHA-256            ec431db7539a75dd0aad786b2b8442f4af1124697590e9650481bd424f0d7a01
selected studies          79
inactive primary studies  36
```

## Translator provenance

The completed run used the frozen reproducible local translator:

```text
backend             Ollama local
model               qwen3:14b
Ollama digest        bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8
quantisation         Q4_K_M
decoding             greedy
seed                 2026
max_new_tokens       4096
prompt SHA-256       086e1daae2843c70712a29662a589dee629d32d7f014a9a51613be496a95ee1a
provenance pinned    yes
translation failures 0
```

The language model was used only to translate the reports into English. The unchanged frozen B6 v1.2.1 parser produced the candidate target states.

## Predeclared gate result

All Phase-6 feasibility conditions passed.

| Criterion | Frozen requirement | Observed | Result |
|---|---:|---:|---|
| Translation failures | 0 | 0 | PASS |
| Overall rescue rate | >=75% | 31/36 = 86.11% | PASS |
| Latin rescue rate | >=50% | 12/12 = 100% | PASS |
| Greek rescue rate | >=50% | 7/12 = 58.33% | PASS |
| Cyrillic rescue rate | >=50% | 12/12 = 100% | PASS |
| Positive + negative cells in each script stratum | yes | yes | PASS |
| Original B6-active controls preserved | 100% | 25/25 preserved | PASS |

Therefore:

```text
coverage-mechanism feasibility   PASS
clinical-label validation        NOT ESTABLISHED
MRI training authorization       NO
model promotion authorization    NO
```

## Recovered supervision

Across the 36 originally B6-inactive primary reports, translation followed by frozen B6 recovered:

```text
rescued studies       31 / 36
added usable cells    112
added positive cells   81
added negative cells   31
```

By script:

| Stratum | Studies | Rescued | Added cells | Positive | Negative |
|---|---:|---:|---:|---:|---:|
| Latin inactive | 12 | 12 | 45 | 34 | 11 |
| Greek inactive | 12 | 7 | 25 | 20 | 5 |
| Cyrillic inactive | 12 | 12 | 42 | 27 | 15 |

The five unrecovered primary studies were all in the Greek-script stratum. Their translations completed successfully; the remaining failure is therefore downstream of translation and reflects residual frozen-B6 English terminology/aggregation limitations rather than translation failure itself. Examples in the local translated sample include grouped normal-structure statements and wording variants for small joint fluid or Baker cyst that do not necessarily map to a definite B6 cell. No rule changes are permitted from this sample.

## Descriptive target distribution of recovered cells

The 112 added cells were distributed across the 12 targets as follows:

| Target | Added usable | Positive | Negative |
|---|---:|---:|---:|
| ACL | 9 | 6 | 3 |
| MCL | 7 | 3 | 4 |
| Medial Meniscus | 23 | 23 | 0 |
| Lateral Meniscus | 17 | 9 | 8 |
| Medial OA | 3 | 3 | 0 |
| Lateral OA | 1 | 1 | 0 |
| PF OA | 6 | 6 | 0 |
| Effusion | 19 | 11 | 8 |
| Synovitis | 1 | 1 | 0 |
| Baker's | 11 | 8 | 3 |
| Contusion | 9 | 6 | 3 |
| Fracture | 6 | 4 | 2 |

This table is descriptive only. It must not be used to invent target-specific rescue thresholds or target-specific inclusion rules.

## Reused-gold safety diagnostic

The pilot translated 18 reused gold reports for diagnostic safety only:

```text
official cells                         216
translated-B6 definite calls           109
translated-B6 definite coverage        50.46%
definite-call accuracy                  74.31%
positive-call precision                 68.54%
negative-call precision                100.00%
```

These are not acceptance or promotion metrics. The gold surface has been repeatedly reused, only six non-Latin gold studies exist in the full dataset, and the Phase-5 investigation directly inspected sampled report text.

The positive-call precision also cautions against treating every recovered translated-B6 positive as expert truth. Phase 6 establishes coverage feasibility, not clinical accuracy.

## Scientific interpretation

Phase 6 supports the narrow mechanism proposed before the run:

> Language normalization by deterministic translation can recover substantial positive and negative supervision from reports that are completely silent to the original B6 parser, while leaving all existing B6-active studies untouched.

The result is especially strong for the Latin and Cyrillic sampled inactive strata and weaker, though still above the frozen gate, for the Greek stratum.

It does **not** justify changing B6 itself, filling partially silent cells in B6-active studies, target-wise rescue rules, B35, or direct MRI retraining yet.

## Decision

```text
Phase-6 feasibility                           GO / PASS
modify B6 v1.2.1                              NO-GO
translate/fill partially silent active cases  NO-GO
per-target rescue tuning                      NO-GO
define B35                                     NO-GO
full 1,229-study inactive-population audit    GO
MRI training from rescued supervision         NOT YET AUTHORIZED
```

The next allowed step is a frozen full-population translation-rescue audit over all 1,229 original B6-inactive report-only studies. That stage must use the exact Phase-6 translator provenance and the same zero-original-cell-only merge rule, then inspect aggregate study coverage, target balance and acquisition-domain recovery before any downstream MRI experiment is defined.

Raw translations remain local-only and must not be committed to GitHub.
