# Phase 7 result — full B6-inactive translation-rescue population audit

## Status

**COMPLETE. GO to freeze a global merged-supervision candidate.**

Phase 7 applied the exact Phase-6 translation mechanism to all 1,229 report-only studies that had zero usable cells under frozen B6 v1.2.1. No B6-active study was modified, no target-specific rescue rule was introduced, and no MRI model was trained.

## Input/output fingerprints

```text
full_population_summary.json
SHA-256  4d15eae7a17807da2624644cc59ea5afb8fe768dd8cb2a83cc592219a920df2e

full_population_rescue_audit.csv
SHA-256  bef4d707727bbe4d793cb506d1810716a66b39949c56d74432970ee0c0c42831

recovered_cells.csv
SHA-256  ed094e5d6f77b1558fe63921f2f22b8e1006443c506f00f921d842cde72025d0
```

The reported translator provenance matches the successful Phase-6 pilot exactly:

```text
backend              Ollama local
model                qwen3:14b
Ollama digest        bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8
quantisation         Q4_K_M
decoding             greedy
seed                 2026
max_new_tokens       4096
prompt SHA-256       086e1daae2843c70712a29662a589dee629d32d7f014a9a51613be496a95ee1a
```

## Integrity checks

The supplied result files are internally consistent:

```text
eligible studies                         1229
successful translations                  1229
translation failures                        0
duplicate study rows                        0
duplicate StudyUID/target recovered cells   0
recovered_cells.csv rows                 3901
positive rows                            2719
negative rows                            1182
other states                                0
recovered-cell confidence                  0.90
```

The audit totals and recovered-cell rows agree exactly.

## Main result

```text
original B6-active report-only studies        3120 / 4349 = 71.74%
newly rescued zero-cell studies               1053 / 1229 = 85.68%
candidate active report-only studies          4173 / 4349 = 95.95%
unrecovered report-only studies                176 / 4349 =  4.05%

original B6 usable cells                     14123
new recovered usable cells                    3901
candidate usable cells                       18024
usable-cell increase                         +27.62%

new positive cells                            2719
new negative cells                            1182
```

Phase 7 therefore recovers supervision for most of the population that had been completely absent from B6, while preserving every original B6-active study unchanged.

## Recovery by report script

| Script | Eligible | Rescued | Rescue rate | Added cells | Positive | Negative |
|---|---:|---:|---:|---:|---:|---:|
| Latin | 733 | 610 | 83.22% | 2367 | 1510 | 857 |
| Greek | 280 | 228 | 81.43% | 705 | 632 | 73 |
| Cyrillic | 216 | 215 | 99.54% | 829 | 577 | 252 |

The full-population Greek result is substantially stronger than the small Phase-6 pilot estimate (7/12), showing why the Phase-7 population audit was necessary before downstream modelling.

## Recovery by target

| Target | Original B6 usable | Added | Candidate usable | Candidate positive | Candidate negative |
|---|---:|---:|---:|---:|---:|
| ACL | 1661 | 435 | 2096 | 844 | 1252 |
| MCL | 1360 | 267 | 1627 | 325 | 1302 |
| Medial Meniscus | 1662 | 699 | 2361 | 1711 | 650 |
| Lateral Meniscus | 1630 | 592 | 2222 | 717 | 1505 |
| Medial OA | 818 | 206 | 1024 | 683 | 341 |
| Lateral OA | 784 | 137 | 921 | 534 | 387 |
| PF OA | 1054 | 276 | 1330 | 935 | 395 |
| Effusion | 2095 | 597 | 2692 | 1771 | 921 |
| Synovitis | 416 | 35 | 451 | 434 | 17 |
| Baker's | 1033 | 356 | 1389 | 783 | 606 |
| Contusion | 855 | 243 | 1098 | 602 | 496 |
| Fracture | 755 | 58 | 813 | 251 | 562 |

### Important imbalance warning

The translation-rescue mechanism is not class-balanced target by target. In particular:

```text
Synovitis additions       35 positive / 0 negative
Medial OA additions      199 positive / 7 negative
Lateral OA additions     132 positive / 5 negative
PF OA additions          253 positive / 23 negative
```

This must **not** be corrected by post-hoc target filtering or target-specific rescue rules derived from Phase-7 outcomes. The first downstream experiment must use one global frozen merge policy. The imbalance is a property to measure, not a reason to cherry-pick rescued targets.

## Acquisition-domain recovery

Among the 1,229 originally B6-inactive studies, Phase 4 found 41 with known 3D series, 41 with >78-slice series and 37 with >100-slice series. Phase 7 rescues:

```text
known-3D inactive studies rescued             38 / 41 = 92.68%
>78-slice inactive studies rescued            38 / 41 = 92.68%
>100-slice inactive studies rescued           34 / 37 = 91.89%
```

Combining original B6-active studies with the rescued population gives supervised coverage of:

```text
known-3D report-only studies                 652 / 655 = 99.54%
>78-slice report-only studies                584 / 587 = 99.49%
>100-slice report-only studies               558 / 561 = 99.47%
>200-slice report-only studies                87 / 87  = 100%
```

The rescued population's dominant manufacturer families are:

```text
Siemens   590
Philips   394
GE         69
```

The 176 unrecovered studies are predominantly Philips (111), followed by Siemens (62) and Canon/Toshiba (3).

Thus the original B6 supervision/acquisition-domain selection problem is greatly reduced by the global rescue mechanism, although not eliminated completely.

## Scientific interpretation

Phase 7 supports the mechanism that deterministic translation followed by unchanged frozen B6 can recover a large amount of otherwise missing report supervision and substantially broaden the acquisition-domain representation of the supervised population.

It does **not** establish that every recovered cell is clinically correct. The Phase-6 reused-gold diagnostic remains the only direct safety signal available for the translation pathway and is not independent validation. Therefore the Phase-7 result authorizes a **frozen global merged-supervision candidate and matched downstream experiment**, not model promotion.

## Decision

```text
freeze B6 + Phase-7 global merged supervision candidate     GO
preserve every original B6 cell exactly                     REQUIRED
include all Phase-7 recovered cells globally                REQUIRED
post-hoc target/script filtering                            NO-GO
fill partially silent B6-active studies                     NO-GO
modify B6 v1.2.1                                             NO-GO
add 58 gold studies to matched training comparison           NO-GO
define a new architecture because of Phase 7                NO-GO
run matched same-architecture B6 vs B6+rescue experiment    GO after merge artifact is frozen
model promotion from Phase 7 alone                           NO-GO
```

The next step is to build and fingerprint the merged supervision artifact over all 4,349 report-only studies, while keeping the 58 official gold studies outside training gradients for the matched comparison.
