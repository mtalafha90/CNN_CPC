# Phase 8 result — frozen global merged supervision

## Status

**COMPLETE / PASS.** The supplied Phase-8 artifact matches the predeclared global B6 + Phase-7 merge contract and is frozen for the first matched MRI supervision experiment.

Phase 8 itself does **not** establish model improvement and does not authorize model promotion.

## Frozen artifact fingerprints

```text
training_targets.csv
SHA-256  c59d78c74743112f09946fd18b64d7726947e6f75b83aabd1f585389a89d045a

merge_audit.json
SHA-256  7f3af0a70758de67d9dc5baf7e997f9ecc7119a6c5ae97602928c2c48402ed5a

policy.json
SHA-256  07055654bc53bf1e369e29b09b9e4b246346faa53a43e967c3bc926171d40ae1

submitted ZIP package
SHA-256  1eefd2611ed37f55f5f7fd43f8c583b70305898f013ae46be56ff294f1d3a70e
```

The merge audit points to the exact frozen Phase-7 recovered-cell artifact:

```text
Phase-7 recovered_cells.csv
SHA-256  ed094e5d6f77b1558fe63921f2f22b8e1006443c506f00f921d842cde72025d0
```

## Verified population totals

```text
report-only studies                 4349
unique StudyInstanceUID values      4349
duplicate StudyInstanceUID values      0
candidate active studies            4173
candidate inactive studies           176
usable cells                       18024
positive cells                      9590
negative cells                      8434
gold studies                           0
```

The merged population therefore changes report-only supervision coverage from:

```text
original B6 active        3120 / 4349 = 71.74%
Phase-8 candidate active  4173 / 4349 = 95.95%
```

and changes usable cells from:

```text
14123 -> 18024
+3901 cells (+27.62%)
```

## Target-level reproduction

The final CSV reproduces the frozen Phase-7 target totals exactly:

| Target | Original B6 usable | Added | Phase-8 usable |
|---|---:|---:|---:|
| ACL | 1661 | 435 | 2096 |
| MCL | 1360 | 267 | 1627 |
| Medial Meniscus | 1662 | 699 | 2361 |
| Lateral Meniscus | 1630 | 592 | 2222 |
| Medial OA | 818 | 206 | 1024 |
| Lateral OA | 784 | 137 | 921 |
| PF OA | 1054 | 276 | 1330 |
| Effusion | 2095 | 597 | 2692 |
| Synovitis | 416 | 35 | 451 |
| Baker's | 1033 | 356 | 1389 |
| Contusion | 855 | 243 | 1098 |
| Fracture | 755 | 58 | 813 |

The previously recorded imbalance remains part of the frozen candidate. In particular, Synovitis receives 35 positive and zero negative recovered cells, and the OA additions are strongly positive-skewed. These observations must not be used for post-hoc target filtering.

## State encoding check

Definite cells in the supplied merged artifact use the expected frozen encoding:

```text
positive  confidence 0.90   probability 0.97
negated   confidence 0.90   probability 0.03
```

Uncertain and unmentioned states remain below the frozen B7 usable-cell threshold and therefore receive zero supervised BCE weight.

## Guardrails certified by the builder

The Phase-8 merge audit certifies:

```text
all original B6-active rows preserved exactly       true
original usable B6 cells overwritten                   0
partially silent B6-active cells filled             false
target-specific filtering                           false
script-specific filtering                           false
gold studies in training                            false
MRI model trained during Phase 8                    false
```

The final Phase-8 ZIP does not contain the original B6 table and Phase-7 recovered-cell table, so those source-to-output preservation assertions cannot be independently reconstructed from the final package alone. They are enforced by the frozen merge builder, which aborts if an original active row changes or an original usable cell is overwritten, and the builder pins the exact Phase-7 recovered-cell SHA-256.

## Decision

```text
freeze Phase-8 merged supervision                    GO / COMPLETE
repeat Phase 8                                       NO-GO
post-hoc target filtering                            NO-GO
post-hoc script filtering                            NO-GO
change frozen B6                                     NO-GO
promote an MRI model from Phase 8                    NO-GO
run matched MRI supervision experiment               GO
```

The next stage is Phase 9: a matched same-architecture original-B6 versus Phase-8-supervision MRI experiment.
