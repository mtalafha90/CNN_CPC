# Completed Phase 4 supervision/acquisition-domain intersection audit

> **Descriptive data audit.** This result records the frozen B6 supervision-coverage intersection with MRI acquisition-domain metadata. It does not modify B6, identify institutions, define B35, authorize target-specific model changes, or promote a checkpoint.

## Population cross-check

```text
training studies                  4407
gold studies                        58
report-only studies               4349
B6-active report-only studies     3120
B6-inactive report-only studies   1229
B6 usable cells                  14123
```

The audit version was `official_dataset_domain_intersection_audit_v1`.

## Strong acquisition-domain association with B6 coverage

The B6-active and B6-inactive report-only populations differ substantially in acquisition composition:

| Metric | B6 active | B6 inactive |
|---|---:|---:|
| studies | 3120 | 1229 |
| mean series/study | 5.601 | 5.338 |
| studies with any known 3D series | 614 (19.68%) | 41 (3.34%) |
| studies with any >78-slice series | 546 (17.50%) | 41 (3.34%) |
| studies with any >100-slice series | 524 (16.79%) | 37 (3.01%) |
| studies with any >200-slice series | 87 (2.79%) | 0 (0%) |
| 3D series fraction | 4.47% | 0.75% |

Thus the report-label coverage selection is associated with a materially different MRI acquisition distribution. This is an association in the released dataset, not evidence that the B6 parser directly uses or causes MRI acquisition properties.

## Manufacturer-family composition

At the series level:

```text
B6 active
Siemens             38.19%
Philips             26.96%
GE                  26.58%
Canon/Toshiba        7.53%
Fujifilm/Hitachi     0.74%

B6 inactive
Siemens             51.77%
Philips             42.41%
GE                   5.58%
Canon/Toshiba        0.24%
Fujifilm/Hitachi     0.00%
```

The B6-active weak-supervision population therefore does not preserve the manufacturer-family composition of the B6-inactive population.

Within Latin-script report-only studies, a similar but smaller manufacturer-associated coverage difference remains at study level. The active fractions by dominant manufacturer family are approximately:

```text
Siemens             76.91%
Philips             72.89%
GE                  92.24%
Canon/Toshiba       98.66%
Fujifilm/Hitachi   100.00%
```

Therefore the supervision-selection effect is not explained by report script alone.

## Script × acquisition-domain intersection

The script buckets are also strongly associated with acquisition families in this release:

```text
Cyrillic report-only studies    217
  dominant/series manufacturer family: Philips only
  known 3D studies:               0
  >78-slice studies:              0

Greek report-only studies       318
  dominant/series manufacturer family: Siemens only
  known 3D studies:               0
  >78-slice studies:              0

Latin report-only studies      3814
  known 3D studies:             655 (17.17%)
  >78-slice studies:            587 (15.39%)
```

This does **not** establish hospital/site identity from report script. It does establish that the currently almost-unsupervised Greek/Cyrillic report strata occupy acquisition-domain regions that differ from much of the B6-active Latin population.

B6 coverage within those non-Latin strata remains extremely sparse:

```text
Cyrillic   1 active / 217 total   = 0.46%
Greek     38 active / 318 total   = 11.95%
Latin   3081 active / 3814 total  = 80.78%
```

Neither Greek nor Cyrillic report-only groups contain a known 3D or >78-slice study in this release.

## Gold-anchor composition

The 58 official fully labelled studies provide only a small direct anchor for non-Latin reports:

```text
Latin gold studies       52
Greek gold studies        3
Cyrillic gold studies     3
```

The three Greek gold studies are Siemens-family and the three Cyrillic gold studies are Philips-family, matching the broad manufacturer-family association of their report-only script strata. None of the six non-Latin gold studies contains a known 3D or >78-slice series.

This is useful for case-level validation but far too small to support high-precision script-specific performance claims by itself.

## Scientific interpretation

The Phase-1 supervision gap is now confirmed to be a **supervision-domain selection problem**, not merely a character-script coverage statistic. The 1,229 B6-inactive report-only studies are drawn from a different mixture of scanner/acquisition families than the 3,120 B6-active studies.

A multilingual-only repair would recover an important missing region (535 Greek/Cyrillic report-only studies), but it would not address all missing supervision because 733 Latin-script report-only studies are also B6-inactive. Future report-supervision work should therefore target the full B6-inactive population while explicitly auditing script and acquisition-domain coverage.

## Decision after Phase 4

```text
define B35 from these data                       NO-GO
globally increase slice sampling                 NO-GO
modify frozen B6 v1.2.1                         NO-GO
inspect actual report-text failure modes         GO
develop a separately versioned supervision candidate after inspection  GO
```

The next data step is a deterministic, local-only report-gap sample across Latin/Greek/Cyrillic active/inactive strata plus the six non-Latin gold cases. Raw report text from that sample must remain a local analysis artifact and must not be committed to the repository.
