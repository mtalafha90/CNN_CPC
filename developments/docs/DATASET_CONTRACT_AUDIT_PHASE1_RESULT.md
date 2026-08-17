# Dataset contract audit — Phase 1 result

> **Descriptive data result.** This audit is independent of model selection. It does not change B6, PV1, PV2, B20, B31, or B34 status.

## Exact local release fingerprints

```text
train.csv SHA256
8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33

train_series.csv SHA256
573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3
```

All results below are tied to those files.

## 1. Official-label structure

The local training table contains:

```text
training studies                 4407
fully labelled for all 12          58
partially labelled                  0
zero official labels             4349
non-empty reports                4407
empty reports                       0
```

Therefore the historical statement that there are 58 fully labelled expert/gold studies is confirmed for this exact release. The repository `gold_mask()` definition is technically `ANY` populated target, but in this release every gold row has all twelve targets populated, so `ANY` and `ALL` select the same 58 studies.

Official positives/negatives among the 58 fully labelled studies:

| Target | Positive | Negative | Positive prevalence |
|---|---:|---:|---:|
| ACL | 24 | 34 | 0.4138 |
| MCL | 9 | 49 | 0.1552 |
| Medial Meniscus | 26 | 32 | 0.4483 |
| Lateral Meniscus | 23 | 35 | 0.3966 |
| Medial OA | 15 | 43 | 0.2586 |
| Lateral OA | 11 | 47 | 0.1897 |
| PF OA | 21 | 37 | 0.3621 |
| Effusion | 35 | 23 | 0.6034 |
| Synovitis | 27 | 31 | 0.4655 |
| Baker's | 12 | 46 | 0.2069 |
| Contusion | 19 | 39 | 0.3276 |
| Fracture | 18 | 40 | 0.3103 |

The gold subset is small but every target has both classes, so per-target ROC AUC is defined on the complete 58-study set. It remains a repeatedly reused development surface, not independent validation.

## 2. Report character-system composition

The audit uses Unicode script buckets, not language inference. A script bucket must not be interpreted as a specific language, country, site, or institution.

All 4,407 reports:

| Script bucket | Studies | Fraction | Gold |
|---|---:|---:|---:|
| Latin | 3866 | 87.72% | 52 |
| Greek | 321 | 7.28% | 3 |
| Cyrillic | 220 | 4.99% | 3 |

All 4,407 reports are non-empty.

Among the 4,349 report-only studies:

```text
Latin       3814  (87.70%)
Greek        318  (7.31%)
Cyrillic     217  (4.99%)
```

## 3. Critical B6 coverage finding

B6 v1.2.1 at the frozen 0.75 usable-cell threshold produces:

```text
report-only studies             4349
B6-active studies               3120
inactive / zero usable cells    1229
usable cells                   14123
positive cells                  6871
negative cells                  7252
```

Coverage by Unicode script bucket:

| Script bucket | Report-only studies | B6-active | Active fraction | Usable cells | Positive | Negative | Usable cells/study |
|---|---:|---:|---:|---:|---:|---:|---:|
| Latin | 3814 | 3081 | 80.78% | 14083 | 6831 | 7252 | 3.6924 |
| Greek | 318 | 38 | 11.95% | 39 | 39 | 0 | 0.1226 |
| Cyrillic | 217 | 1 | 0.46% | 1 | 1 | 0 | 0.0046 |

This is the dominant Phase-1 data finding.

Although Latin-script reports are 87.70% of the report-only population, they become 98.75% of the 3,120-study B6-active population. They also contribute 14,083 of 14,123 usable cells (99.72%). Every one of the 7,252 B6 negated cells comes from the Latin-script bucket; the Greek and Cyrillic buckets contribute only 40 usable cells total, all positive.

The result demonstrates a large **script-associated weak-supervision coverage shift**. It does not by itself identify the language, site, cause, or clinical distribution of the excluded studies. It is therefore a data-selection-bias warning, not proof of institution leakage.

The current B20/PV1/PV2 downstream training surface is consequently dominated by Latin-script reports even though approximately 12.3% of report-only studies use Greek or Cyrillic characters.

## 4. Supplied MRI-series contract

`train_series.csv` contains:

```text
series rows / unique series      24371
studies represented               4407
studies without a listed series      0
series per study mean             5.530
median                             5
95th percentile                    9
99th percentile                   10
maximum                           14
minimum                            3
```

The 58 gold studies are similar in listed series count to the report-only population:

```text
                         gold     report-only
mean series/study        5.79        5.53
median                   5           5
95th percentile          9           9
```

This does not show a major gold-vs-report-only difference in the number of supplied MRI acquisitions. It does not yet test scanner, resolution, slice count, or sequence-content differences.

Anatomical-plane distribution:

```text
Sagittal     9864  40.47%
Coronal      8609  35.32%
Axial        5898  24.20%
```

No supplied plane, fluid-sensitive, or fat-suppression metadata values are missing in `train_series.csv`.

## 5. Fluid-sensitive / fat-suppression redundancy in the training table

The supplied training metadata show:

```text
Fluid_Sensitive=True     14010
Fluid_Sensitive=False    10361
Fat_Suppression=True     14010
Fat_Suppression=False    10361
```

More importantly, all observed train-series combinations are either:

```text
fluid=True  AND fat=True
or
fluid=False AND fat=False
```

There are no `fluid=True/fat=False` or `fluid=False/fat=True` rows among the 24,371 listed training series.

This means the two supplied flags are perfectly redundant on the local training release, even though the competition data contract warns they are not necessarily equivalent for every case. Any model that learns from both independently should therefore be tested for robustness to unseen combinations before hidden-test inference; no test-time equivalence should be assumed.

Plane × flag counts:

```text
Axial     fluid=False fat=False   1179
Axial     fluid=True  fat=True    4719
Coronal   fluid=False fat=False   3985
Coronal   fluid=True  fat=True    4624
Sagittal  fluid=False fat=False   5197
Sagittal  fluid=True  fat=True    4667
```

## 6. Interpretation and next data work

Phase 1 changes the priority of the data investigation.

The highest-priority issue is not another architecture variant. It is the severe script-associated B6 coverage imbalance. Before any B35 definition, the project should determine whether multilingual report processing can recover scientifically defensible supervision from the currently almost-unused Greek- and Cyrillic-script studies without changing the existing frozen B6/PV1/PV2 evidence retroactively.

The next descriptive pass remains the physical DICOM slice-count scan. It will quantify the actual long tail relative to the fixed 16-position sampling policy.

After the slice-count pass, the next report audit should evaluate the 58 fully labelled reports under a separately versioned multilingual label-extraction candidate and compare it globally with the official labels. Any new parser must be treated as a new supervision experiment; B6 v1.2.1 remains frozen for all historical results.

## Governance

```text
B20  remains active historical/predictive model
B31  remains PV1-selected downstream architecture
B34  remains frozen successful PV2 mechanism architecture
B6   remains frozen v1.2.1 for all historical experiments
B35  not defined
```

Do not retroactively reinterpret PV1/PV2 as multilingual validation, do not relabel old runs, and do not use target-specific Phase-1 findings to construct switches or blends.
