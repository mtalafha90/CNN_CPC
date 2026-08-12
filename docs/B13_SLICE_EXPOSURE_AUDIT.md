# B13 exact slice-exposure audit

> **Status — 2026-08-12:** **COMPLETE. Slice-count undersampling remains rejected as a primary B13 bottleneck.** Subsequent B15 also failed to improve global reused-gold macro AUC, so no slice-count sweep has been reopened.

## Frozen surface

```text
active B6 studies       3120
eligible real series   17475
readable series        17475 / 17475
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

The audit reproduces B13's actual sampling:

```text
16 center positions / real series
2.5D triplet around every center
training gap choices [1,2]
training center jitter +/-2
evaluation gap 1
evaluation TTA offsets [-1,0,1]
orientation-aware through-plane DICOM coordinates
```

No gold labels or pixels entered the audit.

## Full 17,475-series result

```text
series audited/readable  17475 / 17475
slices/series median     30
slices/series p95        50
slices/series max        320

frozen B13 exposure
  eval unique fraction   median 100.0% (p25 100.0%)
  eval max skipped run   median 0.0 slices (p95 0.0)
  training expected/view median 87.0%
  complete eval exposure 95.9%
  eval run >=2 slices    3.9%
  eval run >=3 slices    3.8%
  skipped-run length     median 0.0 mm (p95 0.0 mm)
```

Plane breakdown:

```text
Axial      n=4455   eval=100.0% max-run=0.0 train/view=85.2%
Coronal    n=5815   eval=100.0% max-run=0.0 train/view=87.0%
Sagittal   n=7205   eval=100.0% max-run=0.0 train/view=87.0%
```

## Interpretation

The exact B13 evaluation sampler exposes essentially the complete ordinary MRI series. `95.9%` of eligible series have complete evaluation exposure, and the median/p95 longest skipped runs are both zero.

Decision:

```text
slice-count undersampling as primary B13 bottleneck -> REJECT
```

Do not launch a 24/32/48-slice sweep from the repeatedly reused gold surface. This result does **not** rule out in-plane resolution loss at `224x224`, representation limitations, weak-label noise/sparsity or optimization limitations.

## Successor context through B15

```text
B13 gold     0.6293565948  retained champion
B14 gold     0.6197914249  rejected
B15 gold     0.6209002783  no global improvement
```

B15 nevertheless raised frozen weak-v2 teacher agreement from matched-control `0.5652498118` to `0.7319060415`, with paired median `+0.1675245839` and 95% CI `[+0.1124433208,+0.2165156305]`. The lack of expert-gold transfer shifts the immediate diagnostic priority toward supervision quality rather than slice count.

## Reproduction

```bash
rsna-knee-slice-audit \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out runs/slice_audit_b13
```

Artifacts:

```text
runs/slice_audit_b13/slice_audit.csv
runs/slice_audit_b13/slice_audit.json
```

Current campaign status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). Post-B15 roadmap: [`RAISING_AUC.md`](RAISING_AUC.md).