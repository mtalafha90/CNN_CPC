# B13 exact slice-exposure audit

> **Status — 2026-08-11:** **COMPLETE. Slice-count undersampling rejected as a primary B13 bottleneck.**

This diagnostic used the corrected audit in package `0.22.1` on the exact frozen B13 non-gold surface.

## Frozen surface

```text
active B6 studies       3120
eligible real series   17475
readable series        17475 / 17475
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

The audit reads DICOM headers only and reproduces B13 sampling rather than using the retired `16 / n_slices` proxy:

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

The exact B13 evaluation sampler exposes essentially the complete ordinary MRI series. `95.9%` of all eligible series have complete evaluation exposure, the median and p95 longest skipped runs are both zero, and median evaluation exposure is `100%` in every plane.

The remaining long-tail series include acquisitions as long as 320 frames, but only `3.9%` of all series have an evaluation gap of at least two slices. This is not evidence for a global slice-count bottleneck.

Therefore the controlled development conclusion is:

```text
slice-count undersampling as primary B13 bottleneck -> REJECT
```

Do not launch a 24/32/48-slice sweep from the reused 58-study gold surface. If a later slice-budget experiment is undertaken for another reason, it must be globally predeclared and evaluated without target-specific tuning.

This result does **not** rule out in-plane resolution loss at `224x224`, MRI representation limitations, weak-label noise/sparsity, or optimization limitations.

## Command used

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-slice-audit \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out runs/slice_audit_b13
```

Local artifacts:

```text
runs/slice_audit_b13/slice_audit.csv
runs/slice_audit_b13/slice_audit.json
```
