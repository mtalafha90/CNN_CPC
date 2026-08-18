# Phase 9 v2 — descriptive target-level follow-up

## Status

**POST-HOC DESCRIPTIVE ANALYSIS ONLY.**

This follow-up was authorized only after the frozen Phase-9 v2 global result had been observed. It therefore cannot redefine the Phase-9 endpoint, change the Phase-8 merge, select translated cells, retune target weights, or create target-specific model mixtures.

It uses no additional MRI training and does not load a checkpoint. It consumes only:

```text
runs/phase9_matched_supervision_v2/eval/paired_pv2_predictions.csv
frozen B6 v1.2.1 supervision
frozen Phase-8 merged supervision
```

## Questions

1. Are the observed per-target AUC changes robust under paired whole-study resampling?
2. How strongly does each target influence the global macro-AUC difference?
3. Which target-level supervision additions produced the frozen +3,901-cell treatment?
4. Is there any descriptive association between the amount/class balance of rescued supervision and the observed target-level AUC shift?

The fourth question is exploratory only. There are only 12 target-level observations, so any correlation is descriptive and must not be interpreted causally.

## Frozen analyses

### Per-target paired AUC bootstrap

For each of the 12 targets:

```text
truth = original B6 positive / negated state on the 499-study PV2 holdout
resampling unit = StudyInstanceUID
control and candidate use the identical sampled studies
requested bootstrap replicates = 5000
```

A replicate is used for a target only when that target retains both classes after resampling.

Outputs include:

```text
control AUC
candidate AUC
candidate - control point difference
bootstrap median difference
95% percentile interval
P(candidate > control)
valid replicate count
holdout positive/negative cell counts
```

These target-wise intervals are descriptive. No multiplicity-adjusted target-level confirmatory claims are made.

### Macro target influence

The analysis reconstructs the 12-target macro point difference and computes the macro difference after removing each target one at a time.

This identifies whether the sign of the global point estimate depends strongly on one pathology. It does not justify removing that pathology from the competition metric or training objective.

### Exact rescued-supervision counts

The code loads the frozen B6 and Phase-8 artifacts and derives, per target:

```text
original usable / positive / negative cells
added usable / positive / negative cells
candidate usable / positive / negative cells
added-positive fraction
```

The analysis aborts unless the derived global additions are exactly:

```text
added usable      3901
added positive    2719
added negative    1182
```

### Descriptive association

Two target-level Pearson correlations are reported only as exploratory descriptors:

```text
added usable cells vs target delta AUC
added positive fraction vs target delta AUC
```

With only 12 targets, these are not inferential evidence and must not drive policy changes.

## Command

```bash
cd /media/talafha/Disk_1/CNN_CPC_current
conda activate rsna-knee
git pull --ff-only origin main

PYTHONPATH=developments/src \
python -m rsna_knee.phase9_v2_target_analysis \
  --paired-predictions runs/phase9_matched_supervision_v2/eval/paired_pv2_predictions.csv \
  --b6-root "$B6_ROOT" \
  --phase8-root "$PHASE8_ROOT" \
  --out-root runs/phase9_matched_supervision_v2/target_analysis \
  --n-bootstrap 5000
```

## Outputs

```text
runs/phase9_matched_supervision_v2/target_analysis/
├── per_target_bootstrap.csv
├── rescued_supervision_by_target.csv
├── macro_target_influence.csv
└── target_analysis.json
```

## Governance

Permitted interpretation:

> The global Phase-9 v2 weak-label macro-AUC result was heterogeneous across pathologies, and this descriptive analysis quantifies where the point differences and uncertainty lie.

Not permitted:

```text
remove translated cells from a target because its point AUC fell
up-weight a target because its point AUC rose
construct a control/candidate target-wise ensemble
retrain B34 from these target results
change the Phase-8 merge
call a per-target result clinically validated
```

Any future target-specific supervision experiment must be separately predeclared and validated on a fresh surface.
