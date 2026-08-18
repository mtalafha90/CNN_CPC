# Phase 9 v2 pre-evaluation AUC addendum

## Status

**FROZEN AFTER BOTH FIXED-E2 CHECKPOINTS WERE WRITTEN AND BEFORE ANY 499-STUDY PV2-HOLDOUT PREDICTIONS OR METRICS WERE GENERATED OR INSPECTED.**

This addendum does not modify either Phase-9 v2 training arm, checkpoint, split, supervision table, architecture, seed, endpoint, or the original primary endpoint.

The motivation is metric alignment. The RSNA Knee Abnormality Detection competition is scored by macro ROC AUC across the 12 targets, whereas the frozen PV1/PV2 weak-label protocols used macro original-B6-weighted soft-label BCE as the primary metric and weak-state AUC only as an unbootstrapped secondary point estimate.

The historical PV1 result already showed why both views matter: B31 and B33 were clearly separated by soft BCE while their weak-state macro AUC point estimates were nearly identical. That observation was known before this addendum and motivates adding uncertainty for the ranking metric without rewriting the original Phase-9 v2 endpoint.

## Original Phase-9 v2 primary remains unchanged

```text
macro of per-target original-B6-weighted soft-label BCE
lower is better
```

Paired difference:

```text
candidate BCE - control BCE
negative favors Phase-8 supervision
```

The original paired 5,000-replicate study bootstrap remains the primary inferential test.

## Added key secondary: paired macro ROC AUC

The addendum adds:

```text
macro ROC AUC across all 12 targets
truth = original frozen B6 positive/negated states on the 499-study PV2 holdout
candidate AUC - control AUC
positive favors Phase-8 supervision
```

This is **competition-aligned in metric form only**. The truth remains B6 weak supervision, not expert labels and not Kaggle hidden labels. PV2 remains historically exposed and is not independent clinical validation.

### Bootstrap policy

- resampling unit: whole `StudyInstanceUID`;
- paired resampling: the same sampled studies are used for control and candidate;
- requested replicates: 5,000;
- every accepted replicate must define AUC for **all 12 targets** for both arms;
- a replicate that loses one class for any target is discarded rather than silently changing the macro estimand;
- report point difference, median bootstrap difference, 95% percentile interval, probability candidate > control, and number of valid replicates.

No equivalence or superiority threshold beyond zero is introduced after training.

## Per-target output

For transparency the addendum records, for each target:

```text
active original-B6 holdout cells
positive cells
negative cells
control AUC
candidate AUC
candidate - control AUC
```

These are **descriptive only**. They may not be used to:

```text
remove translated cells
change target weights
create target-specific model mixtures
retune B34
change the Phase-8 merge
```

This restriction is important because the Phase-7 additions are strongly non-uniform by target.

## Interpretation matrix

```text
BCE improves, AUC improves      supervision treatment helps both weak-label fit and ranking
BCE improves, AUC flat         better weak-label calibration/soft-target fit without clear ranking gain
BCE flat, AUC improves         ranking gain not captured well by soft BCE
BCE and AUC disagree in sign   weak-label loss and rank discrimination point in different directions
```

None of these outcomes alone authorizes model promotion. Hidden Kaggle evaluation or new external expert-labelled evidence remains required.

## Frozen command

Use this command instead of calling `phase9_matched_supervision_v2_eval` directly:

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.phase9_v2_auc_addendum \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --parent-pv1-manifest "$PV1_MANIFEST" \
  --pv2-manifest "$PV2_MANIFEST" \
  --control-checkpoint runs/phase9_matched_supervision_v2/control/model.pt \
  --candidate-checkpoint runs/phase9_matched_supervision_v2/candidate/model.pt \
  --out-root runs/phase9_matched_supervision_v2/eval \
  --n-bootstrap 5000
```

The wrapper first executes the unchanged frozen Phase-9 v2 evaluator, then computes the predeclared paired AUC addendum from the exact saved paired predictions and exact original-B6 holdout truth.

## Outputs

In addition to the original Phase-9 v2 evaluation files:

```text
runs/phase9_matched_supervision_v2/eval/
├── control_pv2_predictions.csv
├── candidate_pv2_predictions.csv
├── paired_pv2_predictions.csv
├── comparison.json
└── auc_addendum.json
```

`comparison.json` is augmented with `competition_aligned_auc_addendum`; `auc_addendum.json` stores the same addendum separately for auditability.

## Governance

The original BCE endpoint is not retroactively demoted. The AUC addendum is frozen before holdout inspection specifically to avoid selecting an inferential metric after seeing the Phase-9 v2 result.

Do not run an alternative TTA, ensemble, calibration, target filter, translated-cell subset, or loss weighting on the 499-study PV2 holdout after inspecting this result. Those are separate future experiments and require separate validation governance.
