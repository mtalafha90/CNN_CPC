# B43 precursor: target × series × plane evidence audit

## Status

Diagnostic only. This is **not** a trained B43 model and is **not** independent test evidence. It reuses the post-B42 Expert-58 surface to diagnose the frozen B42 endpoint. No parameter, threshold, blend, crop, slice count, sparse-MIL setting, checkpoint, or submission choice may be selected from this audit.

## Question

B42 recovered much of the B41 aspect-ratio loss without exceeding B37. The next mechanistic question is whether weak targets fail because useful evidence is absent from every MRI series, or because useful per-series evidence is diluted/misrouted during study-level aggregation.

The audit therefore records, for every Expert-58 study, target, TTA view and acquired series:

- series UID and anatomical plane;
- fluid-sensitive and fat-suppression metadata;
- B42 token-level top-1 evidence for that series;
- B42 series-only top-k log-mean-exp evidence;
- number of the study-level global top-k locations supplied by that series;
- exact selected sparse-MIL slice/grid locations;
- frozen B42 base/local/combined prediction quantities;
- leave-one-series-out change in the frozen combined logit and probability.

## Leave-one-series-out definition

The image encoder is **not rerun**. The original B42 global and spatial features are encoded once. For a leave-one-series-out diagnostic, one series is removed only from the `present` mask and the already-frozen B34 aggregation and frozen B42 sparse head are reevaluated. Consequently this measures routing/aggregation sensitivity without creating a new model or a new image representation.

## Frozen inference contract

The audit uses the completed B42 fixed-E2 checkpoint, the existing Expert-58 MRI surface, 32 deterministic centres, gap 1, TTA centre offsets `[-1, 0, +1]`, 6×6 local grid, top-k 8, temperature 1, native-aspect constant-area resize, reflection stride padding, and ragged per-series encoding. The audit must reproduce the already-recorded B42 combined Expert-58 predictions to `max|delta| <= 1e-6` before writing a valid PASS.

## Outputs

`series_evidence_by_view.csv` is the raw per-view, per-series evidence table. `selected_locations.csv` contains every selected top-k sparse location with series, plane, slice position and 6×6 grid coordinates. `series_evidence_tta_mean.csv` averages series-level quantities over the three frozen offsets. `strongest_series_by_study_target.csv` identifies the strongest-evidence series for every study-target pair. `target_plane_summary.csv` reports the distribution of strongest planes separately for positive and negative labels. `target_summary.csv` records the reproduced AUC plus positive/negative evidence and leave-one-out summaries. `audit.json` records governance, counts and reproduction checks.

## Interpretation

If positive cases show strong evidence in an individual series and removing that series materially reduces the prediction, the representation can see the pathology and future work should investigate target-conditioned series routing/aggregation. If positive cases show no useful evidence in any series, routing alone is unlikely to help; the next investigation should instead address representation, slice coverage or label supervision. This diagnostic does not by itself authorize a B43 architecture or hidden submission.
