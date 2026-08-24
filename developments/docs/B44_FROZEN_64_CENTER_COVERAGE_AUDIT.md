# B44 frozen B42 32 -> 64 center coverage audit

This is a mechanistic diagnostic only. It is not an independently valid test, is not a tuning surface, and cannot promote a model or choose a hidden submission. It reuses the post-B42 Expert-58 set to answer whether the fixed 32-center sampling is hiding useful local evidence for weak targets.

The critical design constraint is nested coverage. For every series and every frozen B42 TTA offset `[-1, 0, +1]`, the first 32 centers are exactly the historical B42 centers. Thirty-two additional deterministic centers are appended from a denser candidate grid. The historical first-16 B34 base path is therefore unchanged as well. Image normalization, 90% native crop, constant-area aspect-preserving resize, reflection stride padding, ConvNeXt checkpoint, metadata, 6x6 spatial grid, pathology-specific evidence scorer, top-k=8, temperature=1, residual gate, and three-view probability averaging all remain frozen.

The audit compares the existing B43 32-center per-series evidence table against a new 64-center pass. The main scientific output is `target_plane_signal_32_vs_64.csv`, which contains plane-specific best-series evidence AUC and evidence separation for both center counts. `target_combined_auc_32_vs_64.csv` is recorded as a secondary mechanistic quantity only and must not be interpreted as independent model-selection evidence.

A mandatory integrity check compares the 64-center reconstructed B34/base probability against the recorded B43 32-center base probability. Because the first 16 centers are unchanged, this maximum absolute difference must be <= 1e-6. Failure means the diagnostic changed more than slice coverage and must not be interpreted.

The pre-declared focus targets are ACL, MCL, Contusion, and Fracture. ACL is included as a routing-control target because the preceding plane-normalized diagnostic found sagittal evidence AUC substantially above axial. MCL and Contusion test the coverage/representation hypothesis, while Fracture is a mixed routing/coverage case. Results for all 12 targets are still written to avoid selective computation.

Outputs are written under `runs/077_Experiment_B42_constant_area_aspect_sparse_mil/b42_constant_area_aspect_sparse_mil/expert58/coverage_32_vs_64_audit/`:

- `audit.json`
- `series_evidence_64_by_view.csv`
- `series_evidence_64_tta_mean.csv`
- `target_plane_signal_32_vs_64.csv`
- `focus_target_plane_signal_32_vs_64.csv`
- `target_combined_auc_32_vs_64.csv`

Interpretation: a material increase in plane-specific evidence AUC when going from 32 to 64 centers supports a slice-coverage bottleneck. Little or no increase suggests that denser sampling alone does not recover discriminative local evidence and that representation or evidence calibration must change. Expert-58 remains reused mechanistic evidence only in either case.
