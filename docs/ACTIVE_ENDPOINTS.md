# Active endpoint registry

This registry distinguishes a reproducible operational endpoint from historical
and closed experiments. It does **not** turn a displayed Kaggle tie into a
scientific promotion claim.

## Latest competition and development endpoints

| Endpoint | Current evidence | Entry point |
|---|---|---|
| B52 full-data | best recorded Kaggle `0.716`; inherited training exposure limits its local validation claim | [Submission record](../developments/docs/B52_KAGGLE_SUBMISSION.md) |
| B53 / B55 | below B52 on both recorded report-teacher rulers | [Common-ruler results](../developments/docs/B55_RUNBOOK.md) |
| B56 | geometry-only protocol specified; no result in the reviewed main snapshot | [B56 protocol](../developments/docs/B55_RUNBOOK.md) |
| B57 | implemented, unrun; fresh public initialization, shared development AUCs and gated ensemble | [B57 runbook](../developments/docs/B57_CLEAN_BACKBONE_COMPARISON.md) |

The B57 machine-readable recipe is `config/b57_clean_backbone_comparison.json`.
Its launcher is `developments/scripts/run_b57_clean_comparison.sh`; its output
root is `runs/093_Experiment_B57_clean_backbone_comparison`. The historical
numbered archive registry ends at B50 and does not describe these later runs.

## B42 constant-area native-aspect sparse MIL

| Field | Frozen value |
|---|---|
| Status | maintained operational reference |
| Hidden Kaggle result | displayed macro AUC `0.714` (tied with B37 and B41) |
| Scientific status | completed; do not tune from this result |
| Configuration | `config/b42_constant_area_aspect_sparse.yaml` |
| Model implementation | `rsna_knee.b42_constant_area_aspect_sparse_mil` |
| Fixed-E2 trainer | `rsna_knee.b42_constant_area_aspect_sparse_training` |
| Hidden-safe inference | `rsna_knee.b42_constant_area_aspect_sparse_submission_dualgpu_fast` |
| Required GPUs on Kaggle | two CUDA GPUs, validated on T4 x2 |
| B42 checkpoint SHA-256 | `399f0b04c818ce767af539e4f33226b6f5d6223a389814f508fd8f84c95afce3` |
| Local geometry | native 90% crop, one aspect-preserving constant-area resize, reflection stride padding |
| TTA | exactly `[-1, 0, +1]` |

The submission module verifies the B42 checkpoint fingerprint before inference.
It also verifies the base B34 checkpoint fingerprint stored inside the B42
checkpoint. A checkpoint mismatch is a stop condition, not a warning.

Required external artefacts are intentionally not in Git:

1. the fixed B42 checkpoint above;
2. its exact B34 base checkpoint;
3. frozen report-label artefacts and policy;
4. the official competition data mount.

The full scientific contract is in
[`developments/docs/B42_CONSTANT_AREA_ASPECT_SPARSE_MIL.md`](../developments/docs/B42_CONSTANT_AREA_ASPECT_SPARSE_MIL.md).

## Closed endpoints

| Endpoint | Result | Operational decision |
|---|---|---|
| B46 gold anchor | no support | archive; do not vary gold weight or folds |
| B48 global-query conditioning | no meaningful support | archive; do not tune query source/rank/gate |
| B49 native tiled candidate | no practical support; Kaggle `0.707` | archive; do not tune tiles, overlap, crop, TTA, or blend |
| B47 native grid | implemented but unrun | not an approved successor; do not launch automatically |
| B50 gate / adapted hierarchy | implemented and used by subsequent runs | preserve the existing gate; B57 reads it without regeneration |

The complete result evidence and governance rules are in
[`developments/docs/CURRENT_STATUS.md`](../developments/docs/CURRENT_STATUS.md).

## Future experiment gate

Distinguish development comparisons from independent confirmation. B57 freezes
a reused development surface, excludes its scanner profiles from all new
training, and starts both arms from public encoder weights. It must be
confirmed on another group-disjoint fold before promotion. Expert-58 remains
diagnostic only. See the B57 runbook for the ancestry audit and exact gates.
