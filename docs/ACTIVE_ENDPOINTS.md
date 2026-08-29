# Active endpoint registry

This registry distinguishes a reproducible operational endpoint from historical
and closed experiments. It does **not** turn a displayed Kaggle tie into a
scientific promotion claim.

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
| B50 ordered slice sequence | protocol/gate preparation only | create and hash its fresh gate before model implementation |

The complete result evidence and governance rules are in
[`developments/docs/CURRENT_STATUS.md`](../developments/docs/CURRENT_STATUS.md).

## Future experiment gate

Any new architecture must first declare and hash a **fresh group-disjoint
validation surface**. The reused Expert-58 and the B48/B49 scanner split are
reporting surfaces, not a clean new architecture-selection gate. The next
recommended scientific question is ordered through-plane slice reasoning, not
another crop, tile, resolution, top-k, query, or epoch variation.
