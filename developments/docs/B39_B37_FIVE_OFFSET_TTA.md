# B39 — five-offset inference refinement of frozen B37

## Decision

B39 is the fastest prospective attempt to improve the completed B37 model without
retraining it. The B37 checkpoint remains immutable.

The completed B38 global-only 448 ablation did not improve its reused Expert-58
diagnostic (0.66441 versus the historical base 0.66875). That result does not
justify another global-only training run. B37's reported Kaggle score is 0.714;
B39 therefore retains B37's successful 448 sparse-MIL model exactly and only
broadens its deterministic inference coverage.

## Frozen B39 contract

~~~text
weights
  exact completed B37 fixed-E2 checkpoint
  exact B37 Phase-9 LLM-fill base checkpoint
  no training, calibration, blending, thresholds, or label access

preprocessing and model
  exact B37 full-native-volume normalization
  90% native centre crop -> one 448x448 resize
  32 deterministic 2.5D centres per view
  B37 6x6 target-specific sparse-MIL head, top-k=8
  B37 encoder-tail and sparse-head weights unchanged

inference-only change
  B37's three offsets [-1, 0, 1]
  -> B39's five symmetric offsets [-2, -1, 0, 1, 2]
  raw sigmoid probabilities averaged equally across all five views

operational safety
  batch size = 1
  workers = 0
  pin_memory = false
  no per-worker series cache
  maximum wall-clock budget = 8.25 h
  reserved finalization time = 45 min
  timing includes DICOM/DataLoader work, all views, inference, and memory release
~~~

This is a new hidden-test candidate—not a modification of B37. It must not be
selected, retuned, or reweighted from the reused Expert-58 surface.

## Why this is the efficient next step

The local B37 run already proves that the sparse local branch can add signal;
B38 showed that merely increasing global resolution does not. Five symmetric
through-plane offsets are a zero-training way to make B37's learned sparse
top-k evidence less sensitive to where an abnormality falls relative to the
deterministic slice centres.

The original three-view Kaggle inference completed in about four hours. Five
views are expected to scale to roughly 6 hours 40 minutes, but that is only a
planning estimate. B39 measures full per-study wall time and exits before its
reserved output time would be consumed.

## Exact local or Kaggle procedure

Use the same B37 model and base checkpoint. On Kaggle, first run the preflight
cell; it constructs the test study with the most eligible MRI series and runs
all five views once.

~~~bash
cd /kaggle/working/CNN_CPC
export PYTHONPATH="$PWD/developments/src:$PYTHONPATH"

# These are the mount roots shown by Kaggle for this project.
export DATA_ROOT="/kaggle/input/competitions/rsna-knee-abnormality-detection"
find /kaggle/input/datasets/mohammedtalafha -type f | sort | sed -n '1,160p'
# Set these two paths from the preceding listing. Do not guess their names.
export B37_CHECKPOINT="/kaggle/input/datasets/mohammedtalafha/.../b37_model.pt"
export BASE_CHECKPOINT="/kaggle/input/datasets/mohammedtalafha/.../b34_llm_fill_base_model.pt"

test -s "$B37_CHECKPOINT"
test -s "$BASE_CHECKPOINT"

python -m rsna_knee.b39_b37_five_offset_tta \
  --config config/b39_b37_five_offset_tta.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint "$B37_CHECKPOINT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --preflight-only
~~~

Only after it prints "[B39 preflight] ... PASS", generate the final file:

~~~bash
python -m rsna_knee.b39_b37_five_offset_tta \
  --config config/b39_b37_five_offset_tta.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint "$B37_CHECKPOINT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --out submission.csv
~~~

The output is:

~~~text
submission.csv
submission.csv.manifest.json
~~~

The manifest records the parent B37 checkpoint fingerprint, all five offsets,
the full-wall-clock timing scope, and the exact safety reserve. Upload only
submission.csv to Kaggle.
