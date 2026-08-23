# B41 frozen hidden-test submission

## Status

B41 is a completed fixed-E2 native-aspect candidate.  Its reused Expert-58
combined macro AUC is `0.6778722842`; this is development evidence only and does
not alter the frozen endpoint.  B41 is submitted once unchanged to test whether
preserving rectangular acquisition geometry generalizes on the hidden
competition distribution.

The hidden inference path **must** use B41's aspect-preserving preprocessing.
The B37 submission launcher must not be reused directly because it constructs
`B37HighResSparseDataset`, which square-stretches rectangular scans.

B41 checkpoint observed during the completed Expert-58 run:

```text
fd8898cb2c642e3695e11c3f2e96057202a4d68c9a17c64abdea85625d44f5c4
```

## Frozen hidden-test endpoint

```text
full native-volume percentile normalization
-> central 90% crop in native coordinates
-> one antialiased aspect-preserving resize-to-fit
-> symmetric zero padding to 448x448
-> 32 deterministic 2.5D centres
-> ConvNeXt final stage adapted for two fixed epochs
-> 6x6 pathology-specific sparse MIL, top-k=8
-> raw sigmoid probabilities
-> three centre-offset views [-1, 0, +1], probability averaged
```

No thresholds, calibration, target-specific changes, blending, or post-hoc
weighting are used.

## 1. Pull the B41 submission code

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
export PYTHONPATH="$PWD/developments/src:${PYTHONPATH:-}"
```

Run the focused tests:

```bash
pytest -q \
  developments/tests/test_b41_highres_aspect_sparse.py \
  developments/tests/test_b41_highres_aspect_sparse_submission.py
```

## 2. Define the exact local artifacts

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export B41_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/076_Experiment_B41_native_aspect_90crop_sparse_mil/b41_highres_aspect_sparse_mil"
export B41_CHECKPOINT="$B41_ROOT/b41_model.pt"
export BASE_CHECKPOINT="/media/talafha/Disk_1/CNN_CPC/runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt"

sha256sum "$B41_CHECKPOINT" "$BASE_CHECKPOINT"
```

The B41 digest should match the completed endpoint above.  The submission code
also verifies that the supplied base checkpoint SHA-256 matches the base digest
recorded inside `b41_model.pt`.

## 3. Local hidden-path preflight

The public/local test surface is a smoke test, not hidden evidence.  First run
only the largest-series test study through the exact B41 inference path:

```bash
python -m rsna_knee.b41_highres_aspect_sparse_submission \
  --config config/b41_highres_aspect_sparse_448.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint "$B41_CHECKPOINT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --out "$B41_ROOT/submission_smoke.csv" \
  --preflight-only
```

The command must end with a B41 preflight `PASS`.

Then generate the local smoke submission:

```bash
python -m rsna_knee.b41_highres_aspect_sparse_submission \
  --config config/b41_highres_aspect_sparse_448.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint "$B41_CHECKPOINT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --out "$B41_ROOT/submission_smoke.csv"
```

Expected outputs:

```text
$B41_ROOT/submission_smoke.csv
$B41_ROOT/submission_smoke.csv.manifest.json
```

Inspect the manifest:

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["B41_ROOT"])
m = json.loads((root / "submission_smoke.csv.manifest.json").read_text())
for key in [
    "checkpoint_base_sha256_verified",
    "completed_epochs",
    "tta_center_offsets",
    "eval_batch_size",
    "workers",
    "pin_memory",
    "strict_dicom",
    "thresholding_used",
    "blending_used",
]:
    print(key, m[key])
print("resize_policy", m["preprocessing"]["resize_policy"])
print("preserves_aspect", m["preprocessing"]["preserves_in_plane_aspect_ratio"])
PY
```

Required values:

```text
checkpoint_base_sha256_verified  True
completed_epochs                  2
tta_center_offsets                [-1, 0, 1]
eval_batch_size                   1
workers                           0
pin_memory                        False
strict_dicom                      True
thresholding_used                 False
blending_used                     False
resize_policy                     aspect_preserving_pad
preserves_aspect                  True
```

## 4. Build the private Kaggle artifact dataset

Create a private Kaggle Dataset containing the code and the two exact
checkpoints.  Do not include the competition DICOM data in this private dataset.
A convenient local staging tree is:

```bash
cd /media/talafha/Disk_1/CNN_CPC
rm -rf kaggle_b41_artifacts
mkdir -p kaggle_b41_artifacts/CNN_CPC/config
mkdir -p kaggle_b41_artifacts/CNN_CPC/developments/src
mkdir -p kaggle_b41_artifacts/CNN_CPC/models

cp config/b41_highres_aspect_sparse_448.yaml \
  kaggle_b41_artifacts/CNN_CPC/config/
cp -a developments/src/rsna_knee \
  kaggle_b41_artifacts/CNN_CPC/developments/src/
cp "$B41_CHECKPOINT" \
  kaggle_b41_artifacts/CNN_CPC/models/b41_model.pt
cp "$BASE_CHECKPOINT" \
  kaggle_b41_artifacts/CNN_CPC/models/b34_llm_fill_base_model.pt
```

Upload `kaggle_b41_artifacts` as a **private** Kaggle Dataset.  Keep the model
files byte-identical; do not resave them through PyTorch before upload.

## 5. Kaggle code-competition notebook

Create a Kaggle notebook and attach:

1. the official `rsna-knee-abnormality-detection` competition dataset; and
2. the private B41 artifact dataset created above.

Assuming the artifact dataset is mounted as
`/kaggle/input/cnn-cpc-b41-artifacts`, use:

```python
from pathlib import Path
import os
import sys

ARTIFACT_ROOT = Path("/kaggle/input/cnn-cpc-b41-artifacts/CNN_CPC")
DATA_ROOT = Path("/kaggle/input/rsna-knee-abnormality-detection")
CODE_ROOT = ARTIFACT_ROOT / "developments" / "src"
B41_CHECKPOINT = ARTIFACT_ROOT / "models" / "b41_model.pt"
BASE_CHECKPOINT = ARTIFACT_ROOT / "models" / "b34_llm_fill_base_model.pt"

sys.path.insert(0, str(CODE_ROOT))
os.environ["PYTHONPATH"] = f"{CODE_ROOT}:{os.environ.get('PYTHONPATH', '')}"
```

Run the exact frozen inference:

```python
from rsna_knee.b7_weak_supervision import _read_config
from rsna_knee.b41_highres_aspect_sparse_submission import generate_b41_submission

config = dict(
    _read_config(ARTIFACT_ROOT / "config" / "b41_highres_aspect_sparse_448.yaml")
)

generate_b41_submission(
    config,
    data_root=DATA_ROOT,
    checkpoint=B41_CHECKPOINT,
    base_checkpoint=BASE_CHECKPOINT,
    out_path="/kaggle/working/submission.csv",
)
```

Verify the final files before committing the notebook:

```python
import json
import pandas as pd

submission = pd.read_csv("/kaggle/working/submission.csv")
manifest = json.loads(
    Path("/kaggle/working/submission.csv.manifest.json").read_text()
)

print(submission.shape)
print(submission.columns.tolist())
print("NaNs:", int(submission.isna().sum().sum()))
print("base verified:", manifest["checkpoint_base_sha256_verified"])
print("resize:", manifest["preprocessing"]["resize_policy"])
print("aspect preserved:", manifest["preprocessing"]["preserves_in_plane_aspect_ratio"])
print("TTA:", manifest["tta_center_offsets"])
```

The notebook output submitted to the competition must be exactly:

```text
/kaggle/working/submission.csv
```

Commit the notebook version and choose **Submit to Competition** from that
committed run.

## Governance

This hidden submission is an independent test of the already frozen B41 E2
candidate.  Do not change the B41 checkpoint, crop, aspect-preserving resize,
padding value, TTA, grid, top-k, threshold, calibration, or target behavior after
seeing the score.  Any changed geometry is a new experiment.
