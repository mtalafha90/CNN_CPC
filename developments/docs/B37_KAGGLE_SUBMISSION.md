# B37 frozen hidden-test submission

> **Status — 2026-08-22.** This is the exact fixed-E2 B37 high-resolution
> sparse-MIL endpoint. Its 58-study expert result is a reused development
> diagnostic, not hidden-test evidence. Submit it once unchanged to obtain
> independent competition evidence; do not tune it after seeing the score.

## What is submitted

This competition is a Kaggle **code competition**. Do not upload a local CSV,
the Expert-58 plots, `expert58.json`, or a Colab notebook. Instead, attach a
private Kaggle Dataset containing the local source and checkpoints, then run a
minimal Kaggle notebook that creates:

```text
/kaggle/working/submission.csv
```

Kaggle scores that notebook output against its hidden test data. The output has
one row per test study, in the exact `test.csv` order, with these 13 columns:

```text
StudyInstanceUID
ACL
MCL
Medial Meniscus
Lateral Meniscus
Medial OA
Lateral OA
PF OA
Effusion
Synovitis
Baker's
Contusion
Fracture
```

Every target value is the **raw B37 combined sparse-MIL probability** in
`[0, 1]`. There is no 0.50 threshold, calibration, target-specific rule,
ensemble, or blend.

## Required local artifacts

Create a private Kaggle Dataset from your local files. It must contain both
checkpoints because B37 reconstructs the B34 base and verifies its SHA-256
fingerprint before inference:

```text
CNN_CPC/
  config/b37_highres_sparse_448.yaml
  developments/src/rsna_knee/...
  models/
    b37_model.pt
    b34_llm_fill_base_model.pt
```

Copy the complete `developments/src/rsna_knee` source tree rather than only the
new submission file, because the B37 model has several internal dependencies.
Do **not** include the competition DICOM data in this private dataset; attach
the official competition dataset separately in Kaggle.

The two model files must be:

```text
b37_model.pt                 # /.../b37_highres_sparse_mil/b37_model.pt
b34_llm_fill_base_model.pt   # exact Phase-9 llm_fill model.pt used by B37
```

The base file may have a different filename or Kaggle path from the local one.
Its contents must be byte-identical: B37 rejects a base checkpoint whose
SHA-256 fingerprint differs from the value recorded in `b37_model.pt`.

## Local smoke test

First update the local repository and test on the public placeholder test set.
The public `test.csv` is a smoke test only; it is not the hidden leaderboard
surface.

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
export PYTHONPATH="$PWD/developments/src:${PYTHONPATH:-}"

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export B37_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/071_Experiment_B37_highres_448_sparse_mil/b37_highres_sparse_mil"
export BASE_CKPT="/path/to/the/exact/llm_fill/model.pt"

python -m rsna_knee.b37_highres_sparse_submission \
  --config config/b37_highres_sparse_448.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint "$B37_ROOT/b37_model.pt" \
  --base-checkpoint "$BASE_CKPT" \
  --out "$B37_ROOT/submission_smoke.csv"
```

Expected local outputs:

```text
$B37_ROOT/submission_smoke.csv
$B37_ROOT/submission_smoke.csv.manifest.json
```

The command deliberately prints a reminder that a Kaggle final output must be
named `submission.csv` when a different local smoke-test filename is used.

## Kaggle launcher notebook

Create a new Kaggle notebook, attach:

1. the official `rsna-knee-abnormality-detection` competition data; and
2. the private dataset containing the local B37 code and model files.

The notebook is only a launcher for the local files; it is not a Colab notebook
and does not train or change the model. With the artifact dataset mounted as
`/kaggle/input/cnn-cpc-b37-artifacts`, use one cell like this:

```python
from pathlib import Path
import os
import sys

ARTIFACT_ROOT = Path("/kaggle/input/cnn-cpc-b37-artifacts/CNN_CPC")
DATA_ROOT = Path("/kaggle/input/rsna-knee-abnormality-detection")
CODE_ROOT = ARTIFACT_ROOT / "developments" / "src"
B37_CHECKPOINT = ARTIFACT_ROOT / "models" / "b37_model.pt"
BASE_CHECKPOINT = ARTIFACT_ROOT / "models" / "b34_llm_fill_base_model.pt"

sys.path.insert(0, str(CODE_ROOT))
os.environ["PYTHONPATH"] = f"{CODE_ROOT}:{os.environ.get('PYTHONPATH', '')}"
```

Then run inference in the next cell:

```python
from rsna_knee.b37_highres_sparse_submission import generate_b37_submission
from rsna_knee.b7_weak_supervision import _read_config

config = dict(_read_config(ARTIFACT_ROOT / "config" / "b37_highres_sparse_448.yaml"))
submission_path = generate_b37_submission(
    config,
    data_root=DATA_ROOT,
    checkpoint=B37_CHECKPOINT,
    base_checkpoint=BASE_CHECKPOINT,
    out_path="/kaggle/working/submission.csv",
)
print(submission_path)
```

Before committing the notebook, verify its result:

```python
import json
import pandas as pd

submission = pd.read_csv("/kaggle/working/submission.csv")
manifest = json.loads(
    Path("/kaggle/working/submission.csv.manifest.json").read_text()
)
print(submission.shape)
print(submission.columns.tolist())
print(submission.isna().sum().sum())
print(manifest["checkpoint_base_sha256_verified"])
print(manifest["tta_center_offsets"])
```

The manifest must report:

```text
checkpoint_base_sha256_verified  True
tta_center_offsets               [-1, 0, 1]
eval_batch_size                  1
workers                          0
pin_memory                       false
strict_dicom                     true
thresholding_used                false
blending_used                    false
```

Commit the Kaggle notebook version and select **Submit to Competition** from
that committed run. The implementation keeps a maximum 8.25-hour budget with a
30-minute output reserve, remains on one GPU, disables DataLoader workers and
pinned buffers, and drops completed 448-resolution batches before loading the
next one.

## Submission governance

This submission tests one frozen candidate: B37 combined logits after the
predeclared 448 / 90% native crop / 6x6 grid / top-k 8 / three-view-TTA route.
Record the public and private score, but do not use either score to alter this
candidate. Any changed method must receive a new experiment number and a new
submission path.
