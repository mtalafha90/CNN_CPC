# B41 Kaggle dual-T4 resubmission after hidden scoring failure

## Diagnosis

The original B41 notebook completed successfully on the three visible example
studies and wrote a valid `submission.csv`, but it used only one of the two
visible T4 GPUs.  The observed inference section took about 74.4 seconds for
three studies, or about 24.8 seconds per study.  Kaggle replaces the three-study
example set with roughly 1,300 studies during scoring, so this extrapolates to
about nine hours before safety margin.  The B41 runtime guard allows 8.25 hours
with a 30-minute reserve, so the single-GPU hidden run is structurally at risk
of aborting when the first hidden-study timing projects the remaining work.

The B41 checkpoint and scientific endpoint are not changed.  The resubmission
uses two identical B41 replicas, one on each visible T4, and shards complete
studies by test-row index modulo two.  Every study still uses the exact frozen
B41 preprocessing, all three centre offsets `[-1, 0, +1]`, the same sparse-MIL
model, sigmoid probabilities per view, and mean probability aggregation.

Implementation:

```text
developments/src/rsna_knee/b41_highres_aspect_sparse_submission_dualgpu.py
```

## 1. Update the private artifact dataset

Pull current `main`, then rebuild only the code payload.  The checkpoint bytes
must remain unchanged.

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee

export B41_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/076_Experiment_B41_native_aspect_90crop_sparse_mil/b41_highres_aspect_sparse_mil"
export B41_CHECKPOINT="$B41_ROOT/b41_model.pt"
export BASE_CHECKPOINT="/media/talafha/Disk_1/CNN_CPC/runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt"

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

sha256sum \
  kaggle_b41_artifacts/CNN_CPC/models/b41_model.pt \
  kaggle_b41_artifacts/CNN_CPC/models/b34_llm_fill_base_model.pt
```

Required B41 SHA-256:

```text
fd8898cb2c642e3695e11c3f2e96057202a4d68c9a17c64abdea85625d44f5c4
```

Required base SHA-256:

```text
0caadf22935cad72df9515be8f4e09c8144745de1c1e16cb72cd7d8acabca9a6
```

Upload this as a new version of the existing private Kaggle B41 artifact
dataset and attach that new version to the submission notebook.

## 2. Resolve the mounted artifacts in Kaggle

The current Kaggle mount layout is nested below `/kaggle/input/datasets`.
Use discovery rather than hard-coding the dataset slug.

```python
from pathlib import Path
import os
import sys

DATA_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
USER_DATASETS_ROOT = Path("/kaggle/input/datasets/mohammedtalafha")

b41_matches = list(USER_DATASETS_ROOT.rglob("b41_model.pt"))
base_matches = list(USER_DATASETS_ROOT.rglob("b34_llm_fill_base_model.pt"))
config_matches = list(USER_DATASETS_ROOT.rglob("b41_highres_aspect_sparse_448.yaml"))
package_matches = [
    p for p in USER_DATASETS_ROOT.rglob("rsna_knee")
    if p.is_dir() and (p / "__init__.py").exists()
]

assert len(b41_matches) == 1
assert len(base_matches) == 1
assert len(config_matches) == 1
assert len(package_matches) >= 1

B41_CHECKPOINT = b41_matches[0]
BASE_CHECKPOINT = base_matches[0]
CONFIG_PATH = config_matches[0]
CODE_ROOT = package_matches[0].parent
sys.path.insert(0, str(CODE_ROOT))
os.environ["PYTHONPATH"] = f"{CODE_ROOT}:{os.environ.get('PYTHONPATH', '')}"

print("DATA_ROOT", DATA_ROOT)
print("CODE_ROOT", CODE_ROOT)
print("B41_CHECKPOINT", B41_CHECKPOINT)
print("BASE_CHECKPOINT", BASE_CHECKPOINT)
print("visible GPUs", __import__("torch").cuda.device_count())
```

The final line must report at least two GPUs for the dual-T4 path.

## 3. Verify required compressed-DICOM decoders

The competition documentation states that the hidden DICOM set contains mixed
transfer syntaxes including JPEG Lossless and JPEG 2000.  The public 15-series
example is too small to prove that every required decoder is available.

```python
from pydicom.pixels import get_decoder

required = {
    "JPEG Lossless P14": "1.2.840.10008.1.2.4.57",
    "JPEG Lossless SV1": "1.2.840.10008.1.2.4.70",
    "JPEG2000 Lossless": "1.2.840.10008.1.2.4.90",
    "JPEG2000": "1.2.840.10008.1.2.4.91",
}

for name, uid in required.items():
    decoder = get_decoder(uid)
    print(
        name,
        "available=", decoder.is_available,
        "plugins=", decoder.available_plugins,
        "missing=", decoder.missing_dependencies,
    )
    assert decoder.is_available, f"Missing DICOM decoder for {name} ({uid})"

print("Compressed DICOM decoder preflight: PASS")
```

If this cell fails, do not resubmit yet.  Bundle the missing offline decoder
wheels in the private artifact dataset before scoring; internet installation
cannot be relied on during code-competition scoring.

## 4. Verify checkpoint hashes in Kaggle

```python
import hashlib

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

assert sha256(B41_CHECKPOINT) == "fd8898cb2c642e3695e11c3f2e96057202a4d68c9a17c64abdea85625d44f5c4"
assert sha256(BASE_CHECKPOINT) == "0caadf22935cad72df9515be8f4e09c8144745de1c1e16cb72cd7d8acabca9a6"
print("Frozen B41 artifacts: VERIFIED")
```

## 5. Public three-study numerical-equivalence check

On the visible three-study example only, compare the original audited
single-GPU implementation against the new dual-T4 infrastructure.  The hidden
run must not execute the single-GPU comparator.

```python
import numpy as np
import pandas as pd
from rsna_knee.b7_weak_supervision import _read_config
from rsna_knee.b41_highres_aspect_sparse_submission import generate_b41_submission
from rsna_knee.b41_highres_aspect_sparse_submission_dualgpu import (
    generate_b41_submission_dual_gpu,
)

config = dict(_read_config(CONFIG_PATH))
visible_test = pd.read_csv(DATA_ROOT / "test.csv")

if len(visible_test) == 3:
    generate_b41_submission(
        config,
        data_root=DATA_ROOT,
        checkpoint=B41_CHECKPOINT,
        base_checkpoint=BASE_CHECKPOINT,
        out_path="/kaggle/working/b41_single_public.csv",
    )
    generate_b41_submission_dual_gpu(
        config,
        data_root=DATA_ROOT,
        checkpoint=B41_CHECKPOINT,
        base_checkpoint=BASE_CHECKPOINT,
        out_path="/kaggle/working/b41_dual_public.csv",
    )
    single = pd.read_csv("/kaggle/working/b41_single_public.csv")
    dual = pd.read_csv("/kaggle/working/b41_dual_public.csv")
    cols = [c for c in single.columns if c != "StudyInstanceUID"]
    assert single["StudyInstanceUID"].tolist() == dual["StudyInstanceUID"].tolist()
    delta = np.max(np.abs(single[cols].to_numpy(float) - dual[cols].to_numpy(float)))
    print("single-vs-dual max|probability delta| =", delta)
    assert delta <= 1e-5
```

This block runs only on the three-row example.  When Kaggle swaps in the hidden
test set, `len(visible_test) != 3`, so the expensive single-GPU comparator is
skipped.

## 6. Final scoring call

This is the only B41 inference call that should execute on the hidden set.

```python
from rsna_knee.b41_highres_aspect_sparse_submission_dualgpu import (
    generate_b41_submission_dual_gpu,
)

output = generate_b41_submission_dual_gpu(
    config,
    data_root=DATA_ROOT,
    checkpoint=B41_CHECKPOINT,
    base_checkpoint=BASE_CHECKPOINT,
    out_path="/kaggle/working/submission.csv",
)
print(output)
```

## 7. Validate the final output

```python
from pathlib import Path
import json
import pandas as pd

submission = pd.read_csv("/kaggle/working/submission.csv")
manifest = json.loads(
    Path("/kaggle/working/submission.csv.manifest.json").read_text()
)

assert submission.isna().sum().sum() == 0
assert manifest["checkpoint_base_sha256_verified"] is True
assert manifest["completed_epochs"] == 2
assert manifest["tta_center_offsets"] == [-1, 0, 1]
assert manifest["thresholding_used"] is False
assert manifest["blending_used"] is False
assert manifest["preprocessing"]["resize_policy"] == "aspect_preserving_pad"
assert manifest["preprocessing"]["preserves_in_plane_aspect_ratio"] is True
assert manifest.get("execution_version") == "b41_hidden_dual_t4_study_shards_v1"
assert manifest.get("gpu_count") == 2

print("B41 DUAL-T4 SUBMISSION VALIDATION: PASS")
print("shape:", submission.shape)
print("runtime hours:", manifest["runtime_elapsed_hours"])
```

Save a new notebook version with GPU **T4 x2** selected and submit that committed
version to the competition.

## Governance

The failed hidden execution is not model evidence and does not justify changing
B41.  This resubmission changes only execution parallelism.  The checkpoint,
preprocessing, 32 centres, three TTA offsets, sparse-MIL settings, sigmoid
probabilities and view aggregation remain frozen.  If the dual-T4 run still
fails, preserve B41 and diagnose the explicit hidden-run exception rather than
changing the model endpoint.
