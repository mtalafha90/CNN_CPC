# B51 Kaggle submission runbook

B51 is B42 with one training change: the study hierarchy was allowed to learn at
0.05x the head's rate instead of staying frozen. At inference the two are the
same model — B50's class alters only `requires_grad`, which a forward pass
ignores — so this submission reuses B42's proven dual-T4 path unchanged and
changes only which checkpoint it runs.

```text
b51_full_population_adapted_hierarchy_model.pt      the trained artefact
  -> b51_checkpoint_to_b42_format.py                metadata only, weights untouched
  -> b51_as_b42_for_submission.pt                   what Kaggle runs
  -> b51_submission_dualgpu_fast.py                 B42's path, B51's fingerprint
```

Expect a modest move. B50 measured `+0.011221` on 548 unseen-scanner studies.
Kaggle rounds to three decimals, so B51 may display `0.72`, or may display
`0.714` again and be indistinguishable from B42. Neither outcome authorises
tuning the hierarchy learning rate, epoch count, seed or geometry from the
hidden result.

## 0. Fingerprints this run depends on

```text
converted checkpoint  ede12675801838c15cdadf11190d5f6582ff315fa7394b67c97e30f967556266
source checkpoint     9ff78da33ec2a302332c0e05c1d0c0c207833fe7a3bd157c8fc510d1a3b03d65
```

The base checkpoint's required SHA-256 is whatever B51 recorded at training
time; the loader refuses a mismatch. Read it rather than assume it:

```bash
cd /media/talafha/Disk_1/CNN_CPC
R=runs/085_Experiment_B51_full_population_adapted_hierarchy
python -c "
import json
a = json.load(open('$R/training_audit.json'))
print('base path  :', a['base_checkpoint'])
print('base sha256:', a['base_checkpoint_sha256'])
"
```

## 1. Build the artifact dataset

The checkpoint bytes must not change. Only the code payload is rebuilt.

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee

export R="/media/talafha/Disk_1/CNN_CPC/runs/085_Experiment_B51_full_population_adapted_hierarchy"
export B51_CHECKPOINT="$R/b51_as_b42_for_submission.pt"
export BASE_CHECKPOINT="$(python -c "import json;print(json.load(open('$R/training_audit.json'))['base_checkpoint'])")"

rm -rf kaggle_b51_artifacts
mkdir -p kaggle_b51_artifacts/CNN_CPC/config
mkdir -p kaggle_b51_artifacts/CNN_CPC/developments/src
mkdir -p kaggle_b51_artifacts/CNN_CPC/models

cp config/b42_constant_area_aspect_sparse.yaml kaggle_b51_artifacts/CNN_CPC/config/
cp -a developments/src/rsna_knee kaggle_b51_artifacts/CNN_CPC/developments/src/
cp "$B51_CHECKPOINT" kaggle_b51_artifacts/CNN_CPC/models/b51_as_b42_for_submission.pt
cp "$BASE_CHECKPOINT" kaggle_b51_artifacts/CNN_CPC/models/b34_llm_fill_base_model.pt

find kaggle_b51_artifacts -name "__pycache__" -type d -exec rm -rf {} +
sha256sum \
  kaggle_b51_artifacts/CNN_CPC/models/b51_as_b42_for_submission.pt \
  kaggle_b51_artifacts/CNN_CPC/models/b34_llm_fill_base_model.pt
```

The first hash must be `ede12675…`. If it is not, the wrong file was copied;
stop and re-run the converter.

Upload `kaggle_b51_artifacts` as a new private Kaggle dataset (or a new version
of the existing artifact dataset) and attach it to the submission notebook.

## 2. Resolve the mounted artifacts

Discovery rather than a hard-coded slug, because the mount layout is nested.

```python
from pathlib import Path
import os
import sys

DATA_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
USER_DATASETS_ROOT = Path("/kaggle/input/datasets/mohammedtalafha")

ckpt_matches = list(USER_DATASETS_ROOT.rglob("b51_as_b42_for_submission.pt"))
base_matches = list(USER_DATASETS_ROOT.rglob("b34_llm_fill_base_model.pt"))
config_matches = list(USER_DATASETS_ROOT.rglob("b42_constant_area_aspect_sparse.yaml"))
package_matches = [
    p for p in USER_DATASETS_ROOT.rglob("rsna_knee")
    if p.is_dir() and (p / "__init__.py").exists()
]

assert len(ckpt_matches) == 1, ckpt_matches
assert len(base_matches) == 1, base_matches
assert len(config_matches) == 1, config_matches
assert len(package_matches) >= 1

B51_CHECKPOINT = ckpt_matches[0]
BASE_CHECKPOINT = base_matches[0]
CONFIG_PATH = config_matches[0]
CODE_ROOT = package_matches[0].parent
sys.path.insert(0, str(CODE_ROOT))
os.environ["PYTHONPATH"] = f"{CODE_ROOT}:{os.environ.get('PYTHONPATH', '')}"

import torch
print("CODE_ROOT      ", CODE_ROOT)
print("B51_CHECKPOINT ", B51_CHECKPOINT)
print("visible GPUs   ", torch.cuda.device_count())
```

The last line must report **at least two GPUs**. The dual-T4 path is not an
optimisation here: a single-GPU run projects past the runtime guard and aborts
during hidden scoring, which is exactly how B41's first submission failed.

## 3. Verify the compressed-DICOM decoders

The hidden set contains mixed transfer syntaxes. The visible example is too
small to prove every decoder is present.

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
    print(name, "available=", decoder.is_available, "missing=", decoder.missing_dependencies)
    assert decoder.is_available, f"Missing DICOM decoder for {name} ({uid})"
print("Compressed DICOM decoder preflight: PASS")
```

If this fails, bundle the offline decoder wheels in the artifact dataset before
scoring. Internet installation cannot be relied on during code-competition runs.

## 4. Verify the checkpoint is the one you meant

```python
import hashlib

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

B51_SHA = "ede12675801838c15cdadf11190d5f6582ff315fa7394b67c97e30f967556266"
assert sha256(B51_CHECKPOINT) == B51_SHA, "uploaded checkpoint is not the converted B51 file"
print("B51 artifact: VERIFIED")
```

`require_converted_b51` additionally refuses any file the converter did not
produce, so an ordinary B42 checkpoint cannot be run through this path by
accident.

## 5. Run

```python
from rsna_knee.b7_weak_supervision import _read_config
from rsna_knee.b51_submission_dualgpu_fast import generate_b51_submission_dual_gpu_fast

config = dict(_read_config(CONFIG_PATH))

generate_b51_submission_dual_gpu_fast(
    config,
    data_root=DATA_ROOT,
    checkpoint=B51_CHECKPOINT,
    base_checkpoint=BASE_CHECKPOINT,
    expected_checkpoint_sha256=B51_SHA,
    out_path="/kaggle/working/submission.csv",
)
```

On the visible three-study example this completes in about a minute and writes a
valid `submission.csv`. During hidden scoring the same cell processes roughly
1,300 studies across both T4s.

## 6. Check the output before submitting

```python
import pandas as pd

submission = pd.read_csv("/kaggle/working/submission.csv")
print(submission.shape)
print(submission.head())
assert submission.notna().all().all(), "submission contains blanks"
assert submission.iloc[:, 1:].to_numpy().min() >= 0.0
assert submission.iloc[:, 1:].to_numpy().max() <= 1.0
```

The launcher already validates the submission against the sample and writes a
manifest beside it recording the checkpoint fingerprint, the TTA offsets and the
execution version.

## 7. Submit

Save a version of the notebook with internet **off** and both T4s enabled, then
submit it to the competition.

## After the score appears

Record it against B42's `0.714` and stop. Whatever it shows, do not tune the
hierarchy learning rate, epoch count, seed, geometry or target subset from the
hidden result — B51 is a production run of a mechanism B50 already validated,
and its control is B42's existing hidden score.
