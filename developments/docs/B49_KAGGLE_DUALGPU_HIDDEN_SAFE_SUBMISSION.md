# B49 candidate-only Kaggle submission: hidden-safe dual T4

## Scope

This is the one permitted Kaggle endpoint for completed B49: the
`post_cross_attention_candidate` checkpoint. It is an **exploratory hidden-test
submission**, not a promotion of B49's matched-domain result.

## Completed result

The candidate-only hidden submission completed on **GPU T4 x2** and received a
displayed Kaggle score of **`0.707`**. The previously successful B42 endpoint
displayed `0.714`, so B49 is `−0.007` on that leaderboard scale. This result is
recorded, not tuned against: it does not change B49's predeclared matched-domain
verdict of no support and does not permit a B49 rerun, blend, calibration,
control-arm submission, or architecture change.

Use GPU **T4 x2**. Do not use P100 (one GPU) or TPU v5e-8 (this is a PyTorch
CUDA endpoint).

The candidate preserves full native in-plane pixels in its local branch with
640px overlapping tiles. The Kaggle score must not be used to tune architecture,
tile geometry, preprocessing, calibration, or a blend.

## Why this launcher exists

An earlier B41 hidden submission passed Kaggle's three-study visible surface
but a single-GPU path was structurally unsafe on the much larger hidden rerun.
B42 completed after using two T4 replicas and complete-study sharding.

B49 also has a host-memory risk: a high-series-count study can otherwise retain
full-FOV context images for all three TTA offsets. This launcher uses two replica
GPUs and materializes only one whole-study TTA context view at a time. It decodes
and normalizes each native volume once per study, then reuses that exact
normalized array for the fixed context image and fixed local native-tile stream.

Nothing scientific changes: candidate checkpoint, full-FOV/no-crop local tiles,
context resize, tile chunk of two, TTA `[-1, 0, 1]`, raw sigmoid probabilities,
and mean aggregation remain frozen.

## 1. Update and test locally

```bash
cd /media/talafha/Disk_1/CNN_CPC_b49_run
git fetch origin main
git checkout --detach origin/main
conda activate rsna-knee
export PYTHONPATH="$PWD/developments/src:${PYTHONPATH:-}"

python -m pytest -q \
  developments/tests/test_b49_native_tiled_multiscale_submission.py \
  developments/tests/test_b49_native_tiled_multiscale_submission_dualgpu_streaming.py
```

## 2. Build the private Kaggle artifact dataset

The private dataset contains code and immutable non-competition artifacts. Do
**not** include competition DICOMs; Kaggle mounts them from the official
competition dataset.

```bash
cd /media/talafha/Disk_1/CNN_CPC_b49_run

PAIR="$PWD/runs/082_Experiment_B49_native_tiled_multiscale_mil/b49_native_tiled_multiscale_mil/seed_2026"
export B49_CANDIDATE="$PAIR/post_cross_attention_candidate/b49_post_cross_attention_candidate_model.pt"
: "${BASE_CHECKPOINT:?BASE_CHECKPOINT is missing}"
: "${LABELS_ROOT:?LABELS_ROOT is missing}"
export DOMAIN_SPLIT_ROOT="/media/talafha/Disk_1/CNN_CPC_b48_run/runs/domain_shift_split"

rm -rf kaggle_b49_candidate_artifacts
mkdir -p kaggle_b49_candidate_artifacts/CNN_CPC/{config,developments/src,models,labels,domain_split}

cp config/b49_native_tiled_multiscale.yaml kaggle_b49_candidate_artifacts/CNN_CPC/config/
cp -a developments/src/rsna_knee kaggle_b49_candidate_artifacts/CNN_CPC/developments/src/
cp "$B49_CANDIDATE" kaggle_b49_candidate_artifacts/CNN_CPC/models/b49_post_cross_attention_candidate_model.pt
cp "$BASE_CHECKPOINT" kaggle_b49_candidate_artifacts/CNN_CPC/models/b34_llm_fill_base_model.pt
cp "$LABELS_ROOT/training_targets.csv" "$LABELS_ROOT/policy.json" "$LABELS_ROOT/audit.json" \
  kaggle_b49_candidate_artifacts/CNN_CPC/labels/
cp "$DOMAIN_SPLIT_ROOT/domain_split.json" \
  "$DOMAIN_SPLIT_ROOT/domain_split_by_study.csv" \
  "$DOMAIN_SPLIT_ROOT/domain_split.sha256" \
  kaggle_b49_candidate_artifacts/CNN_CPC/domain_split/

sha256sum \
  kaggle_b49_candidate_artifacts/CNN_CPC/models/b49_post_cross_attention_candidate_model.pt \
  kaggle_b49_candidate_artifacts/CNN_CPC/models/b34_llm_fill_base_model.pt \
  kaggle_b49_candidate_artifacts/CNN_CPC/domain_split/domain_split.json
```

Upload this staging tree as a **new version** of a private Kaggle Dataset. Keep
the checkpoint bytes unchanged. The official competition `train.csv`,
`train_series.csv`, `test.csv`, `test_series.csv`, and DICOM files come from
the official competition mount, not this private dataset.

## 3. Kaggle notebook setup

Attach both the official `rsna-knee-abnormality-detection` competition dataset
and the private B49 artifact dataset. Select **GPU T4 x2**, leave Internet
disabled, and run this discovery cell:

```python
from pathlib import Path
import os, sys, torch

DATA_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
USER_DATASETS_ROOT = Path("/kaggle/input/datasets/mohammedtalafha")

candidate_matches = list(USER_DATASETS_ROOT.rglob("b49_post_cross_attention_candidate_model.pt"))
base_matches = list(USER_DATASETS_ROOT.rglob("b34_llm_fill_base_model.pt"))
config_matches = list(USER_DATASETS_ROOT.rglob("b49_native_tiled_multiscale.yaml"))
labels_matches = list(USER_DATASETS_ROOT.rglob("training_targets.csv"))
domain_matches = list(USER_DATASETS_ROOT.rglob("domain_split.json"))
package_matches = [p for p in USER_DATASETS_ROOT.rglob("rsna_knee") if p.is_dir() and (p / "__init__.py").exists()]

assert len(candidate_matches) == len(base_matches) == len(config_matches) == 1
assert len(labels_matches) == len(domain_matches) == 1 and package_matches

B49_CANDIDATE = candidate_matches[0]
BASE_CHECKPOINT = base_matches[0]
CONFIG_PATH = config_matches[0]
LABELS_ROOT = labels_matches[0].parent
DOMAIN_SPLIT = domain_matches[0]
CODE_ROOT = package_matches[0].parent
sys.path.insert(0, str(CODE_ROOT))
os.environ["PYTHONPATH"] = f"{CODE_ROOT}:{os.environ.get('PYTHONPATH', '')}"

print("DATA_ROOT", DATA_ROOT)
print("B49 candidate", B49_CANDIDATE)
print("visible GPUs", torch.cuda.device_count())
assert torch.cuda.device_count() >= 2, "Select GPU T4 x2 before running"
```

## 4. Required preflights

Verify all compressed-DICOM decoders are available offline:

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
```

The completed run initially lacked the JPEG Lossless P14 decoder. It was fixed
operationally by attaching an offline GDCM/Python package and installing
`python-gdcm` before importing `pydicom`; all four decoder assertions then
passed. This is a runtime dependency fix only. It must not change the candidate
checkpoint, preprocessing, TTA, or prediction policy.

If a decoder is still unavailable after installing an approved offline decoder
package, stop. Internet installation is not reliable during code competition
scoring.

## 5. Visible three-study numerical-equivalence check

Run this only while Kaggle presents its visible three-row test set. It
automatically skips when Kaggle swaps in the hidden test set.

```python
import numpy as np
import pandas as pd
from rsna_knee.b7_weak_supervision import _read_config
from rsna_knee.b49_native_tiled_multiscale_submission import generate_b49_candidate_submission
from rsna_knee.b49_native_tiled_multiscale_submission_dualgpu_streaming import generate_b49_candidate_submission_dual_gpu_streaming

config = dict(_read_config(CONFIG_PATH))
if len(pd.read_csv(DATA_ROOT / "test.csv")) == 3:
    common = dict(
        config=config, data_root=DATA_ROOT, labels_root=LABELS_ROOT,
        base_checkpoint=BASE_CHECKPOINT, domain_split=DOMAIN_SPLIT,
        candidate_checkpoint=B49_CANDIDATE,
    )
    generate_b49_candidate_submission(**common, out_path="/kaggle/working/b49_single_public.csv")
    generate_b49_candidate_submission_dual_gpu_streaming(**common, out_path="/kaggle/working/b49_dual_public.csv")
    single = pd.read_csv("/kaggle/working/b49_single_public.csv")
    dual = pd.read_csv("/kaggle/working/b49_dual_public.csv")
    columns = [c for c in single.columns if c != "StudyInstanceUID"]
    assert single["StudyInstanceUID"].tolist() == dual["StudyInstanceUID"].tolist()
    delta = np.max(np.abs(single[columns].to_numpy(float) - dual[columns].to_numpy(float)))
    print("single-vs-dual max|probability delta| =", delta)
    assert delta <= 1e-5
```

## 6. Final scoring call (historical reproducibility)

This is the only B49 inference call that should execute on Kaggle's hidden
test set:

```python
from rsna_knee.b49_native_tiled_multiscale_submission_dualgpu_streaming import generate_b49_candidate_submission_dual_gpu_streaming

output = generate_b49_candidate_submission_dual_gpu_streaming(
    config,
    data_root=DATA_ROOT,
    labels_root=LABELS_ROOT,
    base_checkpoint=BASE_CHECKPOINT,
    domain_split=DOMAIN_SPLIT,
    candidate_checkpoint=B49_CANDIDATE,
    out_path="/kaggle/working/submission.csv",
)
print(output)
```

## 7. Validate and submit

```python
from pathlib import Path
import json, pandas as pd

submission = pd.read_csv("/kaggle/working/submission.csv")
manifest = json.loads(Path("/kaggle/working/submission.csv.manifest.json").read_text())

assert submission.isna().sum().sum() == 0
assert manifest["candidate_arm"] == "post_cross_attention_candidate"
assert manifest["completed_epochs"] == 2
assert manifest["gpu_count"] == 2
assert manifest["tta_offsets"] == [-1, 0, 1]
assert manifest["trained_source_sha256_verified_before_adapter"] is True
assert manifest["execution_only_change"] is True
assert manifest["hidden_safe_execution"]["native_volume_normalizations_per_series"] == 1
assert manifest["hidden_safe_execution"]["all_tta_context_views_materialized"] is False
assert "no_thresholding" in manifest["prediction_policy"]
assert "no_blending" in manifest["prediction_policy"]
print("B49 candidate dual-T4 validation: PASS", submission.shape)
```

Save a new notebook version with **T4 x2** selected, then submit that committed
notebook version. Kaggle Code Competition submits the notebook-generated
`/kaggle/working/submission.csv`; it does not accept a local CSV as a direct
hidden-test submission.

## If scoring fails

Keep B49's checkpoint and scientific result frozen. Copy the exact error and
the most recent `[B49 hidden-safe gpu...]` line. Do not reduce tile size, remove
TTA, switch to the control arm, calibrate, blend, or alter B49 to work around
an execution failure.
