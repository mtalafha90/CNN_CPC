# B52 Kaggle submission runbook

B52 is B42's architecture, fine-tuned for the competition rather than for a
frozen scientific endpoint: six epochs instead of two, five trainable encoder
stages instead of one, and the checkpoint chosen by the best validation epoch
instead of a fixed one. Its selected epoch scored **0.834998** macro AUC on 548
unseen-scanner studies — a selection statistic, not a leaderboard prediction.

At inference none of that matters. `requires_grad` and `encoder_trainable_stages`
have no effect on a forward pass, so B52 runs through B42's proven dual-T4 path
unchanged, at exactly B42's cost per study.

```text
b52_best_model.pt                              the trained artefact, unmodified
  -> b52_competition_submission_dualgpu_fast.py B42's inference path, B52's identity
  -> submission.csv + submission.csv.manifest.json
```

**There is no conversion step.** The file you submit is byte-for-byte the file
that was trained. See "Why there is no converter" below.

## 0. The nine-hour limit, which is the real constraint

Kaggle allows nine hours. The launcher does not use all of it:

```text
competition ceiling             9.00 h
runtime guard (B37_SUBMISSION_MAX_HOURS)    8.25 h
internal reserve                            0.50 h   for writing the output
```

Every ten studies each GPU projects its own remaining time from the mean of its
last five studies, multiplied by a **1.35 safety factor**. For B52 that
projection is **telemetry and cannot raise**. It used to abort the run, and that
is one of the two documented causes of B39's, B41's and B51's hidden failures: in
a 650-study shard a single slow early study forecasts past the budget and kills a
run that would in fact have finished. B42's own launcher keeps the abort, because
its `0.714` hidden run was made under it.

You will see this line, per GPU, throughout the run, now carrying host memory as
well:

```text
[B42 dual submit gpu0] 250/650 shard elapsed=71.3 min estimated_remaining=115.6 min rss=2.9GiB rss_peak=3.1GiB
```

**The single thing that decides whether you finish: both T4s must be on.**

```text
~1,300 hidden studies, one T4     ~24.8 s/study  ->  ~9 h   aborts
~1,300 hidden studies, two T4s    sharded by test-row index modulo 2  ->  ~4.5 h
```

Those per-study seconds are B41's measured figure on the same geometry, and the
single-GPU projection is exactly how B41's first hidden submission failed. The
launcher now refuses to start at all if fewer than two CUDA devices are visible,
so this fails in the first seconds rather than in the eighth hour.

B52 does not change any of this. Same 448² reference area, same 90% native crop,
same three centre offsets `[-1, 0, +1]`, same encoder chunk of 4. The launcher
**refuses a checkpoint whose trained chunk size is not 4**, because that is the
number the budget above was calibrated against.

Three things that would break the budget, all of them refused rather than
silently allowed: a single visible GPU, a chunk size other than 4, and
`num_workers` or `pin_memory` set in the config.

B52 also inherits B51's memory contract, and for the same reason: one TTA view is
materialised at a time rather than `[3, K, 32, 3, H, W]` for a whole study. At
the 14 series this dataset actually contains that is roughly 0.6 GiB against 3.2,
doubled because two shards run at once. Nothing about B52 changes this either
way -- its five trainable encoder stages and `requires_grad` flags cost nothing
under `torch.inference_mode()`, so its per-study memory and time are B42's.

## 1. Fingerprints this run depends on

Read them; do not assume them.

```bash
cd /media/talafha/Disk_1/CNN_CPC
R=runs/087_Experiment_B52_full_data    # the 3,801-study run, NOT 086

sha256sum "$R/b52_best_model.pt"

python -c "
import torch
p = torch.load('$R/b52_best_model.pt', map_location='cpu', weights_only=False)
print('experiment    :', p['experiment'])
print('version       :', p['version'])
print('selected epoch:', p['selected_epoch'], 'at', p['selection_value'])
print('train studies :', p['training_studies'])
print('chunk size    :', p['model_state']['encoder_chunk_size'])
print('base ckpt     :', p['base_checkpoint'])
print('base sha256   :', p['base_checkpoint_sha256'])
"
```

The first hash is what you will declare as `B52_SHA`. The base checkpoint path
and hash are what you must upload alongside it — the loader refuses a mismatch.

**Check the identity, not just the hash.** There are two completed B52 runs and
both are legitimate checkpoints, so nothing will refuse the wrong one:

```text
runs/086_Experiment_B52_competition_full_finetune   0.802666   1,447 studies
runs/087_Experiment_B52_full_data                   0.834998   3,801 studies   <- this one
```

The hash proves the file is the one you pointed at. It says nothing about whether
you pointed at the right one. To list every B52 run that exists:

```python
from pathlib import Path
import torch

for path in sorted(Path("runs").rglob("b52_best_model.pt")):
    p = torch.load(path, map_location="cpu", weights_only=False)
    print(
        f"{p.get('selection_value', float('nan')):.6f}  "
        f"{p.get('training_studies'):>5} studies  "
        f"epoch {p.get('selected_epoch')}/{p.get('epochs_planned')}  "
        f"splits={p.get('train_splits')}  {path}"
    )
```

The launcher prints the same identity before it spends a GPU, and that line is
the last chance to notice:

```text
[B52 submit] epoch 5 selected at 0.834998, trained on 3801 studies, augmentation=False
```

## 2. Build the artifact dataset

The checkpoint bytes must not change. Only the code payload is rebuilt.

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee

export R="/media/talafha/Disk_1/CNN_CPC/runs/087_Experiment_B52_full_data"
export B52_CHECKPOINT="$R/b52_best_model.pt"
export BASE_CHECKPOINT="$(python -c "import torch;print(torch.load('$R/b52_best_model.pt',map_location='cpu',weights_only=False)['base_checkpoint'])")"

rm -rf kaggle_b52_artifacts
mkdir -p kaggle_b52_artifacts/CNN_CPC/config
mkdir -p kaggle_b52_artifacts/CNN_CPC/developments/src
mkdir -p kaggle_b52_artifacts/CNN_CPC/models

cp config/b42_constant_area_aspect_sparse.yaml kaggle_b52_artifacts/CNN_CPC/config/
cp -a developments/src/rsna_knee kaggle_b52_artifacts/CNN_CPC/developments/src/
cp "$B52_CHECKPOINT" kaggle_b52_artifacts/CNN_CPC/models/b52_best_model.pt
cp "$BASE_CHECKPOINT" kaggle_b52_artifacts/CNN_CPC/models/b34_llm_fill_base_model.pt

find kaggle_b52_artifacts -name "__pycache__" -type d -exec rm -rf {} +
sha256sum \
  kaggle_b52_artifacts/CNN_CPC/models/b52_best_model.pt \
  kaggle_b52_artifacts/CNN_CPC/models/b34_llm_fill_base_model.pt
```

The first hash must equal the `B52_SHA` from step 1. If it does not, the wrong
file was copied; stop.

Upload `kaggle_b52_artifacts` as a new private Kaggle dataset (or a new version
of the existing artifact dataset) and attach it to the submission notebook.

## 3. Resolve the mounted artifacts

Discovery rather than a hard-coded slug, because the mount layout is nested.

```python
from pathlib import Path
import os
import sys

DATA_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
USER_DATASETS_ROOT = Path("/kaggle/input/datasets/mohammedtalafha")

ckpt_matches = list(USER_DATASETS_ROOT.rglob("b52_best_model.pt"))
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

B52_CHECKPOINT = ckpt_matches[0]
BASE_CHECKPOINT = base_matches[0]
CONFIG_PATH = config_matches[0]
CODE_ROOT = package_matches[0].parent
sys.path.insert(0, str(CODE_ROOT))
os.environ["PYTHONPATH"] = f"{CODE_ROOT}:{os.environ.get('PYTHONPATH', '')}"

import torch
print("CODE_ROOT      ", CODE_ROOT)
print("B52_CHECKPOINT ", B52_CHECKPOINT)
print("visible GPUs   ", torch.cuda.device_count())
```

The last line must report **at least two GPUs**. If it reports one, stop and fix
the accelerator setting — the run cannot finish inside the limit on one T4.

## 4. Verify the compressed-DICOM decoders

The hidden set contains mixed transfer syntaxes. The visible example is too
small to prove every decoder is present -- all local and visible data is
uncompressed (`1.2.840.10008.1.2.1`), so it exercises no decoder at all.

**`importlib.invalidate_caches()` is load-bearing, not tidiness.** `pip install`
into a running kernel leaves a stale path cache; pydicom then fails to import
GDCM, and because it resolves plugin availability once when
`pydicom.pixels.decoders` is first imported, that verdict stands for the life
of the kernel. This is what produced a spurious `no decoder for JPEG Lossless
P14` in an earlier B52 notebook, on an environment where GDCM was in fact
installed and working.

```python
import importlib

importlib.invalidate_caches()   # before any pydicom import, after the pip install

import gdcm
print("gdcm", gdcm.Version.GetVersion())

from pydicom.pixels import get_decoder

required = {
    "JPEG Lossless P14": "1.2.840.10008.1.2.4.57",
    "JPEG Lossless SV1": "1.2.840.10008.1.2.4.70",
    "JPEG2000 Lossless": "1.2.840.10008.1.2.4.90",
    "JPEG2000":          "1.2.840.10008.1.2.4.91",
}
for name, uid in required.items():
    decoder = get_decoder(uid)
    print(f"{name:<20} available={decoder.is_available}")
    assert decoder.is_available, f"no decoder for {name} ({uid})"
print("Compressed DICOM decoder preflight: PASS")
```

Measured on the Kaggle image with `python-gdcm 3.0.24.1`: all four report
`available=True`. Their `missing_dependencies` lists name *pylibjpeg*, an
optional second plugin -- GDCM alone covers every syntax the contract
mentions, so a missing pylibjpeg is not a problem.

Because this passes on a healthy environment, a failure now means something
real. Assert it rather than warning: with the hidden-safe contract an
undecodable series is silently dropped and the study still predicts, so the
cost of getting this wrong is a quietly weaker score you cannot measure -- the
hidden run's log and manifest are not visible to you.

## 5. Verify the checkpoint is the one you meant

```python
import hashlib

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

B52_SHA = "<the hash from step 1>"
assert sha256(B52_CHECKPOINT) == B52_SHA, "uploaded checkpoint is not the B52 file"
print("B52 artifact: VERIFIED")
```

## 6. Run

```python
from rsna_knee.b7_weak_supervision import _read_config
from rsna_knee.b52_competition_submission_dualgpu_fast import (
    generate_b52_submission_dual_gpu_fast,
)

config = dict(_read_config(CONFIG_PATH))

generate_b52_submission_dual_gpu_fast(
    config,
    data_root=DATA_ROOT,
    checkpoint=B52_CHECKPOINT,
    base_checkpoint=BASE_CHECKPOINT,
    expected_checkpoint_sha256=B52_SHA,
    out_path="/kaggle/working/submission.csv",
)
```

On the visible three-study example this completes in about a minute. During
hidden scoring the same cell processes roughly 1,300 studies across both T4s.

**Read the first `estimated_remaining` line before leaving it.** It appears after
ten studies per GPU. If the two projections sum to anything near 8 hours,
something is wrong with the accelerator setting and the run will abort later
anyway.

## 7. Check the output before submitting

```python
import pandas as pd, json

submission = pd.read_csv("/kaggle/working/submission.csv")
print(submission.shape)
assert submission.notna().all().all(), "submission contains blanks"
assert submission.iloc[:, 1:].to_numpy().min() >= 0.0
assert submission.iloc[:, 1:].to_numpy().max() <= 1.0

manifest = json.load(open("/kaggle/working/submission.csv.manifest.json"))
print(manifest["experiment"], "| fixed_endpoint =", manifest["fixed_endpoint"])
print("elapsed hours:", round(manifest["runtime_elapsed_hours"], 2))
print("selected epoch:", manifest["selected_epoch"], "at", manifest["selection_value"])
```

`fixed_endpoint` must read **False**. That is B52 telling the truth about
itself, not a fault.

## 8. Submit

Save a version of the notebook with internet **off** and both T4s enabled, then
submit it to the competition.

## If a hidden run throws an exception you cannot see

This has now happened three times: B39, B41 and B51 each passed their visible
three-study notebook and then hit `Notebook Threw Exception` on the hidden
rerun, with no traceback exposed. The same code passed and failed on different
data, so the cause is data or scale.

B39 and B41 were diagnosed in `B39_B41_HIDDEN_SAFE_STREAMING.md` and given a
hidden-safe execution contract. **That contract never reached B42's path**,
which is what B51 and B52 run on. It does now. Three switches, all defaulting to
B42's frozen behaviour and all turned on for B51 and B52:

```text
stream_views=True       one TTA view at a time, not [3, K, 32, 3, H, W] for the
                        whole study. At 14 series -- the maximum in the data --
                        that is about 3.2 GiB against roughly 0.6, and two shards
                        run at once. Host arenas are trimmed after each study.

abort_on_budget=False   the runtime projection becomes telemetry. It is
                        mean(last five) x remaining x 1.35, so one slow early
                        study in a 650-study shard can forecast past the budget
                        and raise -- turning a conservative estimate into the
                        exception it exists to prevent.

on_unreadable=fallback  a study that cannot be read gets 0.5 for all twelve
                        targets and the run carries on; a single unreadable
                        series is dropped and the rest of the study still
                        predicts. An out-of-memory study is retried once with an
                        empty cache first.
```

Streaming changes when memory is held, not what the model sees.
`normalized_view_b42` is asserted bit-identical to the audited normalize-once
helper with `torch.equal`, and the wiring around it -- series order, positions,
per-series metadata, the three offsets -- is compared against that helper too.

Every guessed row is counted and named in the manifest:

```json
"hidden_safe_execution": {"tta_materialization": "one complete study view at a time",
                          "runtime_projection": "telemetry_only_no_exception"},
"studies_predicted_from_fallback": 3,
"fallback_studies": [{"index": 417, "study_uid": "...", "error": "..."}]
```

**Read that number before you read the score.** A handful of guessed rows out of
1,300 costs almost nothing. Hundreds means something systemic went wrong -- most
likely the data path -- and the score is meaningless rather than disappointing.

B42's own launcher keeps all three frozen defaults, because its `0.714` hidden
run was made under them and must stay reproducible.

## Why there is no converter

B51 reaches this path through `b51_checkpoint_to_b42_format.py`, which rewrites
its metadata to present as B42. B52 cannot, and must not.

`load_b42_checkpoint` checks four things that are true of B42's frozen run and
false of B52 by design:

```text
fixed_endpoint, completed_epochs == 2    B52 ran six epochs and selected the best
training_studies == 4349                 B52 held out a scanner-grouped split (3,801 trained)
training_series                          B52 does not record it
training_supervision_cells               B52 does not record it
```

A converter could write those values anyway. It must not: they are assertions
about how a model was trained, and forging them to get past a check would put a
false provenance record inside the one artefact nobody can inspect during a
hidden run.

So B52 has its own loader, checking the things that actually govern a forward
pass and a fair score — the declared SHA-256, the B52 experiment and version, the
model class the weights were trained under, the base checkpoint's fingerprint,
the reconstructed encoder's fingerprint, the head geometry, the encoder chunk,
and that no expert label or expert gradient ever reached the run. The inference
loop itself is B42's, called rather than copied: `generate_b42_submission_dual_gpu_fast`
takes the checkpoint loader and the endpoint manifest as arguments, both
defaulting to B42's own, so no B42 submission behaves differently than before.

`developments/tests/test_b52_submission.py` covers all of it, including that
B42's defaults are unchanged and that no B42 claim can leak into a B52 manifest.

## After the score appears

Record it against B42's and B51's `0.714`. B52's `0.834998` is a validation
selection statistic on 548 unseen-scanner studies and is **not** comparable with
a hidden leaderboard number — different studies, different scale, and chosen on
the very surface that produced it. Expect the hidden score to be lower.

Whatever it shows, do not tune epochs, encoder stages, seed or geometry from the
hidden result.
