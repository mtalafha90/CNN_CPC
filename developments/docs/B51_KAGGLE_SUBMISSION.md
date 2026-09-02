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

## Completed hidden result

**COMPLETED / KAGGLE `0.713` / NOT PROMOTED.**

```text
B42 (frozen hierarchy)      0.714
B51 (adapted hierarchy)     0.713      -0.001
```

B50's controlled comparison measured `+0.011221` on 548 unseen-scanner
studies, all twelve targets improved. On roughly 1,300 hidden studies the
same change measures `-0.001`. The runbook predeclared this outcome as one of
the two it expected -- "may display `0.714` again and be indistinguishable" --
and predeclared that it authorises nothing. It does not.

**Do not read `-0.001` as a decrease.** Kaggle displays three decimals, and one
unit in the last place on this many studies is inside the noise of the
measurement. The honest statement is that the adapted hierarchy is
indistinguishable from the frozen one on hidden data, not that it is worse.

### One confound, narrowed but not closed

This run was the first through the hidden-safe execution contract, which
**drops a series it cannot decode** instead of ending the run. B42's `0.714`
was measured under the strict reader, where an undecodable series was
impossible by construction: the run crashed instead.

All local and visible data is uncompressed (`1.2.840.10008.1.2.1`), and the
competition contract says the hidden set may contain JPEG Lossless and JPEG
2000. A decoder preflight in the B52 notebook raised `no decoder for JPEG
Lossless P14`, which looked like a missing codec. It was not:

```text
gdcm OK 3.0.24
1.2.840.10008.1.2.4.57  available=True
1.2.840.10008.1.2.4.70  available=True
1.2.840.10008.1.2.4.90  available=True
1.2.840.10008.1.2.4.91  available=True
```

The four `missing_dependencies` entries name *pylibjpeg*, an optional second
plugin. GDCM alone covers every syntax the contract mentions.

The earlier failure was an **import-cache artefact**: `pip install` into a
running kernel left a stale path cache, so pydicom's decoder module tried
`import gdcm`, failed, and cached the plugin as unavailable. pydicom resolves
plugin availability when `pydicom.pixels.decoders` is first imported, not per
call, so that verdict then stands for the life of the kernel.

**What is still unknown.** The B51 submission notebook installed the wheel and
then imported pydicom in a later cell without calling
`importlib.invalidate_caches()`. Whether that kernel resolved GDCM depends on
whether its path cache was stale at that moment, and the hidden run's log is
not visible. The visible three studies prove nothing either way -- they are
uncompressed and need no decoder.

So `0.713` is either clean or a floor, and which one cannot be recovered for
this submission. Every future submission should call
`importlib.invalidate_caches()` before the first pydicom import and assert
`is_available` on all four syntaxes, which now passes and therefore means
something when it fails.

### What this does and does not settle

* The hidden-safe execution path works. Three previous attempts on this family
  -- B39, B41 and B51's own first try -- ended in `Notebook Threw Exception`.
* B50's `+0.011` on 548 report-labelled unseen-scanner studies did not
  reproduce on ~1,300 hidden studies with expert labels. Either the effect is
  smaller than the domain-shift split suggested, or the two surfaces measure
  different things. This project has now seen `0.694`, `0.707`, `0.713` and
  `0.714` from four architectures, which is a plateau, not a ranking.
* Nothing here authorises tuning the hierarchy learning rate, the epoch count,
  the seed or the geometry against this number.

## After the score appears

Record it against B42's `0.714` and stop. Whatever it shows, do not tune the
hierarchy learning rate, epoch count, seed, geometry or target subset from the
hidden result — B51 is a production run of a mechanism B50 already validated,
and its control is B42's existing hidden score.
