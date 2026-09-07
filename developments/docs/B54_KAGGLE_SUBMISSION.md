# B54 Kaggle submission runbook

B54 is B52's competition recipe with a rebuilt teacher, the B6 v1.3.1 report
vocabulary, and a spacing term the model was told to use. Its selected epoch
scored **0.800665** macro AUC on 548 unseen-scanner studies.

```text
b52_best_model.pt (in runs/085_B54/train)
  -> b54_competition_submission_dualgpu_fast.py   B42's inference path, B54's identity
  -> submission.csv + submission.csv.manifest.json
```

There is no conversion step. The file submitted is the file that was trained.

## 0. Which run this is

Two B54 runs were trained. **Only the second exists.**

```text
v1   scale 1.0     best epoch 4   0.804168   DELETED by the re-run's rm -rf
v2   scale 90.33   best epoch 4   0.800665   runs/085_B54/train   <- this one
```

They scored the same on 548 studies, so nothing was lost scientifically, but v1
is not recoverable and no path below refers to it. `B54_ONE_RUN_RUNBOOK.md` now
renames rather than deletes, so this cannot happen to v2.

The two differ in one way that matters here: v1's spacing term reached 0.07% of
the sum it joined, and v2's reaches **12.65%**. Section 3 is entirely about the
consequence.

## 1. Read the population before you spend a submission

This is the thing most likely to be misread once a number appears.

```text
runs/086_...b52_competition_full_finetune   0.802666   1,447 studies
runs/087_...b52_full_data                   0.834998   3,801 studies  -> Kaggle 0.716
runs/085_B54/train                          0.800665   1,447 studies  <- this one
```

**B54 is 086's population, not 087's.** It trained on the same 1,447-study B50
selection gate. The standing `0.716` was set by a model that saw 2,354 more
studies — 2.6 times as much data.

So a lower hidden score here is the *expected consequence of the smaller
training set*, not evidence against the teacher rebuild. The two numbers are
not comparable, and nothing about the teacher, the vocabulary or the spacing
can be concluded from the difference between them.

What this submission can honestly establish is a floor: whether the rebuilt
teacher produces a model that works at all on hidden data. If you want a number
that is comparable with `0.716`, the run to do is B54's recipe on the full
3,801-study population, and that is a separate 12-hour job.

The launcher prints this comparison before it spends a GPU, and the manifest
records it, so it cannot be forgotten between now and the score.

## 2. Why the B52 launcher will not take this checkpoint

B54's checkpoint declares itself as B52 — `experiment` is
`B52_COMPETITION_FULL_FINETUNE`, with B54's details in `model_state` and
`spacing` — so nearly all of `require_b52_endpoint` already passes on it.

Exactly one thing fails:

```python
model.base.load_state_dict(payload["base_state"], strict=True)
```

B54's `base_state` carries one key B42's model has never heard of,
`spacing_conditioning.projection.weight`, so the strict load raises
`Unexpected key(s) in state_dict`. That is the whole incompatibility.

`b54_competition_submission_dualgpu_fast` installs the conditioning before the
load and delegates everything else — B52's identity checks by calling them,
B42's inference loop by passing a loader into it. **Strictness stays.** Setting
`strict=False` would make the same call succeed by silently throwing the
trained term away, which is the exact shape of failure this project has already
paid for twice.

## 3. The spacing term is switched off, and that has to be earned

The submitted forward runs with `enabled=False`, because B42's inference loop
supplies no spacing. Feeding one means a new per-series DICOM read inside a
hidden run whose exceptions are invisible, and B39, B41 and B51 each passed a
visible notebook and then threw on the hidden rerun.

**Switching off a trained term is a change to the model.** An earlier version of
this runbook justified it in a sentence: the term is a no-op, worth `+0.000152`
on the 58 experts. That was true of B54 v1, whose conditioning reached 0.07% of
the sum it joined. **It is false of v2, whose scaled conditioning reaches
12.65%** — and the sentence would have shipped v1's evidence attached to v2's
weights.

So the launcher now measures the checkpoint in hand, using the same code the
probe uses, and applies a two-step rule:

```text
term below 1% of its own sum      switching it off is not a change to the
                                  model; no further evidence needed

term at or above 1%               the disabled arm is REFUSED unless you pass
                                  --expert58-ablation with the eval result for
                                  this same checkpoint, and that result shows
                                  the two arms agree within the surface's
                                  resolution
```

An ablation of a *different* checkpoint is refused too: it would print
identically and license nothing. The evidence that survives goes into the
manifest under `spacing_conditioning.evidence`, measured rather than recorded.

**For B54 v2 this means you must run the Expert-58 ablation first** (step 4a
below) and it must come back small. If it does not, the spacing term is doing
measurable work, the disabled arm is not the trained model, and this checkpoint
cannot go through B42's loop as it stands. That is the correct outcome rather
than an obstacle — the alternative is submitting a model in a configuration it
was never trained in and calling it B54.

`assert_conditioning_disabled` separately checks the loaded model rather than
trusting the keyword that was passed.

## 4a. Run the ablation, because v2 needs it

```bash
cd /media/talafha/Disk_1/CNN_CPC
PYTHONPATH=developments/src python -m rsna_knee.b54_expert58_eval \
  --data-root /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection \
  --checkpoint runs/085_B54/train/b52_best_model.pt \
  --base-checkpoint runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt \
  --spacing-geometry-csv runs/slice_geometry_scan/series_geometry.csv \
  --out-root runs/085_B54/expert58_v2 \
  2>&1 | tee runs/085_B54/b54_v2_expert58.log
```

It writes `runs/085_B54/expert58_v2/expert58.json`, which is what
`--expert58-ablation` wants in step 9.

## 4. Fingerprints this run depends on

Read them; do not assume them.

```bash
cd /media/talafha/Disk_1/CNN_CPC
R=runs/085_B54/train

find "$R" -name "b52_best_model.pt"
sha256sum "$R"/**/b52_best_model.pt

PYTHONPATH=developments/src python - <<'PY'
from pathlib import Path
import torch

for path in sorted(Path("runs/085_B54/train").rglob("b52_best_model.pt")):
    p = torch.load(path, map_location="cpu", weights_only=False)
    spacing = p.get("spacing", {})
    print(path)
    print("  experiment    :", p["experiment"])
    print("  version       :", p["version"])
    print("  model version :", p["model_state"].get("version"))
    print("  selected epoch:", p["selected_epoch"], "at", p["selection_value"])
    print("  train studies :", p["training_studies"])
    print("  chunk size    :", p["model_state"]["encoder_chunk_size"])
    print("  spacing on    :", spacing.get("enabled"))
    print("  spacing moved :", spacing.get("conditioning_moved"))
    print("  spacing scale :", spacing.get("conditioning_scale", 1.0))
    print("  base ckpt     :", p["base_checkpoint"])
PY
```

Expected: `model version` starting `b54_`, `spacing on` **True**, `spacing
moved` **True**, `spacing scale` **90.334485**, `train studies` **1447**,
`chunk size` **4**, and `selected epoch` **4** at **0.800665**.

A scale of `1.0` would mean you are looking at a v1 checkpoint, which should not
exist — say so rather than submitting it.

The first hash is what you will declare as `B54_SHA`.

## 5. Build the artifact dataset

Identical to B52's, with B54's paths. The checkpoint bytes must not change.

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee

export B54_CHECKPOINT="$(find runs/085_B54/train -name b52_best_model.pt | head -1)"
export BASE_CHECKPOINT="$(PYTHONPATH=developments/src python -c "import torch;print(torch.load('$B54_CHECKPOINT',map_location='cpu',weights_only=False)['base_checkpoint'])")"

rm -rf kaggle_b54_artifacts
mkdir -p kaggle_b54_artifacts/CNN_CPC/config
mkdir -p kaggle_b54_artifacts/CNN_CPC/developments/src
mkdir -p kaggle_b54_artifacts/CNN_CPC/models

cp config/b42_constant_area_aspect_sparse.yaml kaggle_b54_artifacts/CNN_CPC/config/
cp -a developments/src/rsna_knee kaggle_b54_artifacts/CNN_CPC/developments/src/
cp "$B54_CHECKPOINT" kaggle_b54_artifacts/CNN_CPC/models/b54_best_model.pt
cp "$BASE_CHECKPOINT" kaggle_b54_artifacts/CNN_CPC/models/b34_llm_fill_base_model.pt

find kaggle_b54_artifacts -name "__pycache__" -type d -exec rm -rf {} +
sha256sum \
  kaggle_b54_artifacts/CNN_CPC/models/b54_best_model.pt \
  kaggle_b54_artifacts/CNN_CPC/models/b34_llm_fill_base_model.pt
```

The first hash must equal `B54_SHA` from step 4. If it does not, the wrong file
was copied; stop.

Upload `kaggle_b54_artifacts` as a **new** private Kaggle dataset. Do not
overwrite the B52 artifact dataset — the `0.716` notebook still points at it.

## 6. Resolve the mounted artifacts

```python
from pathlib import Path
import os
import sys

DATA_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
USER_DATASETS_ROOT = Path("/kaggle/input/datasets/mohammedtalafha")

ckpt_matches = list(USER_DATASETS_ROOT.rglob("b54_best_model.pt"))
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

B54_CHECKPOINT = ckpt_matches[0]
BASE_CHECKPOINT = base_matches[0]
CONFIG_PATH = config_matches[0]
CODE_ROOT = package_matches[0].parent
sys.path.insert(0, str(CODE_ROOT))
os.environ["PYTHONPATH"] = f"{CODE_ROOT}:{os.environ.get('PYTHONPATH', '')}"

import torch
print("CODE_ROOT      ", CODE_ROOT)
print("B54_CHECKPOINT ", B54_CHECKPOINT)
print("visible GPUs   ", torch.cuda.device_count())
```

The last line must report **at least two GPUs**. One T4 cannot finish inside
the nine-hour limit, and the launcher refuses to start rather than discovering
it in the eighth hour.

If `b34_llm_fill_base_model.pt` matches more than once, both the B52 and the
B54 artifact datasets are attached. Detach the B52 one.

## 7. Verify the compressed-DICOM decoders

Unchanged from B52, and still load-bearing. `importlib.invalidate_caches()`
before any pydicom import, after the pip install — see
`B52_KAGGLE_SUBMISSION.md` §4 for why, and run that cell verbatim.

## 8. Verify the checkpoint is the one you meant

```python
import hashlib

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

B54_SHA = "<the hash from step 4>"
assert sha256(B54_CHECKPOINT) == B54_SHA, "uploaded checkpoint is not the B54 file"
print("B54 artifact: VERIFIED")
```

## 9. Run

```python
from rsna_knee.b7_weak_supervision import _read_config
from rsna_knee.b54_competition_submission_dualgpu_fast import (
    generate_b54_submission_dual_gpu_fast,
)

config = dict(_read_config(CONFIG_PATH))

generate_b54_submission_dual_gpu_fast(
    config,
    data_root=DATA_ROOT,
    checkpoint=B54_CHECKPOINT,
    base_checkpoint=BASE_CHECKPOINT,
    expected_checkpoint_sha256=B54_SHA,
    expert58_ablation=ABLATION_JSON,   # upload it with the artifacts
    out_path="/kaggle/working/submission.csv",
)
```

`ABLATION_JSON` is `expert58.json` from step 4a. Copy it into the artifact
dataset alongside the checkpoint and resolve it the same way, by `rglob`. It is
tiny, and the launcher refuses the disabled arm without it whenever the trained
term is large enough to matter — which for v2 it is.

Five lines to read before you leave it:

```text
[B54 submit] epoch 4 selected at 0.800665, trained on 1447 studies, augmentation=True
[B54 submit] spacing conditioning trained then DISABLED for inference; scale
             90.334485, term reaches 12.65% of the sum it joins
[B54 submit] the trained term reaches 12.65% ... but the measured ablation on
             these same weights moved the Expert-58 macro AUC by only ...
[B54 submit] POPULATION: this model saw 1447 studies; the standing 0.716 was set
             by a B52 run that saw 3801. ...
[B54 submit] inference path is B42's, unchanged
```

Then the first `estimated_remaining` line, after ten studies per GPU. If the
two projections sum to anything near eight hours, the accelerator setting is
wrong and the run will not finish.

## 10. Check the output before submitting

```python
import pandas as pd, json

submission = pd.read_csv("/kaggle/working/submission.csv")
print(submission.shape)
assert submission.notna().all().all(), "submission contains blanks"
assert submission.iloc[:, 1:].to_numpy().min() >= 0.0
assert submission.iloc[:, 1:].to_numpy().max() <= 1.0

manifest = json.load(open("/kaggle/working/submission.csv.manifest.json"))
print(manifest["experiment"])
print("submitted arm :", manifest["spacing_conditioning"]["submitted_arm"])
print("population    :", manifest["population_warning"])
print("guessed rows  :", manifest.get("studies_predicted_from_fallback"))
print("elapsed hours :", round(manifest["runtime_elapsed_hours"], 2))
```

`submitted_arm` must read `spacing_off`, and `population_warning.comparable`
must read `False` — both are B54 telling the truth about itself, not faults.

**Read `guessed rows` before you read the score.** A handful out of ~1,300
costs almost nothing; hundreds means the data path broke and the score is
meaningless rather than disappointing.

## 11. Submit

Save a version with internet **off** and both T4s enabled, then submit.

## After the score appears

Record it beside the population, never alone:

```text
0.714   B42 / B41 constant-area          4,349 studies
0.716   B52 competition full fine-tune   3,801 studies
?????   B54 teacher rebuild              1,447 studies   <- 2.6x less data
```

A B54 score below `0.716` is the expected result and refutes nothing. A B54
score *at or above* `0.716` would be the interesting outcome, because it would
have been reached with 2.6 times less data — and even that is one number on
~1,300 studies, which this project has already agreed to read as noise at the
`±0.002` scale.

Whatever it shows, do not tune the teacher, the vocabulary, the epochs or the
geometry against it.
