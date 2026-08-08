# CNN_CPC Training From Zero

This is the clean, end-to-end runbook for training `CNN_CPC` on the real **RSNA Knee Abnormality Detection** competition data from a fresh Linux machine.

The intended workflow is:

1. clone/update the repository;
2. create a clean Conda environment;
3. install the project and test dependencies;
4. verify the software and the committed four-study DICOM fixture;
5. download/place the official Kaggle data;
6. create a machine-local production config;
7. inspect the real data and export nested validation manifests;
8. run DICOM preflight and the full audit;
9. run Stage-1 random-initialization smoke tests;
10. train Stage-1 random folds;
11. train the competition-data SSL candidate;
12. choose random vs SSL independently for each outer fold using **inner AUC only**;
13. train Stage 2 using leakage-safe fold-local teachers;
14. evaluate OOF predictions;
15. run final three-fold inference.

> **Important:** do not skip the preflight, audit, or smoke stages. Do not launch folds simultaneously on one GPU. Do not use `torchrun` or DDP. The production configuration is designed for one GPU, CPU multiprocessing for DICOM work, and an 8.5-hour software budget per long GPU job.

---

## 0. Fresh terminal: define the repository path

Choose where the repository will live.

Example:

```bash
export REPO="/media/talafha/Disk_1/CNN_CPC"
```

If your repository is elsewhere, change only that path.

If the repository is not cloned yet:

```bash
cd "$(dirname "$REPO")"
git clone https://github.com/mtalafha90/CNN_CPC.git
cd "$REPO"
```

If it is already cloned:

```bash
cd "$REPO"
git checkout main
git pull --ff-only origin main
```

Check the checkout:

```bash
pwd
git branch --show-current
git status
git log -1 --oneline
```

The active branch should be `main` and the working tree should normally be clean before starting a production run.

---

# Part I — Environment setup

## 1. Create the Conda environment

This repository requires Python 3.10 or newer. The recommended reproducible local setup is Python 3.12 in a dedicated Conda environment.

First check Conda:

```bash
conda --version
```

Create the environment once:

```bash
conda create -n rsna-knee python=3.12 -y
```

Activate it:

```bash
conda activate rsna-knee
```

Check that the active Python belongs to the Conda environment:

```bash
which python
python --version
```

Do not mix this environment with an old project `.venv`.

If a `.venv` is active, leave it before activating Conda:

```bash
deactivate 2>/dev/null || true
conda activate rsna-knee
```

---

## 2. Install the project and required packages

Enter the repository:

```bash
cd "$REPO"
```

Update the Python packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the repository in editable mode. This installs the project dependencies declared in `pyproject.toml`, including NumPy, pandas, scikit-learn, pydicom, PyYAML, PyTorch, and torchvision:

```bash
python -m pip install -e .
```

Install the local testing/utility packages:

```bash
python -m pip install pytest pillow kaggle
```

Check dependency consistency:

```bash
python -m pip check
```

Expected:

```text
No broken requirements found.
```

Verify the package import:

```bash
python - <<'PY'
import rsna_knee
print("rsna_knee import: OK")
print("package path:", rsna_knee.__file__)
PY
```

Verify the CLI:

```bash
python -m rsna_knee.cli --help
```

You should see commands including:

```text
inspect
preflight
audit
pseudo-label
pretrain
train
validation-manifest
select-stage1
evaluate
infer
runtime
```

---

## 3. Check the GPU before real training

Run:

```bash
nvidia-smi
```

Then:

```bash
python - <<'PY'
import torch
print("PyTorch version :", torch.__version__)
print("PyTorch CUDA    :", torch.version.cuda)
print("CUDA available  :", torch.cuda.is_available())
print("GPU count       :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name        :", torch.cuda.get_device_name(0))
PY
```

For production training, `torch.cuda.is_available()` must be `True` and a suitable GPU must be visible.

If CUDA is unavailable, the CPU-only validation/audit steps can still be run, but **do not start production model training** until the driver/PyTorch CUDA combination is corrected or the repository is moved to a suitable GPU machine.

Use exactly one GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Do not launch this project with `torchrun`.

---

# Part II — Software sanity checks before touching the competition data

## 4. Run the committed four-study external fixture test

The repository contains four small external MRI cases for technical DICOM/pipeline validation. These are not competition validation data and must not be used for scientific AUC/model selection.

Run:

```bash
cd "$REPO"
pytest -q tests/test_external_fixture.py
```

Expected: all tests pass.

Check the committed test DICOM count:

```bash
find fixtures/external_validation/test_images -type f -name "*.dcm" -print
find fixtures/external_validation/test_images -type f -name "*.dcm" | wc -l
```

Expected DICOM count:

```text
4
```

Run the strict fixture preflight:

```bash
mkdir -p runs
python -m rsna_knee.cli preflight \
  --data-root fixtures/external_validation \
  --split test \
  --sample-size 4 \
  --max-decode-failure-rate 0 \
  --max-file-decode-failure-rate 0 \
  --out runs/external_test_preflight.json
```

Inspect it:

```bash
python -m json.tool runs/external_test_preflight.json
```

For the committed fixture, the important values are:

```text
studies_sampled          = 4
streams_selected         = 4
streams_decoded          = 4
candidate_files          = 4
file_decode_failures     = 0
decode_failure_rate      = 0.0
file_decode_failure_rate = 0.0
decoded_frames           = 28
```

A high `missing_stream_rate` is expected for this tiny fixture because each external example intentionally provides only one MRI series rather than all six possible competition streams.

---

## 5. Run the complete repository test suite

```bash
pytest -q
```

Then compile all Python modules:

```bash
python -m compileall -q src tests kaggle scripts
```

No output from `compileall` means success.

Re-check dependencies:

```bash
python -m pip check
```

Re-check the CLI:

```bash
python -m rsna_knee.cli --help
```

**Hard gate:** do not move to real-data training if the test suite or CLI fails.

---

# Part III — Official Kaggle data

## 6. Download/place the competition data

Before using the Kaggle CLI, make sure the competition rules have been accepted in your Kaggle account.

If you are already downloading the data from the Kaggle website, let that finish and skip directly to the data-path setup below.

Optional CLI download:

```bash
kaggle auth login
```

Choose a parent directory with enough free disk space, for example:

```bash
export DATA_PARENT="/media/talafha/Disk_1"
mkdir -p "$DATA_PARENT"
```

Then download:

```bash
kaggle competitions download \
  -c rsna-knee-abnormality-detection \
  -p "$DATA_PARENT"
```

If the result is a ZIP archive, create the target directory and extract it:

```bash
mkdir -p "$DATA_PARENT/rsna-knee-abnormality-detection"
unzip "$DATA_PARENT/rsna-knee-abnormality-detection.zip" \
  -d "$DATA_PARENT/rsna-knee-abnormality-detection"
```

Now define the official data root:

```bash
export DATA_ROOT="$DATA_PARENT/rsna-knee-abnormality-detection"
```

If you downloaded/extracted elsewhere, set `DATA_ROOT` to that real location instead.

Example:

```bash
export DATA_ROOT="/media/talafha/Disk_1/rsna-knee-abnormality-detection"
```

Check the CSV files:

```bash
ls -lh \
  "$DATA_ROOT/train.csv" \
  "$DATA_ROOT/train_series.csv" \
  "$DATA_ROOT/test.csv" \
  "$DATA_ROOT/test_series.csv"
```

Check the image directories:

```bash
find "$DATA_ROOT" -maxdepth 1 -type d -printf '%f\n' | sort
```

The extracted dataset must contain the official training image tree and, for final inference, the official test image tree.

Do not copy the competition dataset into the Git repository and do not commit the Kaggle images.

---

# Part IV — Create a machine-local production configuration

## 7. Build `configs/train_local.yaml`

Return to the repository and make sure the Conda environment is active:

```bash
cd "$REPO"
conda activate rsna-knee
export CUDA_VISIBLE_DEVICES=0
```

Confirm `DATA_ROOT`:

```bash
echo "$DATA_ROOT"
```

Copy the production config:

```bash
cp configs/train.yaml configs/train_local.yaml
```

Patch only machine/output fields while preserving the production safety contract:

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml

if "DATA_ROOT" not in os.environ:
    raise SystemExit("DATA_ROOT is not defined")

path = Path("configs/train_local.yaml")
config = yaml.safe_load(path.read_text())

config["data_root"] = os.environ["DATA_ROOT"]
config["output_dir"] = "runs/stage1_random"
config["ssl_output_dir"] = "runs/ssl"

# Competition/runtime safety contract.
config["competition_mode"] = True
config["requested_gpus"] = 1
config["runtime_budget_hours"] = 8.5
config["pretrained"] = False
config["allow_external_pretrained"] = False

# Stage 1 starts without SSL or co-training teachers.
config["ssl_encoder_checkpoint"] = None
config["ssl_checkpoint_source"] = None
config["cotrain_stage1_root"] = None
config["cotrain_stage1_candidates"] = None
config["expected_checkpoint_stage"] = None

path.write_text(yaml.safe_dump(config, sort_keys=False))
print("Written:", path)
print("Data root:", config["data_root"])
PY
```

Verify the important fields:

```bash
grep -E \
'^(data_root|output_dir|ssl_output_dir|competition_mode|requested_gpus|runtime_budget_hours|pretrained|allow_external_pretrained|tta_center_offsets|validation_tta_offsets|cotrain_stage1_root|cotrain_stage1_candidates):' \
configs/train_local.yaml
```

The two TTA lists must remain identical:

```text
tta_center_offsets: [-1, 0, 1]
validation_tta_offsets: [-1, 0, 1]
```

Do not change the production TTA policy after looking at outer validation results.

`configs/train_local.yaml`, `configs/train_local_ssl.yaml`, `configs/train_local_stage2.yaml`, and `configs/final_infer.yaml` are machine/run files. Review them carefully before committing any of them, because they may contain machine-specific absolute paths.

---

## 8. Verify the resolved runtime

```bash
python -m rsna_knee.cli runtime --config configs/train_local.yaml
```

Then:

```bash
nvidia-smi
```

During a GPU run, an optional second terminal can monitor utilization:

```bash
watch -n 2 nvidia-smi
```

---

# Part V — Real-data inspection and validation construction

## 9. Inspect the official CSVs

```bash
python -m rsna_knee.cli inspect --data-root "$DATA_ROOT"
```

The command reports the total study count, gold-label count, unlabeled count, report availability, series count, and target frequencies among gold studies.

If the counts look obviously wrong, stop and fix `DATA_ROOT`/extraction before continuing.

---

## 10. Export the three real nested-validation manifests

The four external fixture images are only a technical sanity test. Real model selection uses the official competition gold labels.

Create the output directory:

```bash
mkdir -p runs/validation
```

Export all three outer folds:

```bash
for f in 0 1 2; do
  python -m rsna_knee.cli validation-manifest \
    --config configs/train_local.yaml \
    --fold "$f" \
    --out "runs/validation/fold${f}.csv"
done
```

Inspect role counts:

```bash
python - <<'PY'
import pandas as pd

for f in range(3):
    path = f"runs/validation/fold{f}.csv"
    df = pd.read_csv(path)
    print(f"\n===== FOLD {f} =====")
    print(df["role"].value_counts())
PY
```

The manifest roles separate:

- `outer_validation` — held-out outer fold;
- `inner_selection` — fold used for inner model/candidate selection;
- `gold_train` — remaining gold training studies.

Do not use outer-fold AUC to decide random-vs-SSL Stage-1 candidates.

---

# Part VI — DICOM gates before training

## 11. Run the official training-data preflight

```bash
mkdir -p runs
python -m rsna_knee.cli preflight \
  --data-root "$DATA_ROOT" \
  --split train \
  --sample-size 24 \
  --max-decode-failure-rate 0.05 \
  --max-file-decode-failure-rate 0.05 \
  --out runs/preflight_train.json
```

Inspect:

```bash
python -m json.tool runs/preflight_train.json
```

If the test image tree is already present locally, also run:

```bash
python -m rsna_knee.cli preflight \
  --data-root "$DATA_ROOT" \
  --split test \
  --sample-size 24 \
  --max-decode-failure-rate 0.05 \
  --max-file-decode-failure-rate 0.05 \
  --out runs/preflight_test.json
```

Inspect:

```bash
python -m json.tool runs/preflight_test.json
```

**Hard gate:** do not continue after a strict preflight failure. Investigate DICOM decoding, extraction, path structure, or series routing first.

---

## 12. Run the full CPU multiprocessing audit

This is intentionally separate from GPU training.

```bash
python -m rsna_knee.cli audit \
  --config configs/train_local.yaml \
  --out-dir runs/audit
```

Inspect the most important sections:

```bash
python - <<'PY'
import json
from pathlib import Path

p = json.loads(Path("runs/audit/audit.json").read_text())

print("===== DECODE AUDIT =====")
print(json.dumps(p["decode_audit"], indent=2))

print("\n===== SELECTED STREAMS =====")
print(json.dumps(p["selected_stream_counts"], indent=2))

print("\n===== MISSING STREAMS =====")
print(json.dumps(p["missing_stream_counts"], indent=2))

print("\n===== TEACHER CONFIDENCE =====")
print(json.dumps(p["teacher_confidence_counts"], indent=2))
PY
```

Review especially:

- global DICOM/file decode failures;
- any individual series with substantial partial corruption;
- selected stream distribution;
- unexpectedly high missing-stream counts;
- report-teacher confidence coverage.

**Hard gate:** do not start GPU smoke training until the audit completes successfully and its diagnostics are acceptable.

---

# Part VII — Stage 1A: random initialization

## 13. Run the first real GPU smoke test: fold 0 only

Before launching:

```bash
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
```

Run:

```bash
python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 0 \
  --smoke
```

The smoke mode is deliberately small: at most 2 epochs, 20 train batches per epoch, a reduced bootstrap count, and a shortened runtime budget.

Inspect the output files:

```bash
find runs/stage1_random/smoke/fold0 \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

Inspect the key diagnostics:

```bash
cat runs/stage1_random/smoke/fold0/selection.json
cat runs/stage1_random/smoke/fold0/runtime.json
cat runs/stage1_random/smoke/fold0/training_diagnostics.json
```

Expected core artifacts include:

```text
best.pt
oof.csv
oof_center.csv
weak_oof.csv
selection.json
history.csv
training_diagnostics.json
supervision_plan.json
runtime.json
bootstrap.json
```

`oof.csv` is the primary TTA OOF using the same inference policy planned for submission. `oof_center.csv` is diagnostic center-only OOF. `weak_oof.csv` is the Stage-1 cross-fitted weak-teacher output.

If fold 0 smoke fails, stop and diagnose it before launching any production fold.

---

## 14. Smoke-test folds 1 and 2

Only after fold 0 succeeds:

```bash
python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 1 \
  --smoke

python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 2 \
  --smoke
```

Do not run these simultaneously on the same GPU.

---

## 15. Train the three random-initialized production folds

Run sequentially:

```bash
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 0
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 1
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 2
```

Every long training command has its own runtime budget and reserves time for the finish path (validation/OOF, weak OOF where applicable, bootstrap, and serialization).

After each fold, inspect:

```bash
for f in 0 1 2; do
  if [ -d "runs/stage1_random/fold${f}" ]; then
    echo "===== RANDOM FOLD $f ====="
    cat "runs/stage1_random/fold${f}/selection.json"
    cat "runs/stage1_random/fold${f}/runtime.json"
    cat "runs/stage1_random/fold${f}/training_diagnostics.json"
  fi
done
```

---

## 16. Evaluate Stage-1 random OOF

Primary TTA OOF:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof \
    runs/stage1_random/fold0/oof.csv \
    runs/stage1_random/fold1/oof.csv \
    runs/stage1_random/fold2/oof.csv \
  --n-bootstrap 2000 \
  --out runs/stage1_random/evaluation.json
```

Diagnostic TTA-vs-center comparison:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof \
    runs/stage1_random/fold0/oof_center.csv \
    runs/stage1_random/fold1/oof_center.csv \
    runs/stage1_random/fold2/oof_center.csv \
  --compare-oof \
    runs/stage1_random/fold0/oof.csv \
    runs/stage1_random/fold1/oof.csv \
    runs/stage1_random/fold2/oof.csv \
  --n-bootstrap 2000 \
  --out runs/stage1_random/tta_vs_center.json
```

This comparison is diagnostic only. Do not change the planned submission TTA after observing outer OOF.

---

# Part VIII — Stage 1B: competition-data SSL candidate

## 17. Pretrain the in-domain SSL encoder

Use only the competition-data SSL path configured by the project:

```bash
python -m rsna_knee.cli pretrain --config configs/train_local.yaml
```

Expected outputs:

```text
runs/ssl/ssl_encoder.pt
runs/ssl/history.json
```

Confirm they exist:

```bash
ls -lh runs/ssl/ssl_encoder.pt runs/ssl/history.json
```

---

## 18. Create the SSL Stage-1 config

```bash
cp configs/train_local.yaml configs/train_local_ssl.yaml
```

Patch it:

```bash
python - <<'PY'
from pathlib import Path
import yaml

path = Path("configs/train_local_ssl.yaml")
config = yaml.safe_load(path.read_text())

config["output_dir"] = "runs/stage1_ssl"
config["ssl_encoder_checkpoint"] = str(Path("runs/ssl/ssl_encoder.pt").resolve())
config["ssl_checkpoint_source"] = "competition_training_data"
config["cotrain_stage1_root"] = None
config["cotrain_stage1_candidates"] = None
config["expected_checkpoint_stage"] = None

path.write_text(yaml.safe_dump(config, sort_keys=False))
print(path)
PY
```

Verify:

```bash
grep -E \
'^(output_dir|ssl_encoder_checkpoint|ssl_checkpoint_source|cotrain_stage1_root|cotrain_stage1_candidates):' \
configs/train_local_ssl.yaml
```

---

## 19. Train all three SSL Stage-1 folds

Run sequentially:

```bash
python -m rsna_knee.cli train --config configs/train_local_ssl.yaml --fold 0
python -m rsna_knee.cli train --config configs/train_local_ssl.yaml --fold 1
python -m rsna_knee.cli train --config configs/train_local_ssl.yaml --fold 2
```

You may evaluate SSL outer OOF for research diagnostics, but the random-vs-SSL candidate supplying a given outer fold must **not** be chosen from outer OOF.

---

# Part IX — Leakage-safe Stage-1 candidate selection

## 20. Select random vs SSL independently for each outer fold

Run:

```bash
python -m rsna_knee.cli select-stage1 \
  --candidate-root "$(pwd)/runs/stage1_random" \
  --candidate-root "$(pwd)/runs/stage1_ssl" \
  --n-folds 3 \
  --out runs/stage1_selection.json
```

Inspect:

```bash
python -m json.tool runs/stage1_selection.json
```

The selector uses **only `inner_macro_auc` for the corresponding outer fold**. It deliberately ignores `outer_macro_auc` when choosing random vs SSL.

This selection rule is mandatory for the downstream Stage-2 teacher choice.

---

# Part X — Stage 2: fold-local image/report co-training

## 21. Create the Stage-2 config

Start from the base random initialization config:

```bash
cp configs/train_local.yaml configs/train_local_stage2.yaml
```

Patch it:

```bash
python - <<'PY'
from pathlib import Path
import yaml

path = Path("configs/train_local_stage2.yaml")
config = yaml.safe_load(path.read_text())

config["output_dir"] = "runs/stage2"
config["ssl_encoder_checkpoint"] = None
config["ssl_checkpoint_source"] = None
config["cotrain_stage1_root"] = None
config["cotrain_stage1_candidates"] = [
    str(Path("runs/stage1_random").resolve()),
    str(Path("runs/stage1_ssl").resolve()),
]
config["expected_checkpoint_stage"] = None

path.write_text(yaml.safe_dump(config, sort_keys=False))
print(path)
PY
```

Verify:

```bash
grep -E \
'^(output_dir|ssl_encoder_checkpoint|ssl_checkpoint_source|cotrain_stage1_root|cotrain_stage1_candidates):' \
configs/train_local_stage2.yaml
```

Stage 2 uses the fold-local candidate chosen from each fold's inner selection score. It does not use the outer fold to select its teacher.

---

## 22. Train Stage-2 folds sequentially

```bash
python -m rsna_knee.cli train --config configs/train_local_stage2.yaml --fold 0
python -m rsna_knee.cli train --config configs/train_local_stage2.yaml --fold 1
python -m rsna_knee.cli train --config configs/train_local_stage2.yaml --fold 2
```

Inspect Stage-2 supervision and training diagnostics:

```bash
for f in 0 1 2; do
  echo "===== STAGE2 FOLD $f ====="
  cat "runs/stage2/fold${f}/stage2_supervision.json"
  cat "runs/stage2/fold${f}/training_diagnostics.json"
done
```

Pay particular attention to `zero_to_nonzero_weight`: it measures report-silent cells for which a very confident cross-fitted image teacher contributed modest BCE supervision.

Stage 2 intentionally does not generate another `weak_oof.csv`.

---

## 23. Evaluate Stage 2

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof \
    runs/stage2/fold0/oof.csv \
    runs/stage2/fold1/oof.csv \
    runs/stage2/fold2/oof.csv \
  --n-bootstrap 2000 \
  --out runs/stage2/evaluation.json
```

For a paired comparison with the nested-selected Stage-1 folds, extract their OOF paths from the selection manifest:

```bash
mapfile -t SELECTED_STAGE1_OOF < <(python - <<'PY'
import json
from pathlib import Path

p = json.loads(Path("runs/stage1_selection.json").read_text())
for f in range(3):
    root = Path(p["folds"][str(f)]["selected_root"])
    print(root / f"fold{f}" / "oof.csv")
PY
)
```

Then run the paired comparison:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof \
    "${SELECTED_STAGE1_OOF[0]}" \
    "${SELECTED_STAGE1_OOF[1]}" \
    "${SELECTED_STAGE1_OOF[2]}" \
  --compare-oof \
    runs/stage2/fold0/oof.csv \
    runs/stage2/fold1/oof.csv \
    runs/stage2/fold2/oof.csv \
  --n-bootstrap 2000 \
  --out runs/stage2/vs_nested_stage1.json
```

Once an outer comparison is used to choose the final method, it is model-selection CV rather than a pristine unbiased generalization estimate. Freeze the final method before producing the final test submission.

---

# Part XI — Final inference

## 24. Freeze Stage 2 as the final checkpoint stage

If Stage 2 is the final method, create the inference config:

```bash
cp configs/train_local_stage2.yaml configs/final_infer.yaml
```

Set the checkpoint identity contract:

```bash
python - <<'PY'
from pathlib import Path
import yaml

path = Path("configs/final_infer.yaml")
config = yaml.safe_load(path.read_text())
config["expected_checkpoint_stage"] = "stage2"
path.write_text(yaml.safe_dump(config, sort_keys=False))
print(path)
PY
```

Verify:

```bash
grep -E '^(data_root|expected_checkpoint_stage|tta_center_offsets):' configs/final_infer.yaml
```

---

## 25. Run final three-fold inference

Make sure the official test images are present under `DATA_ROOT` and the three Stage-2 checkpoints exist:

```bash
ls -lh \
  runs/stage2/fold0/best.pt \
  runs/stage2/fold1/best.pt \
  runs/stage2/fold2/best.pt
```

Run inference:

```bash
python -m rsna_knee.cli infer \
  --config configs/final_infer.yaml \
  --checkpoints \
    runs/stage2/fold0/best.pt \
    runs/stage2/fold1/best.pt \
    runs/stage2/fold2/best.pt \
  --out submission.csv
```

Inspect the result:

```bash
ls -lh submission.csv
head submission.csv
```

Do not manually reorder targets or edit prediction values after inference. The inference path validates checkpoint stage/fold/TTA/model identity before writing the submission.

---

# Part XII — Starting again in a new terminal

After the first setup, a new terminal normally needs only:

```bash
export REPO="/media/talafha/Disk_1/CNN_CPC"
export DATA_ROOT="/media/talafha/Disk_1/rsna-knee-abnormality-detection"

cd "$REPO"
conda activate rsna-knee
export CUDA_VISIBLE_DEVICES=0
```

Then verify:

```bash
which python
python -m rsna_knee.cli --help
nvidia-smi
```

Change the two absolute paths to match the machine being used.

---

# Part XIII — Minimal stop/go checklist

Do not start the next phase until the current one is clean.

```text
[ ] Repository is on the intended main commit
[ ] Conda environment rsna-knee is active
[ ] pip check passes
[ ] rsna_knee imports successfully
[ ] CLI --help works
[ ] External fixture tests pass
[ ] External fixture strict preflight has 0 decode failures
[ ] Full pytest suite passes
[ ] DATA_ROOT points to the official extracted competition data
[ ] Real CSV inspection is sensible
[ ] Three nested validation manifests were exported
[ ] Train DICOM preflight passes
[ ] Test DICOM preflight passes when test data are available
[ ] Full audit completes and diagnostics are acceptable
[ ] CUDA is available on the training machine
[ ] Fold-0 random smoke succeeds
[ ] Fold-1 and fold-2 random smoke succeed
[ ] Three random production folds finish
[ ] Random OOF evaluation is saved
[ ] SSL pretraining finishes
[ ] Three SSL Stage-1 folds finish
[ ] Per-fold Stage-1 selection is generated from inner AUC only
[ ] Three Stage-2 folds finish
[ ] Stage-2 OOF evaluation is saved
[ ] Final method is frozen
[ ] Final inference produces submission.csv
```

---

# Part XIV — Things not to do

- Do not use the four external fixture images as leaderboard/scientific validation data.
- Do not select random vs SSL from outer-fold AUC; use `select-stage1`, which uses fold-local inner AUC only.
- Do not tune submission TTA after inspecting outer OOF.
- Do not use external pretrained weights when running the default competition-only configuration.
- Do not launch multiple folds simultaneously on the same GPU.
- Do not use DDP or `torchrun` for this pipeline.
- Do not ignore DICOM preflight/audit failures.
- Do not start a full production run before fold-0 smoke succeeds.
- Do not commit the Kaggle image dataset to GitHub.
- Do not commit large model checkpoints or generated `runs/` directories unless there is a deliberate reason to version them.
- Do not assume `CUDA available: False` is harmless for production training; it is harmless only for CPU inspection/audit/testing steps.

---

# Expected production output layout

A successful full run will approximately produce:

```text
runs/
├── audit/
│   └── audit.json
├── validation/
│   ├── fold0.csv
│   ├── fold1.csv
│   └── fold2.csv
├── stage1_random/
│   ├── fold0/
│   ├── fold1/
│   └── fold2/
├── ssl/
│   ├── ssl_encoder.pt
│   └── history.json
├── stage1_ssl/
│   ├── fold0/
│   ├── fold1/
│   └── fold2/
├── stage1_selection.json
└── stage2/
    ├── fold0/
    ├── fold1/
    └── fold2/

submission.csv
```

Each trained fold contains its checkpoint, OOF predictions, history, runtime record, bootstrap results, and training diagnostics. Stage 1 also writes weak OOF teacher predictions; Stage 2 writes `stage2_supervision.json` instead.

---

## Recommended execution order in one view

```text
SETUP
  conda -> pip install -e . -> pytest -> external preflight

REAL DATA
  DATA_ROOT -> inspect -> validation manifests -> preflight -> audit

STAGE 1 RANDOM
  fold0 smoke -> folds1/2 smoke -> folds0/1/2 production -> evaluate

SSL CANDIDATE
  pretrain -> SSL config -> folds0/1/2 production

NESTED SELECTION
  select-stage1 (inner_macro_auc only)

STAGE 2
  Stage2 config -> folds0/1/2 -> evaluate/compare

FINAL
  freeze final method -> infer 3 checkpoints -> submission.csv
```

When diagnosing a failure, fix the **first failing gate** rather than continuing to later stages.
