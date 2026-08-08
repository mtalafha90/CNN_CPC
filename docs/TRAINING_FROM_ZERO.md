# CNN_CPC Training From Zero

This is the clean end-to-end guide for setting up and training `CNN_CPC` on the real RSNA Knee Abnormality Detection data from a fresh Linux machine.

The current repository has already been verified on one real-data workstation, but this document is intentionally machine-independent.

## Workflow

```text
clone/update repo
-> create Conda environment
-> install and test
-> verify GPU
-> place competition data
-> create local config
-> inspect CSVs
-> export nested validation manifests
-> train/test DICOM preflight
-> full selected-series audit
-> verify report supervision
-> fold-0 smoke
-> Stage-1 random folds 0/1/2
-> optional competition-data SSL
-> per-fold Stage-1 candidate selection from inner AUC only
-> Stage-2 folds 0/1/2
-> OOF evaluation
-> freeze final method
-> three-fold inference
```

Do not skip preflight, audit, or smoke on a new machine.

## 1. Clone or update the repository

Choose a repository location:

```bash
export REPO="/path/to/CNN_CPC"
```

Clone once:

```bash
cd "$(dirname "$REPO")"
git clone https://github.com/mtalafha90/CNN_CPC.git
cd "$REPO"
```

Or update an existing checkout:

```bash
cd "$REPO"
git checkout main
git pull --ff-only origin main
```

Check:

```bash
pwd
git branch --show-current
git status
git log -5 --oneline
```

## 2. Create the Conda environment

```bash
conda create -n rsna-knee python=3.12 -y
conda activate rsna-knee
```

If another virtual environment is active:

```bash
deactivate 2>/dev/null || true
conda activate rsna-knee
```

Check:

```bash
which python
python --version
```

## 3. Install the project

```bash
cd "$REPO"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install pytest pillow kaggle
python -m pip check
```

Verify import and CLI:

```bash
python - <<'PY'
import rsna_knee
print(rsna_knee.__file__)
PY

python -m rsna_knee.cli --help
```

## 4. Run software tests

```bash
pytest -q
python -m compileall -q src tests kaggle scripts
```

Also run the focused methodology tests when modifying supervision/sampling:

```bash
pytest -q tests/test_oa_report_labels.py
pytest -q tests/test_sampling_pairing.py
pytest -q tests/test_methodology.py
```

## 5. Verify GPU support

```bash
nvidia-smi
```

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
```

Production training requires `CUDA available: True`.

Use one GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Do not launch with `torchrun`.

## 6. Verify the committed external fixture

```bash
pytest -q tests/test_external_fixture.py
```

Strict preflight:

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

This fixture tests software plumbing only. Never use it as a competition validation set.

## 7. Place the competition data

After accepting the competition rules, download through Kaggle or the website.

Set:

```bash
export DATA_ROOT="/path/to/rsna-knee-abnormality-detection"
```

The root should contain at least:

```text
train.csv
train_series.csv
test.csv
test_series.csv
sample_submission.csv
train_series/ or the supported training image tree
test_series/ or the supported test image tree
```

Check:

```bash
ls -lh "$DATA_ROOT"
find "$DATA_ROOT" -maxdepth 2 -type d | head -30
```

Do not commit the competition images or machine-local data paths.

## 8. Create `configs/train_local.yaml`

```bash
cd "$REPO"
cp configs/train.yaml configs/train_local.yaml
```

Patch machine-local fields:

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml

p = Path("configs/train_local.yaml")
c = yaml.safe_load(p.read_text())
c["data_root"] = os.environ["DATA_ROOT"]
c["output_dir"] = "runs/stage1_random"
c["ssl_output_dir"] = "runs/ssl"
c["competition_mode"] = True
c["requested_gpus"] = 1
c["runtime_budget_hours"] = 8.5
c["pretrained"] = False
c["allow_external_pretrained"] = False
c["ssl_encoder_checkpoint"] = None
c["ssl_checkpoint_source"] = None
c["cotrain_stage1_root"] = None
c["cotrain_stage1_candidates"] = None
c["expected_checkpoint_stage"] = None
p.write_text(yaml.safe_dump(c, sort_keys=False))
PY
```

Verify TTA parity:

```bash
python - <<'PY'
import yaml
from pathlib import Path
c = yaml.safe_load(Path("configs/train_local.yaml").read_text())
print(c["tta_center_offsets"])
print(c["validation_tta_offsets"])
print("match:", c["tta_center_offsets"] == c["validation_tta_offsets"])
PY
```

Expected:

```text
[-1, 0, 1]
[-1, 0, 1]
match: True
```

## 9. Inspect official CSVs

```bash
python -m rsna_knee.cli inspect --data-root "$DATA_ROOT"
```

The verified 2026-08-08 data release produced:

```text
studies=4407
gold=58
unlabeled=4349
reports_present=4407
series=24371
```

If your downloaded release differs, record the difference before training rather than forcing these counts.

## 10. Export nested validation manifests

```bash
mkdir -p runs/validation
for f in 0 1 2; do
  python -m rsna_knee.cli validation-manifest \
    --config configs/train_local.yaml \
    --fold "$f" \
    --out "runs/validation/fold${f}.csv"
done
```

For the verified release, role sizes were:

```text
fold 0: 20 gold_train, 20 inner_selection, 18 outer_validation
fold 1: 18 gold_train, 20 inner_selection, 20 outer_validation
fold 2: 20 gold_train, 18 inner_selection, 20 outer_validation
```

## 11. Run train preflight

```bash
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

Verified reference result:

```text
121 selected streams
121 decoded streams
4045 candidate files
0 file failures
```

## 12. Run test preflight

If the test tree is present:

```bash
python -m rsna_knee.cli preflight \
  --data-root "$DATA_ROOT" \
  --split test \
  --sample-size 24 \
  --max-decode-failure-rate 0.05 \
  --max-file-decode-failure-rate 0.05 \
  --out runs/preflight_test.json
```

If fewer than 24 test studies exist locally, the command naturally checks the available set.

The verified release contained three local test studies and decoded all 533 candidate files successfully.

## 13. Run the full audit

```bash
python -m rsna_knee.cli audit \
  --config configs/train_local.yaml \
  --out-dir runs/audit
```

Inspect:

```bash
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("runs/audit/audit.json").read_text())
for key in [
    "decode_audit",
    "selected_stream_counts",
    "missing_stream_counts",
    "teacher_confidence_counts",
]:
    print("\n=====", key, "=====")
    print(json.dumps(p[key], indent=2))
PY
```

Verified full audit reference:

```text
21,886 selected series checked
21,886 decoded
0 failed series
732,556 candidate DICOM files
2 failed DICOM files
2 series with one partial file failure each
0 series above the 20% per-series gate
```

Do not repeat the full 700k-file audit after every documentation/config change. Repeat it when DICOM decoding, routing or the underlying data change.

## 14. Check report supervision

The OA parser should produce nonzero supervision. Quick audit check:

```bash
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("runs/audit/audit.json").read_text())
for t in ["Medial OA", "Lateral OA", "PF OA"]:
    print(t, p["teacher_state_counts"][t])
PY
```

The verified parser produced:

```text
Medial OA  positive=492 negated=339 unmentioned=3576
Lateral OA positive=409 negated=387 unmentioned=3611
PF OA      positive=695 negated=379 unmentioned=3333
```

## 15. Run fold-0 smoke

```bash
python -m rsna_knee.cli train \
  --config configs/train_local.yaml \
  --fold 0 \
  --smoke
```

A successful smoke should generate the full Stage-1 artifact set under:

```text
runs/stage1_random/smoke/fold0/
```

The current paired-sampler reference smoke produced nonzero ranking pairs for all 12 targets and best inner macro-AUC `0.55135`. Treat this only as a software/GPU validation result.

## 16. Verify ranking utilization

```bash
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("runs/stage1_random/smoke/fold0/training_diagnostics.json").read_text())
for phase in ["selection", "retrain"]:
    counts = p[phase]["rank_pairs"]
    print(phase, "total=", sum(counts.values()))
    print(counts)
PY
```

The verified paired-sampler smoke produced 63 selection pairs and 61 retraining pairs.

## 17. Run Stage-1 random production folds

```bash
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 0
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 1
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 2
```

Run sequentially on one GPU.

Do not change the method after seeing fold-0 outer AUC. Inspect fold 0 only for computational/runtime correctness, then run folds 1 and 2 unchanged.

## 18. Evaluate Stage-1 random OOF

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

## 19. Optional competition-data SSL

```bash
python -m rsna_knee.cli pretrain --config configs/train_local.yaml
```

Create an SSL Stage-1 config pointing to `runs/ssl/ssl_encoder.pt` with:

```yaml
ssl_checkpoint_source: competition_training_data
output_dir: runs/stage1_ssl
```

Train all three folds using the same validation policy.

## 20. Select Stage-1 candidate per outer fold

```bash
python -m rsna_knee.cli select-stage1 \
  --candidate-root "$(pwd)/runs/stage1_random" \
  --candidate-root "$(pwd)/runs/stage1_ssl" \
  --n-folds 3 \
  --out runs/stage1_selection.json
```

The only supported criterion is fold-local inner AUC.

## 21. Stage 2

Create a Stage-2 config with the candidate roots and run folds 0, 1 and 2 sequentially.

Stage-2 Phase A is report-only. Phase B starts fresh and consumes only safe fold-local image teachers.

Inspect `stage2_supervision.json` to verify how much image-only supervision was actually added.

## 22. Final inference

After the final stage is frozen, create `configs/final_infer.yaml` and set:

```yaml
expected_checkpoint_stage: stage1
```

or `stage2` as appropriate.

Then:

```bash
python -m rsna_knee.cli infer \
  --config configs/final_infer.yaml \
  --checkpoints \
    runs/<final_stage>/fold0/best.pt \
    runs/<final_stage>/fold1/best.pt \
    runs/<final_stage>/fold2/best.pt \
  --out submission.csv
```

## 23. Reporting discipline

Keep these terms separate:

- audit result;
- smoke result;
- production fold result;
- combined three-fold OOF result;
- model-selection CV result;
- leaderboard result.

Do not fill manuscript production-result placeholders from smoke runs.