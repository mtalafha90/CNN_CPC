#!/usr/bin/env bash
# Package everything B52 needs into one self-contained directory.
#
# The target machine is assumed to have the competition data folder and a Python
# environment, and nothing else from this repository. Everything else B52 reads
# is copied here: the code, the config, the Phase-9 base checkpoint, the merged
# label export, the frozen series policy and the B50 selection gate.
#
# The whole rsna_knee package is copied rather than the handful of B52 files.
# B52 imports fifteen sibling modules and those import more; picking a subset by
# hand is how a bundle arrives on another machine and fails on its first import
# two hours into a run.
#
# Every copied artefact is fingerprinted into MANIFEST.sha256, and verify.sh on
# the far side re-checks them. A checkpoint corrupted by a partial copy would
# otherwise fail as a strange training result rather than as a broken file.
#
#   bash developments/scripts/package_b52_standalone.sh /path/to/b52_standalone
set -euo pipefail

REPO="${REPO:-/media/talafha/Disk_1/CNN_CPC}"
OUT="${1:-$REPO/b52_standalone}"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-$REPO/runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt}"
LABELS_ROOT="${LABELS_ROOT:-$REPO/runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all}"
SERIES_POLICY="${SERIES_POLICY:-$REPO/runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json}"
GATE_ROOT="${GATE_ROOT:-$REPO/runs/083_Experiment_B50_selection_gate/b50_ordered_slice_selection_split}"
CONFIG="${CONFIG:-$REPO/config/b42_constant_area_aspect_sparse.yaml}"

say() { printf '%s\n' "$*"; }
need() {
  if [ ! -e "$1" ]; then
    say "MISSING: $1"
    say "Set the matching variable and re-run, for example:"
    say "  BASE_CHECKPOINT=/path/to/model.pt bash $0 $OUT"
    exit 1
  fi
}

say "== checking sources =="
need "$REPO/developments/src/rsna_knee"
need "$CONFIG"
need "$BASE_CHECKPOINT"
need "$SERIES_POLICY"
for name in training_targets.csv policy.json audit.json; do
  need "$LABELS_ROOT/$name"
done
for name in b50_selection_split.json b50_selection_split_by_study.csv; do
  need "$GATE_ROOT/$name"
done

if [ -e "$OUT" ]; then
  say "refusing to overwrite $OUT -- remove it first or pass another path"
  exit 1
fi

say "== copying into $OUT =="
mkdir -p "$OUT"/{src,config,models,labels,policy,gate,runs}

cp -a "$REPO/developments/src/rsna_knee" "$OUT/src/"
find "$OUT/src" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

cp "$CONFIG" "$OUT/config/"
cp "$BASE_CHECKPOINT" "$OUT/models/phase9_llm_fill_base.pt"
for name in training_targets.csv policy.json audit.json; do
  cp "$LABELS_ROOT/$name" "$OUT/labels/"
done
cp "$SERIES_POLICY" "$OUT/policy/series_policy.json"
for name in b50_selection_split.json b50_selection_split_by_study.csv; do
  cp "$GATE_ROOT/$name" "$OUT/gate/"
done

say "== fingerprinting =="
( cd "$OUT" && find config models labels policy gate -type f -print0 \
    | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )

cat > "$OUT/verify.sh" <<'VERIFY'
#!/usr/bin/env bash
# Confirm every artefact survived the copy before spending a GPU on it.
set -euo pipefail
cd "$(dirname "$0")"
sha256sum -c MANIFEST.sha256
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "src")
import torch
payload = torch.load("models/phase9_llm_fill_base.pt", map_location="cpu", weights_only=False)
print("base checkpoint loads; keys:", len(payload))
from rsna_knee.b52_competition_training import train_b52  # noqa: F401
print("rsna_knee imports")
PY
echo "bundle OK"
VERIFY
chmod +x "$OUT/verify.sh"

cat > "$OUT/run.sh" <<'RUN'
#!/usr/bin/env bash
# Train B52 from this bundle. The only thing not in here is the data folder.
#
#   DATA_ROOT=/path/to/rsna-knee-abnormality-detection ./run.sh [epochs] [extra flags]
#
# The full-data run this bundle is meant for:
#   DATA_ROOT=... ./run.sh 6 --all-data
#
# --no-gradient-checkpointing is faster but needs about 15 GiB. It OOMs on a
# 16 GiB card at this geometry. Try it only on a larger GPU, and preflight it.
set -euo pipefail
cd "$(dirname "$0")"

: "${DATA_ROOT:?set DATA_ROOT to the folder holding train.csv and train_series.csv}"
EPOCHS="${1:-6}"

test -f "$DATA_ROOT/train.csv" || { echo "no train.csv under $DATA_ROOT"; exit 1; }
test -f "$DATA_ROOT/train_series.csv" || { echo "no train_series.csv under $DATA_ROOT"; exit 1; }

PYTHONPATH=src python -m rsna_knee.b52_competition_training \
  --config config/b42_constant_area_aspect_sparse.yaml \
  --data-root "$DATA_ROOT" \
  --labels-root labels \
  --series-policy policy/series_policy.json \
  --base-checkpoint models/phase9_llm_fill_base.pt \
  --domain-split gate \
  --out-root runs/b52 \
  --epochs "$EPOCHS" \
  "${@:2}"
RUN
chmod +x "$OUT/run.sh"

cat > "$OUT/README.md" <<'DOC'
# B52 standalone bundle

Everything B52 needs except the competition data folder.

## On the new machine

```bash
pip install "numpy>=1.26" "pandas>=2.0" "scikit-learn>=1.3" "pydicom>=2.4" \
            "PyYAML>=6.0" "torch>=2.2" "torchvision>=0.17"
```

**On a Blackwell card (RTX 50-series), install torch from the CUDA 12.8 index
instead.** Blackwell is compute capability 12.0, and a wheel built for an older
CUDA carries no kernels for it -- the failure is `no kernel image is available
for execution on the device`, at the first forward pass rather than at import.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

The capability must print for your card before you spend a night on a run.

## Check the bundle, then train

```bash
./verify.sh                      # checksums, then a real import and load

DATA_ROOT=/path/to/rsna-knee-abnormality-detection ./run.sh 6 --all-data
```

`--all-data` trains on every split except the unseen-scanner validation
surface: 3,801 studies rather than the gate's 1,447, scored on the same 548.

Six epochs, not more: on 1,447 studies the peak was epoch 5 of 6 and epoch 6
fell back, so the schedule shape is worth reproducing rather than extending.

**Do not add `--no-gradient-checkpointing` without preflighting it.** It is
identical maths and roughly 30% faster, but it retains every encoder
activation and needs about 15 GiB at this geometry -- it OOMs on a 16 GiB
card. Worth trying only on a larger GPU.

Run `verify.sh` first. It re-checks every fingerprint and actually loads the
base checkpoint and imports the package, so a truncated copy fails in seconds
rather than two hours into training.

## Preflight before committing a long run

```bash
DATA_ROOT=/path/to/data ./run.sh 6 --all-data --preflight-only
```

One forward and backward pass. It fails loudly if no gradient reaches the
encoder, and prints peak memory.

## Feeding a fast GPU

`num_workers: 0` in the config means DICOM decoding and all nine augmentations
run on the main thread, between GPU steps. That was chosen for operational
safety in Kaggle submission, where a worker crash loses the run.

On a fast card the GPU then waits for the CPU. If an epoch is slower than the
card suggests it should be, raise it in `config/b42_constant_area_aspect_sparse.yaml`:

```yaml
num_workers: 6
prefetch_factor: 2
```

This changes no maths -- the loader is seeded through `worker_init_fn`, and the
B42 geometry contract does not cover worker count. Preflight after changing it:
each worker is a separate process under `spawn` and costs host RAM.

## What is in here

```text
src/rsna_knee/     the package (all of it -- B52 imports fifteen siblings)
config/            the frozen B42 geometry contract
models/            the Phase-9 llm_fill base checkpoint
labels/            the merged B6+LLM export B52 trains on
policy/            the frozen B12/B13 series policy
gate/              the B50 scanner-grouped split
runs/              output lands here
MANIFEST.sha256    fingerprints of every artefact above
```

## Notes

The trainer writes a checkpoint whenever an epoch beats the best validation
macro AUC so far, and each checkpoint carries the full history inside it. You
can stop after any epoch with Ctrl-C and keep everything up to that point.

The data folder is not copied. It is large, it is unchanged, and the run reads
`train.csv`, `train_series.csv` and the DICOM directories directly from
`DATA_ROOT`.

The gate's payload records the SHA-256 of the `train.csv` it was built from, and
the trainer refuses to start if the data folder's `train.csv` does not match. If
that check fails, the new machine has a different copy of the data.
DOC

BYTES=$(du -sh "$OUT" | cut -f1)
say ""
say "== done: $OUT ($BYTES) =="
say "verify:  cd $OUT && ./verify.sh"
say "zip:     cd $(dirname "$OUT") && zip -r $(basename "$OUT").zip $(basename "$OUT")"
