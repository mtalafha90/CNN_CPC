#!/usr/bin/env bash
set -euo pipefail

# Run one predeclared matched B48 seed pair only after B46's five fixed folds
# have finished.  B48 does NOT consume those checkpoint weights or gold labels;
# the B46-completion guard prevents a source-tree change from splitting the
# in-flight B46 protocol across revisions.
#
# Required environment variables:
#   DATA_ROOT LABELS_ROOT SERIES_POLICY BASE_CHECKPOINT B46_ROOT
#   DOMAIN_SPLIT_ROOT B48_ROOT
# Optional:
#   B48_SEED (one of 2026, 2037, 2048; default 2026)
#   PYTHON_BIN (default: python)

PYTHON_BIN="${PYTHON_BIN:-python}"
B48_SEED="${B48_SEED:-2026}"
CONFIG="config/b48_global_conditioned_sparse.yaml"
ARMS=(static_prior_control post_cross_attention_candidate)
ALLOWED_SEEDS=(2026 2037 2048)

required_vars=(
  DATA_ROOT
  LABELS_ROOT
  SERIES_POLICY
  BASE_CHECKPOINT
  B46_ROOT
  DOMAIN_SPLIT_ROOT
  B48_ROOT
)

for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: required environment variable $name is not set" >&2
    exit 2
  fi
done

seed_allowed=false
for allowed in "${ALLOWED_SEEDS[@]}"; do
  if [[ "$B48_SEED" == "$allowed" ]]; then
    seed_allowed=true
  fi
done
if [[ "$seed_allowed" != true ]]; then
  echo "ERROR: B48_SEED must be one of: ${ALLOWED_SEEDS[*]}" >&2
  exit 2
fi

DOMAIN_SPLIT="$DOMAIN_SPLIT_ROOT/domain_split.json"
DOMAIN_ROWS="$DOMAIN_SPLIT_ROOT/domain_split_by_study.csv"
DOMAIN_SHA="$DOMAIN_SPLIT_ROOT/domain_split.sha256"
for path in "$DOMAIN_SPLIT" "$DOMAIN_ROWS" "$DOMAIN_SHA"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: B48 requires frozen domain-split artifact: $path" >&2
    exit 2
  fi
done

actual_domain_sha="$(sha256sum "$DOMAIN_SPLIT" | awk '{print $1}')"
recorded_domain_sha="$(tr -d '[:space:]' < "$DOMAIN_SHA")"
if [[ "$actual_domain_sha" != "$recorded_domain_sha" ]]; then
  echo "ERROR: domain_split.sha256 does not match domain_split.json" >&2
  echo "expected: $recorded_domain_sha" >&2
  echo "actual:   $actual_domain_sha" >&2
  exit 3
fi

# B46's runner does not pin a source revision.  Do not let B48 change the live
# tree while even one B46 fold remains incomplete.
for fold in 0 1 2 3 4; do
  checkpoint="$B46_ROOT/fold_${fold}/b46_fold${fold}_model.pt"
  if [[ ! -s "$checkpoint" ]]; then
    echo "ERROR: B46 fold $fold is not complete: $checkpoint" >&2
    echo "Finish all five B46 fixed-E2 checkpoints before B48." >&2
    exit 4
  fi
done

pair_root="$B48_ROOT/seed_${B48_SEED}"
all_complete=true
for arm in "${ARMS[@]}"; do
  checkpoint="$pair_root/$arm/b48_${arm}_model.pt"
  if [[ ! -s "$checkpoint" ]]; then
    all_complete=false
  fi
done
if [[ "$all_complete" == true ]]; then
  echo "B48 seed $B48_SEED pair already complete -- no checkpoints overwritten"
  exit 0
fi

mkdir -p "$pair_root"
printf 'B48 seed: %s\n' "$B48_SEED"
printf 'B48 domain split SHA: %s\n' "$actual_domain_sha"
printf 'B46 completion root: %s\n' "$B46_ROOT"
printf 'B48 pair root: %s\n\n' "$pair_root"

# Both preflights are required before either optimizer starts.  They exercise
# the real rectangular DICOM path, B42's worst-case sequential batch, and the
# B48 detached-query/gate gradient contract without an optimizer step.
for arm in "${ARMS[@]}"; do
  echo "============================================================"
  echo "B48 $arm S${B48_SEED} PREFLIGHT"
  echo "============================================================"
  "$PYTHON_BIN" -m rsna_knee.b48_global_conditioned_sparse_training \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --labels-root "$LABELS_ROOT" \
    --series-policy "$SERIES_POLICY" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --domain-split "$DOMAIN_SPLIT" \
    --arm "$arm" \
    --seed "$B48_SEED" \
    --out-root "$pair_root/$arm" \
    --preflight-only
done

for arm in "${ARMS[@]}"; do
  arm_root="$pair_root/$arm"
  checkpoint="$arm_root/b48_${arm}_model.pt"
  log="$arm_root/training.log"
  mkdir -p "$arm_root"
  if [[ -s "$checkpoint" ]]; then
    echo "B48 $arm S${B48_SEED} already complete -- skipping"
    sha256sum "$checkpoint"
    continue
  fi
  if [[ -e "$checkpoint" ]]; then
    echo "ERROR: B48 checkpoint exists but is empty: $checkpoint" >&2
    echo "Inspect it; removing an interrupted checkpoint is an operator decision." >&2
    exit 5
  fi

  echo "============================================================"
  echo "B48 $arm S${B48_SEED} FIXED-E2 TRAINING"
  echo "============================================================"
  {
    echo
    echo "=== B48 $arm seed $B48_SEED attempt started $(date -Is) ==="
  } >> "$log"

  set +e
  "$PYTHON_BIN" -m rsna_knee.b48_global_conditioned_sparse_training \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --labels-root "$LABELS_ROOT" \
    --series-policy "$SERIES_POLICY" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --domain-split "$DOMAIN_SPLIT" \
    --arm "$arm" \
    --seed "$B48_SEED" \
    --out-root "$arm_root" \
    2>&1 | tee -a "$log"
  python_status=${PIPESTATUS[0]}
  set -e
  if [[ "$python_status" -ne 0 ]]; then
    echo "ERROR: B48 $arm failed with status $python_status" >&2
    exit "$python_status"
  fi
  if [[ ! -s "$checkpoint" ]]; then
    echo "ERROR: B48 $arm exited successfully but checkpoint is missing/empty" >&2
    exit 6
  fi
  sha256sum "$checkpoint"
done

echo
echo "B48 seed $B48_SEED matched pair: COMPLETE"
echo "Run the paired evaluator once, after both arms are complete."
