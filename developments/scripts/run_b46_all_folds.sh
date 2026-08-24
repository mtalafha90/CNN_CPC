#!/usr/bin/env bash
set -euo pipefail

# Sequential fixed-endpoint runner for B46.
# Required environment variables:
#   DATA_ROOT LABELS_ROOT SERIES_POLICY BASE_CHECKPOINT B46_ROOT B46_MANIFEST
# Optional:
#   PYTHON_BIN (default: python)
#
# Governance:
# - verifies the frozen manifest SHA before every fold;
# - refuses to overwrite an already completed fold checkpoint;
# - trains folds sequentially on one GPU;
# - preserves Python failure through tee via pipefail/PIPESTATUS;
# - does not inspect metrics or choose checkpoints between folds.

EXPECTED_MANIFEST_SHA="054c4ce9ab808af714cd4b86f159ef02a2b7e67de0c80e5c930d29fa5fb22e03"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="config/b46_gold_anchored_crossfit.yaml"

required_vars=(
  DATA_ROOT
  LABELS_ROOT
  SERIES_POLICY
  BASE_CHECKPOINT
  B46_ROOT
  B46_MANIFEST
)

for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: required environment variable $name is not set" >&2
    exit 2
  fi
done

if [[ ! -f "$B46_MANIFEST" ]]; then
  echo "ERROR: B46 manifest does not exist: $B46_MANIFEST" >&2
  exit 2
fi

manifest_sha() {
  sha256sum "$B46_MANIFEST" | awk '{print $1}'
}

verify_manifest() {
  local actual
  actual="$(manifest_sha)"
  if [[ "$actual" != "$EXPECTED_MANIFEST_SHA" ]]; then
    echo "ERROR: B46 manifest SHA mismatch" >&2
    echo "expected: $EXPECTED_MANIFEST_SHA" >&2
    echo "actual:   $actual" >&2
    exit 3
  fi
}

verify_manifest
mkdir -p "$B46_ROOT"

printf 'B46 frozen manifest SHA: %s\n' "$(manifest_sha)"
printf 'B46 run root: %s\n' "$B46_ROOT"
printf 'Python: %s\n' "$PYTHON_BIN"
printf '\n'

for FOLD in 0 1 2 3 4; do
  verify_manifest

  FOLD_ROOT="$B46_ROOT/fold_${FOLD}"
  CHECKPOINT="$FOLD_ROOT/b46_fold${FOLD}_model.pt"
  LOG="$FOLD_ROOT/training.log"

  mkdir -p "$FOLD_ROOT"

  if [[ -e "$CHECKPOINT" ]]; then
    echo "ERROR: refusing to overwrite completed B46 fold $FOLD checkpoint:" >&2
    echo "  $CHECKPOINT" >&2
    exit 4
  fi

  echo
  echo "============================================================"
  echo "B46 FOLD $FOLD FIXED-E2 TRAINING"
  echo "============================================================"
  echo "manifest_sha=$(manifest_sha)"
  echo "checkpoint=$CHECKPOINT"
  echo "log=$LOG"
  echo

  set +e
  "$PYTHON_BIN" -m rsna_knee.b46_gold_crossfit_training \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --labels-root "$LABELS_ROOT" \
    --series-policy "$SERIES_POLICY" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --fold-manifest "$B46_MANIFEST" \
    --fold "$FOLD" \
    --out-root "$B46_ROOT" \
    2>&1 | tee "$LOG"
  python_status=${PIPESTATUS[0]}
  set -e

  if [[ "$python_status" -ne 0 ]]; then
    echo "ERROR: B46 fold $FOLD training failed with status $python_status" >&2
    exit "$python_status"
  fi

  if [[ ! -s "$CHECKPOINT" ]]; then
    echo "ERROR: B46 fold $FOLD exited successfully but checkpoint is missing/empty" >&2
    exit 5
  fi

  verify_manifest

  echo
  echo "B46 FOLD $FOLD FIXED-E2 COMPLETE"
  sha256sum "$CHECKPOINT"
done

echo
echo "ALL B46 FIXED-E2 FOLDS: COMPLETE"
