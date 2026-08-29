#!/usr/bin/env bash
set -euo pipefail

# Sequential runner for the B50 matched pair.
#
# Required environment variables:
#   DATA_ROOT LABELS_ROOT SERIES_POLICY BASE_CHECKPOINT B50_SELECTION_GATE B50_ROOT
# Optional:
#   PYTHON_BIN (default: python)
#
# Both arms preflight first, then both train, control before candidate. The
# preflights are cheap and they check the one thing that would silently ruin the
# experiment: that the control has no hierarchy gradient and the candidate does.
# Discovering a wiring fault after two epochs of GPU time would look exactly
# like a null result, which is how B48 and B49 read until their gates were
# measured.
#
# Resumable, like the B46 runner: an arm with a complete checkpoint is skipped
# rather than retrained, so an interrupted run continues where it stopped.
# Nothing resumes inside an arm -- each is a fixed two-epoch run from the common
# base checkpoint, so an interrupted arm simply starts again.

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-config/b50_adapted_hierarchy.yaml}"
ARMS=(frozen_hierarchy_control adapted_hierarchy_candidate)

for name in DATA_ROOT LABELS_ROOT SERIES_POLICY BASE_CHECKPOINT B50_SELECTION_GATE B50_ROOT; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: required environment variable $name is not set" >&2
    exit 2
  fi
done

if [[ ! -f "$B50_SELECTION_GATE/b50_selection_split.json" ]]; then
  echo "ERROR: B50 fresh selection gate not found in: $B50_SELECTION_GATE" >&2
  echo "Build it once with developments/scripts/prepare_b50_ordered_slice_gate.sh" >&2
  exit 2
fi

run_arm() {
  local arm="$1" mode="$2"
  local arm_root="$B50_ROOT/$arm"
  mkdir -p "$arm_root"
  local extra=()
  [[ "$mode" == "preflight" ]] && extra+=(--preflight-only)

  "$PYTHON_BIN" -m rsna_knee.b50_adapted_hierarchy_training \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --labels-root "$LABELS_ROOT" \
    --series-policy "$SERIES_POLICY" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --selection-gate "$B50_SELECTION_GATE" \
    --arm "$arm" \
    --out-root "$arm_root" \
    "${extra[@]}"
}

printf 'B50 run root: %s\n' "$B50_ROOT"
printf 'selection gate: %s\n' "$B50_SELECTION_GATE"
printf 'config:       %s\n\n' "$CONFIG"

echo "============================================================"
echo "B50 PREFLIGHT — both arms"
echo "============================================================"
for ARM in "${ARMS[@]}"; do
  echo
  echo "--- preflight $ARM ---"
  run_arm "$ARM" preflight
done

trained_now=()
already_complete=()

for ARM in "${ARMS[@]}"; do
  ARM_ROOT="$B50_ROOT/$ARM"
  CHECKPOINT="$ARM_ROOT/b50_${ARM}_model.pt"
  LOG="$ARM_ROOT/training.log"

  if [[ -s "$CHECKPOINT" ]]; then
    echo
    echo "B50 ARM $ARM ALREADY COMPLETE -- skipping"
    sha256sum "$CHECKPOINT"
    already_complete+=("$ARM")
    continue
  fi
  if [[ -e "$CHECKPOINT" ]]; then
    echo "ERROR: B50 $ARM checkpoint exists but is empty:" >&2
    echo "  $CHECKPOINT" >&2
    echo "A save was interrupted. Inspect it, then remove it to retrain this arm." >&2
    exit 4
  fi

  echo
  echo "============================================================"
  echo "B50 ARM $ARM FIXED-E2 TRAINING"
  echo "============================================================"
  { echo; echo "=== B50 $ARM attempt started $(date -Is) ==="; } >> "$LOG"

  set +e
  run_arm "$ARM" train 2>&1 | tee -a "$LOG"
  status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "ERROR: B50 arm $ARM training failed with status $status" >&2
    exit "$status"
  fi
  if [[ ! -s "$CHECKPOINT" ]]; then
    echo "ERROR: B50 arm $ARM exited successfully but checkpoint is missing/empty" >&2
    exit 5
  fi

  echo
  echo "B50 ARM $ARM COMPLETE"
  sha256sum "$CHECKPOINT"
  trained_now+=("$ARM")
done

echo
echo "ALL B50 ARMS: COMPLETE"
echo "trained on this invocation: ${trained_now[*]:-none}"
echo "already complete beforehand: ${already_complete[*]:-none}"
