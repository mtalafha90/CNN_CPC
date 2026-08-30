#!/usr/bin/env bash
set -euo pipefail

# B51: the adapted hierarchy on all 4,349 report-only studies.
#
# Required environment variables:
#   DATA_ROOT LABELS_ROOT SERIES_POLICY BASE_CHECKPOINT B51_ROOT
# Optional:
#   PYTHON_BIN (default: python)
#
# One run, no arms: B50 already ran the controlled comparison, and B51's control
# is B42's existing hidden score.
#
# Roughly 8.5 hours, scaled from B50's recorded 85 min per epoch on 1,447
# studies. That is longer than a working session, so the checkpoint is written
# only at the end of epoch 2 and a completed run is never overwritten. An
# interrupted run restarts from the base checkpoint -- there is no partial
# resume inside a fixed-endpoint run, by design.

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-config/b51_full_population_adapted_hierarchy.yaml}"

for name in DATA_ROOT LABELS_ROOT SERIES_POLICY BASE_CHECKPOINT B51_ROOT; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: required environment variable $name is not set" >&2
    exit 2
  fi
done

CHECKPOINT="$B51_ROOT/b51_full_population_adapted_hierarchy_model.pt"
if [[ -s "$CHECKPOINT" ]]; then
  echo "B51 ALREADY COMPLETE -- nothing to do"
  sha256sum "$CHECKPOINT"
  exit 0
fi
if [[ -e "$CHECKPOINT" ]]; then
  echo "ERROR: B51 checkpoint exists but is empty: $CHECKPOINT" >&2
  echo "A save was interrupted. Inspect it, then remove it to retrain." >&2
  exit 4
fi

mkdir -p "$B51_ROOT"
LOG="$B51_ROOT/training.log"

run() {
  "$PYTHON_BIN" -m rsna_knee.b51_full_population_training \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --labels-root "$LABELS_ROOT" \
    --series-policy "$SERIES_POLICY" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --out-root "$B51_ROOT" \
    "$@"
}

echo "============================================================"
echo "B51 PREFLIGHT"
echo "============================================================"
run --preflight-only

echo
echo "============================================================"
echo "B51 FIXED-E2 TRAINING on all 4,349 studies (~8.5 h)"
echo "============================================================"
{ echo; echo "=== B51 attempt started $(date -Is) ==="; } >> "$LOG"

set +e
run 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
if [[ "$status" -ne 0 ]]; then
  echo "ERROR: B51 training failed with status $status" >&2
  exit "$status"
fi
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "ERROR: B51 exited successfully but the checkpoint is missing/empty" >&2
  exit 5
fi

echo
echo "B51 COMPLETE"
sha256sum "$CHECKPOINT"
echo
echo "Next: convert for the proven B42 submission path"
echo "  python -m rsna_knee.b51_checkpoint_to_b42_format \\"
echo "    --source $CHECKPOINT \\"
echo "    --destination $B51_ROOT/b51_as_b42_for_submission.pt"
