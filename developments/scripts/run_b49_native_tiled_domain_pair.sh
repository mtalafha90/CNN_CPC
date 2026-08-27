#!/usr/bin/env bash
set -euo pipefail

# Run one predeclared B49 full-FOV native-tiled matched pair only after the B48
# seed pair is complete.  B49 reuses B48's frozen scanner-domain split but never
# loads B48 model weights; B48 completion is a sequencing/provenance guard.
#
# Required environment variables:
#   DATA_ROOT LABELS_ROOT SERIES_POLICY BASE_CHECKPOINT DOMAIN_SPLIT_ROOT
#   B48_ROOT B49_ROOT
# Optional:
#   B49_SEED (2026, 2037, or 2048; default 2026)
#   PYTHON_BIN (default: python)

PYTHON_BIN="${PYTHON_BIN:-python}"
B49_SEED="${B49_SEED:-2026}"
CONFIG="config/b49_native_tiled_multiscale.yaml"
ARMS=(static_prior_control post_cross_attention_candidate)
ALLOWED_SEEDS=(2026 2037 2048)

required_vars=(
  DATA_ROOT
  LABELS_ROOT
  SERIES_POLICY
  BASE_CHECKPOINT
  DOMAIN_SPLIT_ROOT
  B48_ROOT
  B49_ROOT
)
for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: required environment variable $name is not set" >&2
    exit 2
  fi
done

seed_allowed=false
for allowed in "${ALLOWED_SEEDS[@]}"; do
  if [[ "$B49_SEED" == "$allowed" ]]; then
    seed_allowed=true
  fi
done
if [[ "$seed_allowed" != true ]]; then
  echo "ERROR: B49_SEED must be one of: ${ALLOWED_SEEDS[*]}" >&2
  exit 2
fi

DOMAIN_SPLIT="$DOMAIN_SPLIT_ROOT/domain_split.json"
DOMAIN_ROWS="$DOMAIN_SPLIT_ROOT/domain_split_by_study.csv"
DOMAIN_SHA="$DOMAIN_SPLIT_ROOT/domain_split.sha256"
for path in "$DOMAIN_SPLIT" "$DOMAIN_ROWS" "$DOMAIN_SHA"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: B49 requires B48's frozen domain-split artifact: $path" >&2
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

# B49 must not start while B48 seed 2026 is unfinished.  B49 consumes neither
# checkpoint state nor prediction, so this is intentionally only a completion
# guard.  A later B49 replication must be sequenced after the corresponding B48
# seed pair as well.
b48_pair_root="$B48_ROOT/seed_${B49_SEED}"
for arm in "${ARMS[@]}"; do
  checkpoint="$b48_pair_root/$arm/b48_${arm}_model.pt"
  if [[ ! -s "$checkpoint" ]]; then
    echo "ERROR: B48 seed $B49_SEED arm $arm is incomplete: $checkpoint" >&2
    echo "Finish the matching B48 pair before B49; B49 will not use its weights." >&2
    exit 4
  fi
done

pair_root="$B49_ROOT/seed_${B49_SEED}"
all_complete=true
for arm in "${ARMS[@]}"; do
  checkpoint="$pair_root/$arm/b49_${arm}_model.pt"
  if [[ ! -s "$checkpoint" ]]; then
    all_complete=false
  fi
done
if [[ "$all_complete" == true ]]; then
  echo "B49 seed $B49_SEED pair already complete -- no checkpoints overwritten"
  exit 0
fi

mkdir -p "$pair_root"
printf 'B49 seed: %s\n' "$B49_SEED"
printf 'B49 reused domain split SHA: %s\n' "$actual_domain_sha"
printf 'B48 completion guard root: %s\n' "$b48_pair_root"
printf 'B49 pair root: %s\n\n' "$pair_root"

# Both arms must pass full-FOV native coverage, a real-DICOM high-tile forward
# and backward, frozen-B34 reconstruction, and zero-gate gradient checks before
# either optimizer starts.
for arm in "${ARMS[@]}"; do
  echo "============================================================"
  echo "B49 $arm S${B49_SEED} PREFLIGHT"
  echo "============================================================"
  "$PYTHON_BIN" -m rsna_knee.b49_native_tiled_multiscale_training \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --labels-root "$LABELS_ROOT" \
    --series-policy "$SERIES_POLICY" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --domain-split "$DOMAIN_SPLIT" \
    --arm "$arm" \
    --seed "$B49_SEED" \
    --out-root "$pair_root/$arm" \
    --preflight-only
done

for arm in "${ARMS[@]}"; do
  arm_root="$pair_root/$arm"
  checkpoint="$arm_root/b49_${arm}_model.pt"
  log="$arm_root/training.log"
  mkdir -p "$arm_root"
  if [[ -s "$checkpoint" ]]; then
    echo "B49 $arm S${B49_SEED} already complete -- skipping"
    sha256sum "$checkpoint"
    continue
  fi
  if [[ -e "$checkpoint" ]]; then
    echo "ERROR: B49 checkpoint exists but is empty: $checkpoint" >&2
    echo "Inspect it; removing an interrupted checkpoint is an operator decision." >&2
    exit 5
  fi
  echo "============================================================"
  echo "B49 $arm S${B49_SEED} FIXED-E2 TRAINING"
  echo "============================================================"
  {
    echo
    echo "=== B49 $arm seed $B49_SEED attempt started $(date -Is) ==="
  } >> "$log"
  set +e
  "$PYTHON_BIN" -m rsna_knee.b49_native_tiled_multiscale_training \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --labels-root "$LABELS_ROOT" \
    --series-policy "$SERIES_POLICY" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --domain-split "$DOMAIN_SPLIT" \
    --arm "$arm" \
    --seed "$B49_SEED" \
    --out-root "$arm_root" \
    2>&1 | tee -a "$log"
  python_status=${PIPESTATUS[0]}
  set -e
  if [[ "$python_status" -ne 0 ]]; then
    echo "ERROR: B49 $arm failed with status $python_status" >&2
    exit "$python_status"
  fi
  if [[ ! -s "$checkpoint" ]]; then
    echo "ERROR: B49 $arm exited successfully but checkpoint is missing/empty" >&2
    exit 6
  fi
  sha256sum "$checkpoint"
done

echo
echo "B49 seed $B49_SEED matched pair: COMPLETE"
echo "Run the paired B49 evaluator once, after both arms are complete."
