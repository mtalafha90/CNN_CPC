#!/usr/bin/env bash
set -euo pipefail

# Freeze B50's fresh architecture-selection surface before any B50 model code
# is trained or scored. The script does not modify the B48/B49 split and it
# refuses to overwrite a B50 gate.
#
# Required environment variables:
#   DATA_ROOT LABELS_ROOT DOMAIN_SPLIT_ROOT B49_ROOT
# Optional:
#   HEADER_CSV (default: runs/dataset_header_audit/header_by_series.csv)
#   B50_SELECTION_ROOT (default: B50's numbered run directory)
#   B49_SEED (default: 2026)
#   PYTHON_BIN (default: python)

PYTHON_BIN="${PYTHON_BIN:-python}"
B49_SEED="${B49_SEED:-2026}"
HEADER_CSV="${HEADER_CSV:-$PWD/runs/dataset_header_audit/header_by_series.csv}"
B50_SELECTION_ROOT="${B50_SELECTION_ROOT:-$PWD/runs/083_Experiment_B50_ordered_slice_sequence_mil/b50_ordered_slice_selection_split}"

required_vars=(DATA_ROOT LABELS_ROOT DOMAIN_SPLIT_ROOT B49_ROOT)
for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: required environment variable $name is not set" >&2
    exit 2
  fi
done

for arm in static_prior_control post_cross_attention_candidate; do
  # B50 does not consume these checkpoints. This is only a provenance guard:
  # its validation design is not allowed to change while B49 remains in flight.
  checkpoint="$B49_ROOT/seed_${B49_SEED}/${arm}/b49_${arm}_model.pt"
  if [[ ! -s "$checkpoint" ]]; then
    echo "ERROR: completed B49 arm is required before B50 gate creation: $checkpoint" >&2
    exit 3
  fi
done

for path in \
  "$DOMAIN_SPLIT_ROOT/domain_split.json" \
  "$DOMAIN_SPLIT_ROOT/domain_split_by_study.csv" \
  "$DOMAIN_SPLIT_ROOT/domain_split.sha256" \
  "$HEADER_CSV"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required frozen/input artefact is missing: $path" >&2
    exit 4
  fi
done

actual_parent_sha="$(sha256sum "$DOMAIN_SPLIT_ROOT/domain_split.json" | awk '{print $1}')"
recorded_parent_sha="$(tr -d '[:space:]' < "$DOMAIN_SPLIT_ROOT/domain_split.sha256")"
if [[ "$actual_parent_sha" != "$recorded_parent_sha" ]]; then
  echo "ERROR: B48/B49 domain split SHA mismatch" >&2
  echo "expected: $recorded_parent_sha" >&2
  echo "actual:   $actual_parent_sha" >&2
  exit 5
fi

for name in b50_selection_split.json b50_selection_split_by_study.csv b50_selection_split.sha256; do
  if [[ -e "$B50_SELECTION_ROOT/$name" ]]; then
    echo "ERROR: B50 fresh selection gate already exists; do not regenerate it: $B50_SELECTION_ROOT/$name" >&2
    exit 6
  fi
done

echo "B50 parent B48/B49 domain SHA: $actual_parent_sha"
echo "B50 selection gate root: $B50_SELECTION_ROOT"
"$PYTHON_BIN" -m rsna_knee.b50_ordered_slice_selection_split \
  --data-root "$DATA_ROOT" \
  --header-csv "$HEADER_CSV" \
  --labels-root "$LABELS_ROOT" \
  --parent-domain-split "$DOMAIN_SPLIT_ROOT" \
  --out-root "$B50_SELECTION_ROOT"

actual_b50_sha="$(sha256sum "$B50_SELECTION_ROOT/b50_selection_split.json" | awk '{print $1}')"
recorded_b50_sha="$(tr -d '[:space:]' < "$B50_SELECTION_ROOT/b50_selection_split.sha256")"
if [[ "$actual_b50_sha" != "$recorded_b50_sha" ]]; then
  echo "ERROR: B50 selection split hash verification failed" >&2
  exit 7
fi

echo "B50 fresh selection split SHA: $actual_b50_sha"
echo "B50 fresh selection gate: COMPLETE"
