#!/usr/bin/env bash
set -euo pipefail

# Build a self-contained B39 code/config artifact for Kaggle.
# The frozen B37 model checkpoints remain in the existing cnn-cpc-b37-artifacts
# Kaggle dataset and are intentionally NOT duplicated here.

REPO_ROOT="${1:-$(pwd)}"
OUT_ROOT="${2:-$REPO_ROOT/kaggle_b39_artifacts}"
PROJECT_ROOT="$OUT_ROOT/CNN_CPC"

cd "$REPO_ROOT"

required=(
  "config/b39_b37_five_offset_tta.yaml"
  "developments/src/rsna_knee/kaggle_hidden_streaming_highres.py"
  "developments/src/rsna_knee/b39_b37_five_offset_tta_dualgpu_streaming.py"
  "developments/src/rsna_knee/kaggle_hidden_streaming_equivalence.py"
  "developments/src/rsna_knee/b39_b37_five_offset_tta_dualgpu.py"
  "developments/src/rsna_knee/b39_b37_five_offset_tta.py"
)

for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required B39 artifact source: $path" >&2
    exit 2
  fi
done

rm -rf "$OUT_ROOT"
mkdir -p "$PROJECT_ROOT/config" "$PROJECT_ROOT/developments/src"

cp -f config/b39_b37_five_offset_tta.yaml "$PROJECT_ROOT/config/"
cp -a developments/src/rsna_knee "$PROJECT_ROOT/developments/src/"

# A tiny provenance file lets the Kaggle notebook show exactly what was packed.
git_rev="$(git rev-parse HEAD)"
helper_sha="$(sha256sum developments/src/rsna_knee/kaggle_hidden_streaming_highres.py | awk '{print $1}')"
stream_sha="$(sha256sum developments/src/rsna_knee/b39_b37_five_offset_tta_dualgpu_streaming.py | awk '{print $1}')"
cat > "$PROJECT_ROOT/B39_HIDDEN_SAFE_ARTIFACT.txt" <<EOF
repository_commit=$git_rev
streaming_helper_sha256=$helper_sha
streaming_wrapper_sha256=$stream_sha
execution_version=b39_hidden_dual_t4_streaming_views_normonce_noabort_v5
scientific_endpoint=frozen_B37_fixed_E2_plus_five_offsets_-2_-1_0_1_2
EOF

check=(
  "$PROJECT_ROOT/config/b39_b37_five_offset_tta.yaml"
  "$PROJECT_ROOT/developments/src/rsna_knee/kaggle_hidden_streaming_highres.py"
  "$PROJECT_ROOT/developments/src/rsna_knee/b39_b37_five_offset_tta_dualgpu_streaming.py"
  "$PROJECT_ROOT/developments/src/rsna_knee/kaggle_hidden_streaming_equivalence.py"
  "$PROJECT_ROOT/B39_HIDDEN_SAFE_ARTIFACT.txt"
)

for path in "${check[@]}"; do
  test -f "$path"
  echo "PASS $path"
done

echo
echo "B39 Kaggle hidden-safe artifact ready:"
echo "$OUT_ROOT"
echo
echo "Upload the entire directory as the new cnn-b39 dataset content, preserving:"
echo "  kaggle_b39_artifacts/CNN_CPC/..."
