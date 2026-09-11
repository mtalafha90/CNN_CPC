#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/developments/src${PYTHONPATH:+:$PYTHONPATH}"

stage="${1:-preflight}"
case "$stage" in
  prepare|preflight|train|compare|run) ;;
  *) echo "Usage: bash $0 {prepare|preflight|train|compare|run}" >&2; exit 2 ;;
esac

B57_RUN_ROOT="${B57_RUN_ROOT:-$PROJECT_ROOT/runs/093_Experiment_B57_clean_backbone_comparison}"
B57_DATA_ROOT="${B57_DATA_ROOT:-$PROJECT_ROOT/rsna-knee-abnormality-detection}"
B57_LABELS_ROOT="${B57_LABELS_ROOT:-$PROJECT_ROOT/runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all}"
B57_SERIES_POLICY="${B57_SERIES_POLICY:-$PROJECT_ROOT/runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json}"
B57_NUM_WORKERS="${B57_NUM_WORKERS:-0}"

prepare() {
  if [[ -z "${B57_SELECTION_ROOT:-}" ]]; then
    B57_SELECTION_ROOT="$(python - <<'PY'
from pathlib import Path
paths = sorted(Path('runs').rglob('b50_selection_split.json'))
# Deduplicate archive symlinks; never regenerate the old gate.
roots = sorted({p.resolve().parent for p in paths})
if len(roots) != 1:
    raise SystemExit('Set B57_SELECTION_ROOT to the existing gate directory. Found: '
                     + ', '.join(map(str, roots)))
print(roots[0])
PY
)"
  fi
  python -m rsna_knee.b57_training prepare \
    --run-root "$B57_RUN_ROOT" --data-root "$B57_DATA_ROOT" \
    --labels-root "$B57_LABELS_ROOT" --domain-split "$B57_SELECTION_ROOT" \
    --series-policy "$B57_SERIES_POLICY"
}

preflight() {
  for arm in clean_b52_reference dinov2_slice_candidate; do
    python -m rsna_knee.b57_training preflight --run-root "$B57_RUN_ROOT" \
      --arm "$arm" --device cuda --num-workers "$B57_NUM_WORKERS"
  done
}

train() {
  for arm in clean_b52_reference dinov2_slice_candidate; do
    python -m rsna_knee.b57_training train --run-root "$B57_RUN_ROOT" \
      --arm "$arm" --device cuda --num-workers "$B57_NUM_WORKERS"
  done
}

compare() {
  python -m rsna_knee.b57_evaluation --run-root "$B57_RUN_ROOT"
}

case "$stage" in
  prepare) prepare ;;
  preflight) prepare; preflight ;;
  train) train ;;
  compare) compare ;;
  run) prepare; preflight; train; compare ;;
esac
