#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/developments/src${PYTHONPATH:+:$PYTHONPATH}"
if (($# == 0)); then
    set -- preflight
fi
python -m rsna_knee.b58_external_knee "$@"
