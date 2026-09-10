#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${1:-.}"
PYTHON_BIN="${STAGE42G_PYTHON:-python3}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$PYTHON_BIN" -m mediroad.stage4_2g_candidate_expansion --project-root "$PROJECT_ROOT"
