#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-}"
shift || true
export PYTHONPATH="${PYTHONPATH:-$PWD/src}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
PYTHON_BIN="${MEDIROAD_PYTHON:-python3}"
case "$MODE" in
  prepare|computational)
    "$PYTHON_BIN" 12_scripts/v6/run_model_v1_stage4_2.py --project-root . --config configs/model_v1/stage4_2.yaml "$MODE" "$@"
    ;;
  audit-field|operational-final)
    "$PYTHON_BIN" 12_scripts/v6/run_model_v1_stage4_2.py --project-root . --config configs/model_v1/stage4_2.yaml "$MODE" "$@"
    ;;
  *)
    echo "Usage: $0 {prepare|computational|audit-field|operational-final} [arguments...]" >&2
    exit 2
    ;;
esac
