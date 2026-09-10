#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
PY="${MEDIROAD_PYTHON:-python3}"
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 CUDA_VISIBLE_DEVICES=0
export CUDA_PATH="${CUDA_PATH:-${MEDIROAD_WSL_HOME:?Set MEDIROAD_WSL_HOME}/miniforge3/envs/mediroad-stage4-2c/targets/x86_64-linux}"
export LD_LIBRARY_PATH="${MEDIROAD_WSL_HOME:?Set MEDIROAD_WSL_HOME}/miniforge3/envs/mediroad-stage4-2c/targets/x86_64-linux/lib:${MEDIROAD_WSL_HOME:?Set MEDIROAD_WSL_HOME}/miniforge3/envs/mediroad-stage4-2c/lib:/usr/lib/wsl/lib"
exec "$PY" -m mediroad.stage4_2h_coarse_equity --project-root "$ROOT"
