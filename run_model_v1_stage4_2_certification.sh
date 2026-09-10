#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
CONFIG="${2:-configs/model_v1/stage4_2_certification.yaml}"
cd "$ROOT"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONHASHSEED=42

if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/targets/x86_64-linux" ]]; then
  export CUDA_PATH="$CONDA_PREFIX/targets/x86_64-linux"
  export LD_LIBRARY_PATH="$CUDA_PATH/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

python -m mediroad.stage4_2_certification \
  --project-root "$PWD" \
  --stage42-config configs/model_v1/stage4_2.yaml \
  --certification-config "$CONFIG"
