#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
MODE="${MODE:-official}"
SOLVER="${SOLVER:-cp-sat}"
python -m mediroad.stage4 \
  --package-root "$ROOT" \
  --config "$ROOT/configs/model_v1/stage4.yaml" \
  --mode "$MODE" \
  --solver "$SOLVER" \
  "$@"

