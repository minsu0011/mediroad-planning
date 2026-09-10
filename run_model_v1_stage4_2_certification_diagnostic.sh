#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$(pwd)}"
exec "$(dirname "$0")/run_model_v1_stage4_2_certification.sh" \
  "$ROOT" configs/model_v1/stage4_2_certification_diagnostic.yaml
