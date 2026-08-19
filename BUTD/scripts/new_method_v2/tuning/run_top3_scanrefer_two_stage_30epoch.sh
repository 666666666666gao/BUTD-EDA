#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}" || exit 1

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT}
OUTPUT_ROOT=${OUTPUT_ROOT:-logs/new_method_v2/tuning/scanrefer_two_stage_top3_30epoch}
GPUS=${GPUS:-0}
TOP5_JSON=${TOP5_JSON:-reports/tuning/optuna_scanrefer_two_stage_full_top5.json}

python scripts/new_method_v2/tuning/rerun_top_optuna_scanrefer_two_stage.py \
  --top-k 3 \
  --max-epoch 30 \
  --top5-json "${TOP5_JSON}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --gpus "${GPUS}" \
  "$@"
