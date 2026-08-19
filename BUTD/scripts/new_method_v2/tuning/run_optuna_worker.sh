#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}" || exit 1

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT}
OUTPUT_ROOT=${OUTPUT_ROOT:-logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna}
GPUS=${GPUS:-0}
STUDY_NAME=${STUDY_NAME:-scanrefer_two_stage_full_rapf_quick5}
N_TRIALS=${N_TRIALS:-3}
MAX_EPOCH=${MAX_EPOCH:-5}

if [ -z "${STORAGE:-}" ]; then
  echo "ERROR: STORAGE must be set to a PostgreSQL/MySQL Optuna storage URL." >&2
  exit 2
fi

if [ -z "${WORKER_ID:-}" ]; then
  echo "ERROR: WORKER_ID must be set for multi-worker Optuna." >&2
  exit 2
fi

python scripts/new_method_v2/tuning/optuna_scanrefer_two_stage_full.py \
  --multi-worker \
  --study-name "${STUDY_NAME}" \
  --storage "${STORAGE}" \
  --worker-id "${WORKER_ID}" \
  --n-trials "${N_TRIALS}" \
  --max-epoch "${MAX_EPOCH}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --gpus "${GPUS}" \
  "$@"
