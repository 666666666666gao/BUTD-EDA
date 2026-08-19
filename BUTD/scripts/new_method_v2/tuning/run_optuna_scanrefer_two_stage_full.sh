#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}" || exit 1

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT}
OUTPUT_ROOT=${OUTPUT_ROOT:-logs/new_method_v2/tuning/scanrefer_two_stage_full_optuna}
GPUS=${GPUS:-0}
STUDY_NAME=${STUDY_NAME:-scanrefer_two_stage_full_rapf}
STORAGE=${STORAGE:-sqlite:///reports/tuning/optuna_scanrefer_two_stage_full.db}
N_TRIALS=${N_TRIALS:-20}
MAX_EPOCH=${MAX_EPOCH:-15}
MULTI_WORKER=${MULTI_WORKER:-0}
WORKER_ID=${WORKER_ID:-}

ARGS=(
  --study-name "${STUDY_NAME}"
  --storage "${STORAGE}"
  --n-trials "${N_TRIALS}"
  --max-epoch "${MAX_EPOCH}"
  --data-root "${DATA_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --gpus "${GPUS}"
)

if [ "${MULTI_WORKER}" = "1" ]; then
  ARGS+=(--multi-worker)
  if [ -n "${WORKER_ID}" ]; then
    ARGS+=(--worker-id "${WORKER_ID}")
  fi
fi

python scripts/new_method_v2/tuning/optuna_scanrefer_two_stage_full.py \
  "${ARGS[@]}" \
  "$@"
