#!/bin/bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/gb/new butd/butd_detr-main}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
GPU_INDEX="${GPU%%,*}"
RUN_TAG="${ABLATION_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SCREEN_NAME="${ABLATION_SCREEN_NAME:-scanrefer_ablation_e65_${RUN_TAG}}"
LOG_ROOT="${ABLATION_LOG_ROOT:-${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_e65_reproduction/${RUN_TAG}}"
QUEUE_LOG="${LOG_ROOT}/_package_control/queue.log"

if ! [[ "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: invalid CUDA_VISIBLE_DEVICES=${GPU}" >&2
  exit 60
fi
USED="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [ "${USED}" -ge 500 ]; then
  echo "ERROR: GPU ${GPU_INDEX} is busy (${USED} MiB used); current formal queue must finish first." >&2
  exit 61
fi
if screen -ls | grep -F ".${SCREEN_NAME}" >/dev/null 2>&1; then
  echo "ERROR: screen ${SCREEN_NAME} already exists" >&2
  exit 62
fi

mkdir -p "$(dirname "${QUEUE_LOG}")"
screen -dmS "${SCREEN_NAME}" bash -lc "cd '${PACKAGE_ROOT}' && CUDA_VISIBLE_DEVICES='${GPU}' ABLATION_RUN_TAG='${RUN_TAG}' ABLATION_LOG_ROOT='${LOG_ROOT}' bash run_all_serial.sh 2>&1 | tee '${QUEUE_LOG}'"
sleep 2
screen -ls | grep -F ".${SCREEN_NAME}"
printf 'screen=%s\nlog_root=%s\nqueue_log=%s\n' "${SCREEN_NAME}" "${LOG_ROOT}" "${QUEUE_LOG}"

