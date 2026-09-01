#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/home/gb/new butd/butd_detr-main"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
STAGE1_RECEIPT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage1_reload_verify/goal_receipt.json"
STAGE0_RECEIPT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage0_reload_verify/goal_receipt.json"
QUEUE_PARENT_PID=114992
ORIGINAL_SHA="20f289d98657be242530e379fb23a3bea8137ef392dc7cd8f28675151dd805e4"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
STATUS_FILE="${PACKAGE_ROOT}/stage2_watch_status.txt"

stage1_achieved=""
while [ -z "${stage1_achieved}" ]; do
  if [ -s "${STAGE0_RECEIPT}" ]; then
    stage0_achieved="$("${PYTHON}" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["goal_achieved"])).lower())' "${STAGE0_RECEIPT}" 2>/dev/null || true)"
    if [ "${stage0_achieved}" = "true" ]; then
      printf '%s skipped; stage0_goal_achieved=true\n' "$(date -Is)" > "${STATUS_FILE}"
      exit 0
    fi
  fi
  stage1_status="$(cat "${PACKAGE_ROOT}/stage1_watch_status.txt" 2>/dev/null || true)"
  if printf '%s' "${stage1_status}" | grep -q 'stage1_high_iou_failed'; then
    printf '%s refused; stage1_failed\n' "$(date -Is)" > "${STATUS_FILE}"
    exit 120
  fi
  if [ -s "${STAGE1_RECEIPT}" ]; then
    stage1_achieved="$("${PYTHON}" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["goal_achieved"])).lower())' "${STAGE1_RECEIPT}" 2>/dev/null || true)"
  fi
  if [ -z "${stage1_achieved}" ]; then
    printf '%s waiting_for_stage1_receipt\n' "$(date -Is)" > "${STATUS_FILE}"
    sleep "${WAIT_SECONDS}"
  fi
done

if [ "${stage1_achieved}" = "true" ]; then
  printf '%s skipped; stage1_goal_achieved=true\n' "$(date -Is)" > "${STATUS_FILE}"
  exit 0
fi

while pgrep -af '[t]rain_dist_mod.py' >/dev/null 2>&1; do
  printf '%s waiting_for_gpu_after_stage1\n' "$(date -Is)" > "${STATUS_FILE}"
  sleep "${WAIT_SECONDS}"
done

parent_state="$(awk '/^State:/ {print $2}' "/proc/${QUEUE_PARENT_PID}/status" 2>/dev/null || true)"
if [ "${parent_state}" != "T" ]; then
  printf '%s refused; queue_parent_state=%s\n' "$(date -Is)" "${parent_state:-missing}" > "${STATUS_FILE}"
  exit 121
fi

evaluator_sha="$(sha256sum "${REPO_ROOT}/src/grounding_evaluator.py" | awk '{print $1}')"
if [ "${evaluator_sha}" != "${ORIGINAL_SHA}" ]; then
  printf '%s refused; evaluator_sha=%s\n' "$(date -Is)" "${evaluator_sha}" > "${STATUS_FILE}"
  exit 122
fi

printf '%s starting_stage2_full_finetune\n' "$(date -Is)" > "${STATUS_FILE}"
if "${PACKAGE_ROOT}/run_stage2_full_finetune.sh"; then
  printf '%s stage2_full_finetune_complete\n' "$(date -Is)" > "${STATUS_FILE}"
else
  rc=$?
  printf '%s stage2_full_finetune_failed rc=%s\n' "$(date -Is)" "${rc}" > "${STATUS_FILE}"
  exit "${rc}"
fi
