#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/home/gb/new butd/butd_detr-main"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
STAGE3_RECEIPT="${ROOT}/stage3_reload_verify/goal_receipt.json"
QUEUE_PARENT_PID=114992
ORIGINAL_SHA="20f289d98657be242530e379fb23a3bea8137ef392dc7cd8f28675151dd805e4"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
STATUS_FILE="${PACKAGE_ROOT}/stage4_watch_status.txt"

stage3_achieved=""
while [ -z "${stage3_achieved}" ]; do
  stage3_status="$(cat "${PACKAGE_ROOT}/stage3_watch_status.txt" 2>/dev/null || true)"
  if printf '%s' "${stage3_status}" | grep -q 'stage3_quality_grid_failed'; then
    printf '%s refused; stage3_failed\n' "$(date -Is)" > "${STATUS_FILE}"
    exit 190
  fi
  if [ -s "${STAGE3_RECEIPT}" ]; then
    stage3_achieved="$("${PYTHON}" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["goal_achieved"])).lower())' "${STAGE3_RECEIPT}" 2>/dev/null || true)"
  fi
  if [ -z "${stage3_achieved}" ]; then
    printf '%s waiting_for_stage3_receipt\n' "$(date -Is)" > "${STATUS_FILE}"
    sleep "${WAIT_SECONDS}"
  fi
done

if [ "${stage3_achieved}" = "true" ]; then
  printf '%s skipped; stage3_goal_achieved=true\n' "$(date -Is)" > "${STATUS_FILE}"
  exit 0
fi

while pgrep -af '[t]rain_dist_mod.py' >/dev/null 2>&1; do
  printf '%s waiting_for_gpu_after_stage3\n' "$(date -Is)" > "${STATUS_FILE}"
  sleep "${WAIT_SECONDS}"
done

parent_state="$(awk '/^State:/ {print $2}' "/proc/${QUEUE_PARENT_PID}/status" 2>/dev/null || true)"
if [ "${parent_state}" != "T" ]; then
  printf '%s refused; queue_parent_state=%s\n' "$(date -Is)" "${parent_state:-missing}" > "${STATUS_FILE}"
  exit 191
fi
[ "$(sha256sum "${REPO_ROOT}/src/grounding_evaluator.py" | awk '{print $1}')" = "${ORIGINAL_SHA}" ]

printf '%s starting_stage4_quality_head_top5\n' "$(date -Is)" > "${STATUS_FILE}"
if "${PACKAGE_ROOT}/run_stage4_quality_only.sh"; then
  printf '%s stage4_quality_head_complete\n' "$(date -Is)" > "${STATUS_FILE}"
else
  rc=$?
  printf '%s stage4_quality_head_failed rc=%s\n' "$(date -Is)" "${rc}" > "${STATUS_FILE}"
  exit "${rc}"
fi
