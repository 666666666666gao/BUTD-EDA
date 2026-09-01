#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/home/gb/new butd/butd_detr-main"
DIAG_SUMMARY="/root/autodl-tmp/logs/butd_acc50_target_20260827/text_policy_diagnostic/summary.txt"
STAGE0_RECEIPT="/root/autodl-tmp/logs/butd_acc50_target_20260827/stage0_reload_verify/goal_receipt.json"
QUEUE_PARENT_PID=114992
ORIGINAL_SHA="20f289d98657be242530e379fb23a3bea8137ef392dc7cd8f28675151dd805e4"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
STATUS_FILE="${PACKAGE_ROOT}/stage1_watch_status.txt"

while [ ! -s "${DIAG_SUMMARY}" ]; do
  diag_status="$(cat "${PACKAGE_ROOT}/watch_status.txt" 2>/dev/null || true)"
  if printf '%s' "${diag_status}" | grep -q 'diagnostic_failed'; then
    printf '%s refused; diagnostic_failed\n' "$(date -Is)" > "${STATUS_FILE}"
    exit 91
  fi
  printf '%s waiting_for_text_diagnostic\n' "$(date -Is)" > "${STATUS_FILE}"
  sleep "${WAIT_SECONDS}"
done

[ -s "${STAGE0_RECEIPT}" ] || {
  printf '%s refused; stage0_receipt_missing\n' "$(date -Is)" > "${STATUS_FILE}"
  exit 94
}
stage0_achieved="$("/root/miniconda3/envs/bdetr/bin/python" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["goal_achieved"])).lower())' "${STAGE0_RECEIPT}")"
if [ "${stage0_achieved}" = "true" ]; then
  printf '%s skipped; stage0_goal_achieved=true\n' "$(date -Is)" > "${STATUS_FILE}"
  exit 0
fi

while pgrep -af '[t]rain_dist_mod.py' >/dev/null 2>&1; do
  printf '%s waiting_for_gpu_after_diagnostic\n' "$(date -Is)" > "${STATUS_FILE}"
  sleep "${WAIT_SECONDS}"
done

parent_state="$(awk '/^State:/ {print $2}' "/proc/${QUEUE_PARENT_PID}/status" 2>/dev/null || true)"
if [ "${parent_state}" != "T" ]; then
  printf '%s refused; queue_parent_state=%s\n' "$(date -Is)" "${parent_state:-missing}" > "${STATUS_FILE}"
  exit 92
fi

evaluator_sha=""
for attempt in $(seq 1 60); do
  evaluator_sha="$(sha256sum "${REPO_ROOT}/src/grounding_evaluator.py" | awk '{print $1}')"
  if [ "${evaluator_sha}" = "${ORIGINAL_SHA}" ]; then
    break
  fi
  printf '%s waiting_for_evaluator_restore attempt=%s\n' "$(date -Is)" "${attempt}" > "${STATUS_FILE}"
  sleep 5
done
if [ "${evaluator_sha}" != "${ORIGINAL_SHA}" ]; then
  printf '%s refused; evaluator_sha=%s\n' "$(date -Is)" "${evaluator_sha}" > "${STATUS_FILE}"
  exit 93
fi

printf '%s starting_stage1_high_iou\n' "$(date -Is)" > "${STATUS_FILE}"
if "${PACKAGE_ROOT}/run_stage1_high_iou.sh"; then
  printf '%s stage1_high_iou_complete\n' "$(date -Is)" > "${STATUS_FILE}"
else
  rc=$?
  printf '%s stage1_high_iou_failed rc=%s\n' "$(date -Is)" "${rc}" > "${STATUS_FILE}"
  exit "${rc}"
fi
