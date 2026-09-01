#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE_PARENT_PID=114992
WAIT_SECONDS="${WAIT_SECONDS:-300}"
STATUS_FILE="${PACKAGE_ROOT}/watch_status.txt"

while pgrep -af '[t]rain_dist_mod.py' >/dev/null 2>&1; do
  printf '%s waiting_for_gpu; active_training=1\\n' "$(date -Is)" > "${STATUS_FILE}"
  sleep "${WAIT_SECONDS}"
done

parent_state="$(awk '/^State:/ {print $2}' "/proc/${QUEUE_PARENT_PID}/status" 2>/dev/null || true)"
if [ "${parent_state}" != "T" ]; then
  printf '%s refused; queue_parent_state=%s\\n' "$(date -Is)" "${parent_state:-missing}" > "${STATUS_FILE}"
  echo "ERROR: queue parent ${QUEUE_PARENT_PID} is not safely stopped" >&2
  exit 80
fi

gpu_used="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [ "${gpu_used}" -ge 500 ]; then
  printf '%s refused; gpu_used_mib=%s\\n' "$(date -Is)" "${gpu_used}" > "${STATUS_FILE}"
  echo "ERROR: GPU still busy (${gpu_used} MiB)" >&2
  exit 81
fi

printf '%s starting_text_policy_diagnostic\\n' "$(date -Is)" > "${STATUS_FILE}"
if "${PACKAGE_ROOT}/run_text_policy_diagnostic.sh"; then
  printf '%s text_policy_diagnostic_complete\\n' "$(date -Is)" > "${STATUS_FILE}"
else
  rc=$?
  printf '%s text_policy_diagnostic_failed rc=%s\\n' "$(date -Is)" "${rc}" > "${STATUS_FILE}"
  exit "${rc}"
fi
