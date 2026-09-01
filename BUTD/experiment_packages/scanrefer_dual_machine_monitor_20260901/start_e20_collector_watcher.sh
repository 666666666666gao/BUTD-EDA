#!/usr/bin/env bash
set -euo pipefail

MACHINE_LABEL="${1:?usage: start_e20_collector_watcher.sh MACHINE_LABEL}"
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${DUAL_ABLATION_RUN_ROOT:-/root/autodl-tmp/logs/butd_scanrefer_dual_machine_ablations_20260901}"
SCREEN_NAME="scanrefer_e20_collect_${MACHINE_LABEL}_20260901"
CONTROL_ROOT="${RUN_ROOT}/${MACHINE_LABEL}/control"
LOG="${CONTROL_ROOT}/e20_collector_watcher.log"

if [ -f "${CONTROL_ROOT}/E20_SUMMARY_READY" ]; then
  echo "E20 summary marker already exists"
  exit 0
fi
if screen -ls 2>/dev/null | grep -F ".${SCREEN_NAME}" >/dev/null 2>&1; then
  echo "collector watcher already running: ${SCREEN_NAME}"
  exit 0
fi

mkdir -p "${CONTROL_ROOT}"
screen -dmS "${SCREEN_NAME}" bash -lc "cd '${PACKAGE_ROOT}' && bash wait_and_collect_e20_summary.sh '${MACHINE_LABEL}' > '${LOG}' 2>&1"
sleep 2
screen -ls | grep -F ".${SCREEN_NAME}"
printf 'screen=%s\nlog=%s\n' "${SCREEN_NAME}" "${LOG}"
