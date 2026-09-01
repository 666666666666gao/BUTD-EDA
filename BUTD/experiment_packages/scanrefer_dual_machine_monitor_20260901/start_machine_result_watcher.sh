#!/usr/bin/env bash
set -euo pipefail

MACHINE_LABEL="${1:?usage: start_machine_result_watcher.sh MACHINE ROW...}"
shift
[ "$#" -gt 0 ] || { echo "ERROR: at least one row is required" >&2; exit 2; }

MONITOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCREEN_NAME="scanrefer_result_watch_${MACHINE_LABEL}_20260901"
RUN_ROOT="${DUAL_ABLATION_RUN_ROOT:-/root/autodl-tmp/logs/butd_scanrefer_dual_machine_ablations_20260901}"
CONTROL_ROOT="${RUN_ROOT}/${MACHINE_LABEL}/control"
LOG="${CONTROL_ROOT}/machine_result_watcher_screen.log"

mkdir -p "${CONTROL_ROOT}"
if screen -ls | grep -F ".${SCREEN_NAME}" >/dev/null 2>&1; then
  echo "ERROR: watcher already running: ${SCREEN_NAME}" >&2
  exit 3
fi

screen -L -Logfile "${LOG}" -dmS "${SCREEN_NAME}" \
  bash "${MONITOR_ROOT}/wait_and_collect_machine_rows.sh" "${MACHINE_LABEL}" "$@"
sleep 2
screen -ls | grep -F ".${SCREEN_NAME}"
printf 'screen=%s\nlog=%s\nrows=%s\n' "${SCREEN_NAME}" "${LOG}" "$*"
