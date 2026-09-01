#!/usr/bin/env bash
set -euo pipefail

MACHINE_LABEL="${1:?usage: wait_and_collect_e20_summary.sh MACHINE_LABEL}"
RUN_ROOT="${DUAL_ABLATION_RUN_ROOT:-/root/autodl-tmp/logs/butd_scanrefer_dual_machine_ablations_20260901}"
CONTROL_ROOT="${RUN_ROOT}/${MACHINE_LABEL}/control"
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${E20_PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
INTERVAL="${E20_WATCH_INTERVAL_SECONDS:-300}"
READY="${CONTROL_ROOT}/E20_READY"
SUMMARY="${CONTROL_ROOT}/E20_SUMMARY.json"
SUMMARY_READY="${CONTROL_ROOT}/E20_SUMMARY_READY"
ALERT="${CONTROL_ROOT}/E20_SUMMARY_ALERT"

mkdir -p "${CONTROL_ROOT}"
[ ! -e "${SUMMARY_READY}" ] || exit 0
rm -f "${ALERT}"

while true; do
  if [ -f "${READY}" ]; then
    "${PYTHON_BIN}" "${PACKAGE_ROOT}/collect_e20_once.py" "${READY}" "${SUMMARY}"
    {
      printf 'timestamp=%s\n' "$(date -Is)"
      printf 'machine=%s\n' "${MACHINE_LABEL}"
      printf 'summary=%s\n' "${SUMMARY}"
      sha256sum "${SUMMARY}"
    } > "${SUMMARY_READY}.tmp"
    mv "${SUMMARY_READY}.tmp" "${SUMMARY_READY}"
    chmod 0444 "${SUMMARY_READY}"
    exit 0
  fi

  if [ -f "${CONTROL_ROOT}/E20_WATCH_ALERT" ] || [ -f "${CONTROL_ROOT}/QUEUE_ALERT" ]; then
    printf 'upstream_alert_before_e20_summary timestamp=%s\n' "$(date -Is)" > "${ALERT}"
    exit 1
  fi
  sleep "${INTERVAL}"
done
