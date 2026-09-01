#!/usr/bin/env bash
set -euo pipefail

MACHINE_LABEL="${1:?usage: wait_and_collect_machine_rows.sh MACHINE ROW...}"
shift
[ "$#" -gt 0 ] || { echo "ERROR: at least one row is required" >&2; exit 2; }

MONITOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${DUAL_ABLATION_RUN_ROOT:-/root/autodl-tmp/logs/butd_scanrefer_dual_machine_ablations_20260901}"
CONTROL_ROOT="${RUN_ROOT}/${MACHINE_LABEL}/control"
RECEIPT_DIR="${CONTROL_ROOT}/receipts"
RESULT_DIR="${CONTROL_ROOT}/audited_rows"
COLLECTOR="${MONITOR_ROOT}/collect_completed_row.py"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
EXPECTED_COLLECTOR_SHA="ad5da503a1bc91ef1fdffd85ebdb563b36d412b2b4e242cb8c935e70d82cdcfc"
WAIT_SECONDS="${RESULT_WATCH_SECONDS:-300}"

[ -f "${COLLECTOR}" ] || { echo "ERROR: missing collector" >&2; exit 3; }
[ "$(sha256sum "${COLLECTOR}" | awk '{print $1}')" = "${EXPECTED_COLLECTOR_SHA}" ] || {
  echo "ERROR: collector SHA256 mismatch" >&2
  exit 4
}
mkdir -p "${RESULT_DIR}"

while [ ! -f "${CONTROL_ROOT}/ALL_COMPLETE" ]; do
  if [ -f "${CONTROL_ROOT}/QUEUE_ALERT" ]; then
    cp -p "${CONTROL_ROOT}/QUEUE_ALERT" "${CONTROL_ROOT}/MACHINE_RESULT_WATCH_ALERT"
    exit 5
  fi
  sleep "${WAIT_SECONDS}"
done

rm -f "${CONTROL_ROOT}/MACHINE_RESULTS_READY.tmp"
for row in "$@"; do
  receipt="${RECEIPT_DIR}/${row}.json"
  output="${RESULT_DIR}/${row}.json"
  [ -f "${receipt}" ] || {
    printf 'missing receipt for %s\n' "${row}" > "${CONTROL_ROOT}/MACHINE_RESULT_WATCH_ALERT"
    exit 6
  }
  [ ! -e "${output}" ] || {
    printf 'audited result already exists for %s\n' "${row}" > "${CONTROL_ROOT}/MACHINE_RESULT_WATCH_ALERT"
    exit 7
  }
  PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" "${COLLECTOR}" "${receipt}" "${output}"
  sha256sum "${output}" >> "${CONTROL_ROOT}/MACHINE_RESULTS_READY.tmp"
done

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'machine=%s\n' "${MACHINE_LABEL}"
  printf 'row_count=%s\n' "$#"
  cat "${CONTROL_ROOT}/MACHINE_RESULTS_READY.tmp"
} > "${CONTROL_ROOT}/MACHINE_RESULTS_READY.new"
mv "${CONTROL_ROOT}/MACHINE_RESULTS_READY.new" "${CONTROL_ROOT}/MACHINE_RESULTS_READY"
rm -f "${CONTROL_ROOT}/MACHINE_RESULTS_READY.tmp"
chmod 0444 "${CONTROL_ROOT}/MACHINE_RESULTS_READY" "${RESULT_DIR}"/*.json
