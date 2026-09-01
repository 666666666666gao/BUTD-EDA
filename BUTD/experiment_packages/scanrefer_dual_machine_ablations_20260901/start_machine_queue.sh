#!/usr/bin/env bash
set -euo pipefail

MACHINE_LABEL="${1:?usage: start_machine_queue.sh LABEL ROW...}"
shift
[ "$#" -gt 0 ] || { echo "ERROR: at least one row is required" >&2; exit 2; }

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/gb/new butd/butd_detr-main}"
SCREEN_NAME="scanrefer_ablation_${MACHINE_LABEL}_20260901"
RUN_ROOT="${DUAL_ABLATION_RUN_ROOT:-/root/autodl-tmp/logs/butd_scanrefer_dual_machine_ablations_20260901}"
SCREEN_LOG="${RUN_ROOT}/${MACHINE_LABEL}/control/screen.log"

used="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
[ "${used}" -lt 500 ] || { echo "ERROR: GPU0 busy (${used} MiB)" >&2; exit 20; }
! pgrep -af '[t]rain_dist_mod.py' >/dev/null 2>&1 || { echo "ERROR: training already active" >&2; exit 21; }
! screen -ls 2>/dev/null | grep -F ".${SCREEN_NAME}" >/dev/null 2>&1 || { echo "ERROR: screen exists: ${SCREEN_NAME}" >&2; exit 22; }

mkdir -p "$(dirname "${SCREEN_LOG}")"
printf -v quoted_rows ' %q' "$@"
screen -dmS "${SCREEN_NAME}" bash -lc "cd '${PACKAGE_ROOT}' && REPO_ROOT='${REPO_ROOT}' bash run_machine_queue.sh '${MACHINE_LABEL}'${quoted_rows} > '${SCREEN_LOG}' 2>&1"
sleep 3
screen -ls | grep -F ".${SCREEN_NAME}"
printf 'screen=%s\nlog=%s\nrows=%s\n' "${SCREEN_NAME}" "${SCREEN_LOG}" "$*"
