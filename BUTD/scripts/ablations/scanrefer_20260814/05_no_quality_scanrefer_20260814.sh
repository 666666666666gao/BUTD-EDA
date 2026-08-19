#!/bin/bash
set -euo pipefail

REPO="/home/gb/new butd/butd_detr-main"
ORIGINAL="${REPO}/scripts/ablations/scanrefer_20260814/05_no_quality_scanrefer_20260814.sh.original_20260815"
GATE="${REPO}/logs/butd_universal_target/scanrefer_ablation_extension_20260815_queue/MODULE_PRIORITY_PASS"
ALERT="${REPO}/logs/butd_universal_target/scanrefer_ablation_extension_20260815_queue/WATCHDOG_ALERT"

if [ "${DRY_RUN:-0}" = "1" ]; then
  exec bash "${ORIGINAL}" "$@"
fi

while [ ! -f "${GATE}" ]; do
  if [ -f "${ALERT}" ]; then
    echo "MODULE_PRIORITY_QUEUE_ALERT: refusing to start internal ablation" >&2
    exit 20
  fi
  echo "WAITING_FOR_MODULE_PRIORITY_ROWS $(date -Is)"
  sleep 300
done

exec bash "${ORIGINAL}" "$@"
