#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/gb/new butd/butd_detr-main}"
SCREEN_NAME="scanrefer_three_table_ablations_20260821"
QUEUE_ROOT="${REPO_ROOT}/logs/butd_universal_target/scanrefer_three_table_ablations_20260821_queue"

mkdir -p "${QUEUE_ROOT}"
if screen -ls | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "ERROR: screen ${SCREEN_NAME} already exists" >&2
  exit 2
fi

# This queue intentionally starts while the compatible old M1 is training. It
# performs CPU-only preflight checks, then waits for M1/M2/S1 before takeover.
screen -dmS "${SCREEN_NAME}" -L -Logfile "${QUEUE_ROOT}/screen.log" \
  bash "${PACKAGE_ROOT}/run_serial.sh"
sleep 2
screen -ls | grep -q "[.]${SCREEN_NAME}[[:space:]]"
echo "STARTED ${SCREEN_NAME}"
