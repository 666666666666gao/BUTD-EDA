#!/usr/bin/env bash
set -Eeuo pipefail

STAGE141_PID="${1:?stage141 launcher pid required}"
EXPECTED_START_TICKS="${2:?stage141 launcher start ticks required}"
P='/home/gb/new butd/butd_detr-main/experiment_packages/butd_acc50_target_20260827'
RECEIPT='/root/autodl-tmp/logs/butd_acc50_target_20260827/stage141_stage135c_e12_raw_train_geometry_dump/stage141_receipt.json'
LOG="${P}/stage141_to142_finalizer.log"

test -r "/proc/${STAGE141_PID}/stat"
actual_start_ticks="$(awk '{print $22}' "/proc/${STAGE141_PID}/stat")"
test "${actual_start_ticks}" = "${EXPECTED_START_TICKS}"
tr '\0' ' ' < "/proc/${STAGE141_PID}/cmdline" | \
  grep -Fq 'run_stage141_stage135c_raw_train_dump.sh'
{
  printf 'watch_started=%s\n' "$(date --iso-8601=seconds)"
  printf 'stage141_pid=%s\n' "${STAGE141_PID}"
  printf 'stage141_start_ticks=%s\n' "${EXPECTED_START_TICKS}"
} > "${LOG}"

# Wait for the exact already-verified launcher process to exit.  This does not
# read training logs or poll the remote server over SSH.
tail --pid="${STAGE141_PID}" -f /dev/null
printf 'stage141_process_exited=%s\n' "$(date --iso-8601=seconds)" >> "${LOG}"
test -s "${RECEIPT}"
printf 'stage141_receipt_sha256=%s\n' "$(sha256sum "${RECEIPT}" | awk '{print $1}')" >> "${LOG}"
exec bash "${P}/run_stage142_same_domain_nested_blend.sh" >> "${LOG}" 2>&1
