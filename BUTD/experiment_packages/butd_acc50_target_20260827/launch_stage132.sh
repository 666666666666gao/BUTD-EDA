#!/usr/bin/env bash
set -Eeuo pipefail

P="/home/gb/new butd/butd_detr-main/experiment_packages/butd_acc50_target_20260827"
RUNNER="${P}/run_stage132_relationfree_yaw_targeted_box.sh"
STATUS="${P}/stage132_status.txt"
LOG="${P}/stage132_screen.log"

printf 'stage132_running %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
set +e
bash "${RUNNER}" > "${LOG}" 2>&1
rc=$?
set -e
if [ "${rc}" -eq 0 ]; then
  printf 'stage132_complete rc=0 %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
else
  printf 'stage132_failed rc=%s %s\n' "${rc}" "$(date --iso-8601=seconds)" > "${STATUS}"
fi
exit "${rc}"
