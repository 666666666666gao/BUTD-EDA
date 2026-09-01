#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
STATUS="${P}/stage25_chain_status.txt"
LOG="${P}/stage25_chain.log"
SESSION_PATTERN='butd_stage24_traindump_20260828'

cd "${R}"
echo "waiting_for_stage24 $(date --iso-8601=seconds)" > "${STATUS}"
while screen -ls 2>/dev/null | grep -q "${SESSION_PATTERN}"; do
  sleep 300
done

if ! grep -q 'STAGE24_TRAIN_DUMP_PASS rows=36665' "${P}/stage24_train_dump.log"; then
  echo "stage24_failed $(date --iso-8601=seconds)" > "${STATUS}"
  tail -n 80 "${P}/stage24_train_dump.log" >> "${LOG}"
  exit 253
fi

echo "stage25_running $(date --iso-8601=seconds)" > "${STATUS}"
if "${P}/run_stage25_train_and_locked_eval.sh" >> "${LOG}" 2>&1; then
  echo "stage26_complete $(date --iso-8601=seconds)" > "${STATUS}"
else
  code=$?
  echo "stage25_or_26_failed code=${code} $(date --iso-8601=seconds)" > "${STATUS}"
  exit "${code}"
fi
