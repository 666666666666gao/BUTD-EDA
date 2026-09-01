#!/usr/bin/env bash
set -euo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
STATUS="${P}/stage36_chain_status.txt"
LOG="${P}/stage36_chain.log"

cd "${R}"
echo "waiting_for_stage35 $(date --iso-8601=seconds)" > "${STATUS}"
while screen -ls 2>/dev/null | grep -q 'butd_stage35_augdump_20260828'; do
  sleep 300
done
if ! grep -q 'STAGE35_AUGMENTED_TRAIN_DUMP_PASS rows=36665' \
        "${P}/stage35_augmented_dump.log"; then
  echo "stage35_failed $(date --iso-8601=seconds)" > "${STATUS}"
  tail -n 80 "${P}/stage35_augmented_dump.log" >> "${LOG}"
  exit 363
fi
echo "stage36_running $(date --iso-8601=seconds)" > "${STATUS}"
if "${P}/run_stage36_mixed_binary50_and_locked_eval.sh" >> "${LOG}" 2>&1; then
  echo "stage37_complete $(date --iso-8601=seconds)" > "${STATUS}"
else
  code=$?
  echo "stage36_or_37_failed code=${code} $(date --iso-8601=seconds)" > "${STATUS}"
  exit "${code}"
fi
