#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
TRAIN_OUT="${ROOT}/stage148b_tiered_qahnl_adapter_trainonly"
TRAIN_STATUS="${P}/stage148b_tiered_qahnl_adapter_status.txt"
WATCH_STATUS="${P}/stage148_watch_status.txt"
FORMAL_RUNNER="${P}/run_stage148c_tiered_qahnl_formal_reload.sh"

fail_status() {
  local rc=$?
  printf 'stage148_watch_failed rc=%s at=%s line=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" \
    > "${WATCH_STATUS}"
  exit "${rc}"
}
trap fail_status ERR

printf 'stage148_watch_waiting interval_seconds=60 started_at=%s\n' \
  "$(date --iso-8601=seconds)" > "${WATCH_STATUS}"
while true; do
  state="$(cat "${TRAIN_STATUS}" 2>/dev/null || true)"
  case "${state}" in
    stage148b_complete*) break ;;
    stage148_failed*)
      printf 'stage148_watch_training_failed observed_at=%s source=%s\n' \
        "$(date --iso-8601=seconds)" "${state}" > "${WATCH_STATUS}"
      trap - ERR
      exit 0
      ;;
  esac
  sleep 60
done

mapfile -t BEST_JSONS < <(find "${TRAIN_OUT}" -type f -name best_primary.json | sort)
if [[ "${#BEST_JSONS[@]}" != 1 ]]; then
  printf 'stage148_watch_no_unique_candidate count=%s at=%s\n' \
    "${#BEST_JSONS[@]}" "$(date --iso-8601=seconds)" > "${WATCH_STATUS}"
  trap - ERR
  exit 0
fi

if /root/miniconda3/envs/bdetr/bin/python - "${BEST_JSONS[0]}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))['constraint_values']
raise SystemExit(0 if d['overall_025']>0.5391 and d['overall_050']>0.4241 else 1)
PY
then
  printf 'stage148_watch_candidate_passed launching_formal_at=%s\n' \
    "$(date --iso-8601=seconds)" > "${WATCH_STATUS}"
  bash "${FORMAL_RUNNER}"
  grep -q '^stage148c_complete ' "${P}/stage148c_tiered_qahnl_reload_status.txt"
  printf 'stage148_watch_formal_complete at=%s\n' \
    "$(date --iso-8601=seconds)" > "${WATCH_STATUS}"
else
  printf 'stage148_watch_candidate_below_strict_goal formal_not_started at=%s\n' \
    "$(date --iso-8601=seconds)" > "${WATCH_STATUS}"
fi

trap - ERR
