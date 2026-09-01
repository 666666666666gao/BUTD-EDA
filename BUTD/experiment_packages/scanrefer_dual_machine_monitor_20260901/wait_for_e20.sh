#!/usr/bin/env bash
set -euo pipefail

MACHINE_LABEL="${1:?usage: wait_for_e20.sh MACHINE_LABEL}"
RUN_ROOT="${DUAL_ABLATION_RUN_ROOT:-/root/autodl-tmp/logs/butd_scanrefer_dual_machine_ablations_20260901}"
MACHINE_ROOT="${RUN_ROOT}/${MACHINE_LABEL}"
TRAIN_ROOT="${MACHINE_ROOT}/seed0"
CONTROL_ROOT="${MACHINE_ROOT}/control"
INTERVAL="${E20_WATCH_INTERVAL_SECONDS:-300}"
READY="${CONTROL_ROOT}/E20_READY"
ALERT="${CONTROL_ROOT}/E20_WATCH_ALERT"
REQUIRED_KEYS=(
  last__bbs_acc0.25_top1
  last__bbs_acc0.50_top1
  last__bbs_unique_acc0.25_top1
  last__bbs_unique_acc0.50_top1
  last__bbs_multiple_acc0.25_top1
  last__bbs_multiple_acc0.50_top1
  last__bbs_unique_count_acc0.25
  last__bbs_unique_count_acc0.50
  last__bbs_multiple_count_acc0.25
  last__bbs_multiple_count_acc0.50
)

mkdir -p "${CONTROL_ROOT}"
[ ! -e "${READY}" ] || exit 0
rm -f "${ALERT}"

while true; do
  mapfile -t milestones < <(find "${TRAIN_ROOT}" -type f -name 'eval_epoch_20.log' -print 2>/dev/null | sort)
  if [ "${#milestones[@]}" -gt 0 ]; then
    complete=1
    for key in "${REQUIRED_KEYS[@]}"; do
      if ! grep -qF "${key}:" "${milestones[0]}"; then
        complete=0
        break
      fi
    done
    if [ "${complete}" -eq 1 ]; then
      {
        printf 'timestamp=%s\n' "$(date -Is)"
        printf 'machine=%s\n' "${MACHINE_LABEL}"
        printf 'milestone=%s\n' "${milestones[0]}"
        printf 'required_metric_keys=%s\n' "${#REQUIRED_KEYS[@]}"
        sha256sum "${milestones[0]}"
      } > "${READY}.tmp"
      mv "${READY}.tmp" "${READY}"
      chmod 0444 "${READY}"
      exit 0
    fi
  fi

  if [ -f "${CONTROL_ROOT}/QUEUE_ALERT" ]; then
    printf 'queue_alert_before_e20 timestamp=%s\n' "$(date -Is)" > "${ALERT}"
    exit 1
  fi
  sleep "${INTERVAL}"
done
