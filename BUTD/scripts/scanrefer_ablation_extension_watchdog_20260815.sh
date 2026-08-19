#!/bin/bash
set -u

REPO="/home/gb/new butd/butd_detr-main"
QUEUE="${REPO}/logs/butd_universal_target/scanrefer_ablation_extension_20260815_queue"
TRAIN="${REPO}/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_from_official_init"
NAME="scanrefer_ablation_extension_20260815"
ROW5="${REPO}/scripts/ablations/scanrefer_20260814/05_no_quality_scanrefer_20260814.sh"
ROW5_ORIGINAL="${ROW5}.original_20260815"
ORIGINAL_SHA="89af902e547d11f81e82cb10b98c30243b7025ee0e31aacae490e5954135cc66"
WRAPPER_SHA="780b5a51c3bb316166e59d9dbf206af4b9604dbc59e7f06f14d03676eed6f2be"
GATE="${QUEUE}/MODULE_PRIORITY_PASS"
ALERT="${QUEUE}/WATCHDOG_ALERT"
WATCH_LOG="${QUEUE}/watchdog_v2.tsv"

mkdir -p "${QUEUE}"
if [ ! -f "${WATCH_LOG}" ]; then
  printf 'time\tqueue_screen\ttrain_processes\tgpu_mem_mib\tgpu_util_pct\tcompleted\terror_hits\ttransition_state\ttransition_ok\n' > "${WATCH_LOG}"
fi

while true; do
  now=$(date -Is)
  screen -ls 2>/dev/null | grep -q "\.${NAME}" && state=alive || state=missing
  train_count=$(pgrep -fc 'python -u train_dist_mod.py.*scanrefer_ablation_retrain_20260814_v2_from_official_init' || true)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  mem=${gpu%%,*}; util=${gpu#*,}
  completed=$(find "${QUEUE}/status" -maxdepth 1 -type f -name '*.done' 2>/dev/null | wc -l)
  latest=$(find "${QUEUE}/launcher_logs" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  errors=0
  if [ -n "${latest}" ] && [ -f "${latest}" ]; then
    errors=$(grep -aicE 'Traceback|RuntimeError|CUDA out of memory|Killed' "${latest}" || true)
  fi

  transition_ok=1
  if [ ! -f "${QUEUE}/MODULE_RUNTIME_SMOKE_PASS" ]; then
    transition_state=missing_runtime_smoke
    transition_ok=0
  elif [ ! -f "${ROW5_ORIGINAL}" ] || [ "$(sha256sum "${ROW5_ORIGINAL}" 2>/dev/null | awk '{print $1}')" != "${ORIGINAL_SHA}" ]; then
    transition_state=bad_original_backup
    transition_ok=0
  elif [ ! -f "${GATE}" ]; then
    transition_state=waiting_module_rows
    if [ ! -f "${ROW5}" ] || [ "$(sha256sum "${ROW5}" 2>/dev/null | awk '{print $1}')" != "${WRAPPER_SHA}" ]; then
      transition_state=bad_waiting_wrapper
      transition_ok=0
    fi
  else
    transition_state=module_rows_released
    if [ ! -f "${QUEUE}/status/08_sacr_only.done" ] || [ ! -f "${QUEUE}/status/09_sacr_qahnl.done" ]; then
      transition_state=gate_without_module_receipts
      transition_ok=0
    elif [ ! -f "${ROW5}" ] || [ "$(sha256sum "${ROW5}" 2>/dev/null | awk '{print $1}')" != "${ORIGINAL_SHA}" ]; then
      transition_state=original_not_restored
      transition_ok=0
    fi
  fi

  /root/miniconda3/envs/bdetr/bin/python "${REPO}/tools/render_scanrefer_ablation_paper_table.py" >> "${QUEUE}/paper_table.log" 2>&1 || {
    printf '%s paper table update failed\n' "${now}" > "${ALERT}"; exit 5;
  }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${now}" "${state}" "${train_count}" "${mem:-NA}" "${util:-NA}" \
    "${completed}" "${errors}" "${transition_state}" "${transition_ok}" >> "${WATCH_LOG}"

  if [ "${transition_ok}" -ne 1 ]; then
    printf '%s transition integrity failed: %s\n' "${now}" "${transition_state}" > "${ALERT}"
    exit 6
  fi
  if [ "${completed}" -eq 3 ] && [ -f "${QUEUE}/COMPLETION_AUDIT_PASS" ]; then exit 0; fi
  if [ "${state}" = missing ]; then
    printf '%s extension queue missing before completion\n' "${now}" > "${ALERT}"; exit 2
  fi
  if [ "${errors}" -gt 0 ]; then
    printf '%s error signature in %s\n' "${now}" "${latest}" > "${ALERT}"; exit 3
  fi
  sleep 300
done
