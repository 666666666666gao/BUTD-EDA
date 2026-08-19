#!/bin/bash
set -u

REPO_ROOT="/home/gb/new butd/butd_detr-main"
QUEUE_NAME="scanrefer_ablation_retrain_20260814_v2"
QUEUE_ROOT="${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_queue"
TRAIN_ROOT="${REPO_ROOT}/logs/butd_universal_target/scanrefer_ablation_retrain_20260814_v2_from_official_init"
WATCH_LOG="${QUEUE_ROOT}/watchdog.tsv"
ALERT_FILE="${QUEUE_ROOT}/WATCHDOG_ALERT"
HISTORY_COLLECTOR="${REPO_ROOT}/tools/collect_scanrefer_ablation_validation_history.py"

cd "${REPO_ROOT}"
if [ ! -f "${WATCH_LOG}" ]; then
  printf 'time\tqueue_screen\ttrain_processes\tgpu_mem_mib\tgpu_util_pct\tdisk_avail\tlatest_train_step\tbest_receipts\terror_hits\n' > "${WATCH_LOG}"
fi

while true; do
  now=$(date -Is)
  if screen -ls 2>/dev/null | grep -q "\.${QUEUE_NAME}"; then
    screen_state=alive
  else
    screen_state=missing
  fi
  train_processes=$(pgrep -fc 'python -u train_dist_mod.py.*scanrefer_ablation_retrain_20260814_v2_from_official_init' || true)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  gpu_mem=${gpu%%,*}
  gpu_util=${gpu#*,}
  disk_avail=$(df -Pk /home | awk 'NR==2 {print $4}')
  latest_log=$(find "${QUEUE_ROOT}/launcher_logs" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  latest_step='-'
  error_hits=0
  if [ -n "${latest_log}" ] && [ -f "${latest_log}" ]; then
    latest_step=$(grep -a 'Train:' "${latest_log}" 2>/dev/null | tail -1 | sed 's/.*Train: //' | tr '\t ' '__' || true)
    [ -n "${latest_step}" ] || latest_step='-'
    error_hits=$(grep -aicE 'Traceback|RuntimeError|CUDA out of memory|Killed' "${latest_log}" 2>/dev/null || true)
    # The already-running full row predates native early stopping. Its signed
    # bridge intentionally terminates torchrun after a complete saturated
    # validation, then independently reloads/evaluates the strict-best model.
    # Suppress only that expected termination after the bridge completed.
    if [ -f "${QUEUE_ROOT}/EXPECTED_EARLY_STOP_BRIDGE_COMPLETED" ] \
        && [ "$(basename "${latest_log}")" = "02_full_sacr_rapf_qahnl.log" ]; then
      error_hits=0
    fi
  fi
  best_receipts=$(find "${TRAIN_ROOT}" -type f -name best_primary.json 2>/dev/null | wc -l)
  completed_rows=$(find "${QUEUE_ROOT}/status" -maxdepth 1 -type f -name '*.done' 2>/dev/null | wc -l)

  # Hash a retained best only after its receipt has been committed. Using the
  # receipt mtime as the trigger avoids reading a checkpoint while training is
  # still replacing it. The sidecar itself is also replaced atomically.
  while IFS= read -r receipt; do
    [ -n "${receipt}" ] || continue
    run_dir=$(dirname "${receipt}")
    checkpoint="${run_dir}/ckpt_best_primary.pth"
    checksum="${run_dir}/ckpt_best_primary.sha256"
    if [ -f "${checkpoint}" ] && { [ ! -f "${checksum}" ] || [ "${receipt}" -nt "${checksum}" ]; }; then
      checksum_tmp="${checksum}.tmp.$$"
      if sha256sum "${checkpoint}" > "${checksum_tmp}"; then
        mv "${checksum_tmp}" "${checksum}"
      else
        if [ -f "${checksum_tmp}" ]; then
          mv "${checksum_tmp}" "${checksum_tmp}.failed"
        fi
        printf '%s failed to hash %s\n' "${now}" "${checkpoint}" > "${ALERT_FILE}"
        exit 4
      fi
    fi
  done < <(find "${TRAIN_ROOT}" -type f -name best_primary.json -print 2>/dev/null)

  if ! /root/miniconda3/envs/bdetr/bin/python "${HISTORY_COLLECTOR}" \
      --train-root "${TRAIN_ROOT}" --output-dir "${QUEUE_ROOT}" \
      >> "${QUEUE_ROOT}/validation_history_collector.log" 2>&1; then
    printf '%s validation-history collection failed\n' "${now}" > "${ALERT_FILE}"
    exit 5
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${now}" "${screen_state}" "${train_processes}" "${gpu_mem:-NA}" "${gpu_util:-NA}" \
    "${disk_avail}" "${latest_step}" "${best_receipts}" "${error_hits}" >> "${WATCH_LOG}"

  if [ "${completed_rows}" -eq 7 ]; then
    exit 0
  fi
  if [ "${screen_state}" = missing ]; then
    printf '%s queue screen missing before completion\n' "${now}" > "${ALERT_FILE}"
    exit 2
  fi
  if [ "${error_hits}" -gt 0 ]; then
    printf '%s error signature found in %s\n' "${now}" "${latest_log}" > "${ALERT_FILE}"
    exit 3
  fi
  sleep 300
done
