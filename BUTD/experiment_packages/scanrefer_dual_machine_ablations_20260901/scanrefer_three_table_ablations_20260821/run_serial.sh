#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/gb/new butd/butd_detr-main}"
OLD_PACKAGE="${REPO_ROOT}/experiment_packages/scanrefer_simplified_ablations_20260821"
OLD_QUEUE="${REPO_ROOT}/logs/butd_universal_target/scanrefer_simplified_ablations_20260821_queue"
OLD_TRAIN="/root/autodl-tmp/logs/butd_scanrefer_simplified_ablations_20260821/seed0"
QUEUE_ROOT="${REPO_ROOT}/logs/butd_universal_target/scanrefer_three_table_ablations_20260821_queue"
TRAIN_ROOT="${THREE_TABLE_TRAIN_ROOT:-/root/autodl-tmp/logs/butd_scanrefer_three_table_ablations_20260821/seed0}"
STATUS_DIR="${QUEUE_ROOT}/status"
LAUNCH_LOG_DIR="${QUEUE_ROOT}/launcher_logs"
RECEIPT_DIR="${QUEUE_ROOT}/result_receipts"
HANDOFF="${REPO_ROOT}/docs/EXPERIMENT_HANDOFF_MASTER.md"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"
WAIT_SECONDS="${TAKEOVER_WAIT_SECONDS:-300}"

mkdir -p "${STATUS_DIR}" "${LAUNCH_LOG_DIR}" "${RECEIPT_DIR}" "${TRAIN_ROOT}"
exec > >(tee -a "${QUEUE_ROOT}/queue.log") 2>&1

fail() {
  echo "THREE_TABLE_QUEUE_FAIL $(date -Is) $*" >&2
  printf '%s\n' "$*" > "${QUEUE_ROOT}/QUEUE_ALERT"
  exit 1
}

job_id_for_row() {
  case "$1" in
    M1) echo 08_sacr_only ;;
    M2) echo 04_sacr_rapf_no_qahnl ;;
    S0) echo 11_sacr_no_target_attribute ;;
    S1) echo 07_no_relation_anchor ;;
    S2) echo 12_sacr_no_pairwise_geometry ;;
    S3) echo 13_sacr_hard_top1_anchor ;;
    R0) echo 15_rapf_fixed_fusion_g01 ;;
    R1) echo 05_rapf_no_query_quality ;;
    R2) echo 17_rapf_no_parser_anchor_cues ;;
    R3) echo 06_rapf_no_gate_supervision ;;
    *) return 2 ;;
  esac
}

latest_run() {
  local root="$1" job_id="$2"
  mapfile -t receipts < <(find "${root}/${job_id}/scanrefer_spacy" -mindepth 2 -maxdepth 2 -type f -name best_primary.json -print 2>/dev/null | sort)
  [ "${#receipts[@]}" -eq 1 ] || fail "expected one completed best_primary.json for ${job_id}, found ${#receipts[@]}"
  dirname "${receipts[0]}"
}

record_row() {
  local row="$1" run="$2"
  "${PYTHON}" "${PACKAGE_ROOT}/record_result.py" "${row}" "${run}" \
    --handoff "${HANDOFF}" --receipt-dir "${RECEIPT_DIR}"
  touch "${STATUS_DIR}/${row}.done"
}

import_old_row() {
  local row="$1" old_row="$2" job_id run
  if [ -f "${STATUS_DIR}/${row}.done" ] && [ -f "${RECEIPT_DIR}/${row}.json" ]; then
    echo "SKIP_IMPORTED ${row}"
    return
  fi
  [ -f "${OLD_QUEUE}/status/${old_row}.done" ] || fail "old row ${old_row} is not complete"
  job_id="$(job_id_for_row "${row}")"
  run="$(latest_run "${OLD_TRAIN}" "${job_id}")"
  record_row "${row}" "${run}"
  printf '%s\n' "old_row=${old_row}" "source_run=${run}" > "${STATUS_DIR}/${row}.imported"
  echo "IMPORTED ${old_row}->${row} $(date -Is)"
}

run_one() {
  local row="$1" job_id run rc
  job_id="$(job_id_for_row "${row}")"
  if [ -f "${STATUS_DIR}/${row}.done" ] && [ -f "${RECEIPT_DIR}/${row}.json" ]; then
    echo "SKIP_COMPLETED ${row}"
    return
  fi
  [ ! -e "${STATUS_DIR}/${row}.started" ] || fail "${row} has an incomplete prior attempt; inspect before rerun"
  date -Is > "${STATUS_DIR}/${row}.started"
  echo "START ${row} ${job_id} $(date -Is)"
  set +e
  ABLATION_LOG_ROOT="${TRAIN_ROOT}" bash "${PACKAGE_ROOT}/launch/run_row.sh" "${row}" 2>&1 | tee "${LAUNCH_LOG_DIR}/${row}.log"
  rc="${PIPESTATUS[0]}"
  set -e
  if [ "${rc}" -ne 0 ]; then
    echo "${rc}" > "${STATUS_DIR}/${row}.failed"
    fail "${row} failed with rc=${rc}"
  fi
  run="$(latest_run "${TRAIN_ROOT}" "${job_id}")"
  record_row "${row}" "${run}"
  echo "DONE ${row} $(date -Is)"
}

rm -f "${QUEUE_ROOT}/QUEUE_ALERT"
bash "${PACKAGE_ROOT}/validate.sh" | tee "${QUEUE_ROOT}/preflight.log"
"${PYTHON}" "${PACKAGE_ROOT}/register_known_results.py" --receipt-dir "${RECEIPT_DIR}"
printf '%s\n' M1 M2 S1 S2 S0 S3 R1 R3 R0 R2 > "${QUEUE_ROOT}/FROZEN_ORDER"
sha256sum \
  "${REPO_ROOT}/models/sacr_head.py" "${REPO_ROOT}/models/reliability_fusion.py" \
  "${REPO_ROOT}/models/losses.py" "${REPO_ROOT}/models/bdetr.py" \
  "${REPO_ROOT}/main_utils.py" "${REPO_ROOT}/train_dist_mod.py" \
  "${PACKAGE_ROOT}"/*.sh "${PACKAGE_ROOT}"/*.py "${PACKAGE_ROOT}/launch"/*.sh \
  "${PACKAGE_ROOT}/state/plan_manifest.json" \
  > "${QUEUE_ROOT}/frozen_code_and_launchers.sha256"

echo "TAKEOVER_WAIT_BEGIN $(date -Is): waiting for reusable active S1; corrected M1/M2 receipts are supplied by the monotonic-main queue."
while [ ! -f "${OLD_QUEUE}/status/S0.done" ]; do
  if [ -f "${OLD_QUEUE}/QUEUE_ALERT" ]; then
    fail "old queue failed before reusable S0 completion: $(tr '\n' ' ' < "${OLD_QUEUE}/QUEUE_ALERT")"
  fi
  sleep "${WAIT_SECONDS}"
done

# The armed old launcher fails closed before obsolete R0.  Wait for the old
# queue process to exit and for all DataLoader children to release the GPU.
for _ in $(seq 1 60); do
  if ! pgrep -af '[t]rain_dist_mod.py' >/dev/null 2>&1; then
    break
  fi
  sleep 10
done
pgrep -af '[t]rain_dist_mod.py' >/dev/null 2>&1 && fail "training still active after old S0 completion"
screen -S scanrefer_simplified_ablations_20260821 -X quit >/dev/null 2>&1 || true

import_old_row M1 M1
import_old_row M2 M4
import_old_row S1 S0

for row in S2 S0 S3 R1 R3 R0 R2; do
  run_one "${row}"
done

rm -f "${QUEUE_ROOT}/QUEUE_ALERT"
echo "ALL_THREE_TABLE_SCANREFER_ABLATION_ROWS_COMPLETE $(date -Is)" | tee "${QUEUE_ROOT}/ALL_COMPLETE"
