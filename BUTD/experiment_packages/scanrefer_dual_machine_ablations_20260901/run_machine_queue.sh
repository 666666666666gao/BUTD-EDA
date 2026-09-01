#!/usr/bin/env bash
set -euo pipefail

MACHINE_LABEL="${1:?usage: run_machine_queue.sh LABEL ROW...}"
shift
[ "$#" -gt 0 ] || { echo "ERROR: at least one row is required" >&2; exit 2; }

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/gb/new butd/butd_detr-main}"
BASE_PACKAGE="${BASE_PACKAGE:-${PACKAGE_ROOT}/scanrefer_three_table_ablations_20260821}"
RUN_ROOT="${DUAL_ABLATION_RUN_ROOT:-/root/autodl-tmp/logs/butd_scanrefer_dual_machine_ablations_20260901}"
TRAIN_ROOT="${RUN_ROOT}/${MACHINE_LABEL}/seed0"
CONTROL_ROOT="${RUN_ROOT}/${MACHINE_LABEL}/control"
STATUS_DIR="${CONTROL_ROOT}/status"
RECEIPT_DIR="${CONTROL_ROOT}/receipts"
LOG_DIR="${CONTROL_ROOT}/launcher_logs"
PYTHON="/root/miniconda3/envs/bdetr/bin/python"

export PATH="/root/miniconda3/envs/bdetr/bin:${PATH}"
export LD_LIBRARY_PATH="/root/miniconda3/envs/bdetr/lib/python3.7/site-packages/torch/lib:/root/miniconda3/envs/bdetr/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/pointnet2"

mkdir -p "${STATUS_DIR}" "${RECEIPT_DIR}" "${LOG_DIR}" "${TRAIN_ROOT}"
exec > >(tee -a "${CONTROL_ROOT}/queue.log") 2>&1

fail() {
  printf 'DUAL_QUEUE_FAIL %s %s\n' "$(date -Is)" "$*" >&2
  printf '%s\n' "$*" > "${CONTROL_ROOT}/QUEUE_ALERT"
  exit 1
}

job_id_for_row() {
  case "$1" in
    S0) echo 11_sacr_no_target_attribute ;;
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
  local job_id="$1"
  mapfile -t receipts < <(find "${TRAIN_ROOT}/${job_id}" -mindepth 2 -maxdepth 3 -type f -name best_primary.json -print 2>/dev/null | sort)
  [ "${#receipts[@]}" -eq 1 ] || fail "expected one best_primary.json for ${job_id}, found ${#receipts[@]}"
  dirname "${receipts[0]}"
}

write_receipt() {
  local row="$1" run="$2" checkpoint="${run}/ckpt_best_primary.pth"
  [ -f "${checkpoint}" ] || fail "missing strict-best checkpoint for ${row}: ${checkpoint}"
  mapfile -t weights < <(find "${run}" -maxdepth 1 -type f -name '*.pth' -print)
  [ "${#weights[@]}" -eq 1 ] || fail "${row} retained ${#weights[@]} weights, expected exactly one"
  "${PYTHON}" - "${row}" "${run}" "${RECEIPT_DIR}/${row}.json" <<'PY'
import hashlib, json, os, sys, tempfile
row, run, output = sys.argv[1:]
with open(os.path.join(run, "best_primary.json"), "r") as f:
    best = json.load(f)
with open(os.path.join(run, "config.json"), "r") as f:
    config = json.load(f)
checkpoint = os.path.join(run, "ckpt_best_primary.pth")
digest = hashlib.sha256()
with open(checkpoint, "rb") as f:
    while True:
        block = f.read(8 * 1024 * 1024)
        if not block:
            break
        digest.update(block)
receipt = {
    "row": row,
    "run": os.path.abspath(run),
    "best_primary": best,
    "checkpoint": os.path.abspath(checkpoint),
    "checkpoint_size": os.path.getsize(checkpoint),
    "checkpoint_sha256": digest.hexdigest(),
    "protocol": {
        "rng_seed": config.get("rng_seed"),
        "batch_size": config.get("batch_size"),
        "max_epoch": config.get("max_epoch"),
        "val_freq": config.get("val_freq"),
        "lr": config.get("lr"),
        "lr_backbone": config.get("lr_backbone"),
        "lr_decay_epochs": config.get("lr_decay_epochs")
    }
}
os.makedirs(os.path.dirname(output), exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix=".receipt.", dir=os.path.dirname(output))
try:
    with os.fdopen(fd, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, output)
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)
PY
  chmod 0444 "${checkpoint}" "${RECEIPT_DIR}/${row}.json"
}

[ -x "${PYTHON}" ] || fail "missing bdetr Python: ${PYTHON}"
[ -f "${BASE_PACKAGE}/launch/run_row.sh" ] || fail "missing base ablation package: ${BASE_PACKAGE}"
bash "${BASE_PACKAGE}/validate.sh" | tee "${CONTROL_ROOT}/preflight.log"
find "${PACKAGE_ROOT}" "${BASE_PACKAGE}" -type f \
  ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
  | sort -z | xargs -0 sha256sum \
  > "${CONTROL_ROOT}/frozen_code_and_launchers.sha256"
printf '%s\n' "$@" > "${CONTROL_ROOT}/FROZEN_ORDER"
rm -f "${CONTROL_ROOT}/QUEUE_ALERT"

for row in "$@"; do
  job_id="$(job_id_for_row "${row}")" || fail "unsupported row ${row}"
  if [ -f "${STATUS_DIR}/${row}.done" ] && [ -f "${RECEIPT_DIR}/${row}.json" ]; then
    echo "SKIP_COMPLETED ${row}"
    continue
  fi
  [ ! -e "${STATUS_DIR}/${row}.started" ] || fail "${row} has an incomplete prior attempt"
  date -Is > "${STATUS_DIR}/${row}.started"
  echo "START ${row} ${job_id} $(date -Is)"
  set +e
  REPO_ROOT="${REPO_ROOT}" ABLATION_LOG_ROOT="${TRAIN_ROOT}" \
    bash "${BASE_PACKAGE}/launch/run_row.sh" "${row}" 2>&1 | tee "${LOG_DIR}/${row}.log"
  rc="${PIPESTATUS[0]}"
  set -e
  if [ "${rc}" -ne 0 ]; then
    echo "${rc}" > "${STATUS_DIR}/${row}.failed"
    fail "${row} failed with rc=${rc}"
  fi
  run="$(latest_run "${job_id}")"
  write_receipt "${row}" "${run}"
  date -Is > "${STATUS_DIR}/${row}.done"
  echo "DONE ${row} $(date -Is)"
done

echo "ALL_ASSIGNED_ROWS_COMPLETE $(date -Is)" | tee "${CONTROL_ROOT}/ALL_COMPLETE"
