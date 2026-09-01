#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
SCRIPT="${P}/stage128_cross_checkpoint_safe_gate.py"
S18_TRAIN="${ROOT}/stage44_stage18_e8_train_geometry_dump/stage18_e8_train_geometry.pt"
S95_TRAIN="${ROOT}/stage107_stage95_e11_calibrated_train_geometry_dump/stage95_e11_calibrated_train_geometry.pt"
S18_VAL="${ROOT}/stage42_stage18_e8_geometry_dump/stage18_e8_geometry.pt"
S95_VAL="${ROOT}/stage104_stage95_e11_calibrated_geometry_dump/stage95_e11_geometry.pt"
OUT="${ROOT}/stage128_cross_checkpoint_safe_gate"
LOCK="${OUT}/locked_cross_checkpoint_safe_gate.json"
RECEIPT="${OUT}/stage128_receipt.json"
STATUS="${P}/stage128_cross_checkpoint_safe_gate_status.txt"
LOG="${P}/stage128_cross_checkpoint_safe_gate.log"

check_sha() {
  local path="$1" expected="$2"
  test -s "${path}"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [ "${actual}" != "${expected}" ]; then
    printf 'SHA_MISMATCH %s expected=%s actual=%s\n' "${path}" "${expected}" "${actual}" >&2
    return 1
  fi
}

test ! -e "${OUT}"
check_sha "${SCRIPT}" "c562005bf8e408c74c7e2bed1c9077fc4d23899ef33aa2f6ea7643f376f10003"
check_sha "${S18_TRAIN}" "4c46014302a5a28945ee3394c9c3b733586e9dc450695887a7f7d7dfb8161536"
check_sha "${S95_TRAIN}" "f531bea63f3d24e6e948c6600c2eb4624ac43c1552a18b4ea90499e11b045258"
check_sha "${S18_VAL}" "1bfc39fe687a39cb20a7b6e7ab61377abbf064e68550810ea7263a2f3793a0f3"
check_sha "${S95_VAL}" "5bf6a572e33acb9b3b523286d44b317920c6f3db0d76b000687a8c55d8febca0"

printf 'stage128_running %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${R}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" \
  "${P}" "${S18_TRAIN}" "${S95_TRAIN}" "${S18_VAL}" "${S95_VAL}" "${OUT}" \
  2>&1 | tee "${LOG}"

/root/miniconda3/envs/bdetr/bin/python - "${LOCK}" "${RECEIPT}" <<'PY'
import hashlib
import json
import sys

lock_path, receipt_path = sys.argv[1:]
lock = json.load(open(lock_path, encoding="utf-8"))
receipt = {
    "stage": "128",
    "status": "complete",
    "diagnostic_only_until_integrated_and_reloaded": True,
    "selected_spec": lock["selected_spec"],
    "selected_dev": lock["selected_dev"],
    "test": lock["test"],
    "val": lock["val"],
    "strict_goal_met_offline": lock["strict_goal_met_offline"],
    "gate_model_sha256": lock["gate_model_sha256"],
    "lock_sha256": hashlib.sha256(open(lock_path, "rb").read()).hexdigest(),
}
with open(receipt_path, "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "${LOCK}" "${RECEIPT}"
printf 'stage128_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
