#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
SCRIPT="${P}/stage127_cross_checkpoint_box_audit.py"
S18_TRAIN="${ROOT}/stage44_stage18_e8_train_geometry_dump/stage18_e8_train_geometry.pt"
S95_TRAIN="${ROOT}/stage107_stage95_e11_calibrated_train_geometry_dump/stage95_e11_calibrated_train_geometry.pt"
S18_VAL="${ROOT}/stage42_stage18_e8_geometry_dump/stage18_e8_geometry.pt"
S95_VAL="${ROOT}/stage104_stage95_e11_calibrated_geometry_dump/stage95_e11_geometry.pt"
OUT="${ROOT}/stage127_cross_checkpoint_box_audit"
RESULT="${OUT}/stage127_cross_checkpoint_box_audit.json"
RECEIPT="${OUT}/stage127_receipt.json"
STATUS="${P}/stage127_cross_checkpoint_box_audit_status.txt"
LOG="${P}/stage127_cross_checkpoint_box_audit.log"

test ! -e "${OUT}"
test "$(sha256sum "${SCRIPT}" | awk '{print $1}')" = "49f8573d5513286e2ba4eec33c6c0e0620f7244883219ecafc678607126e0e84"
test "$(sha256sum "${S18_TRAIN}" | awk '{print $1}')" = "4c46014302a5a28945ee3394c9c3b733586e9dc450695887a7f7dfb8161536"
test "$(sha256sum "${S95_TRAIN}" | awk '{print $1}')" = "f531bea63f3d24e6e948c6600c2eb4624ac43c1552a18b4ea90499e11b045258"
test "$(sha256sum "${S18_VAL}" | awk '{print $1}')" = "1bfc39fe687a39cb20a7b6e7ab61377abbf064e68550810ea7263a2f3793a0f3"
test "$(sha256sum "${S95_VAL}" | awk '{print $1}')" = "5bf6a572e33acb9b3b523286d44b317920c6f3db0d76b000687a8c55d8febca0"

printf 'stage127_running %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${R}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" \
  "${P}" "${S18_TRAIN}" "${S95_TRAIN}" "${S18_VAL}" "${S95_VAL}" "${OUT}" \
  2>&1 | tee "${LOG}"

/root/miniconda3/envs/bdetr/bin/python - "${RESULT}" "${RECEIPT}" <<'PY'
import hashlib
import json
import sys

result_path, receipt_path = sys.argv[1:]
result = json.load(open(result_path, encoding="utf-8"))
receipt = {
    "stage": "127",
    "status": "complete",
    "diagnostic_only_until_integrated_and_reloaded": True,
    "train_selected_policy": result["train"]["selected_policy"],
    "val": result["val"],
    "strict_goal_met_offline": result["strict_goal_met_offline"],
    "result_sha256": hashlib.sha256(open(result_path, "rb").read()).hexdigest(),
}
with open(receipt_path, "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "${RECEIPT}"
printf 'stage127_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
