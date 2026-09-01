#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
SCRIPT="${P}/stage126_fixed_box_scale_policy.py"
TRAIN="${ROOT}/stage107_stage95_e11_calibrated_train_geometry_dump/stage95_e11_calibrated_train_geometry.pt"
VAL="${ROOT}/stage104_stage95_e11_calibrated_geometry_dump/stage95_e11_geometry.pt"
OUT="${ROOT}/stage126_fixed_box_scale_policy"
LOCK="${OUT}/locked_fixed_box_scale_policy.json"
RECEIPT="${OUT}/stage126_receipt.json"
STATUS="${P}/stage126_fixed_box_scale_policy_status.txt"
LOG="${P}/stage126_fixed_box_scale_policy.log"

test ! -e "${OUT}"
test "$(sha256sum "${SCRIPT}" | awk '{print $1}')" = "e6a1085913c9e95a4c69b932dfb41f90c31b18d4b2141f0d791e3e4f02287d29"
test "$(sha256sum "${TRAIN}" | awk '{print $1}')" = "f531bea63f3d24e6e948c6600c2eb4624ac43c1552a18b4ea90499e11b045258"
test "$(sha256sum "${VAL}" | awk '{print $1}')" = "5bf6a572e33acb9b3b523286d44b317920c6f3db0d76b000687a8c55d8febca0"

printf 'stage126_running %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${R}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" \
  "${P}" "${TRAIN}" "${VAL}" "${OUT}" 2>&1 | tee "${LOG}"

/root/miniconda3/envs/bdetr/bin/python - "${LOCK}" "${RECEIPT}" <<'PY'
import hashlib
import json
import sys

lock_path, receipt_path = sys.argv[1:]
lock = json.load(open(lock_path, encoding="utf-8"))
receipt = {
    "stage": "126",
    "status": "complete",
    "diagnostic_only_until_integrated_and_reloaded": True,
    "train_baseline": lock["train_baseline"],
    "selected_policy": lock["selected_policy"],
    "val": lock["val"],
    "strict_goal_met_offline": lock["strict_goal_met_offline"],
    "lock_sha256": hashlib.sha256(open(lock_path, "rb").read()).hexdigest(),
}
with open(receipt_path, "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "${LOCK}" "${RECEIPT}"
printf 'stage126_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
