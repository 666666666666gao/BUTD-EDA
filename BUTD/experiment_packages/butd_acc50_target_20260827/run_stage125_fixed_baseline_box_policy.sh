#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
SCRIPT="${P}/stage125_fixed_baseline_box_policy.py"
TRAIN="${ROOT}/stage107_stage95_e11_calibrated_train_geometry_dump/stage95_e11_calibrated_train_geometry.pt"
VAL="${ROOT}/stage104_stage95_e11_calibrated_geometry_dump/stage95_e11_geometry.pt"
OUT="${ROOT}/stage125_fixed_baseline_query_box_policy"
LOCK="${OUT}/locked_fixed_box_policy.json"
RECEIPT="${OUT}/stage125_receipt.json"
STATUS="${P}/stage125_fixed_box_policy_status.txt"
LOG="${P}/stage125_fixed_box_policy.log"

test ! -e "${OUT}"
test "$(sha256sum "${SCRIPT}" | awk '{print $1}')" = "6be2c1ef7082d36b37da6fdeb46a99a2863926e80872fc61e4d82944a8380f44"
test "$(sha256sum "${TRAIN}" | awk '{print $1}')" = "f531bea63f3d24e6e948c6600c2eb4624ac43c1552a18b4ea90499e11b045258"
test "$(sha256sum "${VAL}" | awk '{print $1}')" = "5bf6a572e33acb9b3b523286d44b317920c6f3db0d76b000687a8c55d8febca0"

printf 'stage125_running %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${R}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" \
  "${P}" "${TRAIN}" "${VAL}" "${OUT}" 2>&1 | tee "${LOG}"

/root/miniconda3/envs/bdetr/bin/python - "${LOCK}" "${RECEIPT}" <<'PY'
import hashlib
import json
import sys

lock_path, receipt_path = sys.argv[1:]
lock = json.load(open(lock_path, encoding="utf-8"))
digest = hashlib.sha256(open(lock_path, "rb").read()).hexdigest()
receipt = {
    "stage": "125",
    "status": "complete",
    "diagnostic_only_until_integrated_and_reloaded": True,
    "train_baseline": lock["train_baseline"],
    "selected_policy": lock["selected_policy"],
    "val": lock["val"],
    "strict_goal_met_offline": lock["strict_goal_met_offline"],
    "lock_sha256": digest,
}
with open(receipt_path, "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "${LOCK}" "${RECEIPT}"
printf 'stage125_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
