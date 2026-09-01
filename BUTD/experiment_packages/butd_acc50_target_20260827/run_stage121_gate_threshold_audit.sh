#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
SCRIPT="${P}/stage121_gate_threshold_audit.py"
VAL="${ROOT}/stage104_stage95_e11_calibrated_geometry_dump/stage95_e11_geometry.pt"
CANDIDATE_MODEL="${ROOT}/stage117_stage95_mixed_augmented_ranker/mixed_binary50_ranker.txt"
CANDIDATE_LOCK="${ROOT}/stage117_stage95_mixed_augmented_ranker/locked_mixed_policy.json"
GATE_MODEL="${ROOT}/stage119_stage117_mixed_learned_override_gate/learned_override_gate.txt"
GATE_LOCK="${ROOT}/stage119_stage117_mixed_learned_override_gate/locked_learned_override_gate.json"
OUT="${ROOT}/stage121_stage119_gate_threshold_audit"
RESULT="${OUT}/stage121_gate_threshold_audit.json"
RECEIPT="${OUT}/stage121_receipt.json"
STATUS="${P}/stage121_gate_threshold_audit_status.txt"
LOG="${P}/stage121_gate_threshold_audit.log"

test ! -e "${OUT}"
test "$(sha256sum "${SCRIPT}" | awk '{print $1}')" = "9caa50362d2da5a5db9e1d8c1ef99929e9be2a5e58f60ffbaf13d6583d5e6e1e"
test "$(sha256sum "${VAL}" | awk '{print $1}')" = "5bf6a572e33acb9b3b523286d44b317920c6f3db0d76b000687a8c55d8febca0"
test "$(sha256sum "${CANDIDATE_MODEL}" | awk '{print $1}')" = "227351f70d5148add6311d386e5cc565789adeeba63b88d3c32e7e8a91e7222b"
test "$(sha256sum "${CANDIDATE_LOCK}" | awk '{print $1}')" = "6da9a07c9bf97b9571faadbf5cfd22bd9ecbc16af1dac2f8d4aaa10a038a20ba"
test "$(sha256sum "${GATE_MODEL}" | awk '{print $1}')" = "7793ccce0b7b73463b8b35a1809a1cd6ccf918ca0dcb9a7b2db1efb461b10c43"
test "$(sha256sum "${GATE_LOCK}" | awk '{print $1}')" = "484934bc82c22a4e8e579d0d981d0828785752127d9693c386adcc07d1ba80db"

mkdir -p "${OUT}"
printf 'stage121_running %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${R}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" \
  "${P}" "${VAL}" "${CANDIDATE_MODEL}" "${CANDIDATE_LOCK}" \
  "${GATE_MODEL}" "${GATE_LOCK}" "${RESULT}" 2>&1 | tee "${LOG}"

/root/miniconda3/envs/bdetr/bin/python - "${RESULT}" "${RECEIPT}" <<'PY'
import hashlib
import json
import sys

result_path, receipt_path = sys.argv[1:]
result = json.load(open(result_path, encoding="utf-8"))
digest = hashlib.sha256(open(result_path, "rb").read()).hexdigest()
receipt = {
    "stage": "121",
    "status": "complete",
    "diagnostic_only": True,
    "baseline": result["baseline"],
    "locked": result["locked"],
    "best_with_acc025_gt_05391": result["best_with_acc025_gt_05391"],
    "candidate_action_pool": result["candidate_action_pool"],
    "threshold_can_reach_strict_goal": result["threshold_can_reach_strict_goal"],
    "result_sha256": digest,
}
with open(receipt_path, "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "${RECEIPT}"
printf 'stage121_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
