#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
SCRIPT="${P}/stage118_mixed_threshold_audit.py"
DUMP="${ROOT}/stage104_stage95_e11_calibrated_geometry_dump/stage95_e11_geometry.pt"
MODEL="${ROOT}/stage117_stage95_mixed_augmented_ranker/mixed_binary50_ranker.txt"
LOCK="${ROOT}/stage117_stage95_mixed_augmented_ranker/locked_mixed_policy.json"
OUT="${ROOT}/stage118b_stage117_threshold_audit"
RESULT="${OUT}/stage118b_threshold_audit.json"
RECEIPT="${OUT}/stage118b_receipt.json"
STATUS="${P}/stage118b_stage117_threshold_audit_status.txt"

EXPECTED_SCRIPT_SHA="8851e36e26bbcab0b58fc64a8b115c336f2c8e6d02c2848e2ead5b739c5dfdb7"
EXPECTED_DUMP_SHA="5bf6a572e33acb9b3b523286d44b317920c6f3db0d76b000687a8c55d8febca0"
EXPECTED_MODEL_SHA="227351f70d5148add6311d386e5cc565789adeeba63b88d3c32e7e8a91e7222b"
EXPECTED_LOCK_SHA="6da9a07c9bf97b9571faadbf5cfd22bd9ecbc16af1dac2f8d4aaa10a038a20ba"

test ! -e "${OUT}"
test "$(sha256sum "${SCRIPT}" | awk '{print $1}')" = "${EXPECTED_SCRIPT_SHA}"
test "$(sha256sum "${DUMP}" | awk '{print $1}')" = "${EXPECTED_DUMP_SHA}"
test "$(sha256sum "${MODEL}" | awk '{print $1}')" = "${EXPECTED_MODEL_SHA}"
test "$(sha256sum "${LOCK}" | awk '{print $1}')" = "${EXPECTED_LOCK_SHA}"

mkdir -p "${OUT}"
printf 'stage118b_running %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${R}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" \
  "${P}" "${DUMP}" "${MODEL}" "${LOCK}" "${RESULT}" \
  2>&1 | tee "${P}/stage118b_stage117_threshold_audit.log"

/root/miniconda3/envs/bdetr/bin/python - "${RESULT}" "${MODEL}" "${LOCK}" "${RECEIPT}" <<'PY'
import hashlib
import json
import sys

result_path, model_path, lock_path, receipt_path = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

result = json.load(open(result_path, encoding="utf-8"))
best = result["best_with_acc025_gt_05391"]
receipt = {
    "stage": "118b",
    "status": "complete",
    "diagnostic_only": True,
    "baseline": result["baseline"],
    "locked": result["locked"],
    "best_with_acc025_gt_05391": best,
    "threshold_can_reach_strict_goal": bool(
        best["acc025"] > 0.5391 and best["acc050"] > 0.4241
    ),
    "model_sha256": sha256(model_path),
    "lock_sha256": sha256(lock_path),
    "result_sha256": sha256(result_path),
}
with open(receipt_path, "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "${RECEIPT}"
printf 'stage118b_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
