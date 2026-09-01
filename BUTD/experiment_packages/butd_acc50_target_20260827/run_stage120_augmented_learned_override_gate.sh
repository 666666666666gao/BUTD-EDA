#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
WRAPPER="${P}/stage120_augmented_learned_override_gate.py"
SOURCE="${P}/stage114_learned_override_gate.py"
TRAIN="${ROOT}/stage116_stage95_e11_augmented_train_geometry_dump/stage95_e11_augmented_train_geometry.pt"
MODEL="${ROOT}/stage117_stage95_mixed_augmented_ranker/mixed_binary50_ranker.txt"
LOCK="${ROOT}/stage117_stage95_mixed_augmented_ranker/locked_mixed_policy.json"
VAL="${ROOT}/stage104_stage95_e11_calibrated_geometry_dump/stage95_e11_geometry.pt"
OUT="${ROOT}/stage120_stage117_augmented_learned_override_gate"
RECEIPT="${OUT}/stage120_receipt.json"
STATUS="${P}/stage120_augmented_learned_override_gate_status.txt"
LOG="${P}/stage120_augmented_learned_override_gate.log"

EXPECTED_WRAPPER_SHA="5bd19c0ddf5245649f728c18a59e25611a6b5fd1b0d2912e6802d7aae1ea7cfc"
EXPECTED_SOURCE_SHA="d791e55adebb4bc68be8ddbeb86c42dbe7baea8a1bb1b1ce3c3b6f6addadbdca"
EXPECTED_TRAIN_SHA="088cb6b662a889c2bb01a421c694019ae2afd29accdcc089f49559911cea85f4"
EXPECTED_MODEL_SHA="227351f70d5148add6311d386e5cc565789adeeba63b88d3c32e7e8a91e7222b"
EXPECTED_LOCK_SHA="6da9a07c9bf97b9571faadbf5cfd22bd9ecbc16af1dac2f8d4aaa10a038a20ba"
EXPECTED_VAL_SHA="5bf6a572e33acb9b3b523286d44b317920c6f3db0d76b000687a8c55d8febca0"

test ! -e "${OUT}"
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = "${EXPECTED_WRAPPER_SHA}"
test "$(sha256sum "${SOURCE}" | awk '{print $1}')" = "${EXPECTED_SOURCE_SHA}"
test "$(sha256sum "${TRAIN}" | awk '{print $1}')" = "${EXPECTED_TRAIN_SHA}"
test "$(sha256sum "${MODEL}" | awk '{print $1}')" = "${EXPECTED_MODEL_SHA}"
test "$(sha256sum "${LOCK}" | awk '{print $1}')" = "${EXPECTED_LOCK_SHA}"
test "$(sha256sum "${VAL}" | awk '{print $1}')" = "${EXPECTED_VAL_SHA}"

printf 'stage120_running %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${R}"
/root/miniconda3/envs/bdetr/bin/python "${WRAPPER}" \
  "${SOURCE}" "${P}" "${TRAIN}" "${MODEL}" "${LOCK}" "${VAL}" "${OUT}" \
  2>&1 | tee "${LOG}"

/root/miniconda3/envs/bdetr/bin/python - "${OUT}" "${RECEIPT}" <<'PY'
import hashlib
import json
import os
import sys

out, receipt_path = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

lock_path = os.path.join(out, "locked_learned_override_gate.json")
gate_path = os.path.join(out, "learned_override_gate.txt")
lock = json.load(open(lock_path, encoding="utf-8"))
assert lock["stage"] == "120"
baseline = lock["val"]["baseline"]
selected = lock["val"]["selected"]
assert abs(baseline["acc025"] - 0.5477492637778713) < 1e-12
assert abs(baseline["acc050"] - 0.41533445519562473) < 1e-12
receipt = {
    "stage": "120",
    "status": "complete",
    "diagnostic_only_until_integrated_and_reloaded": True,
    "baseline": baseline,
    "selected": selected,
    "strict_goal_met_offline": bool(
        selected["acc025"] > 0.5391 and selected["acc050"] > 0.4241
    ),
    "gate_model_sha256": sha256(gate_path),
    "lock_sha256": sha256(lock_path),
}
with open(receipt_path, "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "${RECEIPT}"
printf 'stage120_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
