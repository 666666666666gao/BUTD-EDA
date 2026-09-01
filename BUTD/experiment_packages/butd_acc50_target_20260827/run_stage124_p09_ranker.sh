#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
SCRIPT="${P}/train_joint_option_ranker.py"
P09="${ROOT}/stage123_stage95_e11_p09_augmented_train_geometry_dump/stage95_e11_p09_augmented_train_geometry.pt"
P09_RECEIPT="${ROOT}/stage123_stage95_e11_p09_augmented_train_geometry_dump/stage123_receipt.json"
VAL="${ROOT}/stage104_stage95_e11_calibrated_geometry_dump/stage95_e11_geometry.pt"
OUT="${ROOT}/stage124_stage95_p09_ranker"
RESULT="${OUT}/stage124_val.json"
STATUS="${P}/stage124_p09_ranker_status.txt"

EXPECTED_SCRIPT_SHA="67b0c8ea0f0baaab57ca961bc4cd01c6f6128d21fb3b82db5e670d07e293b407"
EXPECTED_P09_SHA="a87ccec208cb8a769b39491d2dffa71972a4f38854a58a37215d7fb8c2395690"
EXPECTED_VAL_SHA="5bf6a572e33acb9b3b523286d44b317920c6f3db0d76b000687a8c55d8febca0"

test ! -e "${OUT}"
test "$(sha256sum "${SCRIPT}" | awk '{print $1}')" = "${EXPECTED_SCRIPT_SHA}"
test "$(sha256sum "${P09}" | awk '{print $1}')" = "${EXPECTED_P09_SHA}"
test "$(sha256sum "${VAL}" | awk '{print $1}')" = "${EXPECTED_VAL_SHA}"
test -s "${P09_RECEIPT}"

/root/miniconda3/envs/bdetr/bin/python - "${P09}" "${P09_RECEIPT}" <<'PY'
import hashlib
import json
import sys

dump, receipt_path = sys.argv[1:]
receipt = json.load(open(receipt_path, encoding="utf-8"))
assert receipt["stage"] == "123"
assert receipt["status"] == "complete"
assert receipt["rows"] == 36665
assert receipt["scene_count"] == 562
assert receipt["detector_corruption_probability"] == 0.9
digest = hashlib.sha256()
with open(dump, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
assert digest.hexdigest() == receipt["dump_sha256"]
print("STAGE123_P09_DUMP_RECEIPT_PASS", digest.hexdigest())
PY

printf 'stage124_training %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${R}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" mixed-binary50-train \
  "${P09}" "${P09}" "${OUT}" --max-candidates 8 --num-threads 16 \
  2>&1 | tee "${P}/stage124_p09_ranker_train.log"

printf 'stage124_evaluating %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" evaluate \
  "${VAL}" "${OUT}/mixed_binary50_ranker.txt" \
  "${OUT}/locked_mixed_policy.json" "${RESULT}" \
  2>&1 | tee "${P}/stage124_p09_ranker_val.log"

/root/miniconda3/envs/bdetr/bin/python - "${RESULT}" "${OUT}" <<'PY'
import hashlib
import json
import os
import sys

result_path, out = sys.argv[1:]
result = json.load(open(result_path, encoding="utf-8"))
expected = (0.5477492637778713, 0.41533445519562473)
actual = (result["baseline"]["acc025"], result["baseline"]["acc050"])
assert all(abs(a - b) < 1e-12 for a, b in zip(actual, expected))

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

receipt = {
    "stage": "124",
    "status": "complete",
    "baseline": result["baseline"],
    "selected": result["selected"],
    "strict_goal_met_offline": bool(
        result["selected"]["acc025"] > 0.5391
        and result["selected"]["acc050"] > 0.4241
    ),
    "diagnostic_only_until_integrated_and_reloaded": True,
    "p09_dump_sha256": "a87ccec208cb8a769b39491d2dffa71972a4f38854a58a37215d7fb8c2395690",
    "result_sha256": sha256(result_path),
    "model_sha256": sha256(os.path.join(out, "mixed_binary50_ranker.txt")),
    "lock_sha256": sha256(os.path.join(out, "locked_mixed_policy.json")),
}
path = os.path.join(out, "stage124_receipt.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "${OUT}/stage124_receipt.json"
printf 'stage124_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
