#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
SCRIPT="${P}/train_joint_option_ranker.py"
CLEAN="${ROOT}/stage107_stage95_e11_calibrated_train_geometry_dump/stage95_e11_calibrated_train_geometry.pt"
AUG="${ROOT}/stage116_stage95_e11_augmented_train_geometry_dump/stage95_e11_augmented_train_geometry.pt"
AUG_RECEIPT="${ROOT}/stage116_stage95_e11_augmented_train_geometry_dump/stage116_receipt.json"
VAL="${ROOT}/stage104_stage95_e11_calibrated_geometry_dump/stage95_e11_geometry.pt"
OUT="${ROOT}/stage117_stage95_mixed_augmented_ranker"
RESULT="${OUT}/stage117_val.json"
STATUS="${P}/stage117_mixed_augmented_ranker_status.txt"

EXPECTED_SCRIPT_SHA="67b0c8ea0f0baaab57ca961bc4cd01c6f6128d21fb3b82db5e670d07e293b407"
EXPECTED_CLEAN_SHA="f531bea63f3d24e6e948c6600c2eb4624ac43c1552a18b4ea90499e11b045258"
EXPECTED_VAL_SHA="5bf6a572e33acb9b3b523286d44b317920c6f3db0d76b000687a8c55d8febca0"

test ! -e "${OUT}"
test "$(sha256sum "${SCRIPT}" | awk '{print $1}')" = "${EXPECTED_SCRIPT_SHA}"
test "$(sha256sum "${CLEAN}" | awk '{print $1}')" = "${EXPECTED_CLEAN_SHA}"
test "$(sha256sum "${VAL}" | awk '{print $1}')" = "${EXPECTED_VAL_SHA}"
test -s "${AUG}" && test -s "${AUG_RECEIPT}"

/root/miniconda3/envs/bdetr/bin/python - "${AUG}" "${AUG_RECEIPT}" <<'PY'
import hashlib
import json
import sys

dump, receipt_path = sys.argv[1:]
receipt = json.load(open(receipt_path, encoding='utf-8'))
assert receipt['stage'] == '116'
assert receipt['status'] == 'complete'
assert receipt['rows'] == 36665
h = hashlib.sha256()
with open(dump, 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
assert h.hexdigest() == receipt['dump_sha256']
print('STAGE116_AUGMENTED_DUMP_RECEIPT_PASS', h.hexdigest())
PY

mkdir -p "$(dirname "${OUT}")"
printf 'stage117_training %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
cd "${R}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" mixed-binary50-train \
  "${CLEAN}" "${AUG}" "${OUT}" --max-candidates 8 --num-threads 16 \
  2>&1 | tee "${P}/stage117_mixed_augmented_train.log"

printf 'stage117_evaluating %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" evaluate \
  "${VAL}" "${OUT}/mixed_binary50_ranker.txt" \
  "${OUT}/locked_mixed_policy.json" "${RESULT}" \
  2>&1 | tee "${P}/stage117_mixed_augmented_val.log"

/root/miniconda3/envs/bdetr/bin/python - "${RESULT}" "${OUT}" <<'PY'
import hashlib
import json
import os
import sys

result_path, out = sys.argv[1:]
result = json.load(open(result_path, encoding='utf-8'))
expected = (0.5477492637778713, 0.41533445519562473)
actual = (result['baseline']['acc025'], result['baseline']['acc050'])
assert all(abs(a - b) < 1e-12 for a, b in zip(actual, expected))
receipt = {
    'stage': '117',
    'status': 'complete',
    'baseline': result['baseline'],
    'selected': result['selected'],
    'strict_goal_met_offline': bool(
        result['selected']['acc025'] > 0.5391
        and result['selected']['acc050'] > 0.4241
    ),
    'diagnostic_only_until_integrated_and_reloaded': True,
}
for key, path in (
    ('result_sha256', result_path),
    ('model_sha256', os.path.join(out, 'mixed_binary50_ranker.txt')),
    ('lock_sha256', os.path.join(out, 'locked_mixed_policy.json')),
):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    receipt[key] = h.hexdigest()
path = os.path.join(out, 'stage117_receipt.json')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(receipt, f, indent=2, sort_keys=True)
    f.write('\n')
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

chmod 0444 "${OUT}/stage117_receipt.json"
printf 'stage117_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
