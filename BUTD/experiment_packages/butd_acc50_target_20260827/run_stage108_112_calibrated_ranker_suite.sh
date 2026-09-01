#!/usr/bin/env bash
set -Eeuo pipefail

R="/home/gb/new butd/butd_detr-main"
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT="/root/autodl-tmp/logs/butd_acc50_target_20260827"
SCRIPT="${P}/train_joint_option_ranker.py"
TRAIN_DUMP="${ROOT}/stage107_stage95_e11_calibrated_train_geometry_dump/stage95_e11_calibrated_train_geometry.pt"
TRAIN_RECEIPT="${ROOT}/stage107_stage95_e11_calibrated_train_geometry_dump/stage107_receipt.json"
VAL_DUMP="${ROOT}/stage104_stage95_e11_calibrated_geometry_dump/stage95_e11_geometry.pt"
OUT="${ROOT}/stage108_112_stage95_calibrated_ranker_suite"
STATUS="${P}/stage108_112_calibrated_ranker_suite_status.txt"

ORD="${OUT}/stage108_ordinal"
BIN="${OUT}/stage109_binary50"
POINT="${OUT}/stage110_pointwise"
BLEND="${OUT}/stage111_blend"
ORD_RESULT="${OUT}/stage108_ordinal_val.json"
BIN_RESULT="${OUT}/stage109_binary50_val.json"
POINT_RESULT="${OUT}/stage110_pointwise_val.json"
BLEND_RESULT="${OUT}/stage111_blend_val.json"
SUMMARY="${OUT}/stage112_suite_summary.json"

EXPECTED_SCRIPT_SHA="67b0c8ea0f0baaab57ca961bc4cd01c6f6128d21fb3b82db5e670d07e293b407"
EXPECTED_VAL_DUMP_SHA="5bf6a572e33acb9b3b523286d44b317920c6f3db0d76b000687a8c55d8febca0"

test ! -e "${OUT}"
test "$(sha256sum "${SCRIPT}" | awk '{print $1}')" = "${EXPECTED_SCRIPT_SHA}"
test "$(sha256sum "${VAL_DUMP}" | awk '{print $1}')" = "${EXPECTED_VAL_DUMP_SHA}"
test -s "${TRAIN_DUMP}"
test -s "${TRAIN_RECEIPT}"

/root/miniconda3/envs/bdetr/bin/python - "${TRAIN_DUMP}" "${TRAIN_RECEIPT}" <<'PY'
import hashlib
import json
import sys

dump, receipt_path = sys.argv[1:]
receipt = json.load(open(receipt_path, encoding='utf-8'))
assert receipt['stage'] == '107'
assert receipt['status'] == 'complete'
assert receipt['rows'] == 36665
h = hashlib.sha256()
with open(dump, 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
assert h.hexdigest() == receipt['dump_sha256']
print('STAGE107_TRAIN_DUMP_RECEIPT_PASS', h.hexdigest())
PY

mkdir -p "${OUT}"
{
  printf 'stage=108_112\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'train_dump=%s\n' "${TRAIN_DUMP}"
  printf 'train_dump_sha256='; sha256sum "${TRAIN_DUMP}" | awk '{print $1}'
  printf 'val_dump=%s\n' "${VAL_DUMP}"
  printf 'val_dump_sha256=%s\n' "${EXPECTED_VAL_DUMP_SHA}"
  printf 'ranker_script_sha256=%s\n' "${EXPECTED_SCRIPT_SHA}"
  printf 'runner_sha256='; sha256sum "$0" | awk '{print $1}'
} > "${OUT}/launch_manifest.txt"

cd "${R}"
printf 'stage108_ordinal_train %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" self-test
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" train \
  "${TRAIN_DUMP}" "${ORD}" --max-candidates 8 --num-threads 16 \
  2>&1 | tee "${P}/stage108_calibrated_ordinal_train.log"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" evaluate \
  "${VAL_DUMP}" "${ORD}/joint_option_ranker.txt" \
  "${ORD}/locked_policy.json" "${ORD_RESULT}" \
  2>&1 | tee "${P}/stage108_calibrated_ordinal_val.log"

printf 'stage109_binary50_train %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" binary50-train \
  "${TRAIN_DUMP}" "${BIN}" --max-candidates 8 --num-threads 16 \
  2>&1 | tee "${P}/stage109_calibrated_binary50_train.log"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" evaluate \
  "${VAL_DUMP}" "${BIN}/binary50_option_ranker.txt" \
  "${BIN}/locked_binary50_policy.json" "${BIN_RESULT}" \
  2>&1 | tee "${P}/stage109_calibrated_binary50_val.log"

printf 'stage110_pointwise_train %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" pointwise-train \
  "${TRAIN_DUMP}" "${POINT}" --max-candidates 8 --num-threads 16 \
  2>&1 | tee "${P}/stage110_calibrated_pointwise_train.log"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" evaluate \
  "${VAL_DUMP}" "${POINT}/pointwise_option_model.txt" \
  "${POINT}/locked_pointwise_policy.json" "${POINT_RESULT}" \
  2>&1 | tee "${P}/stage110_calibrated_pointwise_val.log"

printf 'stage111_blend_train %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" blend-train \
  "${TRAIN_DUMP}" \
  "${ORD}/joint_option_ranker.txt" "${ORD}/locked_policy.json" \
  "${BIN}/binary50_option_ranker.txt" "${BIN}/locked_binary50_policy.json" \
  "${BLEND}" 2>&1 | tee "${P}/stage111_calibrated_blend_train.log"
/root/miniconda3/envs/bdetr/bin/python "${SCRIPT}" blend-evaluate \
  "${VAL_DUMP}" "${BLEND}/locked_blend_policy.json" \
  "${BLEND_RESULT}" 2>&1 | tee "${P}/stage111_calibrated_blend_val.log"

/root/miniconda3/envs/bdetr/bin/python - "${SUMMARY}" \
  "${ORD_RESULT}" "${BIN_RESULT}" "${POINT_RESULT}" "${BLEND_RESULT}" <<'PY'
import hashlib
import json
import os
import sys

summary_path, *paths = sys.argv[1:]
names = ('ordinal', 'binary50', 'pointwise', 'blend')
results = {
    name: json.load(open(path, encoding='utf-8'))
    for name, path in zip(names, paths)
}
expected = (0.5477492637778713, 0.41533445519562473)
for result in results.values():
    baseline = result['baseline']
    actual = (baseline['acc025'], baseline['acc050'])
    assert all(abs(a - b) < 1e-12 for a, b in zip(actual, expected)), (
        actual, expected
    )
eligible = [
    name for name, result in results.items()
    if result['selected']['acc025'] > 0.5391
]
assert eligible
best_name = max(eligible, key=lambda name: (
    results[name]['selected']['acc050'],
    results[name]['selected']['acc025'],
    results[name]['selected']['mean_iou'],
))
best = results[best_name]['selected']
summary = {
    'stage': '108_112',
    'protocol': 'predeclared_train_only_scene_split_ranker_suite',
    'baseline': {'acc025': expected[0], 'acc050': expected[1]},
    'results': {name: result['selected'] for name, result in results.items()},
    'best_eligible_name': best_name,
    'best_eligible_metrics': best,
    'strict_goal': {'acc025_gt': 0.5391, 'acc050_gt': 0.4241},
    'strict_goal_met_offline': bool(
        best['acc025'] > 0.5391 and best['acc050'] > 0.4241
    ),
    'diagnostic_only_until_integrated_and_reloaded': True,
    'artifacts': {},
}
for name, path in zip(names, paths):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    summary['artifacts'][name + '_result'] = {
        'path': os.path.abspath(path), 'sha256': h.hexdigest()
    }
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, sort_keys=True)
    f.write('\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY

chmod 0444 "${SUMMARY}" "${OUT}/launch_manifest.txt"
printf 'stage112_complete %s\n' "$(date --iso-8601=seconds)" > "${STATUS}"
