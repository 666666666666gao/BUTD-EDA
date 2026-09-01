#!/usr/bin/env bash
set -Eeuo pipefail

R='/home/gb/new butd/butd_detr-main'
P="${R}/experiment_packages/butd_acc50_target_20260827"
ROOT=/root/autodl-tmp/logs/butd_acc50_target_20260827
PY=/root/miniconda3/envs/bdetr/bin/python
STAGE141_DIR="${ROOT}/stage141_stage135c_e12_raw_train_geometry_dump"
STAGE141_RECEIPT="${STAGE141_DIR}/stage141_receipt.json"
TRAIN_DUMP="${STAGE141_DIR}/stage135c_e12_raw_train_geometry_with_ids.pt"
VAL_DUMP="${ROOT}/stage136c_stage135c_raw_val_geometry_dump/stage135c_e12_raw_val_geometry.pt"
STAGE31_LOCK="${ROOT}/stage31_ordinal_binary_blend/locked_blend_policy.json"
STAGE33_LOCK="${ROOT}/stage33_pointwise_ranker/locked_pointwise_policy.json"
OUTDIR="${ROOT}/stage142_stage135c_same_domain_nested_blend"
POLICY="${OUTDIR}/locked_same_domain_nested_blend_policy.json"
RESULT="${OUTDIR}/stage142_on_stage135c_eval.json"
LOG="${OUTDIR}/stage142.log"

EXPECTED_CKPT_SHA=a367318ccccedfb9fb4345b03044521f67e7cb50dbc9c089c037c9f86f98de2b
EXPECTED_VAL_SHA=6a837f903f69b0ec15f43bf0544230344352adf14303c6ea0d13a7e842825508
EXPECTED_STAGE31_LOCK_SHA=4c1be1199fe2bc62dc3e4679c4ad26af4af193db1f5dda551b8cc559620b83c9
EXPECTED_STAGE33_LOCK_SHA=da1e020bc190d9792a6df57bf83b3d8be41a7e754eb353ab350a40d11588705d
EXPECTED_SCRIPT_SHA=0a17816dc5285dee56fdaab333b3818a856367c3cbd3235ef3cbd8833c86d7ff
EXPECTED_TRAINER_SHA=67b0c8ea0f0baaab57ca961bc4cd01c6f6128d21fb3b82db5e670d07e293b407
EXPECTED_LIVE_EVALUATOR_SHA=50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86

check_sha() {
  local path="$1" expected="$2"
  test -s "${path}"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

test ! -e "${OUTDIR}"
test -s "${STAGE141_RECEIPT}"
test -s "${TRAIN_DUMP}"
check_sha "${VAL_DUMP}" "${EXPECTED_VAL_SHA}"
check_sha "${STAGE31_LOCK}" "${EXPECTED_STAGE31_LOCK_SHA}"
check_sha "${STAGE33_LOCK}" "${EXPECTED_STAGE33_LOCK_SHA}"
check_sha "${P}/stage140_train_eval_nested_blend.py" "${EXPECTED_SCRIPT_SHA}"
check_sha "${P}/train_joint_option_ranker.py" "${EXPECTED_TRAINER_SHA}"
check_sha "${R}/src/grounding_evaluator.py" "${EXPECTED_LIVE_EVALUATOR_SHA}"

TRAIN_DUMP_SHA="$(${PY} - "${STAGE141_RECEIPT}" "${TRAIN_DUMP}" "${EXPECTED_CKPT_SHA}" <<'PY'
import hashlib
import json
import os
import sys
receipt_path, dump_path, expected_checkpoint_sha = sys.argv[1:]
receipt = json.load(open(receipt_path, encoding='utf-8'))
assert receipt['stage'] == '141_stage135c_train_geometry_with_ids'
assert receipt['status'] == 'complete'
assert receipt['rows'] == 36665
assert receipt['unique_stable_keys'] == 36665
assert receipt['source_checkpoint_sha256'] == expected_checkpoint_sha
assert os.path.abspath(dump_path) == receipt['corrected_dump']
h = hashlib.sha256()
with open(dump_path, 'rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        h.update(block)
actual = h.hexdigest()
assert actual == receipt['corrected_dump_sha256']
print(actual)
PY
)"
test -n "${TRAIN_DUMP_SHA}"

mkdir -p "${OUTDIR}"
record_exit() {
  code=$?
  printf '%s\n' "${code}" > "${OUTDIR}/exit_code.txt"
}
trap record_exit EXIT

{
  printf 'stage=142_stage135c_same_domain_nested_blend\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'selection_scope=stage135c_train_scene_hash_dev_only\n'
  printf 'validation_labels_used_for_selection=false\n'
  printf 'source_checkpoint_sha256=%s\n' "${EXPECTED_CKPT_SHA}"
  printf 'train_dump=%s\n' "${TRAIN_DUMP}"
  printf 'train_dump_sha256=%s\n' "${TRAIN_DUMP_SHA}"
  printf 'stage141_receipt_sha256=%s\n' "$(sha256sum "${STAGE141_RECEIPT}" | awk '{print $1}')"
  printf 'val_dump_sha256=%s\n' "${EXPECTED_VAL_SHA}"
  printf 'script_sha256=%s\n' "${EXPECTED_SCRIPT_SHA}"
  printf 'gpu_visible=none\n'
} > "${OUTDIR}/launch_manifest.txt"

cd "${P}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
"${PY}" stage140_train_eval_nested_blend.py train \
  "${TRAIN_DUMP}" "${STAGE31_LOCK}" "${STAGE33_LOCK}" "${POLICY}" \
  2>&1 | tee "${LOG}"
test -s "${POLICY}"
sha256sum "${POLICY}" > "${OUTDIR}/locked_same_domain_nested_blend_policy.json.sha256"

"${PY}" stage140_train_eval_nested_blend.py evaluate \
  "${VAL_DUMP}" "${POLICY}" "${RESULT}" \
  2>&1 | tee -a "${LOG}"
test -s "${RESULT}"
sha256sum "${RESULT}" > "${OUTDIR}/stage142_on_stage135c_eval.json.sha256"
printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >> "${OUTDIR}/launch_manifest.txt"
